# main.py:

# backend/main.py
import os
import json
import tempfile
import uuid
import re
import zipfile
import warnings
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from dotenv import load_dotenv

import whisper
import requests
import numpy as np
import torch
import torchaudio
import librosa
import soundfile as sf
from moviepy.editor import VideoFileClip, AudioFileClip
from scipy.signal import wiener
from sklearn.cluster import AgglomerativeClustering
import yt_dlp

from fpdf import FPDF

warnings.filterwarnings("ignore")

WHISPER_MODEL = whisper.load_model("base")

# Optional dependencies checks
try:
    from speechbrain.pretrained import SpeakerRecognition
    SPEECHBRAIN_AVAILABLE = True
except Exception:
    SPEECHBRAIN_AVAILABLE = False

try:
    import pyannote.audio
    from pyannote.audio import Pipeline
    PYANNOTE_AVAILABLE = True
except Exception:
    PYANNOTE_AVAILABLE = False

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")  # fallback to env var

app = FastAPI(title="MeetingMind API")

# Allow cross-origin requests from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # adjust if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def download_youtube_audio(url: str, target_dir: str = None) -> str:
    """
    Download highest-quality audio from a YouTube URL and return the WAV file path.
    """
    if target_dir is None:
        target_dir = tempfile.gettempdir()

    unique_id = uuid.uuid4().hex
    base_name = f"yt_audio_{unique_id}"
    out_template = os.path.join(target_dir, base_name + ".%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": out_template,
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
            "preferredquality": "192",
        }],
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)

            # The final file should be .wav after postprocessing
            wav_path = os.path.join(target_dir, base_name + ".wav")
            if os.path.exists(wav_path):
                return wav_path

            # fallback: find file manually
            if "requested_downloads" in info:
                for d in info["requested_downloads"]:
                    fp = d.get("filepath")
                    if fp and os.path.exists(fp):
                        return fp

    except Exception as e:
        print("yt-dlp download failed:", e)
        return None

    return None


def extract_audio_ffmpeg(input_path: str) -> str:
    temp_wav = os.path.join(tempfile.gettempdir(), f"audio_extracted_{os.path.basename(input_path)}.wav")
    os.system(f'ffmpeg -i "{input_path}" -vn -acodec pcm_s16le -ar 16000 -ac 1 "{temp_wav}" -y')
    return temp_wav


def extract_audio(input_path: str) -> str:
    if input_path.lower().endswith((".mp4", ".mkv", ".mov", ".webm")):
        video = VideoFileClip(input_path)
        temp_wav = os.path.join(tempfile.gettempdir(), f"audio_extracted_{os.path.basename(input_path)}.wav")
        video.audio.write_audiofile(temp_wav, codec="pcm_s16le", fps=16000, nbytes=2, logger=None)
        return temp_wav
    return input_path


def preprocess_audio(input_path: str, target_sr=16000) -> str:
    try:
        audio, sr = librosa.load(input_path, sr=None, mono=False)
        if audio is None:
            return input_path
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio)
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        audio = librosa.util.normalize(audio) * 0.1
        audio = wiener(audio)
        temp_wav = os.path.join(tempfile.gettempdir(), f"preprocessed_{os.path.basename(input_path)}.wav")
        sf.write(temp_wav, audio, target_sr)
        return temp_wav
    except Exception:
        return input_path


def assign_speakers_to_words(diarization_segments, whisper_segments):
    speaker_map = {}
    next_speaker_id = 1
    result_segments = []

    for wseg in whisper_segments:
        w_start = wseg.get('start', 0)
        w_end = wseg.get('end', 0)
        assigned_speaker = "Unknown"
        max_overlap = 0
        for dseg in diarization_segments:
            d_start = dseg['start']; d_end = dseg['end']
            overlap_start = max(w_start, d_start)
            overlap_end = min(w_end, d_end)
            overlap = max(0, overlap_end - overlap_start)
            if overlap > max_overlap:
                max_overlap = overlap
                assigned_speaker = dseg['speaker']

        if assigned_speaker != "Unknown":
            if assigned_speaker not in speaker_map:
                speaker_map[assigned_speaker] = f"Speaker {next_speaker_id}"
                next_speaker_id += 1
            display_speaker = speaker_map[assigned_speaker]
        else:
            display_speaker = "Unknown Speaker"

        result_segments.append({
            'start': w_start,
            'end': w_end,
            'text': wseg.get('text', ''),
            'speaker': display_speaker,
            'raw_speaker': assigned_speaker
        })

    return result_segments, speaker_map


def fallback_clustering_diarization(audio_path, segments):
    if not SPEECHBRAIN_AVAILABLE:
        return None
    try:
        verification = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="tmpdir_speaker"
        )
        waveform, sr = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        embeddings = []
        valid_segments = []
        for seg in segments:
            start_sample = int(seg['start'] * sr)
            end_sample = int(seg['end'] * sr)
            segment_audio = waveform[:, start_sample:end_sample]
            if segment_audio.shape[1] > sr * 0.5:
                try:
                    emb = verification.encode_batch(segment_audio)
                    embeddings.append(emb.squeeze().cpu().numpy())
                    valid_segments.append(seg)
                except Exception:
                    continue
        if len(embeddings) < 2:
            return None
        embeddings_array = np.vstack(embeddings)
        clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0.7, linkage='average')
        labels = clustering.fit_predict(embeddings_array)
        for i, seg in enumerate(valid_segments):
            seg['speaker'] = f"SPEAKER_{labels[i]:02d}"
        return valid_segments
    except Exception:
        return None


def improved_diarization_pyannote(audio_path):
    try:
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=HF_TOKEN)
        diarization = pipeline(audio_path)
        speaker_segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speaker_segments.append({'start': turn.start, 'end': turn.end, 'speaker': speaker})
        return speaker_segments
    except Exception:
        return None


def build_summary_prompt(content_type: str, transcript: str) -> str:
    """
    Clean, non-conversational HTML summary prompts.
    Used for YouTube summaries.
    """
    if content_type == "Meeting":
        return f"""
You are an expert meeting summarizer.

Produce a structured HTML meeting summary strictly based on the transcript.

<b>Meeting Summary</b>

<b>Overview:</b>
Provide a concise high-level explanation of the meeting’s purpose and discussion flow.

<b>Key Points Discussed:</b>
Summarize major topics and discussions.

<b>Decisions Made:</b>
List any explicit decisions mentioned.

<b>Action Items:</b>
List responsibilities or follow-up tasks mentioned.

<b>Next Steps:</b>
Summarize upcoming plans or intentions.

Transcript:
{transcript}
"""
    elif content_type == "Lecture":
        return f"""
You are an academic lecture summarizer.

Generate a structured HTML summary strictly based on the transcript.

<b>Lecture Summary</b>

<b>Topic Overview:</b>
Describe the main subject of the lecture.

<b>Key Concepts Explained:</b>
Summarize theories, principles, and examples described.

<b>Important Explanations:</b>
Highlight major explanations and reasoning.

<b>Conclusion:</b>
Summarize final remarks if present.

Transcript:
{transcript}
"""
    elif content_type == "Podcast":
        return f"""
You are a podcast summarizer.

Create a structured HTML summary strictly based on the transcript.

<b>Podcast Summary</b>

<b>Overview:</b>
Summarize the topic and purpose of the episode.

<b>Main Topics:</b>
List primary themes discussed.

<b>Guests/Hosts:</b>
Include only names explicitly mentioned in the transcript.

<b>Key Insights:</b>
Summarize meaningful ideas, opinions, or lessons.

<b>Notable Moments:</b>
Highlight interesting statements or impactful points.

Transcript:
{transcript}
"""
    elif content_type == "Interview":
        return f"""
You are an interview summarizer.

Produce a structured HTML summary strictly based on the transcript.

<b>Interview Summary</b>

<b>Introduction:</b>
Describe the context or opening statements.

<b>Questions & Answers:</b>
Summarize major questions and responses.

<b>Insights Shared:</b>
Highlight important viewpoints or experiences.

<b>Closing Remarks:</b>
Summarize final statements if present.

Transcript:
{transcript}
"""
    else:
        return f"""
You are an expert summarizer.

Produce a clean HTML summary based entirely on the transcript.

<b>Summary</b>

Summarize the content concisely and accurately.

Transcript:
{transcript}
"""


def groq_summarize(prompt: str):
    if not GROQ_API_KEY:
        return "❌ GROQ API key not set"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": "llama-3.1-8b-instant", "messages":[{"role":"user","content":prompt}], "temperature": 0.4}
    try:
        resp = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=30)
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"]
        return f"❌ Summarization failed: {resp.text}"
    except Exception as e:
        return f"❌ Connection error: {e}"

# ---------------------
# Pydantic models
# ---------------------
class ProcessResponse(BaseModel):
    transcript: str
    segments: List[dict]
    summary: str
    speaker_map: dict
    generated_clips: List[str] = []
    # processed_audio is the WAV used for transcription (temp file)
    processed_audio_path: Optional[str] = None
    # video_file_path is the ORIGINAL uploaded media file (useful to generate video clips)
    video_file_path: Optional[str] = None

# ---------------------
# Process Endpoint
# ---------------------
@app.post("/process", response_model=ProcessResponse)
async def process_file(
    file: UploadFile = File(...),
    content_type: str = Form("Meeting"),
    preprocess: bool = Form(True)
):
    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)

    # ----- FIX SPACES IN FILENAMES -----
    safe_name = file.filename.replace(" ", "_")
    saved_path = upload_dir / safe_name

    with open(saved_path, "wb") as f:
        f.write(await file.read())

    local_path = str(saved_path)

    if local_path.lower().endswith((".mp4", ".mkv", ".mov", ".webm", ".avi")):
        audio_path = extract_audio_ffmpeg(local_path)
    else:
        audio_path = local_path

    processed_audio = preprocess_audio(audio_path) if preprocess else audio_path

    try:
        whisper_result = WHISPER_MODEL.transcribe(
            processed_audio, verbose=False, beam_size=5, patience=1.0, temperature=0.0
        )
        transcript = whisper_result.get("text", "")
        segments = whisper_result.get("segments", [])
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": f"Whisper transcription failed: {e}"})

    diarization_segments = None
    if PYANNOTE_AVAILABLE and HF_TOKEN:
        diarization_segments = improved_diarization_pyannote(processed_audio)
    if diarization_segments is None and SPEECHBRAIN_AVAILABLE:
        diarization_segments = fallback_clustering_diarization(processed_audio, segments)
    if diarization_segments is None:
        sorted_segments = sorted(segments, key=lambda x: x.get('start', 0))
        current_speaker = 0
        last_end = 0
        speaker_segments = []
        for seg in sorted_segments:
            start = seg.get('start', 0); end = seg.get('end', 0)
            if start - last_end > 2.0:
                current_speaker = (current_speaker + 1) % 2
            seg['speaker'] = f"SPEAKER_{current_speaker:02d}"
            speaker_segments.append(seg)
            last_end = end
        diarization_segments = speaker_segments

    diarized_segments, speaker_map = assign_speakers_to_words(diarization_segments, segments)

    grouped = []
    current_speaker = None
    current_block = []

    for seg in diarized_segments:
        speaker = seg["speaker"]
        text = seg["text"].strip()
        if not text:
            continue
        if speaker != current_speaker:
            if current_block:
                grouped.append(f"{current_speaker}: " + " ".join(current_block))
            current_speaker = speaker
            current_block = [text]
        else:
            current_block.append(text)

    if current_block:
        grouped.append(f"{current_speaker}: " + " ".join(current_block))

    transcript_out = "\n\n".join(grouped)

    # PROMPT (unchanged)
    if content_type == "Meeting":
        prompt = f"""
You are an expert meeting summarizer.

Rewrite the meeting summary using clean HTML formatting.
Use <b> for section titles only.

<b>Meeting Summary</b>

<b>Overview:</b>
Give a short high-level summary of the meeting.

<b>Key Points Discussed:</b>
Write major points as clean sentences.

<b>Action Items:</b>
List responsibilities as full sentences.

<b>Next Steps:</b>
Describe follow-up steps.

Transcript:

{transcript_out}
"""
    elif content_type == "Lecture":
        prompt = f"""
You are an expert academic lecture summarizer.

Summarize the lecture using clean HTML.
Use <b> for section titles only.

<b>Lecture Summary</b>

<b>Topic Overview:</b>
Describe what the lecture is about.

<b>Key Concepts Explained:</b>
Summarize main theories, formulas, concepts, examples.

<b>Important Notes:</b>
Explain critical explanations or ideas.

<b>Conclusion:</b>
Summarize the lecturer's final message.

Transcript:

{transcript_out}
"""
    elif content_type == "Podcast":
        prompt = f"""
You are an expert podcast summarizer.

Produce a structured HTML summary of the podcast episode.
Use <b> only for section headers.

<b>Podcast Summary</b>

<b>Overview:</b>
Provide a short explanation of what this podcast episode talks about.

<b>Main Topics:</b>
List the key ideas or themes discussed in the episode with bullet points.

<b>Guest(s) & Host(s):</b>
If the transcript mentions guests or hosts, list them.
If not, clearly state that the transcript does not provide any guest/host names.

<b>Key Insights:</b>
Summarize the important takeaways, perspectives, or lessons shared.

<b>Notable Moments:</b>
Include any quotes, meaningful statements, or memorable points.

Transcript:

{transcript_out}
"""
    elif content_type == "Interview":
        prompt = f"""
You are an expert interview summarizer.

Produce a structured HTML summary of the interview.

Use <b> for section titles only.

<b>Interview Summary</b>

<b>Introduction:</b>

<b>Questions & Answers:</b>

<b>Insights:</b>

<b>Closing Remarks:</b>

Transcript:

{transcript_out}
"""
    else:
        prompt = f"""
<b>Summary</b>
Write a clean HTML summary.

Transcript:

{transcript_out}
"""

    summary = groq_summarize(prompt)

    # VERY IMPORTANT:
    # Always return ORIGINAL VIDEO PATH for clip generation
    # Never return processed_audio or temp wav
    original_media_path = local_path  # this is always the uploaded MP4/MKV/etc.

    # VERY IMPORTANT: return both the processed audio WAV and the original media path
    return ProcessResponse(
        transcript=transcript_out,
        segments=diarized_segments,
        summary=summary,
        speaker_map=speaker_map,
        generated_clips=[],
        processed_audio_path=processed_audio,   # whisper used this WAV
        video_file_path=local_path               # frontend should use this for clip generation
    )

@app.post("/process_youtube")
async def process_youtube(payload: dict):
    """
    Download audio from a YouTube URL and process it exactly like a normal file upload.
    """
    url = payload.get("url")
    content_type = payload.get("content_type", "Meeting")

    if not url:
        return JSONResponse(status_code=400, content={"detail": "YouTube URL missing"})

    # Step 1 — download audio
    audio_path = download_youtube_audio(url)
    if not audio_path:
        return JSONResponse(status_code=500, content={"detail": "Failed to download YouTube audio"})

    # Step 2 — preprocess audio (denoise/resample)
    processed_audio = preprocess_audio(audio_path)

    # Step 3 — Whisper transcription
    try:
        whisper_result = WHISPER_MODEL.transcribe(
            processed_audio,
            verbose=False,
            beam_size=5,
            patience=1.0,
            temperature=0.0
        )
        transcript_raw = whisper_result.get("text", "")
        whisper_segments = whisper_result.get("segments", [])
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": f"Whisper failed: {e}"})

    # Step 4 — Diarization
    diarization_segments = None

    if PYANNOTE_AVAILABLE and HF_TOKEN:
        diarization_segments = improved_diarization_pyannote(processed_audio)

    if diarization_segments is None and SPEECHBRAIN_AVAILABLE:
        diarization_segments = fallback_clustering_diarization(processed_audio, whisper_segments)

    # If both diarization engines are unavailable → fallback
    if diarization_segments is None:
        current_speaker = 0
        last_end = 0
        diarization_segments = []
        for seg in whisper_segments:
            start = seg.get("start", 0)
            end = seg.get("end", 0)
            text = seg.get("text", "")

            # naive speaker switch
            if start - last_end > 2:
                current_speaker = (current_speaker + 1) % 2

            diarization_segments.append({
                "start": start,
                "end": end,
                "text": text,
                "speaker": f"SPEAKER_{current_speaker:02d}",
            })
            last_end = end

    # Step 5 — assign speakers to sentence blocks
    diarized_segs, speaker_map = assign_speakers_to_words(diarization_segments, whisper_segments)

    # Step 6 — combine segments into readable transcript
    grouped = []
    curr_spk = None
    buffer = []

    for seg in diarized_segs:
        spk = seg["speaker"]
        txt = seg["text"].strip()
        if not txt:
            continue

        if spk != curr_spk:
            if buffer:
                grouped.append(f"{curr_spk}: " + " ".join(buffer))
            curr_spk = spk
            buffer = [txt]
        else:
            buffer.append(txt)

    if buffer:
        grouped.append(f"{curr_spk}: " + " ".join(buffer))

    final_transcript = "\n\n".join(grouped)

    # Step 7 — Build summary prompt (uses same logic as your /process)
    summary_prompt = build_summary_prompt(content_type, final_transcript)
    summary_html = groq_summarize(summary_prompt)

    # Step 8 — return API response
    return {
        "transcript": final_transcript,
        "segments": diarized_segs,
        "summary": summary_html,
        "speaker_map": speaker_map,
        "video_file_path": audio_path,
        "is_youtube": True
    }

# ---------------------------
# CLIP GENERATOR (SUPER-FAST FFmpeg)
# ---------------------------
class ClipsRequest(BaseModel):
    file_path: str
    segments: List[dict]
    clip_padding: int = 1
    max_clips: int = 5



def score_segment(seg):
    """Score transcript segments by importance."""

    text = seg.get("text", "").lower()
    score = 0

    # keyword weighting
    keywords = [
        "important", "key", "summary", "conclusion", "decision", "decided",
        "next step", "action item", "plan", "resolved", "issue", "problem",
        "question", "why", "how", "what", "when"
    ]
    for kw in keywords:
        if kw in text:
            score += 5

    # question indicator
    if "?" in text:
        score += 3

    # length-based importance
    score += min(len(text) / 50, 5)   # up to +5 points

    # numbers (often structured points)
    if re.search(r"\b\d+\b", text):
        score += 2

    return score



@app.post("/generate_clips")
def generate_clips(payload: ClipsRequest):
    video_path = payload.file_path
    video_p = Path(video_path)

    if not video_p.exists():
        alt = Path("uploads") / video_p.name
        if alt.exists():
            video_p = alt
        else:
            return JSONResponse(status_code=400, content={"detail": f"File not found: {video_path}"})

    # -----------------------------
    # 🔥 NEW: Select key moments
    # -----------------------------
    scored = [(score_segment(seg), seg) for seg in payload.segments]
    scored.sort(key=lambda x: x[0], reverse=True)
    key_segments = [seg for _, seg in scored[:payload.max_clips]]
    # -----------------------------

    out_dir = Path("generated_clips")
    out_dir.mkdir(exist_ok=True)

    created = []

    is_video = video_p.suffix.lower() in (".mp4", ".mkv", ".mov", ".webm", ".avi")

    for i, seg in enumerate(key_segments):
        start = float(seg.get("start", 0)) - payload.clip_padding
        end = float(seg.get("end", 0)) + payload.clip_padding

        start = max(0.0, start)
        if end <= start:
            continue

        ext = "mp4" if is_video else "mp3"
        out_path = out_dir / f"clip_{i+1}.{ext}"

        if is_video:
            cmd = (
                f'ffmpeg -y -ss {start} -to {end} -i "{video_p}" '
                f'-c:v libx264 -c:a aac -preset ultrafast "{out_path}"'
            )
        else:
            cmd = (
                f'ffmpeg -y -ss {start} -to {end} -i "{video_p}" '
                f'-vn -acodec copy "{out_path}"'
            )

        print("Running:", cmd)
        os.system(cmd)

        if out_path.exists():
            created.append(out_path.name)

    return {"created": created}

# ---------------------------------------------------
# ZIP ALL CLIPS ENDPOINT
# ---------------------------------------------------
@app.post("/zip_clips")
def zip_clips(filenames: List[str]):
    zip_path = Path("generated_clips/all_clips.zip")

    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for fname in filenames:
            fp = Path("generated_clips") / fname
            if fp.exists():
                zipf.write(fp, arcname=fname)

    return {"zip_file": "all_clips.zip"}

class PDFRequest(BaseModel):
    summary: str
    filename: Optional[str] = None

@app.post("/export_pdf")
def export_pdf(payload: PDFRequest):
    summary = payload.summary
    # pdf_name = payload.filename if payload.filename else "summary.pdf"
    # pdf_path = f"generated_clips/{pdf_name}"

    pdf_name = payload.filename if payload.filename else "summary.pdf"

    # Ensure output directory exists
    os.makedirs("generated_clips", exist_ok=True)
    pdf_path = os.path.join("generated_clips", pdf_name)

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Convert <b>text</b> → bold text and remove unsupported HTML
    def render_html(text):
        parts = re.split(r"(<b>|</b>)", text)

        bold = False
        for part in parts:
            if part == "<b>":
                bold = True
                pdf.set_font("Arial", "B", 12)
            elif part == "</b>":
                bold = False
                pdf.set_font("Arial", "", 12)
            else:
                for line in part.split("\n"):
                    pdf.multi_cell(0, 8, line)

    render_html(summary)

    pdf.output(pdf_path)
    return {"pdf_file": pdf_name}

# ---------------------------
# SERVE CLIPS (unchanged)
# ---------------------------
@app.get("/clips/{filename}")
def serve_clip(filename: str):
    p = Path("generated_clips") / filename
    if not p.exists():
        return JSONResponse(status_code=404, content={"detail": "Clip not found"})

    return FileResponse(
        p,
        media_type="application/octet-stream",
        filename=filename,
    )