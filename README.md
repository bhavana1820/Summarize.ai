
# 🎯 Summarize.AI (MeetingMind)

**Summarize.AI** is an AI-powered web application that automatically **transcribes**, **summarizes**, and **extracts highlight clips** from meetings, lectures, podcasts, interviews, and YouTube videos.
It helps users quickly understand long recordings through clean summaries, speaker-labeled transcripts, and downloadable media outputs.

---

## 🚀 Features

* 🎙️ Upload **audio/video files** or paste **YouTube links**
* 📝 Automatic **speech-to-text transcription** using OpenAI Whisper
* 🧑‍🤝‍🧑 **Speaker diarization**

  * Pyannote (best accuracy)
  * SpeechBrain fallback
  * Heuristic fallback
* ✨ **AI-generated summaries** using Groq (LLaMA 3.1)
* 🎬 **Highlight clip generation** from key moments
* 📦 Download all clips as a **ZIP**
* 📄 Export summaries as **PDF**
* 🌐 Clean **React + FastAPI** architecture

---

## 🏗️ Tech Stack

### Frontend

* React
* React Router
* HTML, CSS

### Backend

* Python
* FastAPI
* OpenAI Whisper
* Groq API (LLaMA 3.1)
* Pyannote / SpeechBrain
* FFmpeg

### Other Tools

* yt-dlp (YouTube audio extraction)
* MoviePy
* FPDF (PDF export)

---

## 📁 Project Structure

```
Summarize.AI/
│
├── backend/
│   └── main.py
│
├── frontend/
│   ├── public/
│   │   |── index.html
|   |   ├──favicon.png
│   └── src/
│       ├── App.jsx
│       ├── UploadPage.jsx
│       ├── ResultPage.jsx
│       └── style.css
│
├...            
├── generated_clips/    # highlight clips + PDFs
├── requirements.txt


---
```
## ⚙️ Requirements

Before running the project, make sure you have:

* **Python 3.9+**
* **Node.js 18+**
* **FFmpeg installed**

  * Windows: add FFmpeg to PATH
  * Linux: `sudo apt install ffmpeg`
  * Mac: `brew install ffmpeg`

---

## 🔐 Environment Variables

Create a file named **`.env`** inside the `backend/` folder.


HF_TOKEN=your_huggingface_token_here


GROQ_API_KEY=your_groq_api_key_here


> `HF_TOKEN` → required for Pyannote diarization


> `GROQ_API_KEY` → required for AI summaries



## 🧩 Installation & Setup

### 1️⃣ Clone the repository


git clone https://github.com/bhavana1820/Summarize.ai.git


cd Summarize.ai

python -m venv env

env\Scripts\activate        # Windows


install requirements.txt (See the file for steps)


### 2️⃣ Backend Setup

Run the backend:

cd backend\
uvicorn main:app --host 0.0.0.0 --port 8000

### 3️⃣ Frontend Setup

Open a new terminal:

cd frontend\
npm install\
npm start

---

## 🧠 System Workflow

1. User uploads file / YouTube link
2. Audio is extracted and preprocessed
3. Whisper performs transcription
4. Speaker diarization assigns speakers
5. Transcript is grouped by speaker
6. Groq AI generates structured summary
7. Important segments are scored
8. FFmpeg creates highlight clips
9. User downloads clips, ZIP, or PDF

---

## 🌱 Future Enhancements

* Real-time live meeting transcription
* Multi-language support
* Cloud deployment (AWS / Azure)
* User authentication
* Team-based meeting dashboards

---

## 📜 License

This project is developed for **academic and learning purposes**.\
You are free to use and modify it for educational and personal projects.

---

## 🙌 Acknowledgements

* OpenAI Whisper
* Groq AI
* Pyannote Audio
* SpeechBrain
* FastAPI & React communities

