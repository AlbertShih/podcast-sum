# 🎙️ Podcast RAG API (FastAPI + FAISS + OpenAI)

A simple RAG (Retrieval-Augmented Generation) backend service that allows you to extract transcripts from YouTube or upload a transcript file, then query it using OpenAI models.

---

## 🚀 Features

- Extract YouTube transcript using `youtube-transcript-api`
- Upload `.txt` transcript files
- Process transcripts into chunks, embed using OpenAI, store in FAISS
- Query processed content via OpenAI Chat completion (RAG style)

---

## 🛠️ Setup

### 1. Create virtual environment
```bash
python -m venv .venv
source .venv/bin/activate  
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the API locally
```bash
uvicorn main:app --reload
```
API will be available at: http://127.0.0.1:8000

