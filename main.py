from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel, Field
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
import traceback
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, CouldNotRetrieveTranscript
from urllib.parse import urlparse, parse_qs
from defusedxml.ElementTree import ParseError
from dotenv import load_dotenv
import yt_dlp
import ffmpeg
import tempfile
import re
from fastapi import APIRouter
from pydantic import BaseModel, Field
from openai import OpenAI



print(shutil.which("ffmpeg"))
print(shutil.which("ffprobe"))

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if openai_api_key:
    print("OPENAI_API_KEY loaded, begins with:", openai_api_key[:15])
else:
    print("OPENAI_API_KEY not found")

# Use OpenAI embeddings to save memory
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI(
    title="Podcast Summarization API with LLM/RAG",
    description="Upload a podcast transcript or YouTube URL, then ask questions using LLM and FAISS vector search.",
    version="1.0.0"
)

# Enable CORS for frontend testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://albertshih.github.io"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Storage paths
FAISS_INDEX_PATH = "faiss_index"
TRANSCRIPT_PATH = "transcript.txt"

class QARequest(BaseModel):
    question: str = Field(..., example="What is this podcast about?")

class YouTubeRequest(BaseModel):
    url: str = Field(..., example="https://www.youtube.com/watch?v=dQw4w9WgXcQ")

@app.post("/upload-transcript", summary="Upload a transcript file", description="Upload a .txt transcript file to process and build a FAISS index.")
async def upload_transcript(file: UploadFile = File(...)):
    with open(TRANSCRIPT_PATH, "wb") as f:
        shutil.copyfileobj(file.file, f)

    with open(TRANSCRIPT_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    return await process_text(text)

@app.post("/upload-youtube", summary="Upload a YouTube URL", description="Fetch transcript from YouTube video and build a FAISS index.")
async def upload_youtube(req: YouTubeRequest):
    video_id = extract_youtube_id(req.url)
    try:
        transcripts = YouTubeTranscriptApi.list_transcripts(video_id)

        print("YouTube Transcript List:")
        for t in transcripts:
            print(f"  Language: {t.language_code}, Translatable: {t.is_translatable}")
        print("End Transcript List")

        preferred_languages = ['zh-TW', 'zh', 'en', 'en-GB']
        found = False
        for lang in preferred_languages:
            try:
                transcript = transcripts.find_transcript([lang])
                print(f"Found transcript in language: {lang}")
                found = True
                break
            except:
                continue

        if not found:
            transcript = transcripts.find_translatable_transcript(['zh', 'en'])
            print(f"Using translatable transcript in language: {transcript.language_code}")

        transcript_list = transcript.fetch()

        print("Transcript Fetch Result (first 3 lines):")
        for item in transcript_list[:3]:
            print(item)
        print("End Fetch Result")

        text = " ".join([item.text for item in transcript_list])

    except Exception as e:
        print("\n--- Transcript Fetch Exception ---")
        print("Error type:", type(e).__name__)
        print("Error message:", str(e))
        print("Traceback:")
        print(traceback.format_exc())
        print("--- End Exception ---\n")
        return {"error": f"Transcript unavailable or corrupted: {str(e)}"}

    with open(TRANSCRIPT_PATH, "w", encoding="utf-8") as f:
        f.write(text)

    return await process_text(text)

def extract_youtube_id(url):
    parsed_url = urlparse(url)
    if parsed_url.hostname == 'youtu.be':
        return parsed_url.path[1:]
    if parsed_url.hostname in ('www.youtube.com', 'youtube.com'):
        if parsed_url.path == '/watch':
            return parse_qs(parsed_url.query)['v'][0]
        if parsed_url.path.startswith('/embed/'):
            return parsed_url.path.split('/')[2]
        if parsed_url.path.startswith('/v/'):
            return parsed_url.path.split('/')[2]
    raise ValueError("Invalid YouTube URL")

async def process_text(text: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = splitter.create_documents([text])

    db = FAISS.from_documents(docs, embeddings)
    db.save_local(FAISS_INDEX_PATH)

    return {"message": "Transcript processed and FAISS index created."}

@app.post("/ask", summary="Ask a question", description="Ask a question related to the uploaded transcript or YouTube video.")
async def ask_question(req: QARequest):
    db = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever()

    retrieved_docs = retriever.invoke(req.question)

    llm = ChatOpenAI(model_name="gpt-4o", openai_api_key=openai_api_key)
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
    result = qa.invoke({"query": req.question})

    print("\n--- GPT REQUEST DEBUG ---")
    print("Question:", req.question)
    print("Answer:", result['result'])
    print("Sources:", result.get('source_documents'))
    print("--- END DEBUG ---\n")

    return {"answer": result['result']}

@app.get("/documents", summary="List all stored documents", description="Returns all documents currently stored in the FAISS vector store.")
async def list_documents():
    db = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    docs = db.similarity_search("", k=1000)
    return {
        "total": len(docs),
        "documents": [doc.page_content for doc in docs]
    }

@app.post("/transcribe-youtube-whisper", summary="Transcribe YouTube video using Whisper >=1.x")
async def transcribe_youtube_whisper(req: YouTubeRequest):
    """Workflow:
    1. Download best audio with yt‑dlp.
    2. Convert to mp3 with ffmpeg.
    3. Whisper transcription (tries zh then en).
    4. Clean transcript of zero‑width / control characters.
    5. Pass cleaned text to process_text().
    """

    # -------- create temp paths --------
    temp_dir = tempfile.mkdtemp(prefix="yt_")
    raw_template = os.path.join(temp_dir, "%(id)s.%(ext)s")

    try:
        # 1. download
        print(f"[1] Downloading audio from {req.url}")
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": raw_template,
            "quiet": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(req.url, download=True)
            raw_path: str = info["requested_downloads"][0]["filepath"]
        print(f"[1] Download complete -> {raw_path}")

        # 2. convert to mp3
        mp3_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
        print(f"[2] Converting to mp3 -> {mp3_path}")
        (
            ffmpeg.input(raw_path)
            .output(mp3_path, acodec="libmp3lame", format="mp3")
            .run(overwrite_output=True, quiet=True)
        )
        print("[2] Conversion done")

        # 3. whisper
        transcript_text = None
        for lang in ("zh", "en"):
            try:
                print(f"[3] Whisper transcription try lang={lang}")
                with open(mp3_path, "rb") as f:
                    resp = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=f,
                        language=lang,
                    )
                transcript_text = resp.text
                print(f"[3] Whisper success lang={lang}")
                break
            except Exception as e:
                print(f"[3] Whisper failed lang={lang}: {e}")

        if not transcript_text:
            return {"error": "Whisper transcription failed for zh and en."}

        print(f"transcript_text=${transcript_text}")

        # 4. clean transcript
        def clean(txt: str) -> str:
            txt = re.sub(r"[\u200B-\u200F\u2028\u2029\u00AD]", "", txt)
            txt = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\u4E00-\u9FFF]+", " ", txt)
            return re.sub(r"\s+", " ", txt).strip()

        clean_text = clean(transcript_text)
        print(f"[4] Transcript cleaned; length={len(clean_text)} chars")

        # save
        with open("transcript.txt", "w", encoding="utf-8") as fp:
            fp.write(clean_text)
        print("[4] Transcript saved to transcript.txt")

        # 5. FAISS processing
        from main import process_text  # deferred import to avoid circular deps
        return await process_text(clean_text)

    except Exception as e:
        print("\n--- Whisper YouTube Exception ---")
        print("Error:", e)
        print(traceback.format_exc())
        print("--- End Exception ---\n")
        return {"error": f"Failed: {e}"}

    finally:
        print(f"[5] Removing temp dir {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)