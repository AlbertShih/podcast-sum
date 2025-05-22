from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel, Field
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, CouldNotRetrieveTranscript
from urllib.parse import urlparse, parse_qs
from defusedxml.ElementTree import ParseError
from dotenv import load_dotenv

load_dotenv(override=True)
openai_api_key = os.getenv("OPENAI_API_KEY")

if openai_api_key:
    print("OPENAI_API_KEY loaded, begins with:", openai_api_key[:15])
else:
    print("OPENAI_API_KEY not found")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

app = FastAPI(
    title="Podcast Summarization API with LLM/RAG",
    description="Upload a podcast transcript or YouTube URL, then ask questions using LLM and FAISS vector search.",
    version="1.0.0"
)

# Enable CORS for frontend testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://albertshih.github.io/podcast-sum-ui"],
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
        if 'zh-TW' in [t.language_code for t in transcripts]:
            transcript = transcripts.find_transcript(['zh-TW'])
        else:
            transcript = transcripts.find_translatable_transcript(['zh'])

        transcript_list = transcript.fetch()
    except (NoTranscriptFound, TranscriptsDisabled, CouldNotRetrieveTranscript, ParseError) as e:
        return {"error": f"Transcript unavailable or corrupted: {str(e)}"}

    text = " ".join([item.text for item in transcript_list])


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
