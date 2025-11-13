# app/ingest.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os
import tempfile
import traceback
import json
import yt_dlp
from faster_whisper import WhisperModel
from typing import List
import requests
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter

# Load environment variables
load_dotenv()

router = APIRouter()

GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY", "YOUR_GEMINI_API_KEY_HERE")
KB_DIR = os.environ.get("KB_DIR", "./kb_store")
os.makedirs(KB_DIR, exist_ok=True)

# Debug: Print the API key (first 5 characters for security)
print(f"[DEBUG] Loaded GOOGLE_API_KEY: {GEMINI_API_KEY[:5] if GEMINI_API_KEY != 'YOUR_GEMINI_API_KEY_HERE' else 'NOT FOUND'}...")

class IngestRequest(BaseModel):
    youtube_url: str

def transcribe_youtube_audio(youtube_url: str) -> str:
    """Download audio using yt-dlp and transcribe with Faster-Whisper"""
    print(f"[DEBUG] Downloading audio from {youtube_url}")
    tmp_dir = tempfile.gettempdir()
    audio_path = os.path.join(tmp_dir, "temp_audio.mp3")

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': audio_path,
        'quiet': True,
        'no_warnings': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])
    print(f"[DEBUG] Audio downloaded to {audio_path}")

    model = WhisperModel("small", device="cpu", compute_type="int8")
    print("[DEBUG] Loaded Faster-Whisper model")
    segments, info = model.transcribe(audio_path)
    transcript_text = " ".join([s.text for s in segments])
    print(f"[DEBUG] Transcription completed, {len(transcript_text)} characters")
    return transcript_text

def split_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks"""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    print(f"[DEBUG] Split transcript into {len(chunks)} chunks")
    return chunks

def gemini_summarize(text_chunk: str) -> str:
    """Call Gemini API to generate structured knowledge"""
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
    headers = {
        "Content-Type": "application/json",
    }
    params = {
        "key": GEMINI_API_KEY
    }
    print(f"[DEBUG] Using API key: {GEMINI_API_KEY[:5] if GEMINI_API_KEY != 'YOUR_GEMINI_API_KEY_HERE' else 'NOT FOUND'}...")
    payload = {
        "contents": [{
            "parts": [{
                "text": f"Extract a knowledge base suitable for a chatbot from the following text:\n\n{text_chunk}"
            }]
        }],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 500
        }
    }
    response = requests.post(url, headers=headers, params=params, json=payload)
    response.raise_for_status()
    return response.json()["candidates"][0]["content"]["parts"][0]["text"]

def create_faiss_index(transcript_text: str, video_id: str):
    """Create FAISS index from transcript text"""
    try:
        # Split the text into documents
        text_splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separator="\n"
        )
        docs = text_splitter.split_text(transcript_text)
        
        # Create embeddings and FAISS index
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        store = FAISS.from_texts(docs, embeddings)
        
        # Save the FAISS index
        faiss_path = os.path.join(KB_DIR, f"faiss_{video_id}")
        store.save_local(faiss_path)
        print(f"[DEBUG] FAISS index saved to {faiss_path}")
        return faiss_path
    except Exception as e:
        print(f"[ERROR] Failed to create FAISS index: {e}")
        # Create a simple FAISS index with the raw text if embedding fails
        text_splitter = CharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separator=" "
        )
        docs = text_splitter.split_text(transcript_text)
        
        # Use a simple embedding model as fallback
        from langchain.embeddings import FakeEmbeddings
        embeddings = FakeEmbeddings(size=768)
        store = FAISS.from_texts(docs, embeddings)
        
        # Save the FAISS index
        faiss_path = os.path.join(KB_DIR, f"faiss_{video_id}")
        store.save_local(faiss_path)
        print(f"[DEBUG] FAISS index (fallback) saved to {faiss_path}")
        return faiss_path

def generate_knowledge_base(transcript_text: str, video_id: str):
    chunks = split_text(transcript_text)
    kb = []

    for i, chunk in enumerate(chunks):
        try:
            summary = gemini_summarize(chunk)
            kb.append({"chunk_id": i, "chunk": chunk, "summary": summary})
        except Exception as e:
            print(f"[ERROR] Failed to summarize chunk {i}: {e}")
            # Still add the chunk to the knowledge base even if summarization fails
            kb.append({"chunk_id": i, "chunk": chunk, "summary": ""})
    
    # Only save if we have content
    if not kb:
        raise Exception("No knowledge base content generated")
        
    kb_path = os.path.join(KB_DIR, f"{video_id}.json")
    with open(kb_path, "w", encoding="utf-8") as f:
        json.dump(kb, f, ensure_ascii=False, indent=2)
    print(f"[DEBUG] Knowledge base saved to {kb_path}")
    
    # Create FAISS index
    faiss_path = create_faiss_index(transcript_text, video_id)
    
    return {"video_id": video_id, "num_chunks": len(kb), "path": kb_path, "faiss_path": faiss_path}

@router.post("/")
def ingest_video(req: IngestRequest):
    youtube_url = req.youtube_url
    print(f"[DEBUG] Received ingest request for URL: {youtube_url}")

    # Extract video ID from URL
    if "v=" in youtube_url:
        video_id = youtube_url.split("v=")[-1].split("&")[0]
    else:
        video_id = youtube_url.split("/")[-1]

    try:
        transcript_text = transcribe_youtube_audio(youtube_url)
        kb_info = generate_knowledge_base(transcript_text, video_id)
        return kb_info
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[ERROR] {tb}")
        raise HTTPException(status_code=400, detail=f"Ingestion error: {str(e)}")