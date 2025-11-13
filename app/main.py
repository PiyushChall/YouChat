# main.py
from fastapi import FastAPI
from app.ingest import router as ingest_router
from app.rag import router as rag_router
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

app = FastAPI(
    title="YouTube KB Generator",
    description="Ingest YouTube videos and generate a structured knowledge base using Gemini-2.5-flash.",
    version="1.0.0"
)

app.include_router(ingest_router, prefix="/ingest", tags=["Ingestion"])
app.include_router(rag_router, prefix="/ask", tags=["Question Answering"])

@app.get("/")
def root():
    return {"status": "ok", "message": "YouTube KB Generator API. Use /ingest to generate knowledge base."}