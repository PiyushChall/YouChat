# rag.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.embeddings import FakeEmbeddings
import os
import glob
from dotenv import load_dotenv
from typing import Optional

# Load environment variables
load_dotenv()

router = APIRouter()

class ChatRequest(BaseModel):
    question: str
    video_id: Optional[str] = None

def load_all_faiss_stores(persist_dir: str):
    """Load all available FAISS stores and merge them"""
    try:
        emb = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        # Test if we can use the embeddings
        emb.embed_query("test")
    except Exception as e:
        print(f"[WARNING] Could not use GoogleGenerativeAIEmbeddings: {e}. Using FakeEmbeddings as fallback.")
        emb = FakeEmbeddings(size=768)
    
    # Find all FAISS indexes (look for directories that start with "faiss_")
    faiss_dirs = [d for d in os.listdir(persist_dir) if d.startswith("faiss_") and os.path.isdir(os.path.join(persist_dir, d))]
    
    if not faiss_dirs:
        raise FileNotFoundError("No knowledge bases found. Run /ingest first.")
    
    # Load the first store
    first_store_path = os.path.join(persist_dir, faiss_dirs[0])
    store = FAISS.load_local(first_store_path, embeddings=emb, allow_dangerous_deserialization=True)
    
    # Merge with other stores if they exist
    for faiss_dir in faiss_dirs[1:]:
        try:
            other_store_path = os.path.join(persist_dir, faiss_dir)
            other_store = FAISS.load_local(other_store_path, embeddings=emb, allow_dangerous_deserialization=True)
            store.merge_from(other_store)
        except Exception as e:
            print(f"[WARNING] Could not merge {faiss_dir}: {e}")
    
    return store

def load_faiss_for_video(video_id: str, persist_dir: str):
    faiss_path = os.path.join(persist_dir, f"faiss_{video_id}")
    if not os.path.exists(faiss_path):
        # Check if we have the JSON knowledge base instead
        kb_path = os.path.join(persist_dir, f"{video_id}.json")
        if not os.path.exists(kb_path):
            raise FileNotFoundError("Knowledge base not found.")
        else:
            raise FileNotFoundError(f"FAISS index not found. Found JSON knowledge base at {kb_path}. The ingestion process may not have completed fully.")
    
    try:
        emb = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        # Test if we can use the embeddings
        emb.embed_query("test")
    except Exception as e:
        print(f"[WARNING] Could not use GoogleGenerativeAIEmbeddings: {e}. Using FakeEmbeddings as fallback.")
        emb = FakeEmbeddings(size=768)
    
    store = FAISS.load_local(faiss_path, embeddings=emb, allow_dangerous_deserialization=True)
    return store

def create_qa_chain(vectorstore, model_name: str = "gemini-1.5-flash"):
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.2)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are a helpful assistant. Use ONLY the following context to answer.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer concisely and accurately."
        ),
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    return qa

@router.post("/")
def chat(req: ChatRequest):
    persist_dir = os.environ.get("PERSIST_DIR", "./kb_store")
    try:
        if req.video_id:
            # Use specific video's knowledge base
            store = load_faiss_for_video(req.video_id, persist_dir)
        else:
            # Use all available knowledge bases
            store = load_all_faiss_stores(persist_dir)
        
        qa = create_qa_chain(store)
        res = qa({"query": req.question})
        return {
            "answer": res["result"],
            "sources": [d.metadata for d in res.get("source_documents", [])]
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))