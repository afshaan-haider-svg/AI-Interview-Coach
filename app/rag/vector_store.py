"""ChromaDB vector store service for indexing and managing document chunks."""

import os
from typing import Any, Dict, Optional
import chromadb
from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import JobDescription, Resume
from app.rag.chunker import chunk_text
from app.rag.embeddings import embed_documents

_chroma_client: Optional[chromadb.PersistentClient] = None


def get_chroma_client() -> chromadb.PersistentClient:
    """Returns persistent ChromaDB client instance."""
    global _chroma_client
    if _chroma_client is None:
        os.makedirs(settings.CHROMA_PERSIST_DIR, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
    return _chroma_client


def get_collection(name: str = "interview_documents"):
    """Returns or creates the Chroma collection configured with cosine similarity."""
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


def delete_existing_chunks(source_type: str, source_id: int) -> None:
    """Deletes existing chunks for a given source to prevent stale/duplicate vectors upon re-indexing."""
    collection = get_collection()
    id_field = "resume_id" if source_type == "resume" else "job_description_id"
    try:
        collection.delete(
            where={
                "$and": [
                    {"source_type": {"$eq": source_type}},
                    {id_field: {"$eq": source_id}},
                ]
            }
        )
    except Exception:
        # Ignore if where condition finds no matching vectors
        pass


def index_resume(resume_id: int, db: Session) -> Dict[str, Any]:
    """Chunks, embeds, and indexes a resume from SQLite into the persistent Chroma vector store."""
    resume = db.query(Resume).filter(Resume.id == resume_id).first()
    if not resume:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Resume with id {resume_id} not found.",
        )

    if not resume.extracted_text or not resume.extracted_text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Resume has no extracted text to index.",
        )

    chunks = chunk_text(
        text=resume.extracted_text,
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )

    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No chunks could be produced from resume text.",
        )

    chunk_texts = [c.text for c in chunks]
    embeddings = embed_documents(chunk_texts)

    # Clean out any previous vectors for this resume before inserting
    delete_existing_chunks(source_type="resume", source_id=resume.id)

    ids = [f"resume_{resume.id}_chunk_{c.chunk_index}" for c in chunks]
    metadatas = [
        {
            "source_type": "resume",
            "user_id": resume.user_id,
            "resume_id": resume.id,
            "chunk_index": c.chunk_index,
            "file_name": resume.file_name,
        }
        for c in chunks
    ]

    collection = get_collection()
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=chunk_texts,
        metadatas=metadatas,
    )

    return {
        "source_type": "resume",
        "source_id": resume.id,
        "chunks_indexed": len(chunks),
    }


def index_job_description(job_description_id: int, db: Session) -> Dict[str, Any]:
    """Chunks, embeds, and indexes a job description from SQLite into Chroma."""
    jd = db.query(JobDescription).filter(JobDescription.id == job_description_id).first()
    if not jd:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job description with id {job_description_id} not found.",
        )

    if not jd.description or not jd.description.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Job description has no text content to index.",
        )

    chunks = chunk_text(
        text=jd.description,
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )

    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No chunks could be produced from job description.",
        )

    chunk_texts = [c.text for c in chunks]
    embeddings = embed_documents(chunk_texts)

    # Clean out any previous vectors for this JD
    delete_existing_chunks(source_type="job_description", source_id=jd.id)

    ids = [f"job_{jd.id}_chunk_{c.chunk_index}" for c in chunks]
    metadatas = [
        {
            "source_type": "job_description",
            "user_id": jd.user_id,
            "job_description_id": jd.id,
            "chunk_index": c.chunk_index,
            "title": jd.title,
        }
        for c in chunks
    ]

    collection = get_collection()
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=chunk_texts,
        metadatas=metadatas,
    )

    return {
        "source_type": "job_description",
        "source_id": jd.id,
        "chunks_indexed": len(chunks),
    }
