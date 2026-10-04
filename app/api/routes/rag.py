"""API endpoints for RAG indexing and semantic context retrieval."""

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.rag.retriever import retrieve_context
from app.rag.vector_store import index_job_description, index_resume
from app.schemas.rag import IndexResponse, RAGSearchRequest, RAGSearchResponse

router = APIRouter(prefix="/rag", tags=["RAG"])


@router.post(
    "/index/resume/{resume_id}",
    response_model=IndexResponse,
    status_code=status.HTTP_200_OK,
    summary="Index a resume into local vector store",
    description="Loads extracted resume text from SQLite, creates chunks, computes local embeddings, and stores in ChromaDB.",
)
def index_resume_endpoint(
    resume_id: int,
    db: Session = Depends(get_db),
) -> IndexResponse:
    """Indexes a resume by ID into ChromaDB."""
    stats = index_resume(resume_id=resume_id, db=db)
    return IndexResponse(**stats)


@router.post(
    "/index/job-description/{job_description_id}",
    response_model=IndexResponse,
    status_code=status.HTTP_200_OK,
    summary="Index a job description into local vector store",
    description="Loads job description from SQLite, chunks, computes local embeddings, and stores in ChromaDB.",
)
def index_job_description_endpoint(
    job_description_id: int,
    db: Session = Depends(get_db),
) -> IndexResponse:
    """Indexes a job description by ID into ChromaDB."""
    stats = index_job_description(job_description_id=job_description_id, db=db)
    return IndexResponse(**stats)


@router.post(
    "/search",
    response_model=RAGSearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Semantic similarity search",
    description="Embeds query locally and retrieves matching context chunks with strict user isolation.",
)
def search_rag_endpoint(
    search_req: RAGSearchRequest,
) -> RAGSearchResponse:
    """Performs semantic similarity retrieval without LLM generation."""
    chunks = retrieve_context(
        query=search_req.query,
        user_id=search_req.user_id,
        resume_id=search_req.resume_id,
        job_description_id=search_req.job_description_id,
        top_k=search_req.top_k,
    )
    return RAGSearchResponse(
        query=search_req.query,
        total_retrieved=len(chunks),
        results=chunks,
    )
