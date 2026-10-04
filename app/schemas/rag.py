"""Pydantic schemas for RAG indexing and retrieval."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class IndexResponse(BaseModel):
    """Response returned when indexing a resume or job description into vector store."""

    source_type: str
    source_id: int
    chunks_indexed: int


class RAGSearchRequest(BaseModel):
    """Request payload for semantic retrieval."""

    query: str = Field(..., min_length=1, description="Natural language search query")
    user_id: int = Field(..., description="ID of the user for strict data isolation")
    resume_id: Optional[int] = Field(None, description="Optional resume ID filter")
    job_description_id: Optional[int] = Field(None, description="Optional job description ID filter")
    top_k: int = Field(4, ge=1, le=20, description="Number of contextual chunks to retrieve")


class RetrievedChunk(BaseModel):
    """Schema representing an individual retrieved context chunk."""

    text: str
    source_type: str
    source_id: int
    chunk_index: int
    metadata: Dict[str, Any]
    score: float


class RAGSearchResponse(BaseModel):
    """Response schema containing retrieved semantic context."""

    query: str
    total_retrieved: int
    results: List[RetrievedChunk]
