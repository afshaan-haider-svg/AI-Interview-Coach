"""Structured Pydantic schemas for LLM grounded analysis."""

from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field


class EvidenceItem(BaseModel):
    """An individual piece of grounded evidence linking a claim to a source."""

    claim: str = Field(..., description="Fact or observation grounded in retrieved context")
    source_type: str = Field(..., description="'resume' or 'job_description'")
    source_id: int = Field(..., description="ID of the resume or job description")

    model_config = ConfigDict(extra="ignore")


class GroundedAnalysis(BaseModel):
    """Generic structured schema for RAG-grounded LLM analysis."""

    summary: str = Field(..., description="Concise analysis grounded strictly in retrieved context")
    key_points: List[str] = Field(
        default_factory=list,
        description="Key factual takeaways supported by the context",
    )
    evidence: List[EvidenceItem] = Field(
        default_factory=list,
        description="Explicit evidence items with source references",
    )
    missing_information: List[str] = Field(
        default_factory=list,
        description="Identified gaps where context was insufficient or silent",
    )

    model_config = ConfigDict(extra="ignore")


class RetrievedSource(BaseModel):
    """Summary of a retrieved source chunk grounding the analysis."""

    source_type: str
    source_id: int
    chunk_index: int
    score: float


class GroundedAnalysisRequest(BaseModel):
    """API request payload for RAG-grounded AI analysis."""

    query: str = Field(..., min_length=1, description="Analysis question or instruction")
    user_id: int = Field(..., description="User ID for strict data isolation")
    resume_id: Optional[int] = Field(None, description="Optional resume ID filter")
    job_description_id: Optional[int] = Field(None, description="Optional job description ID filter")
    top_k: int = Field(4, ge=1, le=10, description="Max context chunks to retrieve")


class GroundedAnalysisResponse(BaseModel):
    """API response containing structured analysis and grounding sources."""

    analysis: GroundedAnalysis
    sources: List[RetrievedSource]
