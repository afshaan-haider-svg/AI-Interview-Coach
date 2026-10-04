"""Pydantic schemas for resume ingestion and retrieval."""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict


class ResumeResponse(BaseModel):
    """Response schema for uploaded or retrieved resume."""

    id: int
    user_id: int
    file_name: str
    extracted_text_preview: str
    character_count: int
    candidate_name: Optional[str] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


def build_resume_response(resume, preview_length: int = 300) -> ResumeResponse:
    """Helper to convert a Resume ORM model instance into a ResumeResponse schema."""
    raw_text = resume.extracted_text or ""
    preview = raw_text[:preview_length] + ("..." if len(raw_text) > preview_length else "")
    return ResumeResponse(
        id=resume.id,
        user_id=resume.user_id,
        file_name=resume.file_name,
        extracted_text_preview=preview,
        character_count=len(raw_text),
        candidate_name=getattr(resume, "candidate_name", None),
        created_at=resume.created_at,
    )
