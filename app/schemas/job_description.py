"""Pydantic schemas for job description creation and retrieval."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class JobDescriptionCreate(BaseModel):
    """Schema for creating a new job description."""

    user_id: int
    title: str = Field(..., min_length=1, description="Target job title")
    company: Optional[str] = Field(None, description="Hiring company name (optional)")
    description: str = Field(..., min_length=1, description="Full job description text")

    @field_validator("title", "description", mode="before")
    @classmethod
    def strip_and_validate_non_empty(cls, v: str) -> str:
        if isinstance(v, str):
            stripped = v.strip()
            if not stripped:
                raise ValueError("Field cannot be empty or contain only whitespace.")
            return stripped
        return v

    @field_validator("company", mode="before")
    @classmethod
    def strip_company(cls, v: Optional[str]) -> Optional[str]:
        if isinstance(v, str):
            stripped = v.strip()
            return stripped if stripped else None
        return v


class JobDescriptionResponse(BaseModel):
    """Response schema for job description data."""

    id: int
    user_id: int
    title: str
    company: Optional[str] = None
    description_preview: str
    character_count: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


def build_job_description_response(jd, preview_length: int = 300) -> JobDescriptionResponse:
    """Helper to convert a JobDescription ORM instance into JobDescriptionResponse."""
    raw_desc = jd.description or ""
    preview = raw_desc[:preview_length] + ("..." if len(raw_desc) > preview_length else "")
    return JobDescriptionResponse(
        id=jd.id,
        user_id=jd.user_id,
        title=jd.title,
        company=jd.company,
        description_preview=preview,
        character_count=len(raw_desc),
        created_at=jd.created_at,
    )
