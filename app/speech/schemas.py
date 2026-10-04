"""Pydantic schemas for local speech-to-text service."""

from typing import Optional
from pydantic import BaseModel, Field


class TranscriptionResponse(BaseModel):
    """Response payload containing recognized speech transcript."""
    transcript: str = Field(..., description="Recognized speech transcript text.")
    language: str = Field(default="en", description="Detected or configured speech language.")
    duration_seconds: Optional[float] = Field(default=None, description="Audio duration in seconds if determined.")
