"""API endpoints for resume upload, parsing, and retrieval."""

from fastapi import APIRouter, Depends, File, Form, UploadFile, status
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.schemas.resume import ResumeResponse
from app.services import resume_service

router = APIRouter(prefix="/resumes", tags=["Resumes"])


@router.post(
    "/upload",
    response_model=ResumeResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and parse a PDF resume",
    description="Uploads a PDF resume (max 5 MB), extracts and normalizes readable text, and stores record in the database.",
)
def upload_resume(
    user_id: int = Form(..., description="ID of the user uploading the resume"),
    file: UploadFile = File(..., description="PDF file to upload"),
    db: Session = Depends(get_db),
) -> ResumeResponse:
    """Uploads a PDF resume, parses its text using PyMuPDF, and returns metadata with a preview."""
    return resume_service.upload_resume(user_id=user_id, file=file, db=db)


@router.get(
    "/{resume_id}",
    response_model=ResumeResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve resume metadata",
    description="Retrieves metadata and text preview for a stored resume by ID.",
)
def get_resume(
    resume_id: int,
    db: Session = Depends(get_db),
) -> ResumeResponse:
    """Fetches resume details and preview by ID without exposing filesystem paths."""
    return resume_service.get_resume_by_id(resume_id=resume_id, db=db)
