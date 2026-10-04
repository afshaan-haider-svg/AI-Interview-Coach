"""API endpoints for job description creation and retrieval."""

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.schemas.job_description import (
    JobDescriptionCreate,
    JobDescriptionResponse,
)
from app.services import job_description_service

router = APIRouter(prefix="/job-descriptions", tags=["Job Descriptions"])


@router.post(
    "",
    response_model=JobDescriptionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new job description",
    description="Validates and persists a job description for a specific user.",
)
def create_job_description(
    jd_in: JobDescriptionCreate,
    db: Session = Depends(get_db),
) -> JobDescriptionResponse:
    """Creates a new job description record in the database."""
    return job_description_service.create_job_description(jd_in=jd_in, db=db)


@router.get(
    "/{job_description_id}",
    response_model=JobDescriptionResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve job description metadata",
    description="Fetches job description details and preview by ID.",
)
def get_job_description(
    job_description_id: int,
    db: Session = Depends(get_db),
) -> JobDescriptionResponse:
    """Fetches a job description by ID."""
    return job_description_service.get_job_description_by_id(
        jd_id=job_description_id, db=db
    )
