"""Job Description service for validating, creating, and retrieving job descriptions."""

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.db.models import JobDescription, User
from app.schemas.job_description import (
    JobDescriptionCreate,
    JobDescriptionResponse,
    build_job_description_response,
)


def create_job_description(
    jd_in: JobDescriptionCreate, db: Session
) -> JobDescriptionResponse:
    """Validates user existence and creates a new JobDescription in the database."""
    # 1. Verify user exists
    user = db.query(User).filter(User.id == jd_in.user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with id {jd_in.user_id} not found.",
        )

    # 2. Insert into database
    jd_record = JobDescription(
        user_id=jd_in.user_id,
        title=jd_in.title,
        company=jd_in.company,
        description=jd_in.description,
    )
    db.add(jd_record)
    db.commit()
    db.refresh(jd_record)

    return build_job_description_response(jd_record)


def get_job_description_by_id(
    jd_id: int, db: Session
) -> JobDescriptionResponse:
    """Retrieves job description metadata and preview by ID."""
    jd = db.query(JobDescription).filter(JobDescription.id == jd_id).first()
    if not jd:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job description with id {jd_id} not found.",
        )
    return build_job_description_response(jd)
