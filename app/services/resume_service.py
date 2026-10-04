from typing import Optional
import os
import re
import uuid
import pymupdf
from fastapi import HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.db.models import Resume, User
from app.schemas.resume import ResumeResponse, build_resume_response

MAX_RESUME_SIZE_MB = 5
MAX_RESUME_SIZE_BYTES = MAX_RESUME_SIZE_MB * 1024 * 1024
UPLOAD_DIR = os.path.join("data", "uploads")

DISALLOWED_NAME_TOKENS = {
    "resume", "cv", "curriculum", "vitae", "profile", "summary", "objective",
    "contact", "experience", "education", "skills", "projects", "certifications",
    "phone", "email", "address", "location", "github", "linkedin", "portfolio",
    "website", "page", "engineer", "developer", "scientist", "manager", "designer",
    "analyst", "architect", "intern", "specialist", "consultant", "student",
    "director", "lead", "junior", "senior", "associate", "fullstack", "frontend",
    "backend", "machine", "learning", "data", "software", "artificial", "intelligence",
    "python", "java", "tech", "technology", "about", "me", "work", "history",
    "references", "personal", "details", "information", "statement", "qualifications",
}


def _is_valid_name_token(token: str) -> bool:
    """Checks if a single word token looks like a valid name part."""
    clean = token.strip(".,'-")
    if not clean:
        return False
    if not re.match(r"^[A-Za-z]+(['-][A-Za-z]+)?$", clean):
        return False
    if clean.lower() in DISALLOWED_NAME_TOKENS:
        return False
    if len(clean) < 2 and not token.endswith("."):
        return False
    return True


def _clean_and_format_name(candidate: str) -> Optional[str]:
    """Validates candidate string as a 2 to 4-word person name and returns formatted title case."""
    if not candidate:
        return None
    if any(char.isdigit() for char in candidate):
        return None
    if any(sym in candidate for sym in ["@", "http", "www", ".com", ".org", ".net", ".io", ":", ";"]):
        return None

    candidate = candidate.strip("'\"`()[]{}")
    tokens = candidate.split()
    if not (2 <= len(tokens) <= 4):
        return None

    for t in tokens:
        if not _is_valid_name_token(t):
            return None

    formatted_tokens = [t.strip(".,").title() for t in tokens]
    return " ".join(formatted_tokens)


def extract_candidate_name(text: str) -> Optional[str]:
    """Deterministically extracts candidate's name from CV text without external LLM calls.

    Examines the top 15 non-empty lines for candidate name patterns.
    """
    if not text:
        return None

    lines = [line.strip() for line in text.split("\n") if line.strip()]
    top_lines = lines[:15]

    for raw_line in top_lines:
        name = _clean_and_format_name(raw_line)
        if name:
            return name

        for delimiter in ["|", "•", "·", " - ", " / ", ","]:
            if delimiter in raw_line:
                segments = raw_line.split(delimiter)
                for seg in segments:
                    name = _clean_and_format_name(seg.strip())
                    if name:
                        return name

    return None


def normalize_extracted_text(raw_text: str) -> str:
    """Normalizes extracted text by standardizing line breaks and cleaning up whitespace

    while preserving paragraph and bullet structure.
    """
    if not raw_text:
        return ""

    # Remove null characters
    text = raw_text.replace("\x00", "")

    # Normalize line breaks
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Clean lines while preserving paragraphs
    lines = [line.strip() for line in text.split("\n")]
    normalized_lines = []
    consecutive_empty = 0

    for line in lines:
        if not line:
            consecutive_empty += 1
            if consecutive_empty <= 1:
                normalized_lines.append("")
        else:
            consecutive_empty = 0
            normalized_lines.append(line)

    return "\n".join(normalized_lines).strip()


def validate_and_extract_pdf_text(contents: bytes, original_filename: str) -> str:
    """Validates PDF file structure and extracts readable text using PyMuPDF.

    Raises HTTPException with appropriate status codes on validation failure.
    """
    # 1. Check file size
    if len(contents) > MAX_RESUME_SIZE_BYTES:
        raise HTTPException(
            status_code=getattr(status, "HTTP_413_CONTENT_TOO_LARGE", 413),
            detail=f"File exceeds maximum allowed size of {MAX_RESUME_SIZE_MB} MB.",
        )

    # 2. Check filename extension
    if not original_filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type. Only PDF documents (.pdf) are accepted.",
        )

    # 3. Parse with PyMuPDF
    try:
        doc = pymupdf.open(stream=contents, filetype="pdf")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file is not a valid or readable PDF document.",
        ) from e

    try:
        if doc.page_count == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="The uploaded PDF contains no pages.",
            )

        extracted_parts = []
        for page in doc:
            page_text = page.get_text()
            if page_text:
                extracted_parts.append(page_text)

        full_text = "\n".join(extracted_parts)
        normalized = normalize_extracted_text(full_text)

        if not normalized:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="This PDF does not contain extractable text. OCR is not supported in the current version.",
            )

        return normalized
    finally:
        doc.close()


def upload_resume(user_id: int, file: UploadFile, db: Session) -> ResumeResponse:
    """Handles the resume upload process: validation, storage, extraction, and database persistence."""
    # 1. Verify user existence
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with id {user_id} not found.",
        )

    # 2. Validate content type header where available
    if file.content_type and file.content_type not in (
        "application/pdf",
        "application/x-pdf",
        "application/octet-stream",
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type. Only PDF documents (.pdf) are accepted.",
        )

    # 3. Read file contents safely
    try:
        contents = file.file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unable to read the uploaded file.",
        ) from e

    original_filename = os.path.basename(file.filename or "resume.pdf")

    # 4. Validate PDF and extract text
    extracted_text = validate_and_extract_pdf_text(contents, original_filename)

    # 5. Save file safely to data/uploads
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    safe_filename = f"{uuid.uuid4().hex}.pdf"
    stored_file_path = os.path.join(UPLOAD_DIR, safe_filename)

    try:
        with open(stored_file_path, "wb") as f:
            f.write(contents)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save uploaded file to storage.",
        ) from e

    # 6. Extract candidate name deterministically from extracted text
    candidate_name = extract_candidate_name(extracted_text)

    # 7. Save metadata to database with rollback/file cleanup on failure
    try:
        resume_record = Resume(
            user_id=user_id,
            file_name=original_filename,
            file_path=stored_file_path,
            extracted_text=extracted_text,
            candidate_name=candidate_name,
        )
        db.add(resume_record)
        db.commit()
        db.refresh(resume_record)
    except Exception as e:
        db.rollback()
        # Clean up saved file if database commit fails
        if os.path.exists(stored_file_path):
            try:
                os.remove(stored_file_path)
            except OSError:
                pass
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to record resume in the database.",
        ) from e

    return build_resume_response(resume_record)


def get_resume_by_id(resume_id: int, db: Session) -> ResumeResponse:
    """Retrieves resume metadata by ID without exposing filesystem paths."""
    resume = db.query(Resume).filter(Resume.id == resume_id).first()
    if not resume:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Resume with id {resume_id} not found.",
        )
    return build_resume_response(resume)
