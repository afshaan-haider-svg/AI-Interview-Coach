"""Automated test suite for Phase 3: Resume and Job Description Ingestion and Parsing."""

import io
import os
import sys
import unittest
import pymupdf

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import HTTPException, UploadFile, status
from pydantic import ValidationError

from app.api.routes.job_description import create_job_description, get_job_description
from app.api.routes.resume import get_resume, upload_resume
from app.db.database import SessionLocal, init_db
from app.db.models import JobDescription, Resume, User
from app.schemas.job_description import JobDescriptionCreate


class TestIngestion(unittest.TestCase):
    """Verifies resume upload, PDF text extraction, job description creation, and error handling."""

    @classmethod
    def setUpClass(cls):
        init_db()
        cls.db = SessionLocal()
        cls.created_files = []

        # Create temporary test user
        cls.test_email = "test_ingestion_user@example.com"
        existing = cls.db.query(User).filter(User.email == cls.test_email).first()
        if existing:
            cls.db.delete(existing)
            cls.db.commit()

        cls.user = User(name="Ingestion Test User", email=cls.test_email)
        cls.db.add(cls.user)
        cls.db.commit()
        cls.db.refresh(cls.user)

    @classmethod
    def tearDownClass(cls):
        # G. Cleanup: Remove temporary test DB records and test files
        try:
            # Query and delete resumes created during testing
            resumes = cls.db.query(Resume).filter(Resume.user_id == cls.user.id).all()
            for r in resumes:
                if r.file_path and os.path.exists(r.file_path):
                    try:
                        os.remove(r.file_path)
                    except OSError:
                        pass
                cls.db.delete(r)

            # Query and delete JDs
            jds = cls.db.query(JobDescription).filter(JobDescription.user_id == cls.user.id).all()
            for jd in jds:
                cls.db.delete(jd)

            # Delete any tracked created files
            for file_path in cls.created_files:
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                    except OSError:
                        pass

            # Delete test user
            test_user = cls.db.query(User).filter(User.id == cls.user.id).first()
            if test_user:
                cls.db.delete(test_user)

            cls.db.commit()
        finally:
            cls.db.close()

    def _create_test_pdf(self, text_content: str = "") -> bytes:
        """Helper to create an in-memory PDF using PyMuPDF."""
        doc = pymupdf.open()
        page = doc.new_page()
        if text_content:
            page.insert_text((50, 72), text_content)
        pdf_bytes = doc.tobytes()
        doc.close()
        return pdf_bytes

    def test_a_valid_resume_upload_and_retrieval(self):
        """A. Valid resume upload, text extraction, DB persistence, and GET retrieval."""
        resume_text = (
            "Alex Smith\n"
            "Software Engineer\n"
            "Experienced in Python, FastAPI, and Database Architecture."
        )
        pdf_bytes = self._create_test_pdf(resume_text)
        upload_file = UploadFile(
            filename="alex_smith_resume.pdf",
            file=io.BytesIO(pdf_bytes),
            headers={"content-type": "application/pdf"},
        )

        response = upload_resume(user_id=self.user.id, file=upload_file, db=self.db)

        # Verify response metadata
        self.assertIsNotNone(response.id)
        self.assertEqual(response.user_id, self.user.id)
        self.assertEqual(response.file_name, "alex_smith_resume.pdf")
        self.assertIn("Alex Smith", response.extracted_text_preview)
        self.assertIn("Python, FastAPI", response.extracted_text_preview)
        self.assertGreater(response.character_count, 0)

        # Verify DB record exists
        db_resume = self.db.query(Resume).filter(Resume.id == response.id).first()
        self.assertIsNotNone(db_resume)
        self.assertTrue(os.path.exists(db_resume.file_path))
        self.created_files.append(db_resume.file_path)

        # Verify GET /resumes/{resume_id}
        retrieved = get_resume(resume_id=response.id, db=self.db)
        self.assertEqual(retrieved.id, response.id)
        self.assertEqual(retrieved.file_name, "alex_smith_resume.pdf")
        self.assertEqual(retrieved.character_count, response.character_count)

    def test_b_invalid_resume_rejected(self):
        """B. Non-PDF files must be rejected with 400 Bad Request."""
        text_file = UploadFile(
            filename="resume.txt",
            file=io.BytesIO(b"Plain text resume content"),
            headers={"content-type": "text/plain"},
        )
        with self.assertRaises(HTTPException) as ctx:
            upload_resume(user_id=self.user.id, file=text_file, db=self.db)
        self.assertEqual(ctx.exception.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("Only PDF documents (.pdf) are accepted", ctx.exception.detail)

    def test_c_empty_or_no_text_pdf_rejected(self):
        """C. PDFs with no extractable text must be rejected with appropriate message."""
        empty_pdf_bytes = self._create_test_pdf("")
        empty_upload = UploadFile(
            filename="scanned_image.pdf",
            file=io.BytesIO(empty_pdf_bytes),
            headers={"content-type": "application/pdf"},
        )
        with self.assertRaises(HTTPException) as ctx:
            upload_resume(user_id=self.user.id, file=empty_upload, db=self.db)
        self.assertEqual(ctx.exception.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn(
            "This PDF does not contain extractable text. OCR is not supported in the current version.",
            ctx.exception.detail,
        )

    def test_d_missing_user_rejected(self):
        """D. Upload with non-existent user_id must be rejected with 404."""
        pdf_bytes = self._create_test_pdf("Sample text")
        upload_file = UploadFile(
            filename="resume.pdf",
            file=io.BytesIO(pdf_bytes),
            headers={"content-type": "application/pdf"},
        )
        non_existent_user_id = 9999999
        with self.assertRaises(HTTPException) as ctx:
            upload_resume(user_id=non_existent_user_id, file=upload_file, db=self.db)
        self.assertEqual(ctx.exception.status_code, status.HTTP_404_NOT_FOUND)

    def test_e_job_description_create_and_retrieve(self):
        """E. POST valid job description, verify DB record, and GET retrieval."""
        jd_in = JobDescriptionCreate(
            user_id=self.user.id,
            title="AI Engineer Intern",
            company="Example Company",
            description="We are looking for a Python / FastAPI / AI enthusiast to join our team.",
        )
        response = create_job_description(jd_in=jd_in, db=self.db)

        self.assertIsNotNone(response.id)
        self.assertEqual(response.user_id, self.user.id)
        self.assertEqual(response.title, "AI Engineer Intern")
        self.assertEqual(response.company, "Example Company")
        self.assertIn("Python / FastAPI", response.description_preview)
        self.assertGreater(response.character_count, 0)

        # Verify DB record
        db_jd = self.db.query(JobDescription).filter(JobDescription.id == response.id).first()
        self.assertIsNotNone(db_jd)
        self.assertEqual(db_jd.title, "AI Engineer Intern")

        # Verify GET /job-descriptions/{id}
        retrieved = get_job_description(job_description_id=response.id, db=self.db)
        self.assertEqual(retrieved.id, response.id)
        self.assertEqual(retrieved.title, "AI Engineer Intern")

    def test_f_invalid_job_description_rejected(self):
        """F. Empty title or empty description must be rejected."""
        # Empty title
        with self.assertRaises(ValidationError):
            JobDescriptionCreate(
                user_id=self.user.id,
                title="   ",
                description="Valid description",
            )

        # Empty description
        with self.assertRaises(ValidationError):
            JobDescriptionCreate(
                user_id=self.user.id,
                title="Valid Title",
                description="    ",
            )

        # Missing user for job description
        valid_jd_missing_user = JobDescriptionCreate(
            user_id=9999999,
            title="Valid Title",
            description="Valid Description",
        )
        with self.assertRaises(HTTPException) as ctx:
            create_job_description(jd_in=valid_jd_missing_user, db=self.db)
        self.assertEqual(ctx.exception.status_code, status.HTTP_404_NOT_FOUND)


if __name__ == "__main__":
    unittest.main()
