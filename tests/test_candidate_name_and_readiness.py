"""Comprehensive tests for Candidate Name Extraction and Deterministic Interview Readiness Assessment.

Verifies:
1. Deterministic CV candidate name extraction (2-word, 3-word, delimited, blacklisted headers, contact rejection).
2. Deterministic Interview Readiness Assessment (exact score boundaries 0.0, 5.99, 6.0, 7.99, 8.0, 10.0).
3. Zero false hiring claims (strictly coaching & preparation readiness).
4. Track-aware headline wording and messaging.
5. Weakness-aware recommendations and deterministic tie-breaking.
6. Report response and SessionSummary schema serialization.
"""

import unittest
from app.db.database import Base, engine, SessionLocal, init_db
from app.db.models import User, Resume, InterviewSession
from app.schemas.resume import build_resume_response, ResumeResponse
from app.services.resume_service import extract_candidate_name
from app.interview.reporting.analytics import (
    compute_readiness_level,
    compute_interview_readiness,
    rank_dimensions,
    TIE_BREAK_ORDER,
    READINESS_NOT_YET_READY,
    READINESS_DEVELOPING,
    READINESS_STRONG,
    LABEL_NOT_YET_READY,
    LABEL_DEVELOPING,
    LABEL_STRONG,
)
from app.interview.reporting.service import resolve_candidate_name
from app.interview.schemas import (
    DimensionAverages,
    InterviewType,
    DifficultyLevel,
    SessionSummary,
    FinalInterviewReportResponse,
)


class TestCandidateNameExtraction(unittest.TestCase):
    """Unit tests for deterministic candidate name extraction from CV text."""

    def test_two_word_name_at_top(self):
        text = "Zain Ahmed\nAI / Machine Learning Engineer\nzain@example.com"
        name = extract_candidate_name(text)
        self.assertEqual(name, "Zain Ahmed")

    def test_three_word_name(self):
        text = "Muhammad Ali Raza\nSenior Data Scientist\nLahore, Pakistan"
        name = extract_candidate_name(text)
        self.assertEqual(name, "Muhammad Ali Raza")

    def test_name_with_delimiter_pipe(self):
        text = "Sara Malik | Software Engineer\nsara.malik@example.com"
        name = extract_candidate_name(text)
        self.assertEqual(name, "Sara Malik")

    def test_name_with_delimiter_dash(self):
        text = "Ayesha Khan - AI Researcher\nSummary of Qualifications..."
        name = extract_candidate_name(text)
        self.assertEqual(name, "Ayesha Khan")

    def test_name_with_delimiter_comma(self):
        text = "Bilal Tariq, Python Developer\nContact: 03001234567"
        name = extract_candidate_name(text)
        self.assertEqual(name, "Bilal Tariq")

    def test_skip_curriculum_vitae_header(self):
        text = "CURRICULUM VITAE\n\nUsman Ghani\nCloud Architect"
        name = extract_candidate_name(text)
        self.assertEqual(name, "Usman Ghani")

    def test_skip_resume_header(self):
        text = "Resume\nFatima Noor\nFullstack Engineer"
        name = extract_candidate_name(text)
        self.assertEqual(name, "Fatima Noor")

    def test_reject_job_title_only(self):
        text = "Senior Software Engineer\nAI / Machine Learning Specialist\nPython Developer"
        name = extract_candidate_name(text)
        self.assertIsNone(name)

    def test_reject_contact_info_lines(self):
        text = "email@domain.com | +92 300 1234567\nhttps://github.com/developer\nLinkedIn: linkedin.com/in/dev"
        name = extract_candidate_name(text)
        self.assertIsNone(name)

    def test_empty_or_whitespace_text(self):
        self.assertIsNone(extract_candidate_name(""))
        self.assertIsNone(extract_candidate_name("   \n\n  \t  "))

    def test_hyphenated_name(self):
        text = "Mary-Jane Watson\nQA Automation Engineer"
        name = extract_candidate_name(text)
        self.assertEqual(name, "Mary-Jane Watson")


class TestInterviewReadinessAssessment(unittest.TestCase):
    """Unit tests for deterministic interview readiness scoring and recommendations."""

    def test_exact_boundary_not_yet_ready_lower(self):
        code, label = compute_readiness_level(0.0)
        self.assertEqual(code, READINESS_NOT_YET_READY)
        self.assertEqual(label, LABEL_NOT_YET_READY)

    def test_exact_boundary_not_yet_ready_upper(self):
        code, label = compute_readiness_level(5.99)
        self.assertEqual(code, READINESS_NOT_YET_READY)
        self.assertEqual(label, LABEL_NOT_YET_READY)

    def test_exact_boundary_developing_lower(self):
        code, label = compute_readiness_level(6.0)
        self.assertEqual(code, READINESS_DEVELOPING)
        self.assertEqual(label, LABEL_DEVELOPING)

    def test_exact_boundary_developing_upper(self):
        code, label = compute_readiness_level(7.99)
        self.assertEqual(code, READINESS_DEVELOPING)
        self.assertEqual(label, LABEL_DEVELOPING)

    def test_exact_boundary_strong_readiness_lower(self):
        code, label = compute_readiness_level(8.0)
        self.assertEqual(code, READINESS_STRONG)
        self.assertEqual(label, LABEL_STRONG)

    def test_exact_boundary_strong_readiness_upper(self):
        code, label = compute_readiness_level(10.0)
        self.assertEqual(code, READINESS_STRONG)
        self.assertEqual(label, LABEL_STRONG)

    def test_no_false_hiring_decisions(self):
        """Ensures feedback never promises hiring, offers, or selection."""
        dim_avg = DimensionAverages(
            relevance=9.0, clarity=9.0, completeness=9.0, technical_correctness=9.0, structure=9.0
        )
        readiness = compute_interview_readiness(9.5, dim_avg, InterviewType.PYTHON)

        full_text = (
            f"{readiness.headline_wording} {readiness.assessment_message} {readiness.next_step}".lower()
        )
        forbidden = ["hired", "offer extended", "guaranteed hire", "selected for the role", "got the job"]
        for phrase in forbidden:
            self.assertNotIn(phrase, full_text)

    def test_track_aware_wording_across_tracks(self):
        dim_avg = DimensionAverages(
            relevance=7.0, clarity=7.0, completeness=7.0, technical_correctness=7.0, structure=7.0
        )
        tracks = [
            (InterviewType.HR, "HR & Behavioral"),
            (InterviewType.PYTHON, "Python Developer"),
            (InterviewType.AI_ML, "AI / Machine Learning"),
            (InterviewType.DATA_SCIENCE, "Data Science"),
            (InterviewType.INTERNSHIP, "Software Engineering Internship"),
        ]
        for itype, expected_label in tracks:
            readiness = compute_interview_readiness(7.2, dim_avg, itype)
            self.assertIn(expected_label, readiness.headline_wording)

    def test_dimension_ranking_and_tie_breaking(self):
        """Tests deterministic tie-breaking when dimensions have identical scores."""
        # All equal scores: tie-break must follow TIE_BREAK_ORDER
        dim_scores = {dim: 7.0 for dim in TIE_BREAK_ORDER}
        strongest, weakest = rank_dimensions(dim_scores)
        self.assertEqual(strongest, TIE_BREAK_ORDER)
        self.assertEqual(weakest, TIE_BREAK_ORDER)

        # Distinct scores
        custom_scores = {
            "relevance": 9.0,
            "clarity": 6.0,
            "completeness": 5.0,
            "technical_correctness": 8.0,
            "structure": 7.0,
        }
        strongest, weakest = rank_dimensions(custom_scores)
        self.assertEqual(strongest[0], "relevance")
        self.assertEqual(weakest[0], "completeness")
        self.assertEqual(weakest[1], "clarity")

    def test_priority_improvements_and_strongest_areas(self):
        dim_avg = DimensionAverages(
            relevance=9.0, clarity=8.0, completeness=5.0, technical_correctness=6.0, structure=7.0
        )
        readiness = compute_interview_readiness(7.0, dim_avg, InterviewType.PYTHON)
        self.assertEqual(len(readiness.priority_improvement_areas), 2)
        self.assertEqual(len(readiness.strongest_areas), 2)
        # Weakest are completeness (5.0) and technical_correctness (6.0)
        self.assertTrue(any("Completeness" in tip for tip in readiness.priority_improvement_areas))
        self.assertTrue(any("Technical" in tip for tip in readiness.priority_improvement_areas))
        # Strongest are relevance (9.0) and clarity (8.0)
        self.assertTrue(any("Relevance" in sa for sa in readiness.strongest_areas))
        self.assertTrue(any("Clarity" in sa for sa in readiness.strongest_areas))


class TestCandidateNameAndReportIntegration(unittest.TestCase):
    """Integration tests for Resume persistence, candidate name resolution, and SessionSummary."""

    def setUp(self):
        init_db()
        self.db = SessionLocal()
        self.user = User(name="Zain Ahmed", email="zain.test@interviewcoach.local")
        self.db.add(self.user)
        self.db.commit()
        self.db.refresh(self.user)

    def tearDown(self):
        self.db.query(InterviewSession).filter(InterviewSession.user_id == self.user.id).delete()
        self.db.query(Resume).filter(Resume.user_id == self.user.id).delete()
        self.db.query(User).filter(User.id == self.user.id).delete()
        self.db.commit()
        self.db.close()

    def test_resume_candidate_name_column_and_response(self):
        resume = Resume(
            user_id=self.user.id,
            file_name="test_cv.pdf",
            file_path="data/uploads/fake.pdf",
            extracted_text="Zain Ahmed\nAI Engineer",
            candidate_name="Zain Ahmed",
        )
        self.db.add(resume)
        self.db.commit()
        self.db.refresh(resume)

        resp = build_resume_response(resume)
        self.assertEqual(resp.candidate_name, "Zain Ahmed")

    def test_resolve_candidate_name_from_resume(self):
        resume = Resume(
            user_id=self.user.id,
            file_name="cv.pdf",
            file_path="data/uploads/fake.pdf",
            extracted_text="Ayesha Khan\nData Scientist",
            candidate_name="Ayesha Khan",
        )
        self.db.add(resume)
        self.db.commit()
        self.db.refresh(resume)

        session = InterviewSession(
            user_id=self.user.id,
            resume_id=resume.id,
            interview_type="python",
            difficulty="medium",
        )
        self.db.add(session)
        self.db.commit()
        self.db.refresh(session)

        resolved = resolve_candidate_name(session)
        self.assertEqual(resolved, "Ayesha Khan")

    def test_resolve_candidate_name_fallback_to_user_name(self):
        session = InterviewSession(
            user_id=self.user.id,
            resume_id=None,
            interview_type="ai_ml",
            difficulty="hard",
        )
        self.db.add(session)
        self.db.commit()
        self.db.refresh(session)

        resolved = resolve_candidate_name(session)
        self.assertEqual(resolved, "Zain Ahmed")

    def test_session_summary_serialization(self):
        summary = SessionSummary(
            session_id=101,
            interview_type=InterviewType.AI_ML,
            initial_difficulty=DifficultyLevel.HARD,
            status="completed",
            candidate_name="Zain Ahmed",
            readiness_label="Strong Interview Readiness",
            total_questions=5,
            completed_questions=5,
            overall_score=8.5,
            started_at=self.user.created_at,
            completed_at=self.user.created_at,
            report_available=True,
        )
        self.assertEqual(summary.candidate_name, "Zain Ahmed")
        self.assertEqual(summary.readiness_label, "Strong Interview Readiness")


if __name__ == "__main__":
    unittest.main()
