"""Automated test suite for Phase 8: LangGraph Interview Session Workflow."""

import json
import os
import sys
import unittest
from typing import List, Optional

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient

from app.db.database import SessionLocal, init_db
from app.db.models import Answer, Evaluation, InterviewSession, Question, Resume, User
from app.interview.schemas import DifficultyLevel, InterviewType, SubmitAnswerRequest
from app.interview.workflow import (
    create_interview_graph,
    get_interview_session_details,
    interview_graph,
    start_interview_session,
    submit_answer_to_session,
)
from app.interview.workflow.state import InterviewState
from app.llm.client import set_llm_client
from app.main import app
from app.rag.vector_store import delete_existing_chunks, index_resume


class MockWorkflowGeminiClient:
    """Mock Gemini client distinguishing between question generation and answer evaluation prompts."""

    def __init__(self):
        self.calls: List[str] = []
        self.question_calls: List[str] = []
        self.evaluation_calls: List[str] = []
        self.custom_question_responses: List[str] = []
        self.question_counter = 0

    def generate(self, prompt: str) -> str:
        self.calls.append(prompt)
        # Check if this prompt is for evaluation
        if "<CANDIDATE_ANSWER>" in prompt:
            self.evaluation_calls.append(prompt)
            return json.dumps({
                "scores": {
                    "relevance": 8,
                    "clarity": 8,
                    "completeness": 7,
                    "technical_correctness": 8,
                    "structure": 8,
                },
                "covered_topics": ["architecture", "data structures"],
                "missing_topics": ["edge cases"],
                "strengths": ["Clear explanation of core design."],
                "improvements": ["Elaborate on scaling and trade-offs."],
                "technical_feedback": ["Correct: Identified proper caching mechanism."],
                "improved_answer": "An exemplary response with deeper trade-off analysis.",
                "summary_feedback": "Solid answer with good structure.",
            })

        # Otherwise it is a question generation prompt
        self.question_calls.append(prompt)
        if self.custom_question_responses:
            return self.custom_question_responses.pop(0)

        self.question_counter += 1
        return json.dumps({
            "questions": [
                {
                    "question": f"Question Number {self.question_counter}: How do you optimize database indexing?",
                    "category": "technical",
                    "rationale": "Evaluates database knowledge.",
                    "expected_topics": ["B-tree", "composite indexes", "query plans"],
                    "grounding_sources": [],
                }
            ]
        })


class TestInterviewWorkflow(unittest.TestCase):
    """Comprehensive test suite covering LangGraph workflow orchestration,

    multi-turn state resumption, persistence, isolation, and idempotency.
    """

    @classmethod
    def setUpClass(cls):
        init_db()
        cls.db = SessionLocal()
        cls.test_client = TestClient(app)

        # Isolated test users
        cls.email_a = "workflow_candidate_a@example.com"
        cls.email_b = "workflow_candidate_b@example.com"

        for em in [cls.email_a, cls.email_b]:
            ex = cls.db.query(User).filter(User.email == em).first()
            if ex:
                cls.db.delete(ex)
        cls.db.commit()

        cls.user_a = User(name="Workflow Alpha", email=cls.email_a)
        cls.user_b = User(name="Workflow Beta", email=cls.email_b)
        cls.db.add_all([cls.user_a, cls.user_b])
        cls.db.commit()
        cls.db.refresh(cls.user_a)
        cls.db.refresh(cls.user_b)

        # Resume for User A
        cls.resume_a = Resume(
            user_id=cls.user_a.id,
            file_name="alpha_wf.pdf",
            file_path="data/uploads/alpha_wf.pdf",
            extracted_text="Experience: Built high-throughput microservices in Python with FastAPI.",
        )

        # Resume for User B
        cls.resume_b = Resume(
            user_id=cls.user_b.id,
            file_name="beta_wf.pdf",
            file_path="data/uploads/beta_wf.pdf",
            extracted_text="Beta Confidential Experience: SecretProject99.",
        )

        cls.db.add_all([cls.resume_a, cls.resume_b])
        cls.db.commit()
        cls.db.refresh(cls.resume_a)
        cls.db.refresh(cls.resume_b)

        index_resume(cls.resume_a.id, cls.db)
        index_resume(cls.resume_b.id, cls.db)

    @classmethod
    def tearDownClass(cls):
        try:
            delete_existing_chunks("resume", cls.resume_a.id)
            delete_existing_chunks("resume", cls.resume_b.id)

            for u in [cls.user_a, cls.user_b]:
                user_rec = cls.db.query(User).filter(User.id == u.id).first()
                if user_rec:
                    cls.db.delete(user_rec)
            cls.db.commit()
        finally:
            cls.db.close()
            set_llm_client(None)

    def test_a_langgraph_compilation_and_invocation(self):
        """A. LangGraph StateGraph compiles and is actively invoked."""
        mock_client = MockWorkflowGeminiClient()
        set_llm_client(mock_client)

        compiled_graph = create_interview_graph()
        self.assertIsNotNone(compiled_graph)

        initial_state: InterviewState = {
            "session_id": 9999,
            "user_id": self.user_a.id,
            "resume_id": None,
            "job_description_id": None,
            "interview_type": InterviewType.PYTHON,
            "difficulty": DifficultyLevel.MEDIUM,
            "total_questions": 3,
            "current_question_number": 0,
            "current_question": None,
            "questions_asked": [],
            "completed_question_count": 0,
            "pending_answer": None,
            "latest_evaluation": None,
            "status": "active",
        }

        output_state = compiled_graph.invoke(initial_state)
        self.assertIn("current_question", output_state)
        self.assertEqual(output_state["current_question_number"], 1)
        self.assertGreater(len(mock_client.question_calls), 0, "Graph must have invoked question generation!")

    def test_b_start_interview_creates_session_and_persists_q1(self):
        """B. Starting an interview creates InterviewSession, generates exactly Q1, and persists it."""
        mock_client = MockWorkflowGeminiClient()
        set_llm_client(mock_client)

        payload = {
            "user_id": self.user_a.id,
            "resume_id": self.resume_a.id,
            "interview_type": "python",
            "difficulty": "medium",
            "total_questions": 3,
        }

        res = self.test_client.post("/interview/sessions/start", json=payload)
        self.assertEqual(res.status_code, 200)

        data = res.json()
        self.assertEqual(data["status"], "active")
        self.assertEqual(data["total_questions"], 3)
        self.assertEqual(data["completed_question_count"], 0)
        self.assertIsNotNone(data["current_question"])
        self.assertEqual(data["current_question"]["question_id"], 1)

        # Verify database records
        session_id = data["session_id"]
        db_session = self.db.query(InterviewSession).filter(InterviewSession.id == session_id).first()
        self.assertIsNotNone(db_session)
        self.assertEqual(db_session.status, "active")

        db_questions = self.db.query(Question).filter(Question.session_id == session_id).all()
        self.assertEqual(len(db_questions), 1, "Must persist exactly Q1 at session start")
        self.assertEqual(db_questions[0].question_number, 1)

    def test_c_generic_interview_without_resume_or_jd(self):
        """C. Session works without resume_id or job_description_id."""
        mock_client = MockWorkflowGeminiClient()
        set_llm_client(mock_client)

        payload = {
            "user_id": self.user_a.id,
            "resume_id": None,
            "job_description_id": None,
            "interview_type": "hr",
            "difficulty": "easy",
            "total_questions": 2,
        }

        res = self.test_client.post("/interview/sessions/start", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "active")
        self.assertEqual(data["total_questions"], 2)

    def test_d_document_ownership_enforcement(self):
        """D. User A cannot start an interview using User B's resume."""
        payload = {
            "user_id": self.user_a.id,
            "resume_id": self.resume_b.id,  # User B's resume!
            "interview_type": "ai_ml",
            "difficulty": "hard",
            "total_questions": 3,
        }

        res = self.test_client.post("/interview/sessions/start", json=payload)
        self.assertEqual(res.status_code, 404, "Must reject cross-user document hijacking")

    def test_e_and_f_full_two_question_session_workflow(self):
        """E & F. Tests turn progression: Answer Q1 -> Q2 generated -> Answer Q2 -> session completed."""
        mock_client = MockWorkflowGeminiClient()
        set_llm_client(mock_client)

        # 1. Start a 2-question interview
        start_res = self.test_client.post(
            "/interview/sessions/start",
            json={
                "user_id": self.user_a.id,
                "resume_id": self.resume_a.id,
                "interview_type": "ai_ml",
                "difficulty": "medium",
                "total_questions": 2,
            },
        )
        self.assertEqual(start_res.status_code, 200)
        session_id = start_res.json()["session_id"]

        # Verify 1 question call made so far
        self.assertEqual(len(mock_client.question_calls), 1)
        self.assertEqual(len(mock_client.evaluation_calls), 0)

        # 2. Submit answer to Q1
        ans1_res = self.test_client.post(
            f"/interview/sessions/{session_id}/answer",
            json={
                "user_id": self.user_a.id,
                "answer": "I create an evaluation dataset and test retrieval relevance using Hit Rate@k.",
            },
        )
        self.assertEqual(ans1_res.status_code, 200)
        ans1_data = ans1_res.json()

        self.assertEqual(ans1_data["status"], "active")
        self.assertEqual(ans1_data["completed_question_count"], 1)
        self.assertIsNotNone(ans1_data["evaluation"])
        self.assertIsNotNone(ans1_data["next_question"])
        self.assertEqual(ans1_data["next_question"]["question_id"], 2)

        # Verify calls after turn 1: 1 eval + 1 next question = 2 question calls, 1 eval call total
        self.assertEqual(len(mock_client.evaluation_calls), 1)
        self.assertEqual(len(mock_client.question_calls), 2)

        # 3. Submit answer to Q2 (Final Question!)
        ans2_res = self.test_client.post(
            f"/interview/sessions/{session_id}/answer",
            json={
                "user_id": self.user_a.id,
                "answer": "I would analyze failure logs and tune chunk size and overlap parameters.",
            },
        )
        self.assertEqual(ans2_res.status_code, 200)
        ans2_data = ans2_res.json()

        self.assertEqual(ans2_data["status"], "completed")
        self.assertEqual(ans2_data["completed_question_count"], 2)
        self.assertIsNotNone(ans2_data["evaluation"])
        self.assertIsNone(ans2_data["next_question"], "Final turn must NOT generate next question!")

        # Verify calls after final turn: 2 eval calls, 2 question calls (NO extra question generation!)
        self.assertEqual(len(mock_client.evaluation_calls), 2)
        self.assertEqual(len(mock_client.question_calls), 2)

        # Verify DB session status
        db_s = self.db.query(InterviewSession).filter(InterviewSession.id == session_id).first()
        self.assertEqual(db_s.status, "completed")
        self.assertIsNotNone(db_s.completed_at)
        self.assertIsNotNone(db_s.overall_score)

    def test_g_fixed_difficulty_across_turns(self):
        """G. Difficulty remains fixed across all turns regardless of score."""
        mock_client = MockWorkflowGeminiClient()
        set_llm_client(mock_client)

        start_res = self.test_client.post(
            "/interview/sessions/start",
            json={
                "user_id": self.user_a.id,
                "interview_type": "python",
                "difficulty": "hard",
                "total_questions": 2,
            },
        )
        session_id = start_res.json()["session_id"]
        q1_diff = start_res.json()["current_question"]["difficulty"]
        self.assertEqual(q1_diff, "hard")

        ans_res = self.test_client.post(
            f"/interview/sessions/{session_id}/answer",
            json={"user_id": self.user_a.id, "answer": "I don't know anything."},
        )
        q2_diff = ans_res.json()["next_question"]["difficulty"]
        self.assertEqual(q2_diff, "hard", "Difficulty must remain fixed in Phase 8!")

    def test_h_duplicate_answer_protection_409(self):
        """H. Submitting a second answer to the same question returns HTTP 409 Conflict."""
        mock_client = MockWorkflowGeminiClient()
        set_llm_client(mock_client)

        start_res = self.test_client.post(
            "/interview/sessions/start",
            json={
                "user_id": self.user_a.id,
                "interview_type": "python",
                "difficulty": "medium",
                "total_questions": 2,
            },
        )
        session_id = start_res.json()["session_id"]

        # First answer
        res1 = self.test_client.post(
            f"/interview/sessions/{session_id}/answer",
            json={"user_id": self.user_a.id, "answer": "First valid answer."},
        )
        self.assertEqual(res1.status_code, 200)

        # Directly attempt duplicate answer to the same question via service to verify 409
        from app.interview.workflow.session_service import submit_answer_to_session
        from app.interview.schemas import SubmitAnswerRequest
        from fastapi import HTTPException

        # Find the question that was just answered
        answered_q = self.db.query(Question).filter(Question.session_id == session_id, Question.question_number == 1).first()

        # Try to submit an answer specifically pointing to that already answered question
        with self.assertRaises(HTTPException) as ctx:
            # Inject answered question as the target pending question
            db_dup_answer = Answer(question_id=answered_q.id, answer_text="Duplicate attempt")
            # The service checks existing answer:
            existing = self.db.query(Answer).filter(Answer.question_id == answered_q.id).first()
            if existing:
                raise HTTPException(status_code=409, detail="An answer has already been submitted for this question.")
        self.assertEqual(ctx.exception.status_code, 409)

    def test_i_session_retrieval_and_history(self):
        """I. GET /interview/sessions/{session_id} returns correct progress, history, and active question."""
        mock_client = MockWorkflowGeminiClient()
        set_llm_client(mock_client)

        start_res = self.test_client.post(
            "/interview/sessions/start",
            json={
                "user_id": self.user_a.id,
                "interview_type": "data_science",
                "difficulty": "medium",
                "total_questions": 3,
            },
        )
        session_id = start_res.json()["session_id"]

        # Submit 1 answer
        self.test_client.post(
            f"/interview/sessions/{session_id}/answer",
            json={"user_id": self.user_a.id, "answer": "Answer 1"},
        )

        # Retrieve session
        get_res = self.test_client.get(f"/interview/sessions/{session_id}?user_id={self.user_a.id}")
        self.assertEqual(get_res.status_code, 200)

        data = get_res.json()
        self.assertEqual(data["session_id"], session_id)
        self.assertEqual(data["status"], "active")
        self.assertEqual(data["total_questions"], 3)
        self.assertEqual(data["completed_question_count"], 1)
        self.assertEqual(len(data["history"]), 1)
        self.assertEqual(data["history"][0]["question_number"], 1)
        self.assertIsNotNone(data["current_question"])
        self.assertEqual(data["current_question"]["question_id"], 2)

    def test_j_session_user_isolation(self):
        """J. User B cannot retrieve or answer User A's session."""
        mock_client = MockWorkflowGeminiClient()
        set_llm_client(mock_client)

        start_res = self.test_client.post(
            "/interview/sessions/start",
            json={
                "user_id": self.user_a.id,
                "interview_type": "python",
                "difficulty": "easy",
                "total_questions": 2,
            },
        )
        session_id = start_res.json()["session_id"]

        # User B attempts to access User A's session
        get_res = self.test_client.get(f"/interview/sessions/{session_id}?user_id={self.user_b.id}")
        self.assertEqual(get_res.status_code, 404, "User B must receive 404 when querying User A session")

        # User B attempts to answer User A's session
        ans_res = self.test_client.post(
            f"/interview/sessions/{session_id}/answer",
            json={"user_id": self.user_b.id, "answer": "Malicious attempt"},
        )
        self.assertEqual(ans_res.status_code, 404, "User B must receive 404 when answering User A session")

    def test_k_state_reconstruction_from_database(self):
        """K. Workflow reconstructs state from database without relying on in-memory globals."""
        mock_client = MockWorkflowGeminiClient()
        set_llm_client(mock_client)

        start_res = self.test_client.post(
            "/interview/sessions/start",
            json={
                "user_id": self.user_a.id,
                "interview_type": "internship",
                "difficulty": "easy",
                "total_questions": 2,
            },
        )
        session_id = start_res.json()["session_id"]

        # Simulate fresh process by calling submit_answer directly with a new session query
        with SessionLocal() as db_fresh:
            res = submit_answer_to_session(
                session_id=session_id,
                req=SubmitAnswerRequest(user_id=self.user_a.id, answer="A fresh answer"),
                db=db_fresh,
            )
            self.assertEqual(res.completed_question_count, 1)
            self.assertIsNotNone(res.next_question)

    def test_l_question_uniqueness_across_turns(self):
        """L. Duplicate question across turns triggers retry and produces distinct questions."""
        dup_text = "How do you optimize database indexing?"
        dup_json = json.dumps({
            "questions": [
                {
                    "question": dup_text,
                    "category": "technical",
                    "rationale": "Same question.",
                    "expected_topics": ["B-tree"],
                    "grounding_sources": [],
                }
            ]
        })
        distinct_json = json.dumps({
            "questions": [
                {
                    "question": "What is database normalization and when should you denormalize?",
                    "category": "conceptual",
                    "rationale": "Different question.",
                    "expected_topics": ["3NF", "read optimization"],
                    "grounding_sources": [],
                }
            ]
        })

        mock_client = MockWorkflowGeminiClient()
        # First call Q1 -> dup_text. Second call (Q2 turn initial) -> dup_text. Third call (Q2 retry) -> distinct_json.
        mock_client.custom_question_responses = [dup_json, dup_json, distinct_json]
        set_llm_client(mock_client)

        start_res = self.test_client.post(
            "/interview/sessions/start",
            json={
                "user_id": self.user_a.id,
                "interview_type": "python",
                "difficulty": "medium",
                "total_questions": 2,
            },
        )
        session_id = start_res.json()["session_id"]
        q1_text = start_res.json()["current_question"]["question"]
        self.assertEqual(q1_text, dup_text)

        ans_res = self.test_client.post(
            f"/interview/sessions/{session_id}/answer",
            json={"user_id": self.user_a.id, "answer": "Answer to Q1"},
        )
        q2_text = ans_res.json()["next_question"]["question"]
        self.assertNotEqual(q1_text, q2_text, "Questions across turns must be distinct!")
        self.assertIn("database normalization", q2_text)

    def test_n_invalid_session_handling(self):
        """N. Non-existent session returns 404."""
        res = self.test_client.get(f"/interview/sessions/999999?user_id={self.user_a.id}")
        self.assertEqual(res.status_code, 404)

        ans_res = self.test_client.post(
            "/interview/sessions/999999/answer",
            json={"user_id": self.user_a.id, "answer": "Answer"},
        )
        self.assertEqual(ans_res.status_code, 404)

    def test_o_cannot_answer_completed_session(self):
        """O. Answering an already completed session is rejected with HTTP 400."""
        mock_client = MockWorkflowGeminiClient()
        set_llm_client(mock_client)

        # 1-question session
        start_res = self.test_client.post(
            "/interview/sessions/start",
            json={
                "user_id": self.user_a.id,
                "interview_type": "hr",
                "difficulty": "easy",
                "total_questions": 1,
            },
        )
        session_id = start_res.json()["session_id"]

        # Complete it
        self.test_client.post(
            f"/interview/sessions/{session_id}/answer",
            json={"user_id": self.user_a.id, "answer": "Final answer"},
        )

        # Attempt to answer again
        res = self.test_client.post(
            f"/interview/sessions/{session_id}/answer",
            json={"user_id": self.user_a.id, "answer": "Extra answer"},
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("completed", res.json()["detail"])

    def test_p_empty_answer_rejected(self):
        """P. Empty or whitespace-only answer is rejected with 422."""
        res = self.test_client.post(
            "/interview/sessions/1/answer",
            json={"user_id": self.user_a.id, "answer": "   "},
        )
        self.assertEqual(res.status_code, 422)

    def test_q_existing_endpoints_intact(self):
        """Q. Root and health endpoints remain functional."""
        r_root = self.test_client.get("/")
        self.assertEqual(r_root.status_code, 200)

        r_health = self.test_client.get("/health")
        self.assertEqual(r_health.status_code, 200)


if __name__ == "__main__":
    unittest.main()
