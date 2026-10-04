"""Automated test suite for Phase 9: Adaptive Difficulty & Intelligent Next-Question Strategy."""

import json
import os
import sys
import unittest
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient

from app.db.database import SessionLocal, init_db
from app.db.models import Answer, Evaluation, InterviewSession, Question, Resume, User
from app.interview.adaptive import decide_next_difficulty
from app.interview.schemas import (
    AdaptiveDecision,
    AnswerEvaluation,
    DifficultyLevel,
    DimensionScores,
    InterviewType,
    SubmitAnswerRequest,
)
from app.interview.workflow import (
    get_interview_session_details,
    start_interview_session,
    submit_answer_to_session,
)
from app.llm.client import set_llm_client
from app.main import app


class MockAdaptiveGeminiClient:
    """Mock Gemini client allowing dynamic evaluation scores and question generation."""

    def __init__(self, eval_scores: Optional[Dict[str, int]] = None):
        self.calls: List[str] = []
        self.question_calls: List[str] = []
        self.evaluation_calls: List[str] = []
        self.eval_scores = eval_scores or {
            "relevance": 8,
            "clarity": 8,
            "completeness": 7,
            "technical_correctness": 8,
            "structure": 8,
        }
        self.question_counter = 0

    def set_scores(self, scores: Dict[str, int]):
        self.eval_scores = scores

    def generate(self, prompt: str) -> str:
        self.calls.append(prompt)
        if "<CANDIDATE_ANSWER>" in prompt:
            self.evaluation_calls.append(prompt)
            return json.dumps({
                "scores": self.eval_scores,
                "covered_topics": ["core concepts"],
                "missing_topics": ["advanced optimization"],
                "strengths": ["Well structured."],
                "improvements": ["Deepen edge cases."],
                "technical_feedback": ["Accurate explanation."],
                "improved_answer": "Benchmark response with fine-grained performance notes.",
                "summary_feedback": "Solid response overall.",
            })
        else:
            self.question_counter += 1
            self.question_calls.append(prompt)
            return json.dumps({
                "questions": [
                    {
                        "category": "technical",
                        "question": f"Adaptive Question {self.question_counter}: Explain system design and algorithmic trade-offs.",
                        "rationale": "Evaluates candidate proficiency at the chosen difficulty.",
                        "expected_topics": ["complexity", "trade-offs"],
                        "grounding_sources": [],
                    }
                ]
            })


class TestPhase9AdaptiveDifficulty(unittest.TestCase):
    """Phase 9 Comprehensive Test Suite: Tests A through V."""

    @classmethod
    def setUpClass(cls):
        init_db()
        cls.test_client = TestClient(app)

    def setUp(self):
        self.db = SessionLocal()
        # Create isolated test users
        self.user_a = User(email="adaptive_user_a@example.com", name="User A Adaptive")
        self.user_b = User(email="adaptive_user_b@example.com", name="User B Adaptive")
        self.db.add_all([self.user_a, self.user_b])
        self.db.commit()
        self.db.refresh(self.user_a)
        self.db.refresh(self.user_b)

    def tearDown(self):
        self.db.query(Evaluation).delete()
        self.db.query(Answer).delete()
        self.db.query(Question).delete()
        self.db.query(InterviewSession).delete()
        self.db.query(Resume).delete()
        self.db.query(User).filter(
            User.email.in_(["adaptive_user_a@example.com", "adaptive_user_b@example.com"])
        ).delete()
        self.db.commit()
        self.db.close()

    # ==========================================
    # PURE POLICY UNIT TESTS (Tests A - L)
    # ==========================================

    def test_a_policy_increase_easy_to_medium(self):
        """A. High score (overall >= 8.0, tech >= 7, comp >= 6) increases easy -> medium."""
        eval_dict = {
            "overall_score": 8.5,
            "scores": {"technical_correctness": 8, "completeness": 7},
        }
        decision = decide_next_difficulty(DifficultyLevel.EASY, InterviewType.PYTHON, eval_dict)
        self.assertEqual(decision.previous_difficulty, DifficultyLevel.EASY)
        self.assertEqual(decision.next_difficulty, DifficultyLevel.MEDIUM)
        self.assertEqual(decision.action, "increase")
        self.assertEqual(decision.reason_code, "high_performance")

    def test_b_policy_increase_medium_to_hard(self):
        """B. High score increases medium -> hard."""
        eval_dict = {
            "overall_score": 8.2,
            "scores": {"technical_correctness": 8, "completeness": 8},
        }
        decision = decide_next_difficulty(DifficultyLevel.MEDIUM, InterviewType.AI_ML, eval_dict)
        self.assertEqual(decision.previous_difficulty, DifficultyLevel.MEDIUM)
        self.assertEqual(decision.next_difficulty, DifficultyLevel.HARD)
        self.assertEqual(decision.action, "increase")
        self.assertEqual(decision.reason_code, "high_performance")

    def test_c_policy_upper_bound_hard_maintains_hard(self):
        """C. Upper bound: High score at hard maintains hard."""
        eval_dict = {
            "overall_score": 9.5,
            "scores": {"technical_correctness": 10, "completeness": 9},
        }
        decision = decide_next_difficulty(DifficultyLevel.HARD, InterviewType.PYTHON, eval_dict)
        self.assertEqual(decision.previous_difficulty, DifficultyLevel.HARD)
        self.assertEqual(decision.next_difficulty, DifficultyLevel.HARD)
        self.assertEqual(decision.action, "maintain")
        self.assertEqual(decision.reason_code, "upper_bound")

    def test_d_policy_decrease_hard_to_medium(self):
        """D. Low score (overall <= 5.0) decreases hard -> medium."""
        eval_dict = {
            "overall_score": 4.5,
            "scores": {"technical_correctness": 4, "completeness": 4},
        }
        decision = decide_next_difficulty(DifficultyLevel.HARD, InterviewType.DATA_SCIENCE, eval_dict)
        self.assertEqual(decision.previous_difficulty, DifficultyLevel.HARD)
        self.assertEqual(decision.next_difficulty, DifficultyLevel.MEDIUM)
        self.assertEqual(decision.action, "decrease")
        self.assertEqual(decision.reason_code, "low_performance")

    def test_e_policy_decrease_medium_to_easy(self):
        """E. Low score decreases medium -> easy."""
        eval_dict = {
            "overall_score": 5.0,
            "scores": {"technical_correctness": 5, "completeness": 5},
        }
        decision = decide_next_difficulty(DifficultyLevel.MEDIUM, InterviewType.PYTHON, eval_dict)
        self.assertEqual(decision.previous_difficulty, DifficultyLevel.MEDIUM)
        self.assertEqual(decision.next_difficulty, DifficultyLevel.EASY)
        self.assertEqual(decision.action, "decrease")
        self.assertEqual(decision.reason_code, "low_performance")

    def test_f_policy_lower_bound_easy_maintains_easy(self):
        """F. Lower bound: Low score at easy maintains easy."""
        eval_dict = {
            "overall_score": 3.0,
            "scores": {"technical_correctness": 2, "completeness": 3},
        }
        decision = decide_next_difficulty(DifficultyLevel.EASY, InterviewType.PYTHON, eval_dict)
        self.assertEqual(decision.previous_difficulty, DifficultyLevel.EASY)
        self.assertEqual(decision.next_difficulty, DifficultyLevel.EASY)
        self.assertEqual(decision.action, "maintain")
        self.assertEqual(decision.reason_code, "lower_bound")

    def test_g_policy_stable_performance_maintains(self):
        """G. Stable performance (5.0 < overall < 8.0) maintains current difficulty."""
        eval_dict = {
            "overall_score": 6.8,
            "scores": {"technical_correctness": 7, "completeness": 6},
        }
        decision = decide_next_difficulty(DifficultyLevel.MEDIUM, InterviewType.PYTHON, eval_dict)
        self.assertEqual(decision.previous_difficulty, DifficultyLevel.MEDIUM)
        self.assertEqual(decision.next_difficulty, DifficultyLevel.MEDIUM)
        self.assertEqual(decision.action, "maintain")
        self.assertEqual(decision.reason_code, "stable_performance")

    def test_h_policy_max_one_level_jump_limitation(self):
        """H. High score at easy only moves to medium, NEVER directly to hard."""
        eval_dict = {
            "overall_score": 10.0,
            "scores": {"technical_correctness": 10, "completeness": 10},
        }
        decision = decide_next_difficulty(DifficultyLevel.EASY, InterviewType.AI_ML, eval_dict)
        self.assertEqual(decision.next_difficulty, DifficultyLevel.MEDIUM)
        self.assertNotEqual(decision.next_difficulty, DifficultyLevel.HARD)

    def test_i_policy_technical_guardrail_blocks_increase(self):
        """I. Technical guardrail: overall >= 8.0 but technical_correctness < 7 blocks increase."""
        eval_dict = {
            "overall_score": 8.2,
            "scores": {"technical_correctness": 6, "completeness": 8},
        }
        decision = decide_next_difficulty(DifficultyLevel.EASY, InterviewType.PYTHON, eval_dict)
        self.assertEqual(decision.previous_difficulty, DifficultyLevel.EASY)
        self.assertEqual(decision.next_difficulty, DifficultyLevel.EASY)
        self.assertEqual(decision.action, "maintain")
        self.assertEqual(decision.reason_code, "technical_guardrail")

    def test_j_policy_completeness_guardrail_blocks_increase(self):
        """J. Completeness guardrail: overall >= 8.0, tech >= 7, but completeness < 6 blocks increase."""
        eval_dict = {
            "overall_score": 8.1,
            "scores": {"technical_correctness": 8, "completeness": 5},
        }
        decision = decide_next_difficulty(DifficultyLevel.MEDIUM, InterviewType.AI_ML, eval_dict)
        self.assertEqual(decision.previous_difficulty, DifficultyLevel.MEDIUM)
        self.assertEqual(decision.next_difficulty, DifficultyLevel.MEDIUM)
        self.assertEqual(decision.action, "maintain")
        self.assertEqual(decision.reason_code, "completeness_guardrail")

    def test_k_policy_hr_track_escalates_without_technical_guardrail(self):
        """K. Non-technical HR track escalates with overall >= 8.0 regardless of technical_correctness."""
        eval_dict = {
            "overall_score": 8.5,
            "scores": {"technical_correctness": 5, "completeness": 5},
        }
        decision = decide_next_difficulty(DifficultyLevel.EASY, InterviewType.HR, eval_dict)
        self.assertEqual(decision.previous_difficulty, DifficultyLevel.EASY)
        self.assertEqual(decision.next_difficulty, DifficultyLevel.MEDIUM)
        self.assertEqual(decision.action, "increase")
        self.assertEqual(decision.reason_code, "high_performance")

    def test_l_policy_internship_track_escalates_without_technical_guardrail(self):
        """L. Non-technical Internship track escalates with overall >= 8.0 without technical guardrail."""
        eval_dict = {
            "overall_score": 8.0,
            "scores": {"technical_correctness": 4, "completeness": 5},
        }
        decision = decide_next_difficulty(DifficultyLevel.MEDIUM, InterviewType.INTERNSHIP, eval_dict)
        self.assertEqual(decision.previous_difficulty, DifficultyLevel.MEDIUM)
        self.assertEqual(decision.next_difficulty, DifficultyLevel.HARD)
        self.assertEqual(decision.action, "increase")
        self.assertEqual(decision.reason_code, "high_performance")

    # ==========================================
    # WORKFLOW INTEGRATION TESTS (Tests M - V)
    # ==========================================

    def test_m_first_question_difficulty_preserved(self):
        """M. Q1 difficulty strictly equals the requested initial_difficulty."""
        mock_client = MockAdaptiveGeminiClient()
        set_llm_client(mock_client)

        resp = self.test_client.post(
            "/interview/sessions/start",
            json={
                "user_id": self.user_a.id,
                "interview_type": "python",
                "difficulty": "hard",
                "total_questions": 3,
            },
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["current_question"]["difficulty"], "hard")

        # In DB, Q1 must be saved with difficulty = "hard"
        db_q = self.db.query(Question).filter(Question.session_id == data["session_id"]).first()
        self.assertEqual(db_q.difficulty, "hard")

    def test_n_workflow_adaptive_increase_turn(self):
        """N. Candidate gives high-scoring answer: Q2 adapts from easy -> medium."""
        # 10s all across -> overall 10.0
        mock_client = MockAdaptiveGeminiClient(
            eval_scores={
                "relevance": 10,
                "clarity": 10,
                "completeness": 10,
                "technical_correctness": 10,
                "structure": 10,
            }
        )
        set_llm_client(mock_client)

        start_resp = self.test_client.post(
            "/interview/sessions/start",
            json={
                "user_id": self.user_a.id,
                "interview_type": "python",
                "difficulty": "easy",
                "total_questions": 3,
            },
        )
        session_id = start_resp.json()["session_id"]
        self.assertEqual(start_resp.json()["current_question"]["difficulty"], "easy")

        ans_resp = self.test_client.post(
            f"/interview/sessions/{session_id}/answer",
            json={"user_id": self.user_a.id, "answer": "Exceptional Python memory model answer."},
        )
        self.assertEqual(ans_resp.status_code, 200)
        ans_data = ans_resp.json()
        decision = ans_data["adaptive_decision"]
        self.assertIsNotNone(decision)
        self.assertEqual(decision["previous_difficulty"], "easy")
        self.assertEqual(decision["next_difficulty"], "medium")
        self.assertEqual(decision["action"], "increase")
        self.assertEqual(decision["reason_code"], "high_performance")
        self.assertEqual(ans_data["next_question"]["difficulty"], "medium")

    def test_o_workflow_adaptive_decrease_turn(self):
        """O. Candidate gives low-scoring answer: Q2 adapts from hard -> medium."""
        # 4s all across -> overall 4.0
        mock_client = MockAdaptiveGeminiClient(
            eval_scores={
                "relevance": 4,
                "clarity": 4,
                "completeness": 4,
                "technical_correctness": 4,
                "structure": 4,
            }
        )
        set_llm_client(mock_client)

        start_resp = self.test_client.post(
            "/interview/sessions/start",
            json={
                "user_id": self.user_a.id,
                "interview_type": "ai_ml",
                "difficulty": "hard",
                "total_questions": 3,
            },
        )
        session_id = start_resp.json()["session_id"]
        self.assertEqual(start_resp.json()["current_question"]["difficulty"], "hard")

        ans_resp = self.test_client.post(
            f"/interview/sessions/{session_id}/answer",
            json={"user_id": self.user_a.id, "answer": "I have no idea how gradient descent works."},
        )
        self.assertEqual(ans_resp.status_code, 200)
        ans_data = ans_resp.json()
        decision = ans_data["adaptive_decision"]
        self.assertIsNotNone(decision)
        self.assertEqual(decision["previous_difficulty"], "hard")
        self.assertEqual(decision["next_difficulty"], "medium")
        self.assertEqual(decision["action"], "decrease")
        self.assertEqual(decision["reason_code"], "low_performance")
        self.assertEqual(ans_data["next_question"]["difficulty"], "medium")

    def test_p_workflow_adaptive_maintain_turn(self):
        """P. Candidate gives mid-range answer: Q2 maintains medium."""
        # 7s all across -> overall 7.0 (5.0 < 7.0 < 8.0)
        mock_client = MockAdaptiveGeminiClient(
            eval_scores={
                "relevance": 7,
                "clarity": 7,
                "completeness": 7,
                "technical_correctness": 7,
                "structure": 7,
            }
        )
        set_llm_client(mock_client)

        start_resp = self.test_client.post(
            "/interview/sessions/start",
            json={
                "user_id": self.user_a.id,
                "interview_type": "data_science",
                "difficulty": "medium",
                "total_questions": 3,
            },
        )
        session_id = start_resp.json()["session_id"]
        self.assertEqual(start_resp.json()["current_question"]["difficulty"], "medium")

        ans_resp = self.test_client.post(
            f"/interview/sessions/{session_id}/answer",
            json={"user_id": self.user_a.id, "answer": "Adequate explanation of ROC-AUC."},
        )
        self.assertEqual(ans_resp.status_code, 200)
        ans_data = ans_resp.json()
        decision = ans_data["adaptive_decision"]
        self.assertIsNotNone(decision)
        self.assertEqual(decision["previous_difficulty"], "medium")
        self.assertEqual(decision["next_difficulty"], "medium")
        self.assertEqual(decision["action"], "maintain")
        self.assertEqual(decision["reason_code"], "stable_performance")
        self.assertEqual(ans_data["next_question"]["difficulty"], "medium")

    def test_q_historical_question_difficulty_persistence(self):
        """Q. Each question in SQLite stores its actual asked difficulty."""
        mock_client = MockAdaptiveGeminiClient(
            eval_scores={
                "relevance": 9,
                "clarity": 9,
                "completeness": 8,
                "technical_correctness": 9,
                "structure": 9,
            }
        )
        set_llm_client(mock_client)

        start_resp = self.test_client.post(
            "/interview/sessions/start",
            json={
                "user_id": self.user_a.id,
                "interview_type": "python",
                "difficulty": "easy",
                "total_questions": 2,
            },
        )
        session_id = start_resp.json()["session_id"]

        self.test_client.post(
            f"/interview/sessions/{session_id}/answer",
            json={"user_id": self.user_a.id, "answer": "Great answer."},
        )

        db_questions = (
            self.db.query(Question)
            .filter(Question.session_id == session_id)
            .order_by(Question.question_number.asc())
            .all()
        )
        self.assertEqual(len(db_questions), 2)
        self.assertEqual(db_questions[0].difficulty, "easy")
        self.assertEqual(db_questions[1].difficulty, "medium")

    def test_r_get_session_details_turn_history_and_current_difficulty(self):
        """R. GET /interview/sessions/{id} returns difficulty on CompletedTurnHistory and current_difficulty."""
        mock_client = MockAdaptiveGeminiClient(
            eval_scores={
                "relevance": 9,
                "clarity": 9,
                "completeness": 8,
                "technical_correctness": 9,
                "structure": 9,
            }
        )
        set_llm_client(mock_client)

        start_resp = self.test_client.post(
            "/interview/sessions/start",
            json={
                "user_id": self.user_a.id,
                "interview_type": "python",
                "difficulty": "easy",
                "total_questions": 3,
            },
        )
        session_id = start_resp.json()["session_id"]

        self.test_client.post(
            f"/interview/sessions/{session_id}/answer",
            json={"user_id": self.user_a.id, "answer": "Excellent answer."},
        )

        get_resp = self.test_client.get(
            f"/interview/sessions/{session_id}?user_id={self.user_a.id}"
        )
        self.assertEqual(get_resp.status_code, 200)
        data = get_resp.json()
        self.assertEqual(data["difficulty"], "easy")  # initial session difficulty
        self.assertEqual(data["current_difficulty"], "medium")  # current active question difficulty
        self.assertEqual(len(data["history"]), 1)
        self.assertEqual(data["history"][0]["difficulty"], "easy")

    def test_s_db_state_reconstruction_resumes_adapted_difficulty(self):
        """S. Resuming turn execution reads pending question's adapted difficulty correctly."""
        mock_client = MockAdaptiveGeminiClient(
            eval_scores={
                "relevance": 10,
                "clarity": 10,
                "completeness": 9,
                "technical_correctness": 9,
                "structure": 10,
            }
        )
        set_llm_client(mock_client)

        # Start at easy (3 questions)
        start_resp = self.test_client.post(
            "/interview/sessions/start",
            json={
                "user_id": self.user_a.id,
                "interview_type": "python",
                "difficulty": "easy",
                "total_questions": 3,
            },
        )
        session_id = start_resp.json()["session_id"]

        # Turn 1 -> Adapts easy to medium
        ans1_resp = self.test_client.post(
            f"/interview/sessions/{session_id}/answer",
            json={"user_id": self.user_a.id, "answer": "Answer 1."},
        )
        self.assertEqual(ans1_resp.json()["next_question"]["difficulty"], "medium")

        # Turn 2 -> Resumes from DB (where Q2 is medium), high score adapts medium to hard!
        ans2_resp = self.test_client.post(
            f"/interview/sessions/{session_id}/answer",
            json={"user_id": self.user_a.id, "answer": "Answer 2."},
        )
        self.assertEqual(ans2_resp.status_code, 200)
        ans2_data = ans2_resp.json()
        self.assertEqual(ans2_data["adaptive_decision"]["previous_difficulty"], "medium")
        self.assertEqual(ans2_data["adaptive_decision"]["next_difficulty"], "hard")
        self.assertEqual(ans2_data["next_question"]["difficulty"], "hard")

    def test_t_zero_extra_llm_calls_for_adaptive_policy(self):
        """T. Adaptive difficulty is pure Python and introduces 0 extra LLM calls."""
        mock_client = MockAdaptiveGeminiClient()
        set_llm_client(mock_client)

        start_resp = self.test_client.post(
            "/interview/sessions/start",
            json={
                "user_id": self.user_a.id,
                "interview_type": "python",
                "difficulty": "medium",
                "total_questions": 2,
            },
        )
        session_id = start_resp.json()["session_id"]
        # Exactly 1 LLM call to generate Q1
        self.assertEqual(len(mock_client.calls), 1)

        self.test_client.post(
            f"/interview/sessions/{session_id}/answer",
            json={"user_id": self.user_a.id, "answer": "Standard answer."},
        )
        # Exactly 2 more LLM calls: 1 to evaluate answer, 1 to generate Q2. 0 for adaptive policy!
        self.assertEqual(len(mock_client.calls), 3)
        self.assertEqual(len(mock_client.evaluation_calls), 1)
        self.assertEqual(len(mock_client.question_calls), 2)

    def test_u_final_turn_nullification(self):
        """U. On the final question, next_question and adaptive_decision are None."""
        mock_client = MockAdaptiveGeminiClient()
        set_llm_client(mock_client)

        start_resp = self.test_client.post(
            "/interview/sessions/start",
            json={
                "user_id": self.user_a.id,
                "interview_type": "python",
                "difficulty": "medium",
                "total_questions": 1,
            },
        )
        session_id = start_resp.json()["session_id"]

        ans_resp = self.test_client.post(
            f"/interview/sessions/{session_id}/answer",
            json={"user_id": self.user_a.id, "answer": "Final turn answer."},
        )
        self.assertEqual(ans_resp.status_code, 200)
        data = ans_resp.json()
        self.assertEqual(data["status"], "completed")
        self.assertIsNone(data["next_question"])
        self.assertIsNone(data["adaptive_decision"])

    def test_v_strict_user_isolation(self):
        """V. User B cannot access User A's adaptive session details or submit answers."""
        mock_client = MockAdaptiveGeminiClient()
        set_llm_client(mock_client)

        start_resp = self.test_client.post(
            "/interview/sessions/start",
            json={
                "user_id": self.user_a.id,
                "interview_type": "python",
                "difficulty": "medium",
                "total_questions": 2,
            },
        )
        session_id = start_resp.json()["session_id"]

        # User B cannot submit answer to User A's session -> 404
        ans_b = self.test_client.post(
            f"/interview/sessions/{session_id}/answer",
            json={"user_id": self.user_b.id, "answer": "Intruder answer."},
        )
        self.assertEqual(ans_b.status_code, 404)

        # User B cannot view User A's session -> 404
        get_b = self.test_client.get(
            f"/interview/sessions/{session_id}?user_id={self.user_b.id}"
        )
        self.assertEqual(get_b.status_code, 404)


if __name__ == "__main__":
    unittest.main()
