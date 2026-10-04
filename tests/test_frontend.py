"""Automated test suite for Phase 11: Streamlit UI, API Client, and Backend Frontend Support."""

import inspect
import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import requests

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient

from app.db.database import SessionLocal, init_db
from app.db.models import Answer, Evaluation, FinalReport, InterviewSession, Question, User
from app.main import app
from frontend.api_client import (
    API_BASE_URL,
    APIError,
    _handle_response,
    _map_http_error,
    create_job_description,
    get_or_create_demo_user,
    list_interview_sessions,
    start_interview,
    submit_answer,
    upload_resume,
)
from frontend.state import (
    ADAPTIVE_REASON_LABELS,
    DIFFICULTY_OPTIONS,
    INTERVIEW_TYPE_OPTIONS,
    format_adaptive_reason,
    init_session_state,
    reset_interview_state,
    validate_candidate_answer,
)


class TestPhase11FrontendArchitecture(unittest.TestCase):
    """Phase 11 Automated Test Suite: Testing Frontend Components, API Client, and Backend Endpoints."""

    @classmethod
    def setUpClass(cls):
        init_db()
        cls.test_client = TestClient(app)

    def setUp(self):
        self.db = SessionLocal()
        # Clean up any leftover test data
        self.test_user = User(name="Frontend Tester", email="frontend_tester@example.com")
        self.test_user_b = User(name="Frontend Tester B", email="frontend_tester_b@example.com")
        self.db.add_all([self.test_user, self.test_user_b])
        self.db.commit()
        self.db.refresh(self.test_user)
        self.db.refresh(self.test_user_b)

    def tearDown(self):
        self.db.query(FinalReport).delete()
        self.db.query(Evaluation).delete()
        self.db.query(Answer).delete()
        self.db.query(Question).delete()
        self.db.query(InterviewSession).delete()
        self.db.query(User).filter(
            User.email.in_([
                "frontend_tester@example.com",
                "frontend_tester_b@example.com",
                "demo_candidate@interviewcoach.local",
            ])
        ).delete()
        self.db.commit()
        self.db.close()

    # =========================================================================
    # TESTS A - D: Module Imports & Clean Architecture Boundaries
    # =========================================================================

    def test_a_frontend_modules_import_successfully(self):
        """A. All frontend package modules and sub-components import cleanly."""
        import frontend.api_client
        import frontend.app
        import frontend.components.feedback
        import frontend.components.history
        import frontend.components.home
        import frontend.components.interview
        import frontend.components.report
        import frontend.components.setup
        import frontend.components.sidebar
        import frontend.server_manager
        import frontend.state
        import frontend.styles

        self.assertIsNotNone(frontend.app)
        self.assertIsNotNone(frontend.api_client)
        self.assertIsNotNone(frontend.server_manager)

    def test_b_no_gemini_client_in_frontend(self):
        """B. Frontend code must NEVER directly import Gemini SDK or LLM clients."""
        frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
        for root, _, files in os.walk(frontend_dir):
            for file in files:
                if file.endswith(".py"):
                    filepath = os.path.join(root, file)
                    with open(filepath, "r", encoding="utf-8") as f:
                        content = f.read()
                        self.assertNotIn("google.genai", content, f"Direct Gemini import found in {file}!")
                        self.assertNotIn("app.llm", content, f"Direct LLM client import found in {file}!")

    def test_c_no_sqlalchemy_or_chroma_in_frontend(self):
        """C. Frontend code must NEVER directly access SQLite or ChromaDB."""
        frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
        for root, _, files in os.walk(frontend_dir):
            for file in files:
                if file.endswith(".py"):
                    filepath = os.path.join(root, file)
                    with open(filepath, "r", encoding="utf-8") as f:
                        content = f.read()
                        self.assertNotIn("chromadb", content, f"ChromaDB import found in {file}!")
                        self.assertNotIn("sentence_transformers", content, f"MiniLM import found in {file}!")
                        self.assertNotIn("sqlalchemy", content, f"SQLAlchemy import found in {file}!")

    def test_d_api_base_url_configuration(self):
        """D. API base URL is configurable and defaults cleanly to local host."""
        self.assertTrue(API_BASE_URL.startswith("http"))
        self.assertNotIn(API_BASE_URL[-1], "/")

    # =========================================================================
    # TESTS E - L: State & Helper Functions
    # =========================================================================

    def test_e_blank_answer_prevention_helper(self):
        """E. Candidate answer validation rejects blank or whitespace-only inputs."""
        self.assertFalse(validate_candidate_answer(""))
        self.assertFalse(validate_candidate_answer("   "))
        self.assertFalse(validate_candidate_answer("\n\t  \n"))
        self.assertTrue(validate_candidate_answer("Valid answer with technical content."))

    def test_f_friendly_interview_type_mappings(self):
        """F. Friendly interview type mappings cover all supported backend tracks."""
        expected_tracks = {"hr", "python", "ai_ml", "data_science", "internship"}
        self.assertEqual(set(INTERVIEW_TYPE_OPTIONS.keys()), expected_tracks)
        for k, v in INTERVIEW_TYPE_OPTIONS.items():
            self.assertTrue(len(v) > 2)

    def test_g_difficulty_options_mapping(self):
        """G. Difficulty options map to Easy, Medium, and Hard."""
        expected_diffs = {"easy", "medium", "hard"}
        self.assertEqual(set(DIFFICULTY_OPTIONS.keys()), expected_diffs)

    def test_h_adaptive_reason_code_formatting(self):
        """H. Adaptive reason codes map to friendly UI descriptions."""
        self.assertIn("Strong performance", format_adaptive_reason("high_performance"))
        self.assertIn("More foundation practice", format_adaptive_reason("low_performance"))
        self.assertIn("Difficulty maintained", format_adaptive_reason("stable_performance"))
        self.assertIn("Technical accuracy", format_adaptive_reason("technical_guardrail"))
        self.assertIn("depth", format_adaptive_reason("completeness_guardrail"))
        self.assertIn("Highest", format_adaptive_reason("upper_bound"))
        self.assertIn("Foundation", format_adaptive_reason("lower_bound"))
        # Fallback formatting
        self.assertEqual(format_adaptive_reason(None), "Difficulty maintained")

    def test_i_session_state_reset_preserves_user_profile(self):
        """I. Resetting interview state clears turn data but preserves candidate user_id and profile."""
        import streamlit as st

        st.session_state.user_id = 42
        st.session_state.user_name = "Preserved User"
        st.session_state.user_email = "preserved@example.com"
        st.session_state.session_id = 999
        st.session_state.answer_text = "Temporary answer"
        st.session_state.current_page = "interview"

        reset_interview_state()

        self.assertEqual(st.session_state.user_id, 42)
        self.assertEqual(st.session_state.user_name, "Preserved User")
        self.assertEqual(st.session_state.user_email, "preserved@example.com")
        self.assertIsNone(st.session_state.session_id)
        self.assertEqual(st.session_state.answer_text, "")
        self.assertEqual(st.session_state.current_page, "setup")

    # =========================================================================
    # TESTS J - O: API Client Error Translation & Handling
    # =========================================================================

    def test_j_api_client_error_translations(self):
        """J. HTTP status codes translate into helpful user-facing error messages."""
        self.assertIn("5 MB", _map_http_error(413, "Payload too large"))
        self.assertIn("demand", _map_http_error(429, "Rate limit exceeded"))
        self.assertIn("not found", _map_http_error(404, "Session missing"))
        self.assertIn("unavailable", _map_http_error(503, "Service down"))

    def test_k_api_client_response_handling(self):
        """K. Successful responses return parsed JSON; non-200 raises APIError."""
        mock_resp_ok = MagicMock()
        mock_resp_ok.ok = True
        mock_resp_ok.json.return_value = {"status": "success", "id": 1}
        data = _handle_response(mock_resp_ok)
        self.assertEqual(data["status"], "success")

        mock_resp_err = MagicMock()
        mock_resp_err.ok = False
        mock_resp_err.status_code = 404
        mock_resp_err.json.return_value = {"detail": "Interview session not found."}
        with self.assertRaises(APIError) as ctx:
            _handle_response(mock_resp_err)
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertIn("Resource not found", ctx.exception.user_message)

    # =========================================================================
    # TESTS L - Q: Backend Support Endpoints (Demo User & Session History)
    # =========================================================================

    def test_l_backend_demo_user_endpoint_idempotent(self):
        """L. POST /users/demo initializes and returns persistent demo candidate."""
        res1 = self.test_client.post("/users/demo")
        self.assertEqual(res1.status_code, 200)
        data1 = res1.json()
        self.assertEqual(data1["name"], "Demo Candidate")
        self.assertIn("demo_candidate@", data1["email"])
        user_id = data1["id"]

        # Call again: must return same user ID
        res2 = self.test_client.get("/users/demo")
        self.assertEqual(res2.status_code, 200)
        data2 = res2.json()
        self.assertEqual(data2["id"], user_id)

    def test_m_backend_session_history_endpoint_newest_first(self):
        """M. GET /interview/sessions returns chronological history (newest first)."""
        # Create 2 sessions for test_user
        s1 = InterviewSession(user_id=self.test_user.id, interview_type="python", difficulty="easy", total_questions=3, status="completed")
        s2 = InterviewSession(user_id=self.test_user.id, interview_type="ai_ml", difficulty="hard", total_questions=5, status="active")
        self.db.add_all([s1, s2])
        self.db.commit()

        res = self.test_client.get(f"/interview/sessions?user_id={self.test_user.id}")
        self.assertEqual(res.status_code, 200)
        sessions = res.json()
        self.assertEqual(len(sessions), 2)
        # Newest session (s2) must appear first
        self.assertEqual(sessions[0]["session_id"], s2.id)
        self.assertEqual(sessions[0]["interview_type"], "ai_ml")
        self.assertEqual(sessions[0]["status"], "active")
        self.assertEqual(sessions[1]["session_id"], s1.id)
        self.assertEqual(sessions[1]["interview_type"], "python")
        self.assertEqual(sessions[1]["status"], "completed")

    def test_n_backend_session_history_user_isolation(self):
        """N. User B cannot see User A's session history."""
        # Create session for User A
        s_a = InterviewSession(user_id=self.test_user.id, interview_type="python", difficulty="medium", total_questions=2, status="completed")
        self.db.add(s_a)
        self.db.commit()

        # Query history for User B
        res_b = self.test_client.get(f"/interview/sessions?user_id={self.test_user_b.id}")
        self.assertEqual(res_b.status_code, 200)
        self.assertEqual(len(res_b.json()), 0, "User B must not see User A's session history!")

    def test_o_backend_session_history_invalid_user_404(self):
        """O. GET /interview/sessions for non-existent user returns 404."""
        res = self.test_client.get("/interview/sessions?user_id=999999")
        self.assertEqual(res.status_code, 404)

    def test_p_backend_session_history_report_available_flag(self):
        """P. report_available accurately reflects existence of FinalReport in SQLite."""
        s = InterviewSession(user_id=self.test_user.id, interview_type="python", difficulty="medium", total_questions=1, status="completed")
        self.db.add(s)
        self.db.commit()

        # Before report generation
        res1 = self.test_client.get(f"/interview/sessions?user_id={self.test_user.id}")
        self.assertFalse(res1.json()[0]["report_available"])

        # Add FinalReport record
        fr = FinalReport(
            session_id=s.id,
            overall_score=8.5,
            performance_summary="Solid performance",
            strengths=json.dumps(["Python internals"]),
            weaknesses=json.dumps(["Asyncio"]),
            study_topics=json.dumps(["Event loop"]),
            recommendations=json.dumps(["Practice asyncio"]),
        )
        self.db.add(fr)
        self.db.commit()

        # After report generation
        res2 = self.test_client.get(f"/interview/sessions?user_id={self.test_user.id}")
        self.assertTrue(res2.json()[0]["report_available"])

    def test_q_backend_offline_controlled_error(self):
        """Q. API client gracefully raises APIError when backend cannot be reached."""
        with patch("requests.get", side_effect=requests.exceptions.ConnectionError("Connection refused")):
            from frontend.api_client import check_backend_health
            is_healthy = check_backend_health()
            self.assertFalse(is_healthy)

    def test_r_bridge_secrets_to_env_safe_population(self):
        """R. bridge_secrets_to_env extracts GEMINI_API_KEY into os.environ safely without error."""
        import streamlit as st
        from frontend.server_manager import bridge_secrets_to_env

        # Save existing env if present
        saved_key = os.environ.get("GEMINI_API_KEY")
        if "GEMINI_API_KEY" in os.environ:
            del os.environ["GEMINI_API_KEY"]

        try:
            # Simulate st.secrets with mock dictionary
            with patch.object(st, "secrets", {"GEMINI_API_KEY": "test-mock-key-123", "API_BASE_URL": "http://127.0.0.1:8000"}):
                bridge_secrets_to_env()
                self.assertEqual(os.environ.get("GEMINI_API_KEY"), "test-mock-key-123")
                self.assertEqual(os.environ.get("API_BASE_URL"), "http://127.0.0.1:8000")
        finally:
            # Restore original environment
            if saved_key is not None:
                os.environ["GEMINI_API_KEY"] = saved_key
            elif "GEMINI_API_KEY" in os.environ:
                del os.environ["GEMINI_API_KEY"]

    def test_s_ensure_backend_running_healthy_noop(self):
        """S. ensure_backend_running returns True immediately when backend is already healthy."""
        from frontend.server_manager import ensure_backend_running

        with patch("frontend.server_manager.check_backend_health", return_value=True):
            with patch("frontend.server_manager.get_backend_manager") as mock_get_mgr:
                result = ensure_backend_running(timeout_seconds=1)
                self.assertTrue(result)
                mock_get_mgr.assert_not_called()

    def test_t_backend_process_manager_lifecycle(self):
        """T. BackendProcessManager launches uvicorn subprocess when backend is offline."""
        from frontend.server_manager import BackendProcessManager

        mgr = BackendProcessManager(repo_root="/test/root")
        self.assertFalse(mgr.is_alive())

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None

        with patch("frontend.server_manager.check_backend_health", return_value=False):
            with patch("subprocess.Popen", return_value=mock_proc) as mock_popen:
                mgr.start()
                self.assertTrue(mgr.is_alive())
                mock_popen.assert_called_once()
                args, kwargs = mock_popen.call_args
                cmd = args[0]
                self.assertIn("-m", cmd)
                self.assertIn("uvicorn", cmd)
                self.assertIn("app.main:app", cmd)
                self.assertIn("8000", cmd)
                self.assertEqual(kwargs.get("cwd"), "/test/root")

                # Test graceful stop
                mgr.stop()
                mock_proc.terminate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
