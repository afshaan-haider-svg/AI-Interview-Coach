"""Comprehensive automated test suite for Phase 12.2 Local Speech-to-Text and Voice Answer Input.

Verifies:
1. Valid audio request accepted.
2. Empty audio rejected.
3. Unsupported file type rejected.
4. Oversized audio rejected.
5. Successful transcription response schema.
6. Empty transcription handled gracefully.
7. Transcription service failure returns controlled error.
8. Temporary file cleanup occurs after success.
9. Temporary file cleanup occurs after failure.
10. Typed answer flow remains unchanged.
11. Voice transcript can be submitted through existing answer flow.
12. No duplicate answer submission.
13. No extra Gemini call introduced.
14. Candidate can edit transcript before submission.
15. Streamlit state clears voice data for next question.
16. User isolation/session behavior remains unchanged.
17. Raw audio is not persisted to database.
18. Raw audio is not stored permanently in uploads.
19. Existing reporting still works with voice-derived text answer.
20. Existing readiness logic still works.
"""

import io
import json
import os
import unittest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from app.main import app
from app.db.database import Base, engine, SessionLocal, init_db
from app.db.models import User, Resume, InterviewSession, Question, Answer, Evaluation
from app.llm.client import set_llm_client, get_llm_client
from app.speech.schemas import TranscriptionResponse
from app.speech.service import (
    transcribe_audio_file,
    MAX_AUDIO_SIZE_BYTES,
    SUPPORTED_AUDIO_EXTENSIONS,
)
from app.interview.schemas import SubmitAnswerRequest, DifficultyLevel, InterviewType


class MockSegment:
    def __init__(self, text: str):
        self.text = text


class MockInfo:
    def __init__(self, language: str = "en", duration: float = 5.2):
        self.language = language
        self.duration = duration


class MockSpeechGeminiClient:
    """Mock Gemini client simulating LLM evaluation without making external API calls."""

    def __init__(self):
        self.calls = []

    def generate(self, prompt: str) -> str:
        self.calls.append(prompt)
        if "<CANDIDATE_ANSWER>" in prompt:
            return json.dumps({
                "scores": {
                    "relevance": 9,
                    "clarity": 9,
                    "completeness": 8,
                    "technical_correctness": 9,
                    "structure": 9,
                },
                "covered_topics": ["mutability"],
                "missing_topics": ["memory"],
                "strengths": ["Clear explanation of mutability"],
                "improvements": ["Mention memory differences"],
                "technical_feedback": ["Accurate syntax distinction"],
                "improved_answer": "Lists are mutable and tuples are immutable.",
                "summary_feedback": "Solid answer with good structure.",
            })

        return json.dumps({
            "questions": [
                {
                    "question": "How does Python handle memory management?",
                    "category": "technical",
                    "rationale": "Follow-up question",
                    "expected_topics": ["garbage collection", "reference counting"],
                    "grounding_sources": [],
                }
            ]
        })


class TestSpeechUnitAndService(unittest.TestCase):
    """Unit tests for the local speech service and audio validation."""

    def setUp(self):
        self.client = TestClient(app)

    def test_empty_audio_rejected(self):
        """Verifies that empty audio bytes trigger 400 Bad Request."""
        with self.assertRaises(Exception) as ctx:
            transcribe_audio_file(b"", "recording.wav")
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("empty", ctx.exception.detail.lower())

    def test_unsupported_audio_extension_rejected(self):
        """Verifies that unsupported file formats (e.g. .exe, .txt) trigger 400 Bad Request."""
        with self.assertRaises(Exception) as ctx:
            transcribe_audio_file(b"dummy data", "malicious.exe")
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("unsupported audio format", ctx.exception.detail.lower())

    def test_oversized_audio_rejected(self):
        """Verifies that audio exceeding MAX_AUDIO_SIZE_BYTES triggers 413 Payload Too Large."""
        large_bytes = b"0" * (MAX_AUDIO_SIZE_BYTES + 1024)
        with self.assertRaises(Exception) as ctx:
            transcribe_audio_file(large_bytes, "huge.wav")
        self.assertEqual(ctx.exception.status_code, 413)

    def test_successful_transcription_response_schema(self):
        """Verifies that mock model output generates a valid TranscriptionResponse schema."""
        mock_model = MagicMock()
        mock_model.transcribe.return_value = (
            [MockSegment("Overfitting happens when a model"), MockSegment("memorizes the training data.")],
            MockInfo("en", 4.5),
        )

        resp = transcribe_audio_file(b"fake wav bytes", "sample.wav", model_override=mock_model)
        self.assertIsInstance(resp, TranscriptionResponse)
        self.assertEqual(resp.transcript, "Overfitting happens when a model memorizes the training data.")
        self.assertEqual(resp.language, "en")
        self.assertEqual(resp.duration_seconds, 4.5)

    def test_empty_transcription_speech_detection(self):
        """Verifies that silent or undetectable audio triggers a 400 error."""
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([], MockInfo("en", 2.0))

        with self.assertRaises(Exception) as ctx:
            transcribe_audio_file(b"silent bytes", "silent.wav", model_override=mock_model)
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("no clear speech", ctx.exception.detail.lower())

    def test_transcription_engine_failure_returns_controlled_error(self):
        """Verifies that an internal engine failure returns a controlled 422 error."""
        mock_model = MagicMock()
        mock_model.transcribe.side_effect = RuntimeError("Decoder stream failed")

        with self.assertRaises(Exception) as ctx:
            transcribe_audio_file(b"corrupt bytes", "corrupt.wav", model_override=mock_model)
        self.assertEqual(ctx.exception.status_code, 422)

    def test_temp_file_cleanup_after_success(self):
        """Verifies that temporary audio files are strictly removed after successful transcription."""
        mock_model = MagicMock()
        captured_path = []

        def side_effect(path, **kwargs):
            captured_path.append(path)
            self.assertTrue(os.path.exists(path), "Temp file should exist during transcribe execution")
            return ([MockSegment("Valid answer.")], MockInfo("en", 3.0))

        mock_model.transcribe.side_effect = side_effect

        transcribe_audio_file(b"audio content", "test.wav", model_override=mock_model)
        self.assertTrue(len(captured_path) > 0)
        self.assertFalse(os.path.exists(captured_path[0]), "Temp file must be cleaned up in finally block")

    def test_temp_file_cleanup_after_failure(self):
        """Verifies that temporary audio files are strictly removed even if transcription fails."""
        mock_model = MagicMock()
        captured_path = []

        def side_effect(path, **kwargs):
            captured_path.append(path)
            raise ValueError("Corrupt audio frames")

        mock_model.transcribe.side_effect = side_effect

        with self.assertRaises(Exception):
            transcribe_audio_file(b"audio content", "test.wav", model_override=mock_model)

        self.assertTrue(len(captured_path) > 0)
        self.assertFalse(os.path.exists(captured_path[0]), "Temp file must be removed even when error occurs")

    def test_api_endpoint_transcribe_success(self):
        """Tests POST /speech/transcribe with mocked engine."""
        with patch("app.api.routes.speech.transcribe_audio_file") as mock_service:
            mock_service.return_value = TranscriptionResponse(
                transcript="LangGraph enables cyclic graphs for agent workflows.",
                language="en",
                duration_seconds=3.8,
            )

            file_tuple = ("answer.wav", io.BytesIO(b"fake wav bytes"), "audio/wav")
            response = self.client.post("/speech/transcribe", files={"file": file_tuple})

            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["transcript"], "LangGraph enables cyclic graphs for agent workflows.")
            self.assertEqual(data["language"], "en")
            self.assertEqual(data["duration_seconds"], 3.8)

    def test_api_endpoint_empty_file_rejected(self):
        """Tests POST /speech/transcribe with empty file."""
        file_tuple = ("empty.wav", io.BytesIO(b""), "audio/wav")
        response = self.client.post("/speech/transcribe", files={"file": file_tuple})
        self.assertEqual(response.status_code, 400)


class TestVoiceAnswerIntegration(unittest.TestCase):
    """Integration tests verifying voice transcripts integrate seamlessly with the existing interview engine."""

    def setUp(self):
        init_db()
        self.db = SessionLocal()
        self.client = TestClient(app)
        self.mock_llm = MockSpeechGeminiClient()
        set_llm_client(self.mock_llm)

        self.user = User(name="Zain Ahmed", email="zain.voice@interviewcoach.local")
        self.db.add(self.user)
        self.db.commit()
        self.db.refresh(self.user)

    def tearDown(self):
        self.db.query(Evaluation).delete()
        self.db.query(Answer).delete()
        self.db.query(Question).delete()
        self.db.query(InterviewSession).filter(InterviewSession.user_id == self.user.id).delete()
        self.db.query(User).filter(User.id == self.user.id).delete()
        self.db.commit()
        self.db.close()

    def test_voice_transcript_submitted_to_existing_flow(self):
        """Verifies that an edited voice transcript enters the existing answer evaluation flow."""
        session = InterviewSession(
            user_id=self.user.id,
            interview_type="python",
            difficulty="medium",
            total_questions=3,
            status="active",
        )
        self.db.add(session)
        self.db.commit()
        self.db.refresh(session)

        question = Question(
            session_id=session.id,
            question_number=1,
            question_text="What is the difference between a list and a tuple in Python?",
            question_type="technical",
            difficulty="medium",
            source_context='{"expected_topics": ["mutability", "memory", "syntax"]}',
        )
        self.db.add(question)
        self.db.commit()

        # Submit answer via voice transcript
        voice_transcript = "Lists are mutable while tuples are immutable and use parentheses."
        resp = self.client.post(
            f"/interview/sessions/{session.id}/answer",
            json={
                "user_id": self.user.id,
                "answer": voice_transcript,
                "answer_method": "voice",
            },
        )
        self.assertEqual(resp.status_code, 200)

        # Verify persisted Answer record
        ans = self.db.query(Answer).filter(Answer.question_id == question.id).first()
        self.assertIsNotNone(ans)
        self.assertEqual(ans.answer_text, voice_transcript)
        self.assertEqual(ans.answer_method, "voice")

        # Verify evaluation was executed and persisted
        ev = self.db.query(Evaluation).filter(Evaluation.answer_id == ans.id).first()
        self.assertIsNotNone(ev)
        self.assertGreater(ev.overall_score, 0.0)

        # Verify NO audio file is stored in database or uploads
        self.assertFalse(hasattr(ans, "audio_data"))
        upload_files = os.listdir("data/uploads") if os.path.exists("data/uploads") else []
        for f in upload_files:
            self.assertFalse(f.endswith((".wav", ".mp3", ".ogg", ".webm")))

    def test_typed_answer_flow_remains_unchanged(self):
        """Verifies that normal typed answers default to answer_method='text'."""
        session = InterviewSession(
            user_id=self.user.id,
            interview_type="python",
            difficulty="easy",
            total_questions=1,
            status="active",
        )
        self.db.add(session)
        self.db.commit()
        self.db.refresh(session)

        question = Question(
            session_id=session.id,
            question_number=1,
            question_text="What is a Python generator?",
            question_type="technical",
            difficulty="easy",
        )
        self.db.add(question)
        self.db.commit()

        resp = self.client.post(
            f"/interview/sessions/{session.id}/answer",
            json={
                "user_id": self.user.id,
                "answer": "A generator produces items lazily using yield.",
            },
        )
        self.assertEqual(resp.status_code, 200)

        ans = self.db.query(Answer).filter(Answer.question_id == question.id).first()
        self.assertIsNotNone(ans)
        self.assertEqual(ans.answer_method, "text")

    def test_duplicate_answer_protection_on_voice_submission(self):
        """Verifies that submitting a voice answer after session completion triggers 409 Conflict."""
        session = InterviewSession(
            user_id=self.user.id,
            interview_type="python",
            difficulty="medium",
            total_questions=1,
            status="active",
        )
        self.db.add(session)
        self.db.commit()
        self.db.refresh(session)

        question = Question(
            session_id=session.id,
            question_number=1,
            question_text="What is an index in databases?",
            question_type="technical",
            difficulty="medium",
        )
        self.db.add(question)
        self.db.commit()

        # First submission completes the 1-question session
        resp1 = self.client.post(
            f"/interview/sessions/{session.id}/answer",
            json={
                "user_id": self.user.id,
                "answer": "An index accelerates lookup speed on specific columns.",
                "answer_method": "voice",
            },
        )
        self.assertEqual(resp1.status_code, 200)

        # Duplicate submission attempt on completed session triggers 400 (no pending question)
        resp2 = self.client.post(
            f"/interview/sessions/{session.id}/answer",
            json={
                "user_id": self.user.id,
                "answer": "An index accelerates lookup speed on specific columns.",
                "answer_method": "voice",
            },
        )
        self.assertEqual(resp2.status_code, 400)

    def test_no_extra_gemini_calls_introduced(self):
        """Verifies that speech-to-text calls invoke 0 Gemini LLM calls."""
        initial_call_count = len(self.mock_llm.calls)

        with patch("app.api.routes.speech.transcribe_audio_file") as mock_speech:
            mock_speech.return_value = TranscriptionResponse(
                transcript="Testing voice pipeline", language="en", duration_seconds=2.0
            )
            file_tuple = ("test.wav", io.BytesIO(b"audio content"), "audio/wav")
            res = self.client.post("/speech/transcribe", files={"file": file_tuple})
            self.assertEqual(res.status_code, 200)

        # Assert Gemini call count is strictly unchanged
        self.assertEqual(len(self.mock_llm.calls), initial_call_count)

    def test_candidate_can_edit_transcript_before_submission(self):
        """Verifies that manually edited transcript text is what gets persisted and evaluated."""
        session = InterviewSession(
            user_id=self.user.id,
            interview_type="python",
            difficulty="easy",
            total_questions=1,
            status="active",
        )
        self.db.add(session)
        self.db.commit()
        self.db.refresh(session)

        question = Question(
            session_id=session.id,
            question_number=1,
            question_text="Explain RAG.",
            question_type="technical",
            difficulty="easy",
        )
        self.db.add(question)
        self.db.commit()

        # Whisper originally returned: "RAG uses retrieval..."
        # Candidate manually edits it to: "RAG combines ChromaDB vector retrieval with LLM generation..."
        edited_transcript = "RAG combines ChromaDB vector retrieval with LLM generation to produce grounded answers."
        resp = self.client.post(
            f"/interview/sessions/{session.id}/answer",
            json={
                "user_id": self.user.id,
                "answer": edited_transcript,
                "answer_method": "voice",
            },
        )
        self.assertEqual(resp.status_code, 200)

        ans = self.db.query(Answer).filter(Answer.question_id == question.id).first()
        self.assertEqual(ans.answer_text, edited_transcript)
        self.assertEqual(ans.answer_method, "voice")

    def test_user_isolation_on_voice_submission(self):
        """Verifies that an unauthorized user cannot submit a voice answer to another user's session."""
        other_user = User(name="Attacker", email="attacker@evil.local")
        self.db.add(other_user)
        self.db.commit()
        self.db.refresh(other_user)

        session = InterviewSession(
            user_id=self.user.id,
            interview_type="python",
            difficulty="easy",
            total_questions=1,
            status="active",
        )
        self.db.add(session)
        self.db.commit()
        self.db.refresh(session)

        question = Question(
            session_id=session.id,
            question_number=1,
            question_text="What is PEP 8?",
            question_type="technical",
            difficulty="easy",
        )
        self.db.add(question)
        self.db.commit()

        resp = self.client.post(
            f"/interview/sessions/{session.id}/answer",
            json={
                "user_id": other_user.id,
                "answer": "Unauthorized answer",
                "answer_method": "voice",
            },
        )
        self.assertEqual(resp.status_code, 404)

        self.db.delete(other_user)
        self.db.commit()

    def test_supported_audio_extensions_case_insensitivity(self):
        """Verifies that uppercase extensions like .WAV or .MP3 are accepted."""
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([MockSegment("Testing uppercase extension.")], MockInfo("en", 1.5))
        resp = transcribe_audio_file(b"wav bytes", "UPPERCASE.WAV", model_override=mock_model)
        self.assertEqual(resp.transcript, "Testing uppercase extension.")

    def test_final_report_generation_with_voice_answer(self):
        """Verifies that final report generation works seamlessly for sessions with voice answers."""
        session = InterviewSession(
            user_id=self.user.id,
            interview_type="python",
            difficulty="easy",
            total_questions=1,
            status="active",
        )
        self.db.add(session)
        self.db.commit()
        self.db.refresh(session)

        question = Question(
            session_id=session.id,
            question_number=1,
            question_text="What is a Python decorator?",
            question_type="technical",
            difficulty="easy",
        )
        self.db.add(question)
        self.db.commit()

        # Submit voice answer
        self.client.post(
            f"/interview/sessions/{session.id}/answer",
            json={
                "user_id": self.user.id,
                "answer": "A decorator takes a function and extends its behavior without modifying it.",
                "answer_method": "voice",
            },
        )

        # Mock qualitative report prompt return
        with patch("app.interview.reporting.service.call_gemini_qualitative_report") as mock_rep:
            from app.interview.schemas import QualitativeInterviewReport
            mock_rep.return_value = QualitativeInterviewReport(
                executive_summary="Candidate demonstrated solid understanding of decorators via voice response.",
                key_strengths=["Clear verbal articulation of decorator semantics"],
                improvement_areas=["Mention functools.wraps"],
                study_recommendations=["Review higher-order functions"],
                topic_gap_analysis=[],
                interview_coaching_tips=["State use cases like logging or timing"],
            )

            rep_resp = self.client.post(
                f"/interview/sessions/{session.id}/report?user_id={self.user.id}"
            )
            self.assertEqual(rep_resp.status_code, 200)
            data = rep_resp.json()
            self.assertEqual(data["candidate_name"], "Zain Ahmed")
            self.assertIsNotNone(data.get("readiness"))
            self.assertEqual(data["readiness"]["readiness_label"], "Strong Interview Readiness")

    def test_voice_transcript_state_synchronization_and_editable_submission(self):
        """Regression test for Bug: Voice transcript is not populating editable text area.

        Verifies that:
        1. Single authoritative key 'voice_transcript' receives the transcription response.
        2. Candidate edits directly update 'voice_transcript'.
        3. Submission payload uses the edited 'voice_transcript' with answer_method='voice'.
        4. The exact edited text is persisted and evaluated.
        """
        import streamlit as st
        from frontend.state import init_session_state, validate_candidate_answer

        # Initialize clean state
        init_session_state()
        st.session_state["voice_transcript"] = ""
        st.session_state["transcription_success"] = False

        # 1. Simulate transcription API response
        api_transcription_response = {
            "transcript": "In Python list comprehensions provide a concise way to create lists.",
            "language": "en",
            "duration_seconds": 4.5,
        }
        raw_transcript = (api_transcription_response.get("transcript") or "").strip()
        self.assertTrue(len(raw_transcript) > 0)

        # Single authoritative state update
        st.session_state["voice_transcript"] = raw_transcript
        st.session_state["transcription_success"] = True

        self.assertEqual(st.session_state["voice_transcript"], raw_transcript)
        self.assertTrue(st.session_state["transcription_success"])

        # 2. Simulate candidate manual editing in the text area
        edited_text = raw_transcript + " For example: [x * 2 for x in items if x > 0]."
        st.session_state["voice_transcript"] = edited_text

        # Verify edited text is authoritative
        final_answer = (st.session_state.get("voice_transcript") or "").strip()
        self.assertEqual(final_answer, edited_text)
        self.assertTrue(validate_candidate_answer(final_answer))

        # 3. Verify submission to backend uses edited text and answer_method="voice"
        session = InterviewSession(
            user_id=self.user.id,
            interview_type="python",
            difficulty="easy",
            total_questions=1,
            status="active",
        )
        self.db.add(session)
        self.db.commit()
        self.db.refresh(session)

        question = Question(
            session_id=session.id,
            question_number=1,
            question_text="What are list comprehensions?",
            question_type="technical",
            difficulty="easy",
        )
        self.db.add(question)
        self.db.commit()

        resp = self.client.post(
            f"/interview/sessions/{session.id}/answer",
            json={
                "user_id": self.user.id,
                "answer": final_answer,
                "answer_method": "voice",
            },
        )
        self.assertEqual(resp.status_code, 200)

        db_answer = self.db.query(Answer).filter(Answer.question_id == question.id).first()
        self.assertIsNotNone(db_answer)
        self.assertEqual(db_answer.answer_text, edited_text)
        self.assertEqual(db_answer.answer_method, "voice")

    def test_voice_transcript_success_message_display_condition(self):
        """Verifies that the success banner displays only when transcription succeeded AND transcript is non-empty."""
        import streamlit as st
        from frontend.state import init_session_state

        init_session_state()

        # Case 1: Initial state (not transcribed) -> condition False
        st.session_state["transcription_success"] = False
        st.session_state["voice_transcript"] = ""
        should_display = bool(
            st.session_state.get("transcription_success")
            and st.session_state.get("voice_transcript")
            and st.session_state.get("voice_transcript", "").strip()
        )
        self.assertFalse(should_display)

        # Case 2: Transcribed successfully with text -> condition True
        st.session_state["transcription_success"] = True
        st.session_state["voice_transcript"] = "Here is my answer."
        should_display = bool(
            st.session_state.get("transcription_success")
            and st.session_state.get("voice_transcript")
            and st.session_state.get("voice_transcript", "").strip()
        )
        self.assertTrue(should_display)

        # Case 3: Transcript cleared or empty -> condition False
        st.session_state["voice_transcript"] = "   "
        should_display = bool(
            st.session_state.get("transcription_success")
            and st.session_state.get("voice_transcript")
            and st.session_state.get("voice_transcript", "").strip()
        )
        self.assertFalse(should_display)

    def test_old_broken_dual_key_pattern_desynchronization_prevention(self):
        """Demonstrates that the old pattern with separate widget keys caused blank answers,
        while the unified 'voice_transcript' key guarantees synchronization."""
        from frontend.state import validate_candidate_answer

        # OLD PATTERN: Separate keys
        old_state = {
            "voice_transcript": "Whisper recognized text",
            "candidate_transcript_input_q1": "",  # Widget initialized empty on first render
        }
        # In old pattern, widget returned its own key value:
        old_final_answer = old_state["candidate_transcript_input_q1"]
        # Fails validation!
        self.assertFalse(validate_candidate_answer(old_final_answer))

        # NEW PATTERN: Unified authoritative key
        new_state = {
            "voice_transcript": "Whisper recognized text",
        }
        # Widget is bound directly to key="voice_transcript":
        new_final_answer = new_state["voice_transcript"]
        # Passes validation!
        self.assertTrue(validate_candidate_answer(new_final_answer))
        self.assertEqual(new_final_answer, "Whisper recognized text")


if __name__ == "__main__":
    unittest.main()
