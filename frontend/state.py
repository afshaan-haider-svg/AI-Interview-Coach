"""Session state management and friendly option mappings for Streamlit UI."""

from typing import Optional
import streamlit as st

INTERVIEW_TYPE_OPTIONS = {
    "hr": "HR & Behavioral",
    "python": "Python Core & Advanced",
    "ai_ml": "AI & Machine Learning",
    "data_science": "Data Science & Analytics",
    "internship": "Internship & Early Career",
}

DIFFICULTY_OPTIONS = {
    "easy": "Easy",
    "medium": "Medium",
    "hard": "Hard",
}

ADAPTIVE_REASON_LABELS = {
    "high_performance": "Strong performance",
    "low_performance": "More foundation practice",
    "stable_performance": "Difficulty maintained",
    "technical_guardrail": "Technical accuracy needs reinforcement",
    "completeness_guardrail": "Answer needs more depth",
    "upper_bound": "Highest difficulty maintained",
    "lower_bound": "Foundation difficulty maintained",
}


def validate_candidate_answer(answer: str) -> bool:
    """Validates that a candidate answer is not empty or composed solely of whitespace."""
    if not answer:
        return False
    return bool(answer.strip())


def init_session_state() -> None:
    """Initializes Streamlit session_state keys with sensible defaults."""
    defaults = {
        "current_page": "home",
        "user_id": None,
        "user_name": "Demo Candidate",
        "candidate_name": "Demo Candidate",
        "user_email": "demo_candidate@interviewcoach.local",
        "backend_connected": True,
        # Setup inputs
        "resume_id": None,
        "resume_filename": None,
        "job_description_id": None,
        "job_title": None,
        "selected_interview_type": "python",
        "selected_difficulty": "medium",
        "selected_question_count": 5,
        # Active interview session
        "session_id": None,
        "interview_status": None,
        "current_question": None,
        "current_question_number": 0,
        "current_difficulty": "medium",
        "next_question_buffer": None,
        # Evaluation & feedback
        "latest_evaluation": None,
        "latest_adaptive_decision": None,
        "answer_text": "",
        "show_feedback_view": False,
        # Voice input
        "answer_mode": "type",
        "voice_transcript": "",
        "voice_audio_bytes": None,
        "transcription_error": None,
        "transcription_success": False,
        "is_submitting_answer": False,
        # Final report & history
        "final_report": None,
        "history_sessions": [],
        "error_message": None,
        "success_message": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_interview_state() -> None:
    """Clears all session-specific interview state while preserving user profile and navigation."""
    st.session_state.session_id = None
    st.session_state.interview_status = None
    st.session_state.resume_id = None
    st.session_state.resume_filename = None
    st.session_state.job_description_id = None
    st.session_state.job_title = None
    st.session_state.current_question = None
    st.session_state.current_question_number = 0
    st.session_state.current_difficulty = "medium"
    st.session_state.next_question_buffer = None
    st.session_state.latest_evaluation = None
    st.session_state.latest_adaptive_decision = None
    st.session_state.answer_text = ""
    st.session_state.answer_mode = "type"
    st.session_state.voice_transcript = ""
    st.session_state.voice_audio_bytes = None
    st.session_state.transcription_error = None
    st.session_state.transcription_success = False
    st.session_state.is_submitting_answer = False
    st.session_state.show_feedback_view = False
    st.session_state.final_report = None
    st.session_state.error_message = None
    st.session_state.success_message = None
    st.session_state.current_page = "setup"


def format_adaptive_reason(reason_code: Optional[str]) -> str:
    """Returns a friendly description for adaptive decision reason codes."""
    if not reason_code:
        return "Difficulty maintained"
    return ADAPTIVE_REASON_LABELS.get(reason_code, reason_code.replace("_", " ").capitalize())
