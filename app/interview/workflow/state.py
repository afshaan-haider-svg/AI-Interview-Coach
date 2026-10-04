"""LangGraph state definition for the interview session workflow."""

from typing import Any, Dict, List, Optional, TypedDict
from app.interview.schemas import DifficultyLevel, InterviewType


class InterviewState(TypedDict, total=False):
    """Serializable typed state tracking interview session progress across turns."""
    session_id: int
    user_id: int

    resume_id: Optional[int]
    job_description_id: Optional[int]

    interview_type: InterviewType
    difficulty: DifficultyLevel
    initial_difficulty: DifficultyLevel
    current_difficulty: DifficultyLevel

    total_questions: int
    current_question_number: int

    current_question_db_id: Optional[int]
    current_question: Optional[Dict[str, Any]]

    questions_asked: List[str]

    pending_answer: Optional[str]
    latest_evaluation: Optional[Dict[str, Any]]
    latest_adaptive_decision: Optional[Dict[str, Any]]

    completed_question_count: int

    status: str

    error: Optional[str]
