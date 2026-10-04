"""Interview module exposing schemas, prompts, question generation, and evaluation services."""

from app.interview.adaptive import decide_next_difficulty
from app.interview.evaluation_service import evaluate_answer
from app.interview.question_service import generate_interview_questions
from app.interview.reporting import (
    compute_interview_analytics,
    deduplicate_topics,
    generate_session_final_report,
    get_session_final_report,
)
from app.interview.schemas import (
    AdaptiveDecision,
    AnswerEvaluation,
    DifficultyLevel,
    DimensionScores,
    EvaluationRequest,
    EvaluationResponse,
    FinalInterviewReportResponse,
    InterviewAnalytics,
    InterviewQuestion,
    InterviewType,
    QuestionCategory,
    QuestionGenerationRequest,
    QuestionGenerationResponse,
    QuestionReport,
    QuestionSource,
    calculate_overall_score,
)

__all__ = [
    "AdaptiveDecision",
    "AnswerEvaluation",
    "DifficultyLevel",
    "DimensionScores",
    "EvaluationRequest",
    "EvaluationResponse",
    "FinalInterviewReportResponse",
    "InterviewAnalytics",
    "InterviewQuestion",
    "InterviewType",
    "QuestionCategory",
    "QuestionGenerationRequest",
    "QuestionGenerationResponse",
    "QuestionReport",
    "QuestionSource",
    "calculate_overall_score",
    "compute_interview_analytics",
    "decide_next_difficulty",
    "deduplicate_topics",
    "evaluate_answer",
    "generate_interview_questions",
    "generate_session_final_report",
    "get_session_final_report",
]
