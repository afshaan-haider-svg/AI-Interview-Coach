"""API routes for personalized interview question generation, answer evaluation, and session workflow."""

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import InterviewSession, User
from app.interview.evaluation_service import evaluate_answer
from app.interview.question_service import generate_interview_questions
from app.interview.reporting import (
    generate_session_final_report,
    get_session_final_report,
)
from app.interview.reporting.analytics import compute_readiness_level
from app.interview.schemas import (
    DifficultyLevel,
    EvaluationRequest,
    EvaluationResponse,
    FinalInterviewReportResponse,
    GenerateReportRequest,
    InterviewSessionResponse,
    InterviewType,
    QuestionGenerationRequest,
    QuestionGenerationResponse,
    SessionSummary,
    StartInterviewRequest,
    StartInterviewResponse,
    SubmitAnswerRequest,
    SubmitAnswerResponse,
)
from app.interview.workflow import (
    get_interview_session_details,
    start_interview_session,
    submit_answer_to_session,
)

router = APIRouter(prefix="/interview", tags=["Interview"])


@router.post(
    "/questions/generate",
    response_model=QuestionGenerationResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate personalized interview questions",
    description=(
        "Retrieves relevant candidate and job context via local RAG and generates a structured, "
        "personalized set of interview questions in a single LLM invocation."
    ),
)
def create_interview_questions(
    req: QuestionGenerationRequest,
) -> QuestionGenerationResponse:
    """Coordinates context retrieval and question generation."""
    return generate_interview_questions(req)


@router.post(
    "/answers/evaluate",
    response_model=EvaluationResponse,
    status_code=status.HTTP_200_OK,
    summary="Evaluate candidate interview answer",
    description=(
        "Evaluates a single interview answer across five core dimensions (relevance, clarity, "
        "completeness, technical correctness, structure) using grounded Gemini reasoning and "
        "deterministic Python scoring."
    ),
)
def create_answer_evaluation(
    req: EvaluationRequest,
) -> EvaluationResponse:
    """Coordinates grounding validation and answer evaluation."""
    return evaluate_answer(req)


@router.post(
    "/sessions/start",
    response_model=StartInterviewResponse,
    status_code=status.HTTP_200_OK,
    summary="Start a new interview session",
    description=(
        "Initializes an interview session, invokes LangGraph to generate the first question, "
        "and persists session state in the database."
    ),
)
def create_interview_session(
    req: StartInterviewRequest,
    db: Session = Depends(get_db),
) -> StartInterviewResponse:
    """Initializes session and returns the first question."""
    return start_interview_session(req, db)


@router.post(
    "/sessions/{session_id}/answer",
    response_model=SubmitAnswerResponse,
    status_code=status.HTTP_200_OK,
    summary="Submit answer to current pending question",
    description=(
        "Accepts candidate answer, resumes the LangGraph workflow, evaluates the answer, "
        "persists the evaluation, and either generates the next question or completes the session."
    ),
)
def submit_session_answer(
    session_id: int,
    req: SubmitAnswerRequest,
    db: Session = Depends(get_db),
) -> SubmitAnswerResponse:
    """Resumes workflow turn with candidate answer."""
    return submit_answer_to_session(session_id, req, db)


@router.get(
    "/sessions/{session_id}",
    response_model=InterviewSessionResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve interview session progress and history",
    description="Returns session metadata, current pending question, and all completed turn evaluations.",
)
def get_interview_session(
    session_id: int,
    user_id: int = Query(..., gt=0, description="User ID for strict tenant isolation."),
    db: Session = Depends(get_db),
) -> InterviewSessionResponse:
    """Retrieves session details and turn history."""
    return get_interview_session_details(session_id, user_id, db)


@router.post(
    "/sessions/{session_id}/report",
    response_model=FinalInterviewReportResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate final interview report and coaching analytics",
    description=(
        "Synthesizes deterministic Python metrics and grounded Gemini qualitative feedback "
        "for a completed interview session. Calling again is idempotent and returns the existing report."
    ),
)
def generate_interview_report(
    session_id: int,
    user_id: Optional[int] = Query(None, gt=0, description="User ID for strict tenant isolation."),
    req: Optional[GenerateReportRequest] = None,
    db: Session = Depends(get_db),
) -> FinalInterviewReportResponse:
    """Generates and persists the final interview report."""
    effective_user_id = user_id or (req.user_id if req else None)
    if not effective_user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="user_id is required either as query parameter (?user_id=X) or in request body.",
        )
    return generate_session_final_report(session_id, effective_user_id, db)


@router.get(
    "/sessions/{session_id}/report",
    response_model=FinalInterviewReportResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve final interview report and coaching analytics",
    description="Returns the persisted final report for a completed session without making any LLM calls.",
)
def get_interview_report(
    session_id: int,
    user_id: int = Query(..., gt=0, description="User ID for strict tenant isolation."),
    db: Session = Depends(get_db),
) -> FinalInterviewReportResponse:
    """Retrieves the existing final report for a completed session."""
    return get_session_final_report(session_id, user_id, db)


@router.get(
    "/sessions",
    response_model=List[SessionSummary],
    status_code=status.HTTP_200_OK,
    summary="List interview session history for candidate",
    description="Returns a chronological list (newest first) of interview session summaries for the specified user.",
)
def list_interview_sessions(
    user_id: int = Query(..., gt=0, description="User ID for strict tenant isolation."),
    db: Session = Depends(get_db),
) -> List[SessionSummary]:
    """Lists safe interview session history for the specified user."""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with id {user_id} not found.",
        )

    sessions = (
        db.query(InterviewSession)
        .filter(InterviewSession.user_id == user_id)
        .order_by(InterviewSession.id.desc())
        .all()
    )

    summaries: List[SessionSummary] = []
    for s in sessions:
        completed_count = sum(1 for q in s.questions if q.answer and q.answer.evaluation)
        has_report = s.final_report is not None
        candidate_name = (
            s.resume.candidate_name
            if (s.resume and s.resume.candidate_name)
            else (s.user.name if (s.user and s.user.name) else "Demo Candidate")
        )
        readiness_label = None
        if s.overall_score is not None:
            _, readiness_label = compute_readiness_level(s.overall_score)

        summaries.append(
            SessionSummary(
                session_id=s.id,
                interview_type=InterviewType(s.interview_type),
                initial_difficulty=DifficultyLevel(s.difficulty),
                status=s.status,
                candidate_name=candidate_name,
                readiness_label=readiness_label,
                total_questions=s.total_questions,
                completed_questions=completed_count,
                overall_score=s.overall_score,
                started_at=s.started_at,
                completed_at=s.completed_at,
                report_available=has_report,
            )
        )

    return summaries


