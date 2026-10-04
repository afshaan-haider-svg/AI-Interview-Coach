"""Service layer for final interview report generation, persistence, and retrieval."""

import json
from typing import Any, Dict, List, Tuple
from fastapi import HTTPException, status
from pydantic import ValidationError
from sqlalchemy.orm import Session

from app.db.models import Evaluation, FinalReport, InterviewSession, Question
from app.interview.reporting.analytics import (
    compute_interview_analytics,
    compute_interview_readiness,
)
from app.interview.reporting.prompts import build_qualitative_report_prompt
from app.interview.schemas import (
    DifficultyLevel,
    DimensionScores,
    FinalInterviewReportResponse,
    InterviewAnalytics,
    InterviewType,
    QualitativeInterviewReport,
    QuestionReport,
)
from app.llm.client import get_llm_client


def _clean_json_text(text: str) -> str:
    """Strips markdown code blocks and surrounding whitespace from LLM output."""
    raw = text.strip()
    if raw.startswith("```json"):
        raw = raw[7:]
    elif raw.startswith("```"):
        raw = raw[3:]
    if raw.endswith("```"):
        raw = raw[:-3]
    return raw.strip()


def call_gemini_qualitative_report(prompt: str) -> QualitativeInterviewReport:
    """Executes a single Gemini call to generate the qualitative coaching report.
    
    If the response is malformed, attempts at most ONE retry.
    """
    client = get_llm_client()

    for attempt in range(2):
        try:
            curr_prompt = prompt if attempt == 0 else (
                f"{prompt}\n\nIMPORTANT: Your previous output could not be parsed as valid JSON matching the schema. "
                "Output RAW valid JSON strictly matching the schema with all required fields."
            )
            raw_output = client.generate(curr_prompt)
            cleaned = _clean_json_text(raw_output)
            data = json.loads(cleaned)
            return QualitativeInterviewReport(**data)
        except (json.JSONDecodeError, ValidationError) as e:
            if attempt == 1:
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"Failed to generate structured qualitative report from LLM: {str(e)}",
                )


def extract_turn_data_and_question_reports(
    questions: List[Question],
) -> Tuple[List[Dict[str, Any]], List[QuestionReport]]:
    """Transforms persisted Question, Answer, and Evaluation records into analytics inputs and QuestionReports."""
    analytics_turns: List[Dict[str, Any]] = []
    question_reports: List[QuestionReport] = []

    for q in questions:
        if not q.answer or not q.answer.evaluation:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Cannot generate report: Question {q.question_number} is missing an answer or evaluation.",
            )

        ev: Evaluation = q.answer.evaluation

        # Unpack JSON columns
        strengths = json.loads(ev.strengths or "[]")
        improvements = json.loads(ev.improvement or "[]")
        weaknesses_data = json.loads(ev.weaknesses or "{}")

        missing_topics = weaknesses_data.get("missing_topics", [])
        covered_topics = weaknesses_data.get("covered_topics", [])
        technical_feedback = weaknesses_data.get("technical_feedback", [])

        dim_scores = DimensionScores(
            relevance=int(round(ev.relevance_score)),
            clarity=int(round(ev.clarity_score)),
            completeness=int(round(ev.completeness_score)),
            technical_correctness=int(round(ev.technical_score)),
            structure=int(round(ev.structure_score)),
        )

        analytics_turns.append({
            "question_number": q.question_number,
            "difficulty": q.difficulty,
            "overall_score": ev.overall_score,
            "scores": {
                "relevance": ev.relevance_score,
                "clarity": ev.clarity_score,
                "completeness": ev.completeness_score,
                "technical_correctness": ev.technical_score,
                "structure": ev.structure_score,
            },
            "covered_topics": covered_topics,
            "missing_topics": missing_topics,
        })

        q_diff = DifficultyLevel(q.difficulty) if q.difficulty else DifficultyLevel.MEDIUM

        question_reports.append(
            QuestionReport(
                question_number=q.question_number,
                question=q.question_text,
                difficulty=q_diff,
                category=q.question_type,
                candidate_answer=q.answer.answer_text,
                scores=dim_scores,
                overall_score=ev.overall_score,
                strengths=strengths,
                improvements=improvements,
                covered_topics=covered_topics,
                missing_topics=missing_topics,
                technical_feedback=technical_feedback,
                improved_answer=ev.sample_answer or "",
            )
        )

    return analytics_turns, question_reports


def resolve_candidate_name(session: InterviewSession) -> str:
    """Resolves candidate name from session resume, user profile, or fallback default."""
    if session.resume and session.resume.candidate_name:
        return session.resume.candidate_name
    if session.user and session.user.name and session.user.name.strip():
        return session.user.name
    return "Demo Candidate"


def _build_response_from_db_report(
    report: FinalReport,
    session: InterviewSession,
    db: Session,
) -> FinalInterviewReportResponse:
    """Builds FinalInterviewReportResponse from persisted FinalReport record."""
    questions = (
        db.query(Question)
        .filter(Question.session_id == session.id)
        .order_by(Question.question_number.asc())
        .all()
    )

    _, question_reports = extract_turn_data_and_question_reports(questions)

    # Reconstruct analytics
    if report.analytics_data:
        analytics_dict = json.loads(report.analytics_data)
        analytics = InterviewAnalytics(**analytics_dict)
    else:
        # Fallback to computing analytics from turns
        analytics_turns, _ = extract_turn_data_and_question_reports(questions)
        analytics = compute_interview_analytics(analytics_turns)

    candidate_name = resolve_candidate_name(session)
    readiness = compute_interview_readiness(
        overall_score=analytics.overall_score,
        dimension_averages=analytics.dimension_averages,
        interview_type=session.interview_type,
    )

    strengths = json.loads(report.strengths or "[]")
    weaknesses = json.loads(report.weaknesses or "[]")
    study_topics = json.loads(report.study_topics or "[]")
    topic_gaps = json.loads(report.topic_gap_analysis or "[]") if report.topic_gap_analysis else []
    coaching_tips = json.loads(report.coaching_tips or "[]") if report.coaching_tips else json.loads(report.recommendations or "[]")

    return FinalInterviewReportResponse(
        session_id=session.id,
        interview_type=InterviewType(session.interview_type),
        status=session.status,
        candidate_name=candidate_name,
        readiness=readiness,
        analytics=analytics,
        executive_summary=report.performance_summary,
        key_strengths=strengths,
        improvement_areas=weaknesses,
        study_recommendations=study_topics,
        topic_gap_analysis=topic_gaps,
        interview_coaching_tips=coaching_tips,
        questions=question_reports,
        generated_at=report.created_at,
    )


def generate_session_final_report(
    session_id: int,
    user_id: int,
    db: Session,
) -> FinalInterviewReportResponse:
    """Generates and persists the final interview report combining deterministic analytics and qualitative LLM coaching."""
    # 1. Verify session exists and belongs to user
    session = db.query(InterviewSession).filter(InterviewSession.id == session_id).first()
    if not session or session.user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Interview session not found.",
        )

    # 2. Verify session is completed
    if session.status != "completed":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Cannot generate a final report for an active interview session (status: '{session.status}'). Complete the session first.",
        )

    # 3. Idempotency: Return existing report if already generated (0 Gemini calls!)
    existing_report = db.query(FinalReport).filter(FinalReport.session_id == session_id).first()
    if existing_report:
        return _build_response_from_db_report(existing_report, session, db)

    # 4. Load persisted questions
    questions = (
        db.query(Question)
        .filter(Question.session_id == session_id)
        .order_by(Question.question_number.asc())
        .all()
    )
    if not questions:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Cannot generate report: No questions found for this session.",
        )

    # 5. Extract turns and verify completeness
    analytics_turns, question_reports = extract_turn_data_and_question_reports(questions)

    # 6. Calculate deterministic Python analytics (0 Gemini calls)
    analytics = compute_interview_analytics(analytics_turns)

    # 7. Generate qualitative report via Gemini (exactly 1 call)
    prompt = build_qualitative_report_prompt(
        interview_type=InterviewType(session.interview_type),
        analytics=analytics,
        questions=question_reports,
    )
    qualitative = call_gemini_qualitative_report(prompt)

    # 8. Persist FinalReport in database
    db_report = FinalReport(
        session_id=session.id,
        overall_score=analytics.overall_score,
        performance_summary=qualitative.executive_summary,
        strengths=json.dumps(qualitative.key_strengths),
        weaknesses=json.dumps(qualitative.improvement_areas),
        study_topics=json.dumps(qualitative.study_recommendations),
        recommendations=json.dumps(qualitative.interview_coaching_tips),
        dimension_averages=json.dumps(analytics.dimension_averages.model_dump()),
        strongest_dimension=analytics.strongest_dimension,
        weakest_dimension=analytics.weakest_dimension,
        difficulty_progression=json.dumps(analytics.difficulty_progression),
        covered_topics=json.dumps(analytics.covered_topics),
        missing_topics=json.dumps(analytics.missing_topics),
        topic_gap_analysis=json.dumps(qualitative.topic_gap_analysis),
        coaching_tips=json.dumps(qualitative.interview_coaching_tips),
        analytics_data=json.dumps(analytics.model_dump()),
    )
    db.add(db_report)

    # Update session overall_score to authoritative analytics overall_score
    session.overall_score = analytics.overall_score

    db.commit()
    db.refresh(db_report)

    candidate_name = resolve_candidate_name(session)
    readiness = compute_interview_readiness(
        overall_score=analytics.overall_score,
        dimension_averages=analytics.dimension_averages,
        interview_type=session.interview_type,
    )

    return FinalInterviewReportResponse(
        session_id=session.id,
        interview_type=InterviewType(session.interview_type),
        status=session.status,
        candidate_name=candidate_name,
        readiness=readiness,
        analytics=analytics,
        executive_summary=qualitative.executive_summary,
        key_strengths=qualitative.key_strengths,
        improvement_areas=qualitative.improvement_areas,
        study_recommendations=qualitative.study_recommendations,
        topic_gap_analysis=qualitative.topic_gap_analysis,
        interview_coaching_tips=qualitative.interview_coaching_tips,
        questions=question_reports,
        generated_at=db_report.created_at,
    )


def get_session_final_report(
    session_id: int,
    user_id: int,
    db: Session,
) -> FinalInterviewReportResponse:
    """Retrieves an existing persisted final report for an interview session (0 Gemini calls)."""
    # 1. Verify session exists and belongs to user
    session = db.query(InterviewSession).filter(InterviewSession.id == session_id).first()
    if not session or session.user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Interview session not found.",
        )

    # 2. If session is still active, return controlled 409
    if session.status != "completed":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Interview session is active (status: '{session.status}'). Final report is only available after completion.",
        )

    # 3. Retrieve report
    report = db.query(FinalReport).filter(FinalReport.session_id == session_id).first()
    if not report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Final report has not been generated for this session. Please call POST /interview/sessions/{session_id}/report first.",
        )

    return _build_response_from_db_report(report, session, db)
