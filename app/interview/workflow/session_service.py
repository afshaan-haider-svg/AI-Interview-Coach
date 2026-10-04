"""Service layer managing interview sessions, LangGraph turn execution, and database persistence."""

import json
from typing import List, Optional
from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.db.models import Answer, Evaluation, InterviewSession, JobDescription, Question, Resume, User, get_utc_now
from app.interview.schemas import (
    AdaptiveDecision,
    AnswerEvaluation,
    CompletedTurnHistory,
    DifficultyLevel,
    DimensionScores,
    InterviewQuestion,
    InterviewSessionResponse,
    InterviewType,
    QuestionSource,
    StartInterviewRequest,
    StartInterviewResponse,
    SubmitAnswerRequest,
    SubmitAnswerResponse,
)
from app.interview.workflow.graph import interview_graph
from app.interview.workflow.state import InterviewState


def _reconstruct_question_model(q: Question, session: InterviewSession) -> InterviewQuestion:
    """Helper to convert a DB Question record into an InterviewQuestion Pydantic model."""
    ctx = json.loads(q.source_context or "{}")
    grounding = [
        QuestionSource(**s) if isinstance(s, dict) else s
        for s in ctx.get("grounding_sources", [])
    ]
    q_diff = DifficultyLevel(q.difficulty) if q.difficulty else DifficultyLevel(session.difficulty)
    return InterviewQuestion(
        question_id=q.question_number,
        question=q.question_text,
        interview_type=InterviewType(session.interview_type),
        difficulty=q_diff,
        category=ctx.get("category", "technical"),
        rationale=ctx.get("rationale", ""),
        expected_topics=ctx.get("expected_topics", []),
        grounding_sources=grounding,
    )


def start_interview_session(
    req: StartInterviewRequest,
    db: Session,
) -> StartInterviewResponse:
    """Initializes a new interview session, invokes LangGraph for Q1, and persists state."""
    # 1. Validate user existence
    user = db.query(User).filter(User.id == req.user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with id {req.user_id} not found.",
        )

    # 2. Validate resume ownership if supplied
    if req.resume_id is not None:
        resume = db.query(Resume).filter(Resume.id == req.resume_id).first()
        if not resume or resume.user_id != req.user_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Resume {req.resume_id} not found or does not belong to user {req.user_id}.",
            )

    # 3. Validate job description ownership if supplied
    if req.job_description_id is not None:
        jd = db.query(JobDescription).filter(JobDescription.id == req.job_description_id).first()
        if not jd or jd.user_id != req.user_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job description {req.job_description_id} not found or does not belong to user {req.user_id}.",
            )

    # 4. Create InterviewSession record
    session = InterviewSession(
        user_id=req.user_id,
        resume_id=req.resume_id,
        job_description_id=req.job_description_id,
        interview_type=req.interview_type.value,
        difficulty=req.difficulty.value,
        total_questions=req.total_questions,
        status="active",
    )
    db.add(session)
    db.commit()
    db.refresh(session)

    # 5. Build initial LangGraph state
    initial_state: InterviewState = {
        "session_id": session.id,
        "user_id": req.user_id,
        "resume_id": req.resume_id,
        "job_description_id": req.job_description_id,
        "interview_type": req.interview_type,
        "difficulty": req.difficulty,
        "initial_difficulty": req.difficulty,
        "current_difficulty": req.difficulty,
        "total_questions": req.total_questions,
        "current_question_number": 0,
        "current_question": None,
        "questions_asked": [],
        "completed_question_count": 0,
        "pending_answer": None,
        "latest_evaluation": None,
        "status": "active",
    }

    # 6. Invoke LangGraph to generate Q1 and persist in DB with transaction safety
    try:
        final_state = interview_graph.invoke(initial_state)
        q_dict = final_state.get("current_question") or {}
        q_model = InterviewQuestion(**q_dict)

        # 7. Persist first question in database
        db_question = Question(
            session_id=session.id,
            question_number=1,
            question_text=q_model.question,
            question_type=q_model.category.value,
            difficulty=q_model.difficulty.value,
            source_context=json.dumps({
                "category": q_model.category.value,
                "rationale": q_model.rationale,
                "expected_topics": q_model.expected_topics,
                "grounding_sources": [s.model_dump() for s in q_model.grounding_sources],
            }),
        )
        db.add(db_question)
        db.commit()
        db.refresh(db_question)
    except Exception:
        # Transaction safety: clean up newly created empty session on question generation failure
        try:
            db.delete(session)
            db.commit()
        except Exception:
            db.rollback()
        raise

    return StartInterviewResponse(
        session_id=session.id,
        status=session.status,
        total_questions=session.total_questions,
        completed_question_count=0,
        current_question=q_model,
    )


def submit_answer_to_session(
    session_id: int,
    req: SubmitAnswerRequest,
    db: Session,
) -> SubmitAnswerResponse:
    """Accepts an answer to the current pending question, resumes LangGraph, and updates DB."""
    # 1. Verify session exists and belongs to user
    session = db.query(InterviewSession).filter(InterviewSession.id == session_id).first()
    if not session or session.user_id != req.user_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Interview session not found.",
        )

    # 2. Verify session is active
    if session.status != "active":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot submit answer to a session with status '{session.status}'.",
        )

    # 3. Identify the current unanswered question
    questions = (
        db.query(Question)
        .filter(Question.session_id == session_id)
        .order_by(Question.question_number.asc())
        .all()
    )

    pending_question: Optional[Question] = None
    for q in questions:
        if not q.answer:
            pending_question = q
            break

    if not pending_question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No pending question awaits an answer in this session.",
        )

    # 4. Idempotency / Double answer protection
    existing_answer = db.query(Answer).filter(Answer.question_id == pending_question.id).first()
    if existing_answer:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An answer has already been submitted for this question.",
        )

    # 5. Persist the candidate answer
    db_answer = Answer(
        question_id=pending_question.id,
        answer_text=req.answer.strip(),
        answer_method=getattr(req, "answer_method", "text") or "text",
    )
    db.add(db_answer)
    db.commit()
    db.refresh(db_answer)

    # 6. Reconstruct state from durable DB source of truth
    q_ctx = json.loads(pending_question.source_context or "{}")
    pending_diff = (
        DifficultyLevel(pending_question.difficulty)
        if pending_question.difficulty
        else DifficultyLevel(session.difficulty)
    )
    curr_q_dict = {
        "question_id": pending_question.question_number,
        "question": pending_question.question_text,
        "interview_type": session.interview_type,
        "difficulty": pending_diff.value,
        "category": q_ctx.get("category", "technical"),
        "rationale": q_ctx.get("rationale", ""),
        "expected_topics": q_ctx.get("expected_topics", []),
        "grounding_sources": q_ctx.get("grounding_sources", []),
    }

    questions_asked = [q.question_text for q in questions]
    completed_count = sum(1 for q in questions if q.answer and q.answer.evaluation)

    resumed_state: InterviewState = {
        "session_id": session.id,
        "user_id": session.user_id,
        "resume_id": session.resume_id,
        "job_description_id": session.job_description_id,
        "interview_type": InterviewType(session.interview_type),
        "difficulty": pending_diff,
        "initial_difficulty": DifficultyLevel(session.difficulty),
        "current_difficulty": pending_diff,
        "total_questions": session.total_questions,
        "current_question_number": pending_question.question_number,
        "current_question": curr_q_dict,
        "questions_asked": questions_asked,
        "completed_question_count": completed_count,
        "pending_answer": req.answer.strip(),
        "status": session.status,
    }

    # 7. Resume LangGraph (evaluates answer and either generates next question or completes session)
    final_state = interview_graph.invoke(resumed_state)

    # 8. Persist evaluation in database
    eval_dict = final_state.get("latest_evaluation") or {}
    eval_model = AnswerEvaluation(**eval_dict)

    db_evaluation = Evaluation(
        answer_id=db_answer.id,
        relevance_score=float(eval_model.scores.relevance),
        clarity_score=float(eval_model.scores.clarity),
        structure_score=float(eval_model.scores.structure),
        completeness_score=float(eval_model.scores.completeness),
        technical_score=float(eval_model.scores.technical_correctness),
        overall_score=eval_model.overall_score,
        strengths=json.dumps(eval_model.strengths),
        weaknesses=json.dumps({
            "missing_topics": eval_model.missing_topics,
            "covered_topics": eval_model.covered_topics,
            "technical_feedback": eval_model.technical_feedback,
        }),
        improvement=json.dumps(eval_model.improvements),
        sample_answer=eval_model.improved_answer,
    )
    db.add(db_evaluation)
    db.commit()

    # 9. Handle next question or session completion
    next_q_dict = final_state.get("current_question")
    next_q_model: Optional[InterviewQuestion] = None

    if next_q_dict and final_state.get("status") != "completed":
        next_q_model = InterviewQuestion(**next_q_dict)
        db_next_q = Question(
            session_id=session.id,
            question_number=next_q_model.question_id,
            question_text=next_q_model.question,
            question_type=next_q_model.category.value,
            difficulty=next_q_model.difficulty.value,
            source_context=json.dumps({
                "category": next_q_model.category.value,
                "rationale": next_q_model.rationale,
                "expected_topics": next_q_model.expected_topics,
                "grounding_sources": [s.model_dump() for s in next_q_model.grounding_sources],
            }),
        )
        db.add(db_next_q)
        db.commit()
    else:
        session.status = "completed"
        session.completed_at = get_utc_now()
        # Compute overall session average score
        all_evals = (
            db.query(Evaluation.overall_score)
            .join(Answer, Evaluation.answer_id == Answer.id)
            .join(Question, Answer.question_id == Question.id)
            .filter(Question.session_id == session.id)
            .all()
        )
        if all_evals:
            session.overall_score = round(sum(e[0] for e in all_evals) / len(all_evals), 2)
        db.commit()

    adaptive_dict = final_state.get("latest_adaptive_decision")
    adaptive_decision = AdaptiveDecision(**adaptive_dict) if adaptive_dict else None

    return SubmitAnswerResponse(
        session_id=session.id,
        status=session.status,
        completed_question_count=final_state.get("completed_question_count", completed_count + 1),
        evaluation=eval_model,
        adaptive_decision=adaptive_decision,
        next_question=next_q_model,
    )


def get_interview_session_details(
    session_id: int,
    user_id: int,
    db: Session,
) -> InterviewSessionResponse:
    """Retrieves full interview session details, current pending question, and turn history."""
    session = db.query(InterviewSession).filter(InterviewSession.id == session_id).first()
    if not session or session.user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Interview session not found.",
        )

    questions = (
        db.query(Question)
        .filter(Question.session_id == session_id)
        .order_by(Question.question_number.asc())
        .all()
    )

    history: List[CompletedTurnHistory] = []
    current_question: Optional[InterviewQuestion] = None

    for q in questions:
        if q.answer and q.answer.evaluation:
            ev = q.answer.evaluation
            dim_scores = DimensionScores(
                relevance=int(round(ev.relevance_score)),
                clarity=int(round(ev.clarity_score)),
                completeness=int(round(ev.completeness_score)),
                technical_correctness=int(round(ev.technical_score)),
                structure=int(round(ev.structure_score)),
            )
            history.append(
                CompletedTurnHistory(
                    question_number=q.question_number,
                    difficulty=DifficultyLevel(q.difficulty) if q.difficulty else DifficultyLevel(session.difficulty),
                    question=q.question_text,
                    answer=q.answer.answer_text,
                    overall_score=ev.overall_score,
                    dimension_scores=dim_scores,
                )
            )
        elif not q.answer and current_question is None:
            current_question = _reconstruct_question_model(q, session)

    latest_q = questions[-1] if questions else None
    active_current_diff = (
        DifficultyLevel(latest_q.difficulty)
        if latest_q and latest_q.difficulty
        else DifficultyLevel(session.difficulty)
    )

    return InterviewSessionResponse(
        session_id=session.id,
        status=session.status,
        interview_type=InterviewType(session.interview_type),
        difficulty=DifficultyLevel(session.difficulty),
        current_difficulty=active_current_diff if session.status == "active" else None,
        total_questions=session.total_questions,
        completed_question_count=len(history),
        current_question=current_question if session.status == "active" else None,
        history=history,
    )
