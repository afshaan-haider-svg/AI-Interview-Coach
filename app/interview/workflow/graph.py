"""LangGraph workflow definition for interview session turn orchestration with adaptive difficulty."""

from typing import Any, Dict, List
from langgraph.graph import END, START, StateGraph

from app.interview.adaptive import decide_next_difficulty
from app.interview.evaluation_service import evaluate_answer
from app.interview.question_service import (
    compute_word_similarity,
    generate_interview_questions,
    normalize_question_text,
)
from app.interview.schemas import (
    DifficultyLevel,
    EvaluationRequest,
    InterviewType,
    QuestionGenerationRequest,
    QuestionSource,
)
from app.interview.workflow.state import InterviewState


def _is_duplicate_question(q_text: str, previous_questions: List[str]) -> bool:
    """Checks whether a newly generated question duplicates any question previously asked in this session."""
    norm_new = normalize_question_text(q_text)
    for prev in previous_questions:
        norm_prev = normalize_question_text(prev)
        if norm_new == norm_prev or compute_word_similarity(norm_new, norm_prev) >= 0.85:
            return True
    return False


def generate_question_node(state: InterviewState) -> Dict[str, Any]:
    """Graph node: Generates exactly one personalized interview question using Phase-6 services."""
    raw_type = state.get("interview_type")
    i_type = (
        raw_type
        if isinstance(raw_type, InterviewType)
        else InterviewType(raw_type)
    )

    raw_diff = state.get("current_difficulty") or state.get("difficulty") or DifficultyLevel.MEDIUM
    diff = (
        raw_diff
        if isinstance(raw_diff, DifficultyLevel)
        else DifficultyLevel(raw_diff)
    )

    req = QuestionGenerationRequest(
        user_id=state["user_id"],
        resume_id=state.get("resume_id"),
        job_description_id=state.get("job_description_id"),
        interview_type=i_type,
        difficulty=diff,
        question_count=1,
    )

    resp = generate_interview_questions(req)
    selected_question = resp.questions[0] if resp.questions else None

    # Check for cross-turn duplicates; allow at most 1 regeneration retry
    previous_questions = state.get("questions_asked", [])
    if selected_question and _is_duplicate_question(selected_question.question, previous_questions):
        retry_resp = generate_interview_questions(req)
        if retry_resp.questions and not _is_duplicate_question(retry_resp.questions[0].question, previous_questions):
            selected_question = retry_resp.questions[0]

    next_num = state.get("current_question_number", 0) + 1
    q_dict = selected_question.model_dump() if selected_question else {}
    q_dict["question_id"] = next_num
    q_dict["difficulty"] = diff.value

    updated_history = list(previous_questions)
    if selected_question:
        updated_history.append(selected_question.question)

    return {
        "current_question": q_dict,
        "current_question_number": next_num,
        "current_difficulty": diff,
        "difficulty": diff,
        "questions_asked": updated_history,
    }


def evaluate_answer_node(state: InterviewState) -> Dict[str, Any]:
    """Graph node: Evaluates candidate answer using Phase-7 evaluation engine."""
    raw_type = state.get("interview_type")
    i_type = (
        raw_type
        if isinstance(raw_type, InterviewType)
        else InterviewType(raw_type)
    )

    curr_q = state.get("current_question") or {}
    q_diff_raw = curr_q.get("difficulty") or state.get("current_difficulty") or state.get("difficulty") or DifficultyLevel.MEDIUM
    diff = (
        q_diff_raw
        if isinstance(q_diff_raw, DifficultyLevel)
        else DifficultyLevel(q_diff_raw)
    )

    raw_sources = curr_q.get("grounding_sources", [])
    grounding_sources = [
        s if isinstance(s, QuestionSource) else QuestionSource(**s)
        for s in raw_sources
    ]

    eval_req = EvaluationRequest(
        user_id=state["user_id"],
        question=curr_q.get("question", ""),
        candidate_answer=state.get("pending_answer", ""),
        interview_type=i_type,
        difficulty=diff,
        expected_topics=curr_q.get("expected_topics", []),
        grounding_sources=grounding_sources,
    )

    eval_resp = evaluate_answer(eval_req)
    completed_count = state.get("completed_question_count", 0) + 1

    return {
        "latest_evaluation": eval_resp.evaluation.model_dump(),
        "completed_question_count": completed_count,
        "pending_answer": None,
    }


def adapt_difficulty_node(state: InterviewState) -> Dict[str, Any]:
    """Graph node: Calls the deterministic Python adaptive policy to select the difficulty for Question N+1."""
    c_diff_raw = state.get("current_difficulty") or state.get("difficulty") or DifficultyLevel.MEDIUM
    c_diff = (
        c_diff_raw
        if isinstance(c_diff_raw, DifficultyLevel)
        else DifficultyLevel(c_diff_raw)
    )

    raw_type = state.get("interview_type")
    i_type = (
        raw_type
        if isinstance(raw_type, InterviewType)
        else InterviewType(raw_type)
    )

    eval_data = state.get("latest_evaluation") or {}
    decision = decide_next_difficulty(
        current_difficulty=c_diff,
        interview_type=i_type,
        evaluation=eval_data,
    )

    return {
        "current_difficulty": decision.next_difficulty,
        "difficulty": decision.next_difficulty,
        "latest_adaptive_decision": decision.model_dump(),
    }


def complete_session_node(state: InterviewState) -> Dict[str, Any]:
    """Graph node: Finalizes interview session when all questions have been completed."""
    return {
        "status": "completed",
        "current_question": None,
        "latest_adaptive_decision": None,
    }


def route_entry(state: InterviewState) -> str:
    """Entry conditional edge: Decides whether to evaluate an incoming answer or generate a question."""
    if state.get("pending_answer"):
        return "evaluate_answer"
    return "generate_question"


def route_after_evaluation(state: InterviewState) -> str:
    """Conditional edge: Determines whether to adapt difficulty for the next question or complete the session."""
    completed = state.get("completed_question_count", 0)
    total = state.get("total_questions", 5)
    if completed < total:
        return "adapt_difficulty"
    return "complete_session"


def create_interview_graph():
    """Builds and compiles the LangGraph StateGraph workflow with adaptive difficulty."""
    builder = StateGraph(InterviewState)

    builder.add_node("generate_question", generate_question_node)
    builder.add_node("evaluate_answer", evaluate_answer_node)
    builder.add_node("adapt_difficulty", adapt_difficulty_node)
    builder.add_node("complete_session", complete_session_node)

    builder.add_conditional_edges(
        START,
        route_entry,
        {
            "generate_question": "generate_question",
            "evaluate_answer": "evaluate_answer",
        },
    )

    builder.add_conditional_edges(
        "evaluate_answer",
        route_after_evaluation,
        {
            "adapt_difficulty": "adapt_difficulty",
            "complete_session": "complete_session",
        },
    )

    builder.add_edge("adapt_difficulty", "generate_question")
    builder.add_edge("generate_question", END)
    builder.add_edge("complete_session", END)

    return builder.compile()


interview_graph = create_interview_graph()
