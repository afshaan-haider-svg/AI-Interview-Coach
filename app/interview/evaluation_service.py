"""Service layer orchestrating RAG context lookup, prompt construction,

single-call Gemini evaluation, structured validation, and deterministic scoring.
"""

import json
from typing import Any, Dict, List
from fastapi import HTTPException, status

from app.interview.evaluation_prompts import (
    build_evaluation_correction_prompt,
    build_evaluation_prompt,
)
from app.interview.schemas import (
    AnswerEvaluation,
    DimensionScores,
    EvaluationRequest,
    EvaluationResponse,
    QuestionSource,
    calculate_overall_score,
)
from app.llm.client import get_llm_client
from app.rag.retriever import retrieve_chunks_by_sources
from app.schemas.rag import RetrievedChunk


def _clean_json_text(raw_text: str) -> str:
    """Strips markdown code fences and whitespace from raw LLM output."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()


def _parse_and_validate_evaluation_dict(data: Dict[str, Any]) -> AnswerEvaluation:
    """Parses raw evaluation dictionary into AnswerEvaluation with clamped dimension scores."""
    raw_scores = data.get("scores", {})
    if not isinstance(raw_scores, dict):
        raise ValueError("Field 'scores' must be a JSON object.")

    def _get_score(key: str, default: int = 5) -> int:
        val = raw_scores.get(key, default)
        try:
            int_val = int(round(float(val)))
        except (ValueError, TypeError):
            int_val = default
        # Strict clamp 0 to 10
        return max(0, min(10, int_val))

    scores = DimensionScores(
        relevance=_get_score("relevance"),
        clarity=_get_score("clarity"),
        completeness=_get_score("completeness"),
        technical_correctness=_get_score("technical_correctness"),
        structure=_get_score("structure"),
    )

    # Compute authoritative overall_score deterministically in Python
    overall_score = calculate_overall_score(scores)

    def _get_string_list(key: str) -> List[str]:
        raw_list = data.get(key, [])
        if not isinstance(raw_list, list):
            return []
        return [str(item).strip() for item in raw_list if str(item).strip()]

    covered_topics = _get_string_list("covered_topics")
    missing_topics = _get_string_list("missing_topics")
    strengths = _get_string_list("strengths")
    improvements = _get_string_list("improvements")
    technical_feedback = _get_string_list("technical_feedback")

    improved_answer = str(data.get("improved_answer", "")).strip()
    if not improved_answer:
        improved_answer = "A concise, well-structured answer addressing core concepts and edge cases."

    summary_feedback = str(data.get("summary_feedback", "")).strip()
    if not summary_feedback:
        summary_feedback = "Evaluation complete."

    return AnswerEvaluation(
        scores=scores,
        overall_score=overall_score,
        covered_topics=covered_topics,
        missing_topics=missing_topics,
        strengths=strengths,
        improvements=improvements,
        technical_feedback=technical_feedback,
        improved_answer=improved_answer,
        summary_feedback=summary_feedback,
    )


def evaluate_answer(request: EvaluationRequest) -> EvaluationResponse:
    """Evaluates a single interview answer using grounded Gemini generation and Python scoring."""
    # 1. Retrieve and validate grounding context chunks with strict tenant isolation
    context_chunks: List[RetrievedChunk] = []
    if request.grounding_sources:
        context_chunks = retrieve_chunks_by_sources(
            sources=request.grounding_sources,
            user_id=request.user_id,
        )

    validated_sources = [
        QuestionSource(
            source_type=c.source_type,
            source_id=c.source_id,
            chunk_index=c.chunk_index,
        )
        for c in context_chunks
    ]

    # 2. Build structured prompt with injection defense
    prompt = build_evaluation_prompt(
        question=request.question,
        candidate_answer=request.candidate_answer,
        interview_type=request.interview_type,
        difficulty=request.difficulty,
        expected_topics=request.expected_topics,
        context_chunks=context_chunks,
    )

    # 3. Call LLM (normal path: exactly 1 call)
    client = get_llm_client()
    raw_output = client.generate(prompt)

    # 4. Parse structured output with at most 1 correction retry
    try:
        cleaned_json = _clean_json_text(raw_output)
        data = json.loads(cleaned_json)
        evaluation = _parse_and_validate_evaluation_dict(data)
    except Exception as parse_err:
        correction_prompt = build_evaluation_correction_prompt(
            raw_output=raw_output,
            error_detail=str(parse_err),
        )
        retry_output = client.generate(correction_prompt)
        try:
            cleaned_retry = _clean_json_text(retry_output)
            retry_data = json.loads(cleaned_retry)
            evaluation = _parse_and_validate_evaluation_dict(retry_data)
        except Exception as retry_err:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Failed to obtain valid structured evaluation from LLM: {retry_err}",
            ) from retry_err

    return EvaluationResponse(
        evaluation=evaluation,
        context_sources=validated_sources,
    )
