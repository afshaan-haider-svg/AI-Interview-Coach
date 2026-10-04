"""Service layer for RAG-grounded interview question generation."""

import json
import string
from typing import Any, List, Set, Tuple
from fastapi import HTTPException, status

from app.interview.prompts import build_question_prompt
from app.interview.schemas import (
    InterviewQuestion,
    InterviewType,
    QuestionCategory,
    QuestionGenerationRequest,
    QuestionGenerationResponse,
    QuestionSource,
)
from app.llm.client import get_llm_client
from app.rag.retriever import retrieve_context
from app.schemas.rag import RetrievedChunk


INTERVIEW_TYPE_QUERIES = {
    InterviewType.HR: "work experience teamwork leadership achievements challenges collaboration communication",
    InterviewType.PYTHON: "python programming language core data structures backend APIs frameworks performance async",
    InterviewType.AI_ML: "machine learning deep learning models algorithms training evaluation metrics deployment",
    InterviewType.DATA_SCIENCE: "data science statistics analysis SQL modeling visualization metrics EDA experiments",
    InterviewType.INTERNSHIP: "projects coursework education technical skills learning adaptability problem solving",
}


def clean_json_text(raw_text: str) -> str:
    """Strips markdown code fences and whitespace from raw LLM output."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()


def normalize_question_text(text: str) -> str:
    """Normalizes text by lowercasing and removing punctuation for duplicate detection."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def compute_word_similarity(text1: str, text2: str) -> float:
    """Computes Jaccard word set similarity between two normalized question strings."""
    words1 = set(text1.split())
    words2 = set(text2.split())
    if not words1 or not words2:
        return 0.0
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    return len(intersection) / len(union)


def parse_category(raw_category: Any, default_type: InterviewType) -> QuestionCategory:
    """Safely coerces LLM category string to a valid QuestionCategory enum."""
    if isinstance(raw_category, str):
        val = raw_category.strip().lower()
        for cat in QuestionCategory:
            if cat.value == val:
                return cat

    if default_type == InterviewType.HR:
        return QuestionCategory.BEHAVIORAL
    return QuestionCategory.TECHNICAL


def generate_interview_questions(
    request: QuestionGenerationRequest,
) -> QuestionGenerationResponse:
    """Orchestrates retrieval, single-call LLM generation, validation,

    deduplication, and citation verification for interview questions.
    """
    # 1. Determine retrieval search query
    search_query = INTERVIEW_TYPE_QUERIES.get(
        request.interview_type, "technical skills and experience"
    )

    # 2. Retrieve RAG context if document IDs are provided
    retrieved_chunks: List[RetrievedChunk] = []
    if request.resume_id is not None or request.job_description_id is not None:
        retrieved_chunks = retrieve_context(
            query=search_query,
            user_id=request.user_id,
            resume_id=request.resume_id,
            job_description_id=request.job_description_id,
            top_k=6,
        )

    resume_chunks = [c for c in retrieved_chunks if c.source_type == "resume"]
    jd_chunks = [c for c in retrieved_chunks if c.source_type == "job_description"]

    # Record valid chunk coordinate keys for grounding citation validation
    valid_chunks: Set[Tuple[str, int, int]] = {
        (c.source_type, c.source_id, c.chunk_index) for c in retrieved_chunks
    }

    # 3. Construct prompt
    prompt = build_question_prompt(
        interview_type=request.interview_type,
        difficulty=request.difficulty,
        question_count=request.question_count,
        resume_chunks=resume_chunks,
        jd_chunks=jd_chunks,
    )

    # 4. Single LLM call
    client = get_llm_client()
    raw_output = client.generate(prompt)

    # 5. Parse JSON response
    try:
        cleaned_json = clean_json_text(raw_output)
        data = json.loads(cleaned_json)
        raw_questions = data.get("questions", [])
    except (json.JSONDecodeError, AttributeError) as err:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to parse interview questions JSON from LLM: {err}",
        ) from err

    # 6. Post-processing: deduplication, grounding validation, and formatting
    processed_questions: List[InterviewQuestion] = []
    seen_normalized_texts: List[str] = []

    for item in raw_questions:
        if not isinstance(item, dict):
            continue

        q_text = str(item.get("question", "")).strip()
        if not q_text:
            continue

        # Duplicate detection using exact and word overlap comparison
        norm_text = normalize_question_text(q_text)
        is_duplicate = False
        for seen in seen_normalized_texts:
            if norm_text == seen or compute_word_similarity(norm_text, seen) >= 0.85:
                is_duplicate = True
                break

        if is_duplicate:
            continue
        seen_normalized_texts.append(norm_text)

        category = parse_category(item.get("category"), request.interview_type)
        rationale = str(item.get("rationale", "Standard interview inquiry.")).strip()

        raw_topics = item.get("expected_topics", [])
        expected_topics = [
            str(t).strip() for t in raw_topics if isinstance(t, str) and t.strip()
        ]

        # Grounding source validation: prune hallucinated citations
        validated_sources: List[QuestionSource] = []
        raw_sources = item.get("grounding_sources", [])
        if isinstance(raw_sources, list):
            for src in raw_sources:
                if isinstance(src, dict):
                    src_type = str(src.get("source_type", "")).strip().lower()
                    src_id = src.get("source_id")
                    chunk_idx = src.get("chunk_index")
                    if (
                        src_type in ("resume", "job_description")
                        and isinstance(src_id, int)
                        and isinstance(chunk_idx, int)
                    ):
                        if (src_type, src_id, chunk_idx) in valid_chunks:
                            validated_sources.append(
                                QuestionSource(
                                    source_type=src_type,
                                    source_id=src_id,
                                    chunk_index=chunk_idx,
                                )
                            )

        processed_questions.append(
            InterviewQuestion(
                question_id=len(processed_questions) + 1,
                question=q_text,
                interview_type=request.interview_type,
                difficulty=request.difficulty,
                category=category,
                rationale=rationale,
                expected_topics=expected_topics,
                grounding_sources=validated_sources,
            )
        )

    # 7. Construct final response
    return QuestionGenerationResponse(
        interview_type=request.interview_type,
        difficulty=request.difficulty,
        requested_count=request.question_count,
        generated_count=len(processed_questions),
        questions=processed_questions,
    )
