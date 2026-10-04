"""Service layer orchestrating RAG context retrieval, prompt assembly, LLM execution,

and structured response validation.
"""

import json
from typing import List, Optional
from fastapi import HTTPException, status
from pydantic import ValidationError

from app.llm.client import get_llm_client
from app.llm.prompts import build_correction_prompt, build_grounded_prompt
from app.llm.schemas import (
    GroundedAnalysis,
    GroundedAnalysisResponse,
    RetrievedSource,
)
from app.rag.retriever import retrieve_context
from app.schemas.rag import RetrievedChunk


def parse_and_validate_analysis(raw_text: str) -> GroundedAnalysis:
    """Safely extracts JSON and validates it against the GroundedAnalysis Pydantic model."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    parsed_dict = json.loads(cleaned)
    return GroundedAnalysis.model_validate(parsed_dict)


def generate_grounded_analysis(
    query: str,
    user_id: int,
    resume_id: Optional[int] = None,
    job_description_id: Optional[int] = None,
    top_k: int = 4,
) -> GroundedAnalysisResponse:
    """Coordinates retrieval of user context, grounded prompt construction,

    LLM generation, and verified structured output parsing.
    """
    if not query or not query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query cannot be empty.",
        )

    # 1. Retrieve RAG context (enforcing strict user_id isolation)
    retrieved_chunks: List[RetrievedChunk] = retrieve_context(
        query=query,
        user_id=user_id,
        resume_id=resume_id,
        job_description_id=job_description_id,
        top_k=top_k,
    )

    # 2. Check for insufficient context (zero hallucination guard)
    if not retrieved_chunks:
        return GroundedAnalysisResponse(
            analysis=GroundedAnalysis(
                summary="Insufficient context available to address the request.",
                key_points=[],
                evidence=[],
                missing_information=[
                    f"No relevant document context was found for user {user_id} regarding '{query}'."
                ],
            ),
            sources=[],
        )

    # 3. Categorize chunks
    resume_chunks = [c for c in retrieved_chunks if c.source_type == "resume"]
    jd_chunks = [c for c in retrieved_chunks if c.source_type == "job_description"]

    # 4. Construct grounded prompt with untrusted data fencing
    prompt = build_grounded_prompt(
        query=query,
        resume_chunks=resume_chunks,
        jd_chunks=jd_chunks,
    )

    # 5. Call LLM
    client = get_llm_client()
    raw_output = client.generate(prompt)

    # 6. Parse structured response with at most 1 correction retry
    try:
        analysis = parse_and_validate_analysis(raw_output)
    except (json.JSONDecodeError, ValidationError) as parse_err:
        # One retry with correction prompt
        correction_prompt = build_correction_prompt(
            raw_output=raw_output,
            error_detail=str(parse_err),
        )
        retry_output = client.generate(correction_prompt)
        try:
            analysis = parse_and_validate_analysis(retry_output)
        except (json.JSONDecodeError, ValidationError) as retry_err:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Failed to obtain valid structured JSON analysis from LLM: {retry_err}",
            ) from retry_err

    # 7. Build source traceability list
    sources = [
        RetrievedSource(
            source_type=c.source_type,
            source_id=c.source_id,
            chunk_index=c.chunk_index,
            score=c.score,
        )
        for c in retrieved_chunks
    ]

    return GroundedAnalysisResponse(analysis=analysis, sources=sources)
