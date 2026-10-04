"""API routes for AI reasoning and grounded analysis."""

from fastapi import APIRouter, status

from app.llm.schemas import GroundedAnalysisRequest, GroundedAnalysisResponse
from app.llm.service import generate_grounded_analysis

router = APIRouter(prefix="/ai", tags=["AI"])


@router.post(
    "/grounded-analysis",
    response_model=GroundedAnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate RAG-grounded candidate analysis",
    description="Retrieves candidate/job context via local RAG and generates a structured, grounded analysis via Gemini LLM.",
)
def create_grounded_analysis(
    req: GroundedAnalysisRequest,
) -> GroundedAnalysisResponse:
    """Performs semantic context retrieval followed by structured grounded reasoning."""
    return generate_grounded_analysis(
        query=req.query,
        user_id=req.user_id,
        resume_id=req.resume_id,
        job_description_id=req.job_description_id,
        top_k=req.top_k,
    )
