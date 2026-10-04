"""LLM package for Gemini integration, prompt management, and structured generation."""

from app.llm.client import GeminiLLMClient, get_llm_client, set_llm_client
from app.llm.prompts import build_grounded_prompt
from app.llm.schemas import (
    EvidenceItem,
    GroundedAnalysis,
    GroundedAnalysisRequest,
    GroundedAnalysisResponse,
    RetrievedSource,
)
from app.llm.service import generate_grounded_analysis

__all__ = [
    "GeminiLLMClient",
    "get_llm_client",
    "set_llm_client",
    "build_grounded_prompt",
    "EvidenceItem",
    "GroundedAnalysis",
    "GroundedAnalysisRequest",
    "GroundedAnalysisResponse",
    "RetrievedSource",
    "generate_grounded_analysis",
]
