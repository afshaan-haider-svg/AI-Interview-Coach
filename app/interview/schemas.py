"""Pydantic schemas and enums for personalized interview question generation."""

from datetime import datetime
from enum import Enum
from typing import List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


class InterviewType(str, Enum):
    """Supported interview types for tailored question generation."""
    HR = "hr"
    PYTHON = "python"
    AI_ML = "ai_ml"
    DATA_SCIENCE = "data_science"
    INTERNSHIP = "internship"


class DifficultyLevel(str, Enum):
    """Target difficulty levels for generated questions."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class QuestionCategory(str, Enum):
    """Categories characterizing the focus of each generated question."""
    BEHAVIORAL = "behavioral"
    TECHNICAL = "technical"
    PROJECT_BASED = "project_based"
    CONCEPTUAL = "conceptual"
    SCENARIO_BASED = "scenario_based"
    FUNDAMENTAL = "fundamental"


class QuestionSource(BaseModel):
    """Reference citation linking a question to a specific retrieved context chunk."""
    source_type: Literal["resume", "job_description"]
    source_id: int
    chunk_index: int


class InterviewQuestion(BaseModel):
    """Individual generated interview question with metadata and grounding citations."""
    question_id: int
    question: str
    interview_type: InterviewType
    difficulty: DifficultyLevel
    category: QuestionCategory
    rationale: str = Field(
        ...,
        description="Reasoning linking this question to candidate context, job requirements, or core domain skills.",
    )
    expected_topics: List[str] = Field(
        default_factory=list,
        description="Key technical, conceptual, or behavioral elements expected in a strong answer.",
    )
    grounding_sources: List[QuestionSource] = Field(
        default_factory=list,
        description="Verified citations from retrieved context supporting this question.",
    )


class QuestionGenerationRequest(BaseModel):
    """Request payload for generating personalized interview questions."""
    user_id: int = Field(..., description="ID of the user requesting questions for strict data isolation.")
    resume_id: Optional[int] = Field(None, description="Optional ID of candidate resume to ground questions.")
    job_description_id: Optional[int] = Field(None, description="Optional ID of target job description.")
    interview_type: InterviewType = Field(..., description="Focus area of the interview.")
    difficulty: DifficultyLevel = Field(default=DifficultyLevel.MEDIUM, description="Target difficulty.")
    question_count: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Number of questions to generate (1-10, default 5).",
    )


class QuestionGenerationResponse(BaseModel):
    """Complete response payload containing generated interview questions."""
    interview_type: InterviewType
    difficulty: DifficultyLevel
    requested_count: int
    generated_count: int
    questions: List[InterviewQuestion]


class DimensionScores(BaseModel):
    """0-10 integer scores across 5 core evaluation dimensions."""
    relevance: int = Field(..., ge=0, le=10, description="How directly the answer addresses the question asked.")
    clarity: int = Field(..., ge=0, le=10, description="Coherence, conciseness, and articulation.")
    completeness: int = Field(..., ge=0, le=10, description="Coverage of necessary components and depth for target difficulty.")
    technical_correctness: int = Field(..., ge=0, le=10, description="Accuracy of concepts, facts, syntax, and terminology.")
    structure: int = Field(..., ge=0, le=10, description="Logical organization and flow of thought.")


def calculate_overall_score(scores: DimensionScores) -> float:
    """Calculates deterministic weighted overall score (0.00 to 10.00) in Python.

    Weights:
      relevance: 25%
      technical_correctness: 25%
      completeness: 20%
      clarity: 15%
      structure: 15%
    """
    raw_score = (
        scores.relevance * 0.25
        + scores.technical_correctness * 0.25
        + scores.completeness * 0.20
        + scores.clarity * 0.15
        + scores.structure * 0.15
    )
    return round(float(raw_score), 2)


class AnswerEvaluation(BaseModel):
    """Comprehensive evaluation outcome for a single question-answer pair."""
    scores: DimensionScores
    overall_score: float = Field(..., ge=0.0, le=10.0, description="Deterministic Python-calculated weighted score (0.00-10.00).")
    covered_topics: List[str] = Field(default_factory=list, description="Expected topics identified in candidate answer.")
    missing_topics: List[str] = Field(default_factory=list, description="Expected topics omitted or inadequately addressed.")
    strengths: List[str] = Field(default_factory=list, description="Constructive highlights of what the answer did well.")
    improvements: List[str] = Field(default_factory=list, description="Actionable recommendations to improve the response.")
    technical_feedback: List[str] = Field(default_factory=list, description="Specific feedback on technical/conceptual accuracy or omissions.")
    improved_answer: str = Field(..., description="Exemplary answer demonstrating how to address the question effectively.")
    summary_feedback: str = Field(..., description="Concise overall evaluation summary.")


class EvaluationRequest(BaseModel):
    """Request payload to evaluate a single candidate answer."""
    user_id: int = Field(..., gt=0, description="User ID for strict tenant isolation.")
    question: str = Field(..., min_length=1, max_length=2000, description="The interview question asked.")
    candidate_answer: str = Field(..., min_length=1, max_length=5000, description="Candidate submitted answer.")
    interview_type: InterviewType = Field(..., description="Interview track (e.g. hr, python, ai_ml, data_science, internship).")
    difficulty: DifficultyLevel = Field(..., description="Target difficulty level (easy, medium, hard).")
    expected_topics: List[str] = Field(default_factory=list, description="Anchor topics expected in a complete response.")
    grounding_sources: List[QuestionSource] = Field(default_factory=list, description="Optional RAG context references.")

    @field_validator("question", "candidate_answer")
    @classmethod
    def validate_non_empty_trimmed(cls, v: str) -> str:
        trimmed = v.strip()
        if not trimmed:
            raise ValueError("Field cannot be empty or contain only whitespace.")
        return trimmed

    @field_validator("expected_topics")
    @classmethod
    def clean_expected_topics(cls, v: List[str]) -> List[str]:
        return [t.strip() for t in v if isinstance(t, str) and t.strip()]


class EvaluationResponse(BaseModel):
    """Complete API response for interview answer evaluation."""
    evaluation: AnswerEvaluation
    context_sources: List[QuestionSource] = Field(
        default_factory=list,
        description="Validated grounding sources utilized during evaluation.",
    )


class StartInterviewRequest(BaseModel):
    """Request payload to initiate a new interview session."""
    user_id: int = Field(..., gt=0, description="User ID for strict tenant isolation.")
    resume_id: Optional[int] = Field(None, description="Optional ID of candidate resume.")
    job_description_id: Optional[int] = Field(None, description="Optional ID of target job description.")
    interview_type: InterviewType = Field(..., description="Interview track.")
    difficulty: DifficultyLevel = Field(..., description="Target difficulty level (easy, medium, hard).")
    total_questions: int = Field(default=5, ge=1, le=10, description="Total questions for this session (1 to 10).")


class SubmitAnswerRequest(BaseModel):
    """Request payload to submit an answer to the current pending question."""
    user_id: int = Field(..., gt=0, description="User ID for strict tenant isolation.")
    answer: str = Field(..., min_length=1, max_length=5000, description="Submitted candidate answer text.")
    answer_method: Optional[str] = Field(default="text", description="Method used: 'text' or 'voice'")

    @field_validator("answer")
    @classmethod
    def validate_non_empty_trimmed(cls, v: str) -> str:
        trimmed = v.strip()
        if not trimmed:
            raise ValueError("Answer cannot be empty or contain only whitespace.")
        return trimmed


class StartInterviewResponse(BaseModel):
    """Response payload when an interview session is initialized."""
    session_id: int
    status: str
    total_questions: int
    completed_question_count: int
    current_question: InterviewQuestion


class AdaptiveDecision(BaseModel):
    """Deterministic result of the adaptive difficulty policy."""
    previous_difficulty: DifficultyLevel
    next_difficulty: DifficultyLevel
    action: Literal["increase", "decrease", "maintain"]
    reason_code: str = Field(
        ...,
        description="Reason code: high_performance, low_performance, stable_performance, technical_guardrail, completeness_guardrail, upper_bound, lower_bound.",
    )


class SubmitAnswerResponse(BaseModel):
    """Response payload after an answer turn is evaluated."""
    session_id: int
    status: str
    completed_question_count: int
    evaluation: AnswerEvaluation
    adaptive_decision: Optional[AdaptiveDecision] = None
    next_question: Optional[InterviewQuestion] = None


class CompletedTurnHistory(BaseModel):
    """Structured record of a completed interview turn."""
    question_number: int
    difficulty: DifficultyLevel
    question: str
    answer: str
    overall_score: float
    dimension_scores: Optional[DimensionScores] = None


class InterviewSessionResponse(BaseModel):
    """Full detail of an interview session, its progress, and completed turns."""
    session_id: int
    status: str
    interview_type: InterviewType
    difficulty: DifficultyLevel
    current_difficulty: Optional[DifficultyLevel] = None
    total_questions: int
    completed_question_count: int
    current_question: Optional[InterviewQuestion] = None
    history: List[CompletedTurnHistory] = Field(default_factory=list)


# =====================================================================
# Phase 10: Final Interview Report & Performance Analytics Schemas
# =====================================================================

class DimensionAverages(BaseModel):
    """Arithmetic mean scores for each evaluation dimension across all interview questions."""
    relevance: float
    clarity: float
    completeness: float
    technical_correctness: float
    structure: float


class DifficultyChanges(BaseModel):
    """Summary of difficulty shifts across questions in the session."""
    increases: int = 0
    decreases: int = 0
    maintains: int = 0


class InterviewAnalytics(BaseModel):
    """Deterministic analytics calculated from persisted interview evaluations."""
    question_count: int
    overall_score: float
    dimension_averages: DimensionAverages
    highest_question_score: float
    lowest_question_score: float
    highest_question_number: Optional[int] = None
    lowest_question_number: Optional[int] = None
    strongest_dimension: str
    weakest_dimension: str
    difficulty_progression: List[str]
    difficulty_changes: Optional[DifficultyChanges] = None
    covered_topics: List[str]
    missing_topics: List[str]


class QuestionReport(BaseModel):
    """Per-question turn report containing historical question, answer, and evaluation details."""
    question_number: int
    question: str
    difficulty: DifficultyLevel
    category: str
    candidate_answer: str
    scores: DimensionScores
    overall_score: float
    strengths: List[str]
    improvements: List[str]
    covered_topics: List[str]
    missing_topics: List[str]
    technical_feedback: List[str]
    improved_answer: str


class QualitativeInterviewReport(BaseModel):
    """Structured qualitative evaluation and coaching feedback generated by Gemini."""
    executive_summary: str = Field(
        ..., description="Coaching-oriented summary of candidate performance."
    )
    key_strengths: List[str] = Field(
        ..., description="Key recurring strengths synthesized from interview evidence."
    )
    improvement_areas: List[str] = Field(
        ..., description="Actionable coaching points and areas for improvement."
    )
    study_recommendations: List[str] = Field(
        ..., description="Targeted study recommendations based on missing topics and weakest dimensions."
    )
    topic_gap_analysis: List[str] = Field(
        ..., description="Recurring topic gaps explained. Empty list if none."
    )
    interview_coaching_tips: List[str] = Field(
        ..., description="Practical interview answering guidance matching the interview type."
    )


class GenerateReportRequest(BaseModel):
    """Request payload to generate a final report for an interview session."""
    user_id: int = Field(..., gt=0, description="User ID for strict tenant isolation.")


class InterviewReadiness(BaseModel):
    """Deterministic assessment of interview readiness and priority growth areas."""
    readiness_code: str = Field(..., description="'not_yet_ready', 'developing', or 'strong_readiness'")
    readiness_label: str = Field(..., description="Human-readable label for readiness tier")
    headline_wording: str = Field(..., description="Track-aware coaching headline")
    assessment_message: str = Field(..., description="Nuanced coaching feedback without false hiring decisions")
    priority_improvement_areas: List[str] = Field(default_factory=list, description="Weakest dimensions with coaching guidance")
    strongest_areas: List[str] = Field(default_factory=list, description="Strongest dimensions highlighted")
    next_step: str = Field(..., description="Actionable next step recommendation")


class FinalInterviewReportResponse(BaseModel):
    """Comprehensive final interview report combining deterministic analytics and qualitative coaching."""
    session_id: int
    interview_type: InterviewType
    status: str
    candidate_name: str = "Demo Candidate"
    readiness: Optional[InterviewReadiness] = None
    analytics: InterviewAnalytics
    executive_summary: str
    key_strengths: List[str]
    improvement_areas: List[str]
    study_recommendations: List[str]
    topic_gap_analysis: List[str]
    interview_coaching_tips: List[str]
    questions: List[QuestionReport]
    generated_at: datetime


class SessionSummary(BaseModel):
    """Safe session summary item for user history listings."""
    session_id: int
    interview_type: InterviewType
    initial_difficulty: DifficultyLevel
    status: str
    candidate_name: Optional[str] = None
    readiness_label: Optional[str] = None
    total_questions: int
    completed_questions: int
    overall_score: Optional[float] = None
    started_at: datetime
    completed_at: Optional[datetime] = None
    report_available: bool = False


class DemoUserResponse(BaseModel):
    """Demo user response payload."""
    id: int
    name: str
    email: str





