"""Deterministic adaptive difficulty policy for interview question progression."""

from typing import Any, Dict, Union

from app.interview.schemas import (
    AdaptiveDecision,
    AnswerEvaluation,
    DifficultyLevel,
    InterviewType,
)


DIFFICULTY_ORDER = [
    DifficultyLevel.EASY,
    DifficultyLevel.MEDIUM,
    DifficultyLevel.HARD,
]

TECHNICAL_TRACKS = {
    InterviewType.PYTHON,
    InterviewType.AI_ML,
    InterviewType.DATA_SCIENCE,
}


def decide_next_difficulty(
    current_difficulty: Union[DifficultyLevel, str],
    interview_type: Union[InterviewType, str],
    evaluation: Union[AnswerEvaluation, Dict[str, Any]],
) -> AdaptiveDecision:
    """Pure, deterministic policy selecting the difficulty of the next question.

    Rules:
      - overall_score <= 5.0: decrease by 1 level (down to easy, lower_bound)
      - overall_score >= 8.0:
          - for technical tracks (python, ai_ml, data_science):
              requires technical_correctness >= 7 and completeness >= 6
          - for non-technical tracks (hr, internship):
              increases directly
          - increases by 1 level (up to hard, upper_bound)
      - 5.0 < overall_score < 8.0: maintain current difficulty (stable_performance)
    """
    c_diff = (
        current_difficulty
        if isinstance(current_difficulty, DifficultyLevel)
        else DifficultyLevel(current_difficulty)
    )
    i_type = (
        interview_type
        if isinstance(interview_type, InterviewType)
        else InterviewType(interview_type)
    )

    # Extract score metrics
    if isinstance(evaluation, AnswerEvaluation):
        overall = evaluation.overall_score
        tech_score = evaluation.scores.technical_correctness
        comp_score = evaluation.scores.completeness
    elif isinstance(evaluation, dict):
        overall = float(evaluation.get("overall_score", 0.0))
        scores_dict = evaluation.get("scores", {})
        tech_score = int(scores_dict.get("technical_correctness", 0))
        comp_score = int(scores_dict.get("completeness", 0))
    else:
        overall = 0.0
        tech_score = 0
        comp_score = 0

    curr_idx = DIFFICULTY_ORDER.index(c_diff)

    # Rule 1: Decrease difficulty if overall_score <= 5.0
    if overall <= 5.0:
        if curr_idx == 0:
            return AdaptiveDecision(
                previous_difficulty=c_diff,
                next_difficulty=c_diff,
                action="maintain",
                reason_code="lower_bound",
            )
        return AdaptiveDecision(
            previous_difficulty=c_diff,
            next_difficulty=DIFFICULTY_ORDER[curr_idx - 1],
            action="decrease",
            reason_code="low_performance",
        )

    # Rule 2: Increase difficulty if overall_score >= 8.0
    if overall >= 8.0:
        if i_type in TECHNICAL_TRACKS:
            if tech_score < 7:
                return AdaptiveDecision(
                    previous_difficulty=c_diff,
                    next_difficulty=c_diff,
                    action="maintain",
                    reason_code="technical_guardrail",
                )
            if comp_score < 6:
                return AdaptiveDecision(
                    previous_difficulty=c_diff,
                    next_difficulty=c_diff,
                    action="maintain",
                    reason_code="completeness_guardrail",
                )

        if curr_idx == len(DIFFICULTY_ORDER) - 1:
            return AdaptiveDecision(
                previous_difficulty=c_diff,
                next_difficulty=c_diff,
                action="maintain",
                reason_code="upper_bound",
            )

        return AdaptiveDecision(
            previous_difficulty=c_diff,
            next_difficulty=DIFFICULTY_ORDER[curr_idx + 1],
            action="increase",
            reason_code="high_performance",
        )

    # Rule 3: Maintain current difficulty for stable performance (5.0 < overall < 8.0)
    return AdaptiveDecision(
        previous_difficulty=c_diff,
        next_difficulty=c_diff,
        action="maintain",
        reason_code="stable_performance",
    )
