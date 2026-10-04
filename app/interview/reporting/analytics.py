"""Pure deterministic Python analytics engine for interview sessions."""

from typing import Any, Dict, List, Optional, Set, Tuple

from app.interview.schemas import (
    DifficultyChanges,
    DifficultyLevel,
    DimensionAverages,
    InterviewAnalytics,
    InterviewReadiness,
)

TIE_BREAK_ORDER = [
    "relevance",
    "clarity",
    "completeness",
    "technical_correctness",
    "structure",
]

DIFFICULTY_RANK = {
    "easy": 0,
    "medium": 1,
    "hard": 2,
    DifficultyLevel.EASY: 0,
    DifficultyLevel.MEDIUM: 1,
    DifficultyLevel.HARD: 2,
}

TRACK_DISPLAY_NAMES = {
    "hr": "HR & Behavioral",
    "python": "Python Developer",
    "ai_ml": "AI / Machine Learning",
    "data_science": "Data Science",
    "internship": "Software Engineering Internship",
}

DIMENSION_DISPLAY_NAMES = {
    "relevance": "Answer Relevance",
    "clarity": "Communication Clarity",
    "completeness": "Completeness & Depth",
    "technical_correctness": "Technical Accuracy",
    "structure": "Logical Structure",
}

DIMENSION_COACHING_TIPS = {
    "relevance": "Keep responses tightly anchored to the specific prompt; avoid drifting into unrelated background details.",
    "clarity": "Articulate ideas concisely and explain complex technical concepts with clear, direct terminology.",
    "completeness": "Provide end-to-end depth, touching on edge cases, implementation considerations, and trade-offs.",
    "technical_correctness": "Strengthen technical accuracy, syntax precision, and underlying architectural mechanics.",
    "structure": "Use structured answering patterns like STAR (Situation, Task, Action, Result) to organize thoughts.",
}

READINESS_NOT_YET_READY = "not_yet_ready"
READINESS_DEVELOPING = "developing"
READINESS_STRONG = "strong_readiness"

LABEL_NOT_YET_READY = "Not Yet Ready"
LABEL_DEVELOPING = "Developing — More Preparation Recommended"
LABEL_STRONG = "Strong Interview Readiness"


def rank_dimensions(dim_averages: Dict[str, float]) -> Tuple[List[str], List[str]]:
    """Deterministically ranks dimensions from strongest to weakest, and weakest to strongest.

    Tie-breaking uses TIE_BREAK_ORDER to ensure deterministic sorting.
    """
    strongest_sorted = sorted(
        TIE_BREAK_ORDER,
        key=lambda d: (-dim_averages.get(d, 0.0), TIE_BREAK_ORDER.index(d)),
    )
    weakest_sorted = sorted(
        TIE_BREAK_ORDER,
        key=lambda d: (dim_averages.get(d, 0.0), TIE_BREAK_ORDER.index(d)),
    )
    return strongest_sorted, weakest_sorted


def compute_readiness_level(overall_score: float) -> Tuple[str, str]:
    """Returns (readiness_code, readiness_label) based on deterministic score boundaries.

    Score boundaries:
      < 6.0: NOT_YET_READY ("not_yet_ready", "Not Yet Ready")
      >= 6.0 and < 8.0: DEVELOPING ("developing", "Developing — More Preparation Recommended")
      >= 8.0 and <= 10.0: STRONG_READINESS ("strong_readiness", "Strong Interview Readiness")
    """
    if overall_score < 6.0:
        return READINESS_NOT_YET_READY, LABEL_NOT_YET_READY
    elif overall_score < 8.0:
        return READINESS_DEVELOPING, LABEL_DEVELOPING
    else:
        return READINESS_STRONG, LABEL_STRONG


def compute_interview_readiness(
    overall_score: float,
    dimension_averages: DimensionAverages,
    interview_type: Any,
) -> InterviewReadiness:
    """Calculates deterministic interview readiness and weakness-aware recommendations.

    Zero LLM calls. Framed strictly as coaching readiness with no false hiring claims.
    """
    type_str = interview_type.value if hasattr(interview_type, "value") else str(interview_type)
    track_label = TRACK_DISPLAY_NAMES.get(type_str.lower(), type_str.replace("_", " ").title())

    dim_dict = {
        dim: getattr(dimension_averages, dim, 0.0)
        for dim in TIE_BREAK_ORDER
    }
    strongest_dims, weakest_dims = rank_dimensions(dim_dict)

    readiness_code, readiness_label = compute_readiness_level(overall_score)

    if readiness_code == READINESS_NOT_YET_READY:
        headline = f"Targeted preparation needed before live {track_label} interviews."
        message = (
            f"Your interview simulation shows emerging potential, but requires significant practice on core "
            f"concepts, structured answering, and technical depth before taking a formal {track_label} interview."
        )
        next_step = "Focus on core fundamentals, practice answering aloud with the STAR technique, and review missing topics."
    elif readiness_code == READINESS_DEVELOPING:
        headline = f"Demonstrates developing readiness for {track_label} roles — more preparation recommended."
        message = (
            f"You show a solid foundational understanding of {track_label} concepts. With focused refinement "
            f"on technical depth, edge cases, and structured delivery, you will be well-positioned for live interviews."
        )
        next_step = "Refine technical precision and elaborate on trade-offs and implementation details in weaker areas."
    else:
        headline = f"Demonstrates strong interview readiness for {track_label} roles."
        message = (
            f"You demonstrated excellent command of key concepts, strong communication clarity, and methodical "
            f"problem-solving suited for {track_label} interviews."
        )
        next_step = "Maintain your preparation cadence and conduct mock rounds focused on complex system trade-offs or specialized follow-ups."

    priority_improvements = [
        f"{DIMENSION_DISPLAY_NAMES.get(d, d)}: {DIMENSION_COACHING_TIPS.get(d, '')}"
        for d in weakest_dims[:2]
    ]

    strongest_areas = [
        f"{DIMENSION_DISPLAY_NAMES.get(d, d)} ({dim_dict.get(d, 0.0):.1f}/10)"
        for d in strongest_dims[:2]
    ]

    return InterviewReadiness(
        readiness_code=readiness_code,
        readiness_label=readiness_label,
        headline_wording=headline,
        assessment_message=message,
        priority_improvement_areas=priority_improvements,
        strongest_areas=strongest_areas,
        next_step=next_step,
    )


def deduplicate_topics(topic_lists: List[List[str]]) -> List[str]:
    """Aggregates topic strings across turns with case-insensitive deduplication.
    
    Preserves the first useful display form, strips surrounding whitespace,
    and ignores empty strings.
    """
    seen: Set[str] = set()
    result: List[str] = []

    for group in topic_lists:
        if not group:
            continue
        for topic in group:
            if not isinstance(topic, str):
                continue
            cleaned = topic.strip()
            if not cleaned:
                continue
            lower_key = cleaned.lower()
            if lower_key not in seen:
                seen.add(lower_key)
                result.append(cleaned)

    return result


def compute_difficulty_changes(progression: List[str]) -> DifficultyChanges:
    """Calculates step shifts between consecutive question difficulties."""
    increases = 0
    decreases = 0
    maintains = 0

    if len(progression) <= 1:
        return DifficultyChanges(increases=0, decreases=0, maintains=0)

    for i in range(1, len(progression)):
        prev_diff = progression[i - 1]
        curr_diff = progression[i]
        prev_rank = DIFFICULTY_RANK.get(prev_diff, 1)
        curr_rank = DIFFICULTY_RANK.get(curr_diff, 1)

        if curr_rank > prev_rank:
            increases += 1
        elif curr_rank < prev_rank:
            decreases += 1
        else:
            maintains += 1

    return DifficultyChanges(
        increases=increases,
        decreases=decreases,
        maintains=maintains,
    )


def compute_interview_analytics(turns: List[Dict[str, Any]]) -> InterviewAnalytics:
    """Calculates comprehensive deterministic metrics across completed interview turns.
    
    Expected format per turn dict:
      - question_number: int
      - difficulty: str | DifficultyLevel
      - overall_score: float
      - scores: dict with relevance, clarity, completeness, technical_correctness, structure
      - covered_topics: list[str]
      - missing_topics: list[str]
    """
    if not turns:
        raise ValueError("Cannot calculate analytics on an empty set of turns.")

    question_count = len(turns)

    # 1. Overall Score calculation
    overall_scores = [float(t["overall_score"]) for t in turns]
    overall_score = round(sum(overall_scores) / question_count, 2)
    overall_score = max(0.0, min(10.0, overall_score))

    # 2. Dimension averages
    dim_sums = {dim: 0.0 for dim in TIE_BREAK_ORDER}
    for t in turns:
        scores = t.get("scores") or {}
        for dim in TIE_BREAK_ORDER:
            dim_sums[dim] += float(scores.get(dim, 0.0))

    dim_averages = {
        dim: round(dim_sums[dim] / question_count, 2)
        for dim in TIE_BREAK_ORDER
    }
    # Clamp bounds to [0.0, 10.0]
    for dim in TIE_BREAK_ORDER:
        dim_averages[dim] = max(0.0, min(10.0, dim_averages[dim]))

    dimension_averages_model = DimensionAverages(**dim_averages)

    # 3. Strongest & Weakest Dimension (deterministic tie handling)
    max_val = max(dim_averages.values())
    min_val = min(dim_averages.values())

    strongest_dimension = next(d for d in TIE_BREAK_ORDER if dim_averages[d] == max_val)
    weakest_dimension = next(d for d in TIE_BREAK_ORDER if dim_averages[d] == min_val)

    # 4. Highest & Lowest question scores
    highest_score = round(max(overall_scores), 2)
    lowest_score = round(min(overall_scores), 2)

    highest_q_num: Optional[int] = None
    lowest_q_num: Optional[int] = None
    for t in turns:
        q_num = t.get("question_number")
        s = round(float(t["overall_score"]), 2)
        if s == highest_score and highest_q_num is None:
            highest_q_num = q_num
        if s == lowest_score and lowest_q_num is None:
            lowest_q_num = q_num

    # 5. Difficulty progression
    difficulty_progression = [
        t["difficulty"].value if isinstance(t["difficulty"], DifficultyLevel) else str(t["difficulty"])
        for t in turns
    ]
    diff_changes = compute_difficulty_changes(difficulty_progression)

    # 6. Topic aggregation
    covered_topic_lists = [t.get("covered_topics", []) for t in turns]
    missing_topic_lists = [t.get("missing_topics", []) for t in turns]

    covered_topics = deduplicate_topics(covered_topic_lists)
    missing_topics = deduplicate_topics(missing_topic_lists)

    return InterviewAnalytics(
        question_count=question_count,
        overall_score=overall_score,
        dimension_averages=dimension_averages_model,
        highest_question_score=highest_score,
        lowest_question_score=lowest_score,
        highest_question_number=highest_q_num,
        lowest_question_number=lowest_q_num,
        strongest_dimension=strongest_dimension,
        weakest_dimension=weakest_dimension,
        difficulty_progression=difficulty_progression,
        difficulty_changes=diff_changes,
        covered_topics=covered_topics,
        missing_topics=missing_topics,
    )
