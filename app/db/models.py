"""SQLAlchemy ORM models for AI-Powered Interview Coach."""

from datetime import datetime, timezone
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.db.database import Base


def get_utc_now():
    """Returns current UTC datetime with timezone information."""
    return datetime.now(timezone.utc)


class User(Base):
    """User account model."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        default=get_utc_now,
        server_default=func.now(),
        nullable=False,
    )

    # Relationships
    resumes = relationship("Resume", back_populates="user", cascade="all, delete-orphan")
    job_descriptions = relationship(
        "JobDescription", back_populates="user", cascade="all, delete-orphan"
    )
    interview_sessions = relationship(
        "InterviewSession", back_populates="user", cascade="all, delete-orphan"
    )


class Resume(Base):
    """Uploaded user resume model."""

    __tablename__ = "resumes"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    file_name = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    extracted_text = Column(Text, nullable=False)
    candidate_name = Column(String(255), nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        default=get_utc_now,
        server_default=func.now(),
        nullable=False,
    )

    # Relationships
    user = relationship("User", back_populates="resumes")
    interview_sessions = relationship("InterviewSession", back_populates="resume")


class JobDescription(Base):
    """Target job description model."""

    __tablename__ = "job_descriptions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    title = Column(String(255), nullable=False)
    company = Column(String(255), nullable=True)
    description = Column(Text, nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        default=get_utc_now,
        server_default=func.now(),
        nullable=False,
    )

    # Relationships
    user = relationship("User", back_populates="job_descriptions")
    interview_sessions = relationship("InterviewSession", back_populates="job_description")


class InterviewSession(Base):
    """Interview simulation session model."""

    __tablename__ = "interview_sessions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    resume_id = Column(Integer, ForeignKey("resumes.id"), nullable=True)
    job_description_id = Column(Integer, ForeignKey("job_descriptions.id"), nullable=True)
    interview_type = Column(String(100), nullable=False)
    difficulty = Column(String(50), nullable=False)
    total_questions = Column(Integer, nullable=False, default=5)
    status = Column(String(50), nullable=False, default="active")
    overall_score = Column(Float, nullable=True)
    started_at = Column(
        DateTime(timezone=True),
        default=get_utc_now,
        server_default=func.now(),
        nullable=False,
    )
    completed_at = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    user = relationship("User", back_populates="interview_sessions")
    resume = relationship("Resume", back_populates="interview_sessions")
    job_description = relationship("JobDescription", back_populates="interview_sessions")
    questions = relationship(
        "Question", back_populates="session", cascade="all, delete-orphan"
    )
    final_report = relationship(
        "FinalReport", back_populates="session", uselist=False, cascade="all, delete-orphan"
    )


class Question(Base):
    """Interview question model."""

    __tablename__ = "questions"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("interview_sessions.id"), nullable=False)
    question_number = Column(Integer, nullable=False)
    question_text = Column(Text, nullable=False)
    question_type = Column(String(100), nullable=False)
    difficulty = Column(String(50), nullable=False)
    source_context = Column(Text, nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        default=get_utc_now,
        server_default=func.now(),
        nullable=False,
    )

    # Relationships
    session = relationship("InterviewSession", back_populates="questions")
    answer = relationship(
        "Answer", back_populates="question", uselist=False, cascade="all, delete-orphan"
    )


class Answer(Base):
    """User response to an interview question."""

    __tablename__ = "answers"

    id = Column(Integer, primary_key=True, index=True)
    question_id = Column(Integer, ForeignKey("questions.id"), nullable=False)
    answer_text = Column(Text, nullable=False)
    answer_method = Column(String(20), nullable=False, default="text")
    submitted_at = Column(
        DateTime(timezone=True),
        default=get_utc_now,
        server_default=func.now(),
        nullable=False,
    )

    # Relationships
    question = relationship("Question", back_populates="answer")
    evaluation = relationship(
        "Evaluation", back_populates="answer", uselist=False, cascade="all, delete-orphan"
    )


class Evaluation(Base):
    """Evaluation feedback for a specific answer."""

    __tablename__ = "evaluations"

    id = Column(Integer, primary_key=True, index=True)
    answer_id = Column(Integer, ForeignKey("answers.id"), nullable=False)
    relevance_score = Column(Float, nullable=False)
    clarity_score = Column(Float, nullable=False)
    structure_score = Column(Float, nullable=False)
    completeness_score = Column(Float, nullable=False)
    technical_score = Column(Float, nullable=False)
    overall_score = Column(Float, nullable=False)
    strengths = Column(Text, nullable=False)
    weaknesses = Column(Text, nullable=False)
    improvement = Column(Text, nullable=False)
    sample_answer = Column(Text, nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        default=get_utc_now,
        server_default=func.now(),
        nullable=False,
    )

    # Relationships
    answer = relationship("Answer", back_populates="evaluation")


class FinalReport(Base):
    """Overall summary and recommendations for an interview session."""

    __tablename__ = "final_reports"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("interview_sessions.id"), nullable=False)
    overall_score = Column(Float, nullable=False)
    performance_summary = Column(Text, nullable=False)
    strengths = Column(Text, nullable=False)
    weaknesses = Column(Text, nullable=False)
    study_topics = Column(Text, nullable=False)
    recommendations = Column(Text, nullable=False)

    # Deterministic analytics and coaching extensions
    dimension_averages = Column(Text, nullable=True)
    strongest_dimension = Column(String(50), nullable=True)
    weakest_dimension = Column(String(50), nullable=True)
    difficulty_progression = Column(Text, nullable=True)
    covered_topics = Column(Text, nullable=True)
    missing_topics = Column(Text, nullable=True)
    topic_gap_analysis = Column(Text, nullable=True)
    coaching_tips = Column(Text, nullable=True)
    analytics_data = Column(Text, nullable=True)

    created_at = Column(
        DateTime(timezone=True),
        default=get_utc_now,
        server_default=func.now(),
        nullable=False,
    )

    # Relationships
    session = relationship("InterviewSession", back_populates="final_report")
