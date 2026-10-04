import os
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config import settings

# Handle SQLite's check_same_thread configuration
connect_args = {}
if settings.DATABASE_URL.startswith("sqlite"):
    connect_args["check_same_thread"] = False

engine = create_engine(
    settings.DATABASE_URL,
    connect_args=connect_args,
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy ORM models."""
    pass


def get_db() -> Generator:
    """FastAPI dependency for yielding database sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Creates database directories (if SQLite) and initializes all tables."""
    if settings.DATABASE_URL.startswith("sqlite:///"):
        db_path = settings.DATABASE_URL.replace("sqlite:///", "")
        dir_name = os.path.dirname(db_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

    # Import models here to ensure they are registered with Base.metadata
    from app.db import models  # noqa: F401

    Base.metadata.create_all(bind=engine)

    from sqlalchemy import inspect, text
    inspector = inspect(engine)
    if "interview_sessions" in inspector.get_table_names():
        columns = [c["name"] for c in inspector.get_columns("interview_sessions")]
        if "total_questions" not in columns:
            with engine.connect() as conn:
                conn.execute(text("ALTER TABLE interview_sessions ADD COLUMN total_questions INTEGER DEFAULT 5"))
                conn.commit()

    if "resumes" in inspector.get_table_names():
        res_columns = [c["name"] for c in inspector.get_columns("resumes")]
        if "candidate_name" not in res_columns:
            with engine.connect() as conn:
                conn.execute(text("ALTER TABLE resumes ADD COLUMN candidate_name VARCHAR(255)"))
                conn.commit()

    if "answers" in inspector.get_table_names():
        ans_columns = [c["name"] for c in inspector.get_columns("answers")]
        if "answer_method" not in ans_columns:
            with engine.connect() as conn:
                conn.execute(text("ALTER TABLE answers ADD COLUMN answer_method VARCHAR(20) DEFAULT 'text'"))
                conn.commit()

    if "final_reports" in inspector.get_table_names():
        fr_columns = [c["name"] for c in inspector.get_columns("final_reports")]
        new_cols = [
            ("dimension_averages", "TEXT"),
            ("strongest_dimension", "VARCHAR(50)"),
            ("weakest_dimension", "VARCHAR(50)"),
            ("difficulty_progression", "TEXT"),
            ("covered_topics", "TEXT"),
            ("missing_topics", "TEXT"),
            ("topic_gap_analysis", "TEXT"),
            ("coaching_tips", "TEXT"),
            ("analytics_data", "TEXT"),
        ]
        for col_name, col_type in new_cols:
            if col_name not in fr_columns:
                with engine.connect() as conn:
                    conn.execute(text(f"ALTER TABLE final_reports ADD COLUMN {col_name} {col_type}"))
                    conn.commit()

