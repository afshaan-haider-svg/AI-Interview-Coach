"""Programmatic verification test for database foundation and models."""

import os
import sys
import unittest

# Ensure project root is in Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sqlalchemy import inspect
from app.db.database import SessionLocal, engine, init_db
from app.db.models import User


class TestDatabaseFoundation(unittest.TestCase):
    """Tests for database initialization, tables, and CRUD operations."""

    @classmethod
    def setUpClass(cls):
        init_db()

    def test_database_and_tables(self):
        db_file_path = "data/interview_coach.db"
        self.assertTrue(
            os.path.exists(db_file_path),
            f"Database file not found at {db_file_path}",
        )

        inspector = inspect(engine)
        tables = set(inspector.get_table_names())
        expected_tables = {
            "users",
            "resumes",
            "job_descriptions",
            "interview_sessions",
            "questions",
            "answers",
            "evaluations",
            "final_reports",
        }
        missing_tables = expected_tables - tables
        self.assertEqual(len(missing_tables), 0, f"Missing tables: {missing_tables}")

    def test_user_insert_query_delete(self):
        db = SessionLocal()
        test_email = "test_user_phase2@example.com"
        try:
            # Clean up if existing
            existing = db.query(User).filter(User.email == test_email).first()
            if existing:
                db.delete(existing)
                db.commit()

            # Insert
            user = User(name="Test User", email=test_email)
            db.add(user)
            db.commit()
            db.refresh(user)
            self.assertIsNotNone(user.id)
            user_id = user.id

            # Query
            queried = db.query(User).filter(User.id == user_id).first()
            self.assertIsNotNone(queried)
            self.assertEqual(queried.email, test_email)
            self.assertEqual(queried.name, "Test User")

            # Delete
            db.delete(queried)
            db.commit()

            # Verify deletion
            deleted = db.query(User).filter(User.id == user_id).first()
            self.assertIsNone(deleted)
        finally:
            db.close()


if __name__ == "__main__":
    unittest.main()
