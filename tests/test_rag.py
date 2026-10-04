"""Automated test suite for Phase 4: RAG Retrieval Foundation."""

import os
import sys
import unittest

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.db.database import SessionLocal, init_db
from app.db.models import JobDescription, Resume, User
from app.main import app
from app.rag.chunker import chunk_text
from app.rag.embeddings import embed_documents, embed_query
from app.rag.retriever import retrieve_context
from app.rag.vector_store import (
    delete_existing_chunks,
    get_collection,
    index_job_description,
    index_resume,
)


class TestRAGFoundation(unittest.TestCase):
    """Comprehensive test suite for chunking, local embeddings, vector storage, and semantic retrieval."""

    @classmethod
    def setUpClass(cls):
        init_db()
        cls.db = SessionLocal()
        cls.client = TestClient(app)

        # Synthetic Users
        cls.email_a = "synthetic_user_a@example.com"
        cls.email_b = "synthetic_user_b@example.com"

        # Clean existing test users if any
        for email in [cls.email_a, cls.email_b]:
            existing = cls.db.query(User).filter(User.email == email).first()
            if existing:
                cls.db.delete(existing)
        cls.db.commit()

        cls.user_a = User(name="User A (Test Candidate)", email=cls.email_a)
        cls.user_b = User(name="User B (Other Candidate)", email=cls.email_b)
        cls.db.add_all([cls.user_a, cls.user_b])
        cls.db.commit()
        cls.db.refresh(cls.user_a)
        cls.db.refresh(cls.user_b)

        # Synthetic Resume for User A
        cls.resume_text_a = (
            "Candidate: Test Candidate\n\n"
            "Summary:\n"
            "Passionate AI Software Engineer experienced in building scalable applications.\n\n"
            "Skills:\n"
            "Python, FastAPI, Machine Learning, Computer Vision, SQLite, Vector Databases\n\n"
            "Projects:\n"
            "SafetyCopilot — built a RAG-based industrial safety assistant "
            "using LangGraph and vector retrieval for real-time hazard detection.\n\n"
            "VisionInspect — developed a computer vision defect detection system "
            "using convolutional neural networks for factory assembly lines.\n\n"
            "Experience:\n"
            "Software Developer Intern at Tech Labs. Built high-throughput REST APIs using FastAPI."
        )
        cls.resume_a = Resume(
            user_id=cls.user_a.id,
            file_name="candidate_a_resume.pdf",
            file_path="data/uploads/synthetic_a.pdf",
            extracted_text=cls.resume_text_a,
        )

        # Synthetic Job Description for User A
        cls.jd_text_a = (
            "Title: AI Engineer Intern\n\n"
            "Required skills:\n"
            "Python, machine learning, RAG, FastAPI and vector databases.\n\n"
            "Responsibilities:\n"
            "Responsibilities include building retrieval systems and evaluating LLM applications. "
            "Work with vector retrieval pipelines and optimize prompt contexts."
        )
        cls.jd_a = JobDescription(
            user_id=cls.user_a.id,
            title="AI Engineer Intern",
            company="NextGen AI",
            description=cls.jd_text_a,
        )

        # Synthetic Resume for User B (User Isolation Test)
        cls.resume_text_b = (
            "Candidate: User B Secret\n\n"
            "Proprietary Project:\n"
            "ProjectX_Quantum — quantum algorithmic encryption for secure satellite comms."
        )
        cls.resume_b = Resume(
            user_id=cls.user_b.id,
            file_name="candidate_b_resume.pdf",
            file_path="data/uploads/synthetic_b.pdf",
            extracted_text=cls.resume_text_b,
        )

        cls.db.add_all([cls.resume_a, cls.jd_a, cls.resume_b])
        cls.db.commit()
        cls.db.refresh(cls.resume_a)
        cls.db.refresh(cls.jd_a)
        cls.db.refresh(cls.resume_b)

    @classmethod
    def tearDownClass(cls):
        try:
            # Clean up Chroma vectors
            delete_existing_chunks("resume", cls.resume_a.id)
            delete_existing_chunks("job_description", cls.jd_a.id)
            delete_existing_chunks("resume", cls.resume_b.id)

            # Clean up SQLite records
            for u in [cls.user_a, cls.user_b]:
                user_rec = cls.db.query(User).filter(User.id == u.id).first()
                if user_rec:
                    cls.db.delete(user_rec)
            cls.db.commit()
        finally:
            cls.db.close()

    def test_a_chunking(self):
        """A. Chunking logic verification."""
        long_text = "\n\n".join([f"Paragraph {i}: " + " ".join([f"topic_{i}_token_{j}" for j in range(35)]) for i in range(8)])
        chunks = chunk_text(long_text, chunk_size=300, chunk_overlap=50)

        self.assertGreater(len(chunks), 1, "Expected multiple chunks for long text.")
        for i, c in enumerate(chunks):
            self.assertEqual(c.chunk_index, i, "Chunk indices must be strictly ordered.")
            self.assertTrue(len(c.text.strip()) > 0, "No chunk should be empty.")
            self.assertEqual(c.character_count, len(c.text))

        # Check no accidental duplicate chunks
        chunk_texts = [c.text for c in chunks]
        self.assertEqual(len(chunk_texts), len(set(chunk_texts)), "Chunks should not contain duplicates.")

    def test_b_local_embeddings(self):
        """B. Verify local embedding dimension, batch documents, and query embedding."""
        query_vec = embed_query("FastAPI vector retrieval")
        self.assertEqual(len(query_vec), 384, "MiniLM-L6-v2 must output 384-dimensional vectors.")

        doc_vecs = embed_documents(["Document one content", "Document two content"])
        self.assertEqual(len(doc_vecs), 2)
        self.assertEqual(len(doc_vecs[0]), 384)
        self.assertEqual(len(doc_vecs[1]), 384)

    def test_c_resume_indexing(self):
        """C. Index resume and verify chunks and deterministic IDs in Chroma."""
        stats = index_resume(resume_id=self.resume_a.id, db=self.db)
        self.assertEqual(stats["source_type"], "resume")
        self.assertEqual(stats["source_id"], self.resume_a.id)
        self.assertGreater(stats["chunks_indexed"], 0)

        # Check in Chroma
        collection = get_collection()
        stored = collection.get(
            where={"$and": [{"source_type": "resume"}, {"resume_id": self.resume_a.id}]}
        )
        self.assertEqual(len(stored["ids"]), stats["chunks_indexed"])
        for chunk_id in stored["ids"]:
            self.assertTrue(chunk_id.startswith(f"resume_{self.resume_a.id}_chunk_"))

    def test_d_job_description_indexing(self):
        """D. Index job description and verify vectors in Chroma."""
        stats = index_job_description(job_description_id=self.jd_a.id, db=self.db)
        self.assertEqual(stats["source_type"], "job_description")
        self.assertEqual(stats["source_id"], self.jd_a.id)
        self.assertGreater(stats["chunks_indexed"], 0)

        collection = get_collection()
        stored = collection.get(
            where={"$and": [{"source_type": "job_description"}, {"job_description_id": self.jd_a.id}]}
        )
        self.assertEqual(len(stored["ids"]), stats["chunks_indexed"])

    def test_e_resume_retrieval_safety_copilot(self):
        """E. Resume retrieval: 'Which project used RAG for industrial safety?' -> SafetyCopilot."""
        results = retrieve_context(
            query="Which project used RAG for industrial safety?",
            user_id=self.user_a.id,
            resume_id=self.resume_a.id,
            top_k=2,
        )
        self.assertGreater(len(results), 0)
        top_text = results[0].text
        self.assertIn("SafetyCopilot", top_text)
        self.assertIn("industrial safety", top_text)

    def test_f_computer_vision_retrieval(self):
        """F. Computer Vision retrieval: 'What computer vision project did the candidate build?' -> VisionInspect."""
        results = retrieve_context(
            query="What computer vision project did the candidate build?",
            user_id=self.user_a.id,
            resume_id=self.resume_a.id,
            top_k=2,
        )
        self.assertGreater(len(results), 0)
        found = any("VisionInspect" in r.text or "computer vision" in r.text for r in results)
        self.assertTrue(found, "Expected VisionInspect or computer vision defect detection in results.")

    def test_g_job_description_retrieval(self):
        """G. Job Description retrieval: 'What skills are required for the AI Engineer Intern role?'"""
        results = retrieve_context(
            query="What skills are required for the AI Engineer Intern role?",
            user_id=self.user_a.id,
            job_description_id=self.jd_a.id,
            top_k=2,
        )
        self.assertGreater(len(results), 0)
        top_text = results[0].text
        self.assertTrue(
            "Python" in top_text or "machine learning" in top_text or "RAG" in top_text
        )

    def test_h_user_isolation(self):
        """H. MANDATORY: User isolation. User A search must NEVER return User B chunks."""
        # Index User B's secret resume
        index_resume(resume_id=self.resume_b.id, db=self.db)

        # Search as User A for User B's content
        results_for_a = retrieve_context(
            query="ProjectX_Quantum satellite encryption",
            user_id=self.user_a.id,
            top_k=5,
        )

        for chunk in results_for_a:
            self.assertEqual(
                chunk.metadata.get("user_id"),
                self.user_a.id,
                "SECURITY VIOLATION: User A retrieved a chunk not belonging to User A!",
            )
            self.assertNotIn("ProjectX_Quantum", chunk.text)

        # Search as User B should find it
        results_for_b = retrieve_context(
            query="ProjectX_Quantum satellite encryption",
            user_id=self.user_b.id,
            top_k=2,
        )
        self.assertGreater(len(results_for_b), 0)
        self.assertIn("ProjectX_Quantum", results_for_b[0].text)

    def test_i_reindexing_no_duplicates(self):
        """I. Re-indexing the same resume twice replaces/upserts vectors rather than duplicating."""
        stats_1 = index_resume(resume_id=self.resume_a.id, db=self.db)
        stats_2 = index_resume(resume_id=self.resume_a.id, db=self.db)

        self.assertEqual(stats_1["chunks_indexed"], stats_2["chunks_indexed"])

        collection = get_collection()
        stored = collection.get(
            where={"$and": [{"source_type": "resume"}, {"resume_id": self.resume_a.id}]}
        )
        self.assertEqual(
            len(stored["ids"]),
            stats_2["chunks_indexed"],
            "Chroma collection must not contain duplicated vector IDs after re-indexing.",
        )

    def test_j_top_k_respected(self):
        """J. Verify top_k parameter is respected."""
        results = retrieve_context(
            query="Python developer experience",
            user_id=self.user_a.id,
            top_k=1,
        )
        self.assertLessEqual(len(results), 1)

    def test_k_missing_source_errors(self):
        """K. Indexing non-existent resume or JD raises 404."""
        with self.assertRaises(HTTPException) as ctx_res:
            index_resume(resume_id=999999, db=self.db)
        self.assertEqual(ctx_res.exception.status_code, 404)

        with self.assertRaises(HTTPException) as ctx_jd:
            index_job_description(job_description_id=999999, db=self.db)
        self.assertEqual(ctx_jd.exception.status_code, 404)

    def test_l_fastapi_endpoints(self):
        """L. Verify the RAG HTTP endpoints using TestClient."""
        # Test POST /rag/index/resume/{id}
        res_idx = self.client.post(f"/rag/index/resume/{self.resume_a.id}")
        self.assertEqual(res_idx.status_code, 200)
        data_idx = res_idx.json()
        self.assertEqual(data_idx["source_type"], "resume")
        self.assertEqual(data_idx["source_id"], self.resume_a.id)

        # Test POST /rag/index/job-description/{id}
        res_jd_idx = self.client.post(f"/rag/index/job-description/{self.jd_a.id}")
        self.assertEqual(res_jd_idx.status_code, 200)

        # Test POST /rag/search
        search_payload = {
            "query": "Which project used RAG for industrial safety?",
            "user_id": self.user_a.id,
            "resume_id": self.resume_a.id,
            "top_k": 2,
        }
        res_search = self.client.post("/rag/search", json=search_payload)
        self.assertEqual(res_search.status_code, 200)
        search_data = res_search.json()
        self.assertIn("results", search_data)
        self.assertGreater(len(search_data["results"]), 0)
        self.assertIn("SafetyCopilot", search_data["results"][0]["text"])


if __name__ == "__main__":
    unittest.main()
