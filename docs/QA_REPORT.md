# QA and Hardening Report: AI-Powered Interview Coach

**Project:** AI-Powered Interview Coach  
**Phase:** Phase 12 — Full Integration, QA, Hardening & Final System Testing  
**Date:** October 3, 2026  
**Status:** **PASS (100%)**  

---

## 1. Executive Summary

This Quality Assurance and System Hardening Report certifies the complete, end-to-end verification of the AI-Powered Interview Coach backend and frontend architecture. All core functional modules—from PDF resume ingestion and local RAG vector retrieval, through LangGraph stateful interview orchestration and deterministic adaptive difficulty, to final coaching analytics and the Streamlit candidate interface—have been subjected to regression and hardening suites.

**Total Automated Test Count:** **198 Tests across 13 Test Suites**  
**Pass Rate:** **100% (198 Passed, 0 Failed, 0 Skipped)**  
**Zero-Cost Architecture Verification:** Embeddings (`sentence-transformers/all-MiniLM-L6-v2`), vector storage (`ChromaDB`), and speech-to-text transcription (`faster-whisper` `base.en` with int8 quantization on CPU) run 100% locally. Zero paid third-party APIs are utilized. The Generative AI layer runs via Google Gemini free-tier access with a strict LLM call budget ($2N + 1$ normal calls per $N$-question interview).

---

## 2. Test Suite Breakdown

| Suite # | Test File | Test Count | Scope & Focus | Result |
|---|---|---|---|---|
| **1** | `tests/test_db.py` | 2 | Database initialization, SQLite engine, 8 core tables schema & relationships | **PASS** |
| **2** | `tests/test_ingestion.py` | 6 | PyMuPDF text extraction, normalization, file size / format validation (max 5 MB) | **PASS** |
| **3** | `tests/test_rag.py` | 12 | Local MiniLM embeddings, persistent ChromaDB indexing, cosine similarity, user tenant isolation | **PASS** |
| **4** | `tests/test_llm.py` | 10 | Gemini client integration, API key loading, grounded prompt construction, Pydantic schemas | **PASS** |
| **5** | `tests/test_questions.py` | 11 | RAG-grounded question generation across 5 tracks, duplicate question detection, citation pruning | **PASS** |
| **6** | `tests/test_evaluation.py` | 15 | 5-dimension grounded answer evaluation, deterministic Python score calculation, technical guardrails | **PASS** |
| **7** | `tests/test_workflow.py` | 15 | LangGraph multi-turn orchestration, SQLite session persistence, turn resumption, duplicate answer guards | **PASS** |
| **8** | `tests/test_adaptive.py` | 22 | Deterministic adaptive difficulty policy, progression transitions, upper/lower bounds, guardrail overrides | **PASS** |
| **9** | `tests/test_reporting.py` | 24 | Deterministic performance analytics, Gemini qualitative coaching synthesis, report persistence, idempotency | **PASS** |
| **10** | `tests/test_frontend.py` | 17 | Streamlit UI architecture, API client mapping, error handling, session state transitions | **PASS** |
| **11** | `tests/test_integration.py` | 18 | E2E 15-step interview cycle, context modes, tracks, transaction safety, AFC disable, quota mapping, security | **PASS** |
| **12** | `tests/test_candidate_name_and_readiness.py` | 25 | CV name extraction, exact readiness boundaries (0.0-10.0), no false hiring claims, weakness-aware ranking | **PASS** |
| **13** | `tests/test_speech.py` | 21 | Local faster-whisper STT, CPU int8, zero paid API, audio format validation, privacy deletion, editable review sync, answer_method persistence | **PASS** |
| **TOTAL** | **13 Test Suites** | **198 Tests** | **Comprehensive Full-System & Speech Coverage** | **PASS** |

---

## 3. Integration Scenarios Verified

### 3.1 End-to-End Interview Lifecycle (15 Steps)
1. **User Creation:** Candidate account provisioned via `POST /users` / `POST /users/demo`.
2. **Resume PDF Ingestion:** Candidate PDF uploaded (`POST /resumes/upload`), text normalized, stored in SQLite.
3. **Job Description Ingestion:** Target job description ingested (`POST /job-descriptions`), validated for non-empty text.
4. **Vector Indexing:** Document chunks embedded locally and upserted to ChromaDB with user metadata.
5. **Session Initialization:** `POST /interview/sessions/start` initializes session state, invokes LangGraph to generate Question #1.
6. **Active Turn State:** Session verified with status `active`, `completed_question_count = 0`, question metadata validated.
7. **Answer Submission:** Candidate answers Question #1 via `POST /interview/sessions/{id}/answer`.
8. **Evaluation & Adaptive Shift:** Gemini evaluates answer; Python calculates authoritative scores; difficulty shifts dynamically.
9. **Next Question Generation:** Next personalized question delivered in the same turn response.
10. **Turn Progression:** Turns 2 through $N-1$ progress sequentially with verified persistence.
11. **Session Completion Invariant:** Turn $N$ submission completes the session (`status = completed`, `next_question = None`).
12. **Final Report Synthesis:** `POST /interview/sessions/{id}/report` calculates analytics and generates coaching report.
13. **Score & Coaching Verification:** Overall scores, dimension averages, strengths, weaknesses, and tips verified.
14. **Report Idempotency:** Subsequent report requests (`GET` and `POST`) return the existing report with 0 additional Gemini calls.
15. **Candidate History:** `GET /interview/sessions?user_id=X` confirms the completed session is listed accurately.

### 3.2 Context Modes
All 4 context combinations verified:
- **Mode 1 (Generic):** No resume, no job description. High-quality track questions generated successfully.
- **Mode 2 (Resume-only):** Tailored questions grounded in candidate experience and skills.
- **Mode 3 (JD-only):** Tailored questions focused on role expectations and core competencies.
- **Mode 4 (Resume + JD):** High-precision cross-grounded questions bridging candidate background and job requirements.

### 3.3 Interview Tracks & Difficulty Progression
- **5 Tracks Verified:** `hr`, `python`, `ai_ml`, `data_science`, `internship`.
- **Difficulty Transitions Verified:**
  - Easy $\rightarrow$ Medium (2 consecutive scores $\ge 7.5$)
  - Medium $\rightarrow$ Hard (2 consecutive scores $\ge 7.5$)
  - Hard capped at Hard (`upper_bound` guardrail)
  - Hard $\rightarrow$ Medium (scores $\le 5.0$)
  - Medium $\rightarrow$ Easy (scores $\le 5.0$)
  - Easy bounded at Easy (`lower_bound` guardrail)
  - Technical guardrails override promotion if technical correctness $< 7$ or completeness $< 6$.

---

## 4. Security, Hardening & Guardrails

- **Zero Secret Leakage:** Complete repository scan confirms zero exposed API keys or credentials in code or commit history. `.gitignore` strictly protects `.env`, `.venv/`, `data/uploads/`, `data/vector_store/`, and database binaries.
- **Tenant Isolation:** Rigorously enforced across SQLite tables and ChromaDB vector queries. Cross-tenant reads, answer submissions, and report retrievals return `404 Not Found`.
- **Prompt Injection Defense:** Multi-layer defense handles adversarial strings (`IGNORE ALL INSTRUCTIONS`, `<INJECTION>`, fake JSON fences) in Resume, JD, and Answers as harmless passive data.
- **Malformed LLM Output Handling:** At most 1 structured retry on JSON parse failures. Persistent failures return HTTP `502 Bad Gateway` without crashing the application.
- **Input Boundaries & Validation:** Strict Pydantic constraints reject $N=0$, $N > 10$, invalid enum tracks/difficulties, empty/whitespace strings, non-PDF uploads, corrupt documents, and files $> 5$ MB.
- **Double-Click & Rerun Safety:** Streamlit frontend includes state recovery and submission mutexes to prevent duplicate submissions.

---

## 5. Phase 12.1 Product Enhancements

### 5.1 Deterministic Candidate Name Extraction
- **Zero Cost / Zero LLM Overhead:** Extracts candidate names from the top 15 non-empty lines of uploaded CV PDFs using pure Python pattern matching and token validation.
- **Robust Sanitization:** Automatically ignores section headers (e.g., "Curriculum Vitae", "Resume"), contact information (emails, phones, URLs), and professional titles (e.g., "Senior Software Engineer", "AI Specialist").
- **Delimited Token Support:** Handles names followed by role descriptors separated by `|`, `-`, or `,` (e.g. `Sara Malik | Software Engineer` -> `Sara Malik`).
- **Database & Schema Persistence:** `candidate_name` column added to SQLite `resumes` table with automatic migration support. Fallback gracefully to user profile display name or `Demo Candidate`.

### 5.2 Deterministic Interview Readiness Assessment
- **Zero LLM Calls:** 100% Python calculated from authoritative overall score.
- **Exact Score Boundaries:**
  - `< 6.0`: `not_yet_ready` ("Not Yet Ready")
  - `6.0 <= score < 8.0`: `developing` ("Developing — More Preparation Recommended")
  - `8.0 <= score <= 10.0`: `strong_readiness` ("Strong Interview Readiness")
- **No False Hiring Decisions:** Guaranteed coaching framing without misleading employment promises (zero usage of "hired", "offer", "selected", etc.).
- **Weakness-Aware Guidance:** Ranks evaluation dimensions with deterministic tie-breaking to pinpoint priority improvement areas and top demonstrated strengths.
- **Export Ready:** Full report export option with candidate-specific file naming (`{candidate_name}_Interview_Report.md`).

---

## 6. Free-Tier Architecture & LLM Budget

- **Local Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` runs on CPU without API keys or costs.
- **Local Vector Database:** `ChromaDB` stores embeddings persistently on disk.
- **Deterministic Analytics & Readiness:** 100% computed in Python without LLM calls.
- **LLM Call Budget:** Exactly $2N + 1$ normal calls per $N$-question interview session ($N$ questions + $N$ evaluations + 1 final report).
- **Idempotency:** Report generation is strictly idempotent, preventing redundant Gemini calls on repeated requests.

---

## 7. Known Prototype Limitations (Non-Blocking)

1. **Local Desktop Architecture:** Designed for local single-user developer/demo execution. Concurrency is limited by SQLite's single-writer database model.
2. **Local Vector Storage:** ChromaDB runs in embedded persistent directory mode rather than a distributed cluster.
3. **PyMuPDF Rendering:** Scanned image-only PDFs without OCR text layers yield no extractable text.

---

## 8. QA Verdict

**FINAL STATUS: PASS**  
The AI-Powered Interview Coach is fully integrated, personalized, voice-enabled (optional local faster-whisper STT with editable synchronization), hardened, resilient against invalid inputs, and verified across all 13 test suites (198/198 tests passing).
