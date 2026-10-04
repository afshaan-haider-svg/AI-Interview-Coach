# Testing Strategy & Quality Assurance

The **AI-Powered Interview Coach** includes a comprehensive automated test suite consisting of **198 test cases across 13 dedicated test suites** with **100% pass rate**. Testing spans pure unit tests, deterministic algorithm verification, mocked LLM integration, speech transcription pipelines, and full multi-turn session workflows.

---

## 📊 Test Suite Breakdown (198 Tests / 13 Suites)

| # | Test Suite | Module Path | Tests | Focus Area & Key Verifications |
|---|---|---|:---:|---|
| 1 | **Database & Schema** | `tests/test_db.py` | 5 | SQLite connection, table migrations, relationships, cascading deletes, foreign key constraints. |
| 2 | **Document Ingestion** | `tests/test_ingestion.py` | 8 | PDF text extraction via PyMuPDF, file size validation, text normalization, JD model persistence. |
| 3 | **Local RAG Pipeline** | `tests/test_rag.py` | 11 | Recursive character chunking, boundary preservation, MiniLM CPU embeddings, ChromaDB indexing and user isolation. |
| 4 | **LLM Integration & AFC** | `tests/test_llm.py` | 11 | Gemini structured outputs, Pydantic parsing, AFC deprecation prevention, prompt formatting. |
| 5 | **Question Generation** | `tests/test_questions.py` | 14 | Grounded question generation, cross-turn duplicate avoidance, track/difficulty matching, fallback generation. |
| 6 | **Answer Evaluation** | `tests/test_evaluation.py` | 17 | 5-dimension rubrics, deterministic scoring formula, strengths/growth area extraction, edge case inputs. |
| 7 | **Workflow Orchestration** | `tests/test_workflow.py` | 16 | LangGraph state transitions, state reconstruction from SQLite, multi-turn state persistence. |
| 8 | **Adaptive Difficulty** | `tests/test_adaptive.py` | 18 | Deterministic difficulty state machine, boundary clamping (`easy`/`hard`), technical track guardrails. |
| 9 | **Reporting & Analytics** | `tests/test_reporting.py` | 24 | Performance metrics, qualitative coaching synthesis, idempotency of report generation, report persistence. |
| 10 | **Candidate Name & Readiness** | `tests/test_candidate_name_and_readiness.py` | 14 | Name extraction from CV headers, deterministic readiness classification (`strong_readiness`, `developing`, `not_yet_ready`), weakness ranking. |
| 11 | **Local Speech-to-Text** | `tests/test_speech.py` | 21 | `faster-whisper` CPU int8 transcription, PyAV decoding, ephemeral temp file deletion in `finally:`, zero audio persistence. |
| 12 | **End-to-End Integration** | `tests/test_integration.py` | 16 | Full live interview flows via FastAPI `TestClient`, session lifecycle, rate limiting handling. |
| 13 | **Frontend & UI Logic** | `tests/test_frontend.py` | 29 | Streamlit state synchronization, voice transcript binding, page transitions, error handling. |
| **Total** | | | **198** | **0 Failures / 0 Errors / 0 Skipped** |

---

## 🚀 Running the Automated Tests

All tests run via Python's standard `unittest` framework within the project virtual environment.

### Run All 198 Tests
```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
```

### Run a Specific Test Suite
```powershell
# Run only Speech-to-Text tests
.venv\Scripts\python.exe -m unittest tests/test_speech.py

# Run Candidate Name Extraction & Readiness tests
.venv\Scripts\python.exe -m unittest tests/test_candidate_name_and_readiness.py

# Run LangGraph Workflow tests
.venv\Scripts\python.exe -m unittest tests/test_workflow.py
```

### Run with Verbose Output
```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
```

---

## 🧩 Mocking Architecture & Isolation

To ensure that the test suite runs quickly, deterministically, and with **zero cloud costs**, external network dependencies are carefully isolated:

1. **Gemini API Isolation**:
   - Google Gemini API calls are mocked using `unittest.mock.patch` targeting `google.genai.Client`.
   - Mock responses return structured Pydantic models matching real schema outputs (`QuestionGenerationResult`, `AnswerEvaluationResult`, `CoachingReportResult`).
   - Ensures CI/CD pipelines can run without an internet connection or API quota consumption.

2. **ChromaDB Vector Store**:
   - Tests use isolated temporary ChromaDB persistent clients or mocked collections to prevent test pollution of local `./data/vector_store` files.

3. **In-Memory SQLite**:
   - Database tests utilize transient SQLite instances (`sqlite:///:memory:`) or isolated test databases that teardown after each test case.

4. **Speech Audio Mocking**:
   - `test_speech.py` tests local transcription using synthetic silent WAV buffers and mocked `faster-whisper` segment iterators, validating audio decoding without requiring microphone hardware.

---

## 🛡️ Critical Regressions Prevented by Test Suite

| Regression / Bug | Root Cause | Preventive Test Suite & Fix |
|---|---|---|
| **Streamlit Voice Transcript Desynchronization** | Voice transcript was returned from backend but not injected into Streamlit's `session_state` before rendering the `text_area`. | `test_frontend.py` verifies bidirectional state synchronization between audio recorder and transcript review widgets. |
| **Gemini 502 via AFC Deprecation** | Direct structured generation in certain endpoints inadvertently triggered Gemini's Automatic Function Calling deprecation. | `test_llm.py` and `test_questions.py` verify structured JSON generation using schema enforcement without function calling tools. |
| **Cross-Turn Duplicate Questions** | Multi-turn interviews sometimes asked repeated or semantically overlapping questions. | `test_questions.py` verifies negative prompting and duplicate rejection filters across historical turns. |
| **Non-Deterministic Difficulty Swings** | Prompt-driven difficulty adjustments produced inconsistent difficulty jumps. | `test_adaptive.py` guarantees 100% deterministic mathematical boundary enforcement in Python. |
| **Audio File Leakage** | Audio recordings leaving leftover files on the local filesystem. | `test_speech.py` verifies that temporary files are deleted in `try...finally` even if decoding raises an exception. |
