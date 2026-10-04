# REST API Specification

The **AI-Powered Interview Coach** provides a RESTful API powered by FastAPI. All requests and responses communicate via JSON, with the exception of multipart file uploads for resumes (`application/pdf`) and speech audio (`audio/*`).

---

## 🌐 Base URL & Conventions

- **Base URL**: `http://localhost:8000`
- **Interactive Swagger Docs**: `http://localhost:8000/docs`
- **ReDoc Documentation**: `http://localhost:8000/redoc`
- **Content-Type**: `application/json` (unless uploading files via `multipart/form-data`)

### Common HTTP Status Codes

| Code | Status | Description |
|---|---|---|
| `200` | OK | Request succeeded. Returns requested entity or result. |
| `201` | Created | Resource successfully created (User, Resume, Job Description). |
| `400` | Bad Request | Missing parameter, unsupported file format, or validation failure. |
| `404` | Not Found | Target resource ID does not exist. |
| `422` | Unprocessable Entity | Pydantic payload validation error. |
| `429` | Too Many Requests | Upstream AI rate limit exceeded (includes exponential retry headers). |
| `500` | Internal Server Error | Unexpected server execution exception. |
| `502` | Bad Gateway | Upstream Gemini API unreachable or returned malformed content. |

---

## 📑 API Endpoint Index

1. **System & Health**
   - `GET /` - Root status
   - `GET /health` - Application health check
2. **User Management**
   - `POST /users/demo` - Initialize or retrieve default local demo candidate
   - `GET /users/demo` - Fetch default local demo candidate profile
3. **Resume Management**
   - `POST /resumes/upload` - Upload PDF, extract candidate name & text preview
   - `GET /resumes/{resume_id}` - Retrieve resume metadata and preview
4. **Job Description Management**
   - `POST /job-descriptions` - Persist role and requirements
   - `GET /job-descriptions/{job_description_id}` - Retrieve job description details
5. **RAG Vector Search & Indexing**
   - `POST /rag/index/resume/{resume_id}` - Chunk and index resume into ChromaDB
   - `POST /rag/index/job-description/{job_description_id}` - Chunk and index JD into ChromaDB
   - `POST /rag/search` - Execute similarity search across indexed documents
6. **Speech-to-Text Transcription**
   - `POST /speech/transcribe` - Local CPU speech transcription via `faster-whisper`
7. **Interview Workflow & Orchestration**
   - `POST /interview/questions/generate` - Standalone grounded question generation
   - `POST /interview/answers/evaluate` - Standalone grounded answer evaluation
   - `POST /interview/sessions/start` - Initialize LangGraph session and get Question 1
   - `POST /interview/sessions/{session_id}/answer` - Submit answer, evaluate, and get Next Question
   - `POST /interview/sessions/{session_id}/report` - Generate final qualitative coaching report
   - `GET /interview/sessions/{session_id}/report` - Fetch persisted final report (idempotent, $0 LLM)
   - `GET /interview/sessions/{session_id}` - Retrieve session state and turn history
   - `GET /interview/sessions` - List candidate session history

---

## 1. System & Health

### `GET /`
Returns root service health and running status.

**Response `200 OK`**:
```json
{
  "message": "AI-Powered Interview Coach API",
  "status": "running"
}
```

---

### `GET /health`
Liveness probe verifying that the FastAPI server is active.

**Response `200 OK`**:
```json
{
  "status": "healthy"
}
```

---

## 2. User Management

### `POST /users/demo`
Retrieves or creates the local demo candidate account (`demo_candidate@interviewcoach.local`). Used by the Streamlit frontend for zero-configuration onboarding.

**Response `200 OK`**:
```json
{
  "id": 1,
  "name": "Demo Candidate",
  "email": "demo_candidate@interviewcoach.local"
}
```

---

## 3. Resume Management

### `POST /resumes/upload`
Uploads a candidate PDF resume (up to 5 MB). Extracts text using PyMuPDF (`fitz`), automatically detects the candidate's real name via deterministic multi-tier heuristic extraction, and saves metadata to SQLite.

**Request**: `multipart/form-data`
- `user_id` (int, form parameter): User identifier.
- `file` (binary, file parameter): PDF resume file.

**Response `201 Created`**:
```json
{
  "id": 1,
  "user_id": 1,
  "filename": "Zain_Ahmed_Resume.pdf",
  "file_size": 104857,
  "candidate_name": "Zain Ahmed",
  "text_preview": "Zain Ahmed\nAI / Machine Learning Engineer\nExperienced in Python, PyTorch...",
  "created_at": "2026-10-03T18:00:00Z"
}
```

---

### `GET /resumes/{resume_id}`
Retrieves parsed resume metadata without exposing internal filesystem paths.

**Response `200 OK`**:
```json
{
  "id": 1,
  "user_id": 1,
  "filename": "Zain_Ahmed_Resume.pdf",
  "file_size": 104857,
  "candidate_name": "Zain Ahmed",
  "text_preview": "Zain Ahmed\nAI / Machine Learning Engineer...",
  "created_at": "2026-10-03T18:00:00Z"
}
```

---

## 4. Job Description Management

### `POST /job-descriptions`
Persists a target job description, role title, and optional company name.

**Request Payload `application/json`**:
```json
{
  "user_id": 1,
  "title": "Senior Machine Learning Engineer",
  "company": "Tech Corp",
  "raw_text": "We are seeking a Senior ML Engineer with deep experience in PyTorch, LLM fine-tuning, and RAG pipelines..."
}
```

**Response `201 Created`**:
```json
{
  "id": 1,
  "user_id": 1,
  "title": "Senior Machine Learning Engineer",
  "company": "Tech Corp",
  "text_preview": "We are seeking a Senior ML Engineer with deep experience in PyTorch...",
  "created_at": "2026-10-03T18:02:00Z"
}
```

---

## 5. RAG Vector Search & Indexing

### `POST /rag/index/resume/{resume_id}`
Splits resume text into overlapping character chunks (`chunk_size=800`, `chunk_overlap=150`), computes local 384-dimensional embeddings via `all-MiniLM-L6-v2`, and stores them in persistent ChromaDB with user/document metadata.

**Response `200 OK`**:
```json
{
  "document_id": 1,
  "document_type": "resume",
  "chunks_indexed": 6,
  "collection_name": "interview_documents"
}
```

---

### `POST /rag/index/job-description/{job_description_id}`
Chunks and indexes the specified job description into ChromaDB.

**Response `200 OK`**:
```json
{
  "document_id": 1,
  "document_type": "job_description",
  "chunks_indexed": 4,
  "collection_name": "interview_documents"
}
```

---

### `POST /rag/search`
Performs cosine semantic similarity search over indexed chunks with strict user isolation.

**Request Payload**:
```json
{
  "query": "experience with vector databases and RAG",
  "user_id": 1,
  "resume_id": 1,
  "job_description_id": 1,
  "top_k": 4
}
```

**Response `200 OK`**:
```json
{
  "query": "experience with vector databases and RAG",
  "results": [
    {
      "chunk_id": "resume_1_chunk_2",
      "document_type": "resume",
      "text": "Engineered production RAG pipelines using ChromaDB and sentence-transformers...",
      "score": 0.884,
      "metadata": {
        "user_id": 1,
        "resume_id": 1,
        "chunk_index": 2
      }
    }
  ]
}
```

---

## 6. Speech-to-Text Transcription

### `POST /speech/transcribe`
Accepts recorded audio in any common container format (`wav`, `mp3`, `webm`, `m4a`, `ogg`). Runs local `faster-whisper` (`base.en`, int8 CPU) inference. Raw audio is streamed directly, saved only to an ephemeral temp file, and immediately deleted upon completion.

**Request**: `multipart/form-data`
- `file` (binary, file parameter): Spoken audio recording.

**Response `200 OK`**:
```json
{
  "transcript": "In my previous role, I designed a RAG pipeline utilizing ChromaDB and sentence transformers to ground LLM responses with sub-100 millisecond latency.",
  "duration_seconds": 9.42,
  "language": "en",
  "model": "base.en"
}
```

---

## 7. Interview Workflow & Orchestration

### `POST /interview/sessions/start`
Initializes an interview session, verifies tenant isolation, queries ChromaDB for candidate and JD context, invokes Gemini with structured JSON output, and returns the first question.

**Request Payload**:
```json
{
  "user_id": 1,
  "interview_type": "ai_ml",
  "difficulty": "medium",
  "total_questions": 5,
  "resume_id": 1,
  "job_description_id": 1
}
```
*Allowed `interview_type` values*: `hr`, `python`, `ai_ml`, `data_science`, `internship`  
*Allowed `difficulty` values*: `easy`, `medium`, `hard`

**Response `200 OK`**:
```json
{
  "session_id": 12,
  "status": "in_progress",
  "candidate_name": "Zain Ahmed",
  "current_turn": 1,
  "total_questions": 5,
  "question": {
    "question_id": 45,
    "question_order": 1,
    "question_text": "Could you describe your architectural approach to indexing and chunking technical documents in a RAG pipeline?",
    "difficulty": "medium",
    "topic": "RAG Architecture",
    "competency": "system_design"
  }
}
```

---

### `POST /interview/sessions/{session_id}/answer`
Submits a candidate's answer for evaluation. Triggers LangGraph turn execution:
1. Validates answer quality and enforces duplicate answer prevention.
2. Retrieves relevant RAG context chunks.
3. Invokes Gemini for 5-dimension qualitative scoring.
4. Calculates deterministic weighted overall score in Python.
5. Evaluates deterministic adaptive difficulty policy.
6. Either generates Question $k+1$ OR transitions session to `completed`.

**Request Payload**:
```json
{
  "user_id": 1,
  "question_id": 45,
  "answer_text": "I typically use an 800-character chunk window with a 150-character overlap, splitting on paragraph and sentence boundaries to preserve semantic context before embedding with MiniLM.",
  "answer_method": "text"
}
```
*Allowed `answer_method` values*: `text`, `voice`

**Response `200 OK` (Turn Complete, Next Question Ready)**:
```json
{
  "session_id": 12,
  "status": "in_progress",
  "completed_turn": 1,
  "total_questions": 5,
  "evaluation": {
    "question_id": 45,
    "overall_score": 8.4,
    "dimension_scores": {
      "relevance": 9.0,
      "clarity": 8.5,
      "completeness": 8.0,
      "technical_correctness": 8.5,
      "structure": 8.0
    },
    "feedback": "Strong answer clearly explaining chunking windows and boundary delimiters.",
    "strengths": [
      "Explicitly specified chunk size and overlap parameters.",
      "Highlighted delimiter boundaries to preserve context."
    ],
    "growth_areas": [
      "Could mention handling metadata filtering alongside chunk storage."
    ]
  },
  "adaptive_transition": {
    "previous_difficulty": "medium",
    "next_difficulty": "hard",
    "action": "increased",
    "reason": "High technical accuracy and completeness met advancement thresholds."
  },
  "next_question": {
    "question_id": 46,
    "question_order": 2,
    "question_text": "How would you handle vector re-ranking and cosine similarity thresholding in low-latency environments?",
    "difficulty": "hard",
    "topic": "Vector Search Optimization",
    "competency": "advanced_systems"
  }
}
```

---

### `POST /interview/sessions/{session_id}/report`
Generates a comprehensive final coaching report synthesizing turn-by-turn evaluations into overall metrics, qualitative feedback, and deterministic readiness assessment. **Idempotent**: Calling this endpoint on an already generated report returns the existing database record without consuming LLM tokens.

**Request Payload**:
```json
{
  "user_id": 1
}
```
*(Also accepts `?user_id=1` as query parameter).*

**Response `200 OK`**:
```json
{
  "session_id": 12,
  "candidate_name": "Zain Ahmed",
  "interview_type": "ai_ml",
  "total_questions": 5,
  "overall_score": 8.2,
  "readiness_key": "strong_readiness",
  "readiness_label": "Strong Interview Readiness",
  "dimension_breakdown": {
    "relevance": 8.8,
    "clarity": 8.2,
    "completeness": 7.8,
    "technical_correctness": 8.4,
    "structure": 7.8
  },
  "summary": "Demonstrated comprehensive technical depth in machine learning system design and production RAG implementations.",
  "key_strengths": [
    "Precise understanding of embedding dimensions and retrieval mechanics.",
    "Clear, structured explanations of real-world tradeoffs."
  ],
  "areas_for_improvement": [
    "Provide deeper quantitative benchmarks when describing production systems."
  ],
  "recommended_focus": [
    "Review cross-encoder re-ranking latency implications."
  ],
  "question_evaluations": [
    {
      "question_order": 1,
      "question_text": "Could you describe your architectural approach...",
      "difficulty": "medium",
      "overall_score": 8.4,
      "feedback": "Strong answer clearly explaining chunking windows..."
    }
  ],
  "generated_at": "2026-10-03T18:15:00Z"
}
```

---

### `GET /interview/sessions/{session_id}/report`
Retrieves a previously generated final interview report from SQLite with **zero LLM cost**.

**Parameters**:
- `session_id` (path, int): Session ID.
- `user_id` (query, int): User ID for strict authorization check.

**Response `200 OK`**: Same schema as `POST .../report`.

---

### `GET /interview/sessions/{session_id}`
Fetches active session progress, current pending question (if session is in progress), and all submitted turns with evaluations.

**Parameters**:
- `session_id` (path, int): Session ID.
- `user_id` (query, int): User ID.

**Response `200 OK`**:
```json
{
  "session_id": 12,
  "status": "in_progress",
  "interview_type": "ai_ml",
  "difficulty": "hard",
  "candidate_name": "Zain Ahmed",
  "total_questions": 5,
  "completed_questions": 2,
  "started_at": "2026-10-03T18:05:00Z",
  "completed_at": null,
  "current_question": {
    "question_id": 47,
    "question_order": 3,
    "question_text": "...",
    "difficulty": "hard"
  },
  "turns": [
    {
      "turn_order": 1,
      "question_text": "...",
      "answer_text": "...",
      "answer_method": "voice",
      "overall_score": 8.4
    }
  ]
}
```

---

### `GET /interview/sessions`
Lists chronological session history for the specified candidate.

**Parameters**:
- `user_id` (query, int): User ID.

**Response `200 OK`**:
```json
[
  {
    "session_id": 12,
    "interview_type": "ai_ml",
    "initial_difficulty": "medium",
    "status": "completed",
    "candidate_name": "Zain Ahmed",
    "readiness_label": "Strong Interview Readiness",
    "total_questions": 5,
    "completed_questions": 5,
    "overall_score": 8.2,
    "started_at": "2026-10-03T18:05:00Z",
    "completed_at": "2026-10-03T18:15:00Z",
    "report_available": true
  }
]
```
