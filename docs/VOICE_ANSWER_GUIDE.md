# Voice Answer & Local Speech-to-Text Developer Guide

## 1. Overview

Phase 12.2 adds optional voice-answer capability to the AI-Powered Interview Coach without introducing any third-party paid speech APIs ($0.00 cost architecture preserved). Candidates can seamlessly alternate between typing and recording spoken answers for any question in an interview session.

```
Interview Question
       ↓
Record Voice Answer (Native st.audio_input)
       ↓
Local Speech-to-Text (faster-whisper base.en int8 on CPU)
       ↓
Editable Transcript (Review & Terminology Correction)
       ↓
Submit Answer (answer_method="voice")
       ↓
Existing Evaluation & Scoring Pipeline (Unchanged)
```

## 2. Architecture & Design Principles

1. **Local & Free ($0.00 Cost):**
   - Uses `faster-whisper` (`base.en` model) with CPU-optimized `int8` quantization.
   - Zero paid cloud speech APIs (no Google Speech, Whisper API, or Azure Speech).
   - Zero extra Gemini calls used for transcription.

2. **Candidate Review & Terminology Correction (Never Auto-Submit):**
   - Transcribed text is placed directly into an editable text area (`st.text_area`).
   - Candidates can review, edit, fix technical jargon or package names (e.g., PyTorch, LangGraph, scikit-learn), or re-record before submitting.
   - Submission is triggered ONLY when the candidate explicitly clicks the submit button.

3. **Strict Candidate Privacy:**
   - Audio recordings are processed in a temporary file and deleted immediately in a `finally:` block.
   - Raw audio bytes are NEVER permanently stored in the database or filesystem.
   - Only the final (reviewed/edited) text is persisted in SQLite.

4. **Bi-modal Parity & Metadata Tracking:**
   - Every answer records `answer_method = "text" | "voice"`.
   - The downstream evaluation engine evaluates the content identically regardless of input method.
   - Evaluation strictly judges semantic/technical quality—no voice accent, audio tone, or personal attribute profiling.

5. **First-Run Behavior & Offline Fallback:**
   - On first execution, `faster-whisper` downloads the `base.en` weights (~140MB) once to the local HuggingFace cache and caches the instantiated model in memory.
   - Subsequent calls execute in ~2-4 seconds on standard multi-core CPUs.
   - If audio recording or microphone access is unavailable, candidates can switch back to `[⌨️ Type Answer]` at any time with a single click.

## 3. Endpoints & Schemas

### `POST /speech/transcribe`
- **Request:** `multipart/form-data` with `file: UploadFile` (WAV, MP3, M4A, OGG, WEBM, FLAC, AAC, max 25 MB).
- **Response:**
  ```json
  {
    "transcript": "In Python, list comprehension provides a concise way to create lists...",
    "language": "en",
    "duration_seconds": 6.81
  }
  ```

### `POST /interview/sessions/{session_id}/answer`
- **Request:**
  ```json
  {
    "user_id": 1,
    "answer": "In Python, list comprehension provides a concise way to create lists...",
    "answer_method": "voice"
  }
  ```
