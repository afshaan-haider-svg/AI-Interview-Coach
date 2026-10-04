# Portfolio Demo Video Plan & Narration Script

**Project**: AI-Powered Interview Coach  
**Target Duration**: 3 minutes (approx. 2.5 to 3.5 minutes)  
**Target Audience**: Technical Recruiters, Engineering Hiring Managers, AI / ML Interviewers  
**Resolution**: 1080p (1920×1080) at 60fps or 30fps  
**Audio**: Crisp microphone recording with normalized voice levels  

---

## 1. Timestamped Storyboard

| Timestamp | Duration | Screen Action | Visual Focus | Core Message |
|---|:---:|---|---|---|
| **0:00 – 0:15** | 15s | Browser on Landing Page (`01-home.png`) | Hero banner, key capability cards | Hook: The challenge of generic, hallucinated AI coaching. |
| **0:15 – 0:35** | 20s | Split screen or quick pan to Architecture diagram | Mermaid architecture diagram | How local RAG, LangGraph, and deterministic scoring solve this. |
| **0:35 – 0:55** | 20s | Setup Interview view (`02-setup-resume.png`) | Upload synthetic resume (`Zain Ahmed`) and target JD | Name extraction and local embedding generation into ChromaDB. |
| **0:55 – 1:20** | 25s | Active Interview view (`03-interview-question.png`) | Question 1 card, difficulty and category badges | Question probing authentic RAG experience rather than a template. |
| **1:20 – 1:45** | 25s | Submit typed answer $\rightarrow$ Turn feedback (`06-answer-evaluation.png`) | 5-dimension score breakdown, progress bars | Deterministic weighted formula and qualitative coaching feedback. |
| **1:45 – 2:15** | 30s | Voice recording mode (`04-voice-recording.png` & `05-editable-transcript.png`) | Speak audio, click "Transcribe Answer", edit technical term | On-device `faster-whisper` transcription with editable review buffer. |
| **2:15 – 2:40** | 25s | Turn 2 evaluation and Adaptive Shift banner | Adaptive banner: Medium $\rightarrow$ Easy / Maintain | Policy guardrails preventing arbitrary difficulty spikes. |
| **2:40 – 3:10** | 30s | Final Report view (`07-final-report.png`) | "Zain Ahmed" report header, readiness badge, analytics | Deterministic arithmetic mean and actionable coaching roadmap. |
| **3:10 – 3:30** | 20s | Terminal showing test run + GitHub repository | 198 tests passing, clean modular structure | Reliability, rigorous testing, and local-first architecture. |

---

## 2. Natural English Narration Script

> **Tone Guideline**: Professional, thoughtful, engineering-focused. Avoid exaggerated claims like "revolutionary", "enterprise-grade", or "100% accurate". Speak at a calm, natural pace.

---

### [0:00 – 0:15] Introduction & The Problem
*(Screen: Landing page at `http://localhost:8501`)*

> *"Hi everyone. Most AI interview preparation tools have a major flaw: they rely on generic prompts, hallucinate arbitrary scores without a rubric, and send your confidential resume and voice data across third-party cloud APIs. I built the AI-Powered Interview Coach to solve this: an end-to-end, privacy-conscious interview practice platform built with local RAG, LangGraph state orchestration, on-device speech recognition, and deterministic scoring policies."*

---

### [0:15 – 0:35] Architecture Overview
*(Screen: System Architecture diagram in documentation or split view)*

> *"The system couples a FastAPI backend with a reactive Streamlit interface. Document embeddings and vector search run entirely locally using sentence-transformers and ChromaDB. Workflow state transitions are managed as a formal LangGraph state machine, while Google Gemini 2.5 Flash Lite is used strictly for structured semantic reasoning within bounded API quotas."*

---

### [0:35 – 0:55] Document Ingestion & Candidate Personalization
*(Screen: Setup Interview page, uploading `zain_ahmed_resume.pdf`)*

> *"Let's set up an interview. I'll upload a synthetic resume for an AI and Machine Learning Engineer named Zain Ahmed, along with a target job description. The system automatically extracts the candidate's name from the document header, splits the text into 800-character chunks with a 150-character overlap, and indexes the embeddings locally into ChromaDB."*

---

### [0:55 – 1:20] RAG-Grounded Question Generation
*(Screen: Active Interview page, Question 1 displayed)*

> *"When I launch the session, LangGraph initiates Question 1. Notice that rather than asking a generic textbook question, it retrieves relevant context from both the candidate's CV and the job description to craft a targeted scenario on semantic search optimization and imbalanced classification."*

---

### [1:20 – 1:45] Typed Answer & 5-Dimension Evaluation
*(Screen: Candidate submits typed answer, evaluation card appears)*

> *"After submitting a response, the answer is evaluated across five explicit dimensions: Relevance, Technical Correctness, Completeness, Clarity, and Structure. Crucially, the overall score isn't an arbitrary number hallucinated by the model; it is calculated deterministically in Python using a calibrated weighted formula. Alongside the scores, the candidate receives actionable strengths, missing topics, and an exemplary model response."*

---

### [1:45 – 2:15] On-Device Speech-to-Text & Editable Transcript
*(Screen: Switch to Voice Input radio, record short audio, click Transcribe)*

> *"For question two, let's practice speaking. The candidate records their answer directly in the browser. When they click 'Transcribe Answer', local faster-whisper with int8 CPU quantization decodes the audio offline in about two seconds with zero audio sent over the internet. Because speech models can occasionally mishear specialized technical acronyms like LangGraph or ChromaDB, the candidate can review and edit their transcript directly before submitting it for evaluation."*

---

### [2:15 – 2:40] Deterministic Adaptive Difficulty
*(Screen: Turn 2 evaluation and adaptive banner)*

> *"Notice the adaptive difficulty banner. If a candidate excels, the difficulty dynamically escalates; if foundational gaps are detected, it eases up or maintains the challenge with strict technical guardrails to ensure realistic practice."*

---

### [2:40 – 3:10] Final Performance Report & Readiness Assessment
*(Screen: Final Coaching Report page)*

> *"At the end of the session, the coach generates a comprehensive personalized report for Zain Ahmed. The overall score is the exact arithmetic mean across all completed questions. Based on this, a deterministic readiness assessment classifies the candidate's preparation as 'Developing' or 'Not Yet Ready' with weakness-prioritized study recommendations."*

---

### [3:10 – 3:30] Persistence, Testing & Wrap-Up
*(Screen: Session history table and terminal running tests)*

> *"All completed sessions, questions, and reports persist durably in SQLite for historical review. The entire codebase is backed by 198 automated unit and integration tests across 13 suites, and verified with real end-to-end acceptance runs. The project is completely open and documented on GitHub with full setup guides and Docker packaging. Thanks for watching!"*

---

## 3. Recording Checklist

### Before Recording
- [ ] Set browser window to 1080p (1920×1080) at 100% zoom.
- [ ] Use a clean browser profile with bookmarks toolbar hidden.
- [ ] Start backend (`uvicorn app.main:app --port 8000`) and frontend (`streamlit run frontend/app.py --server.port 8501`).
- [ ] Use synthetic test files only (`zain_ahmed_resume.pdf`).
- [ ] Test microphone input level to avoid clipping or low volume.
- [ ] Close all unrelated browser tabs, chat apps, and system notifications.
- [ ] Ensure `.env` and terminal credentials are not visible on screen.

### During Recording
- [ ] Move mouse smoothly and deliberately; avoid rapid erratic clicking.
- [ ] Allow 1–2 seconds of pause after UI state transitions before speaking next line.
- [ ] Speak clearly and maintain an engaging, professional tone.
- [ ] Showcase both typed input and speech recording modes.
- [ ] Highlight the editable transcript buffer.

### After Recording
- [ ] Cut out any silent pauses or API latency waiting periods.
- [ ] Verify that no API keys or local file paths appear in video frames.
- [ ] Export in 1080p MP4 (H.264 video, AAC audio).
- [ ] Upload to YouTube (Unlisted or Public) or Google Drive / Loom.
- [ ] Add the video link to `README.md` and portfolio documents.
