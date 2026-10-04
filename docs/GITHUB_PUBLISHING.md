# GitHub Publication & Repository Packaging Guide

This guide specifies the repository metadata, visibility, description variants, and safe manual command sequence for publishing the **AI-Powered Interview Coach** to GitHub as an AI / Generative AI Engineer portfolio repository.

---

## 1. Recommended Repository Metadata

| Parameter | Recommended Value |
|---|---|
| **Repository Name** | `AI-Interview-Coach` |
| **Visibility** | **Public** (recommended for portfolio showcasing; do not change account settings automatically) |
| **Primary Language** | Python (3.11+) |
| **License Status** | **No license present** (unlicensed/all rights reserved by default; author may optionally add MIT or Apache-2.0 upon publication) |

### Recommended Topics (Tags)
```text
python, generative-ai, rag, llm, langgraph, fastapi, streamlit, chromadb, machine-learning, nlp, speech-to-text, gemini, ai-engineering, interview-preparation
```
*(Exact 14 relevant technical keywords; zero keyword spam).*

---

## 2. GitHub About Description Variants

Choose one of the following variants for the GitHub repository "About" field (under 350 characters):

### Variant A: Recommended (Balanced & Accurate) — 243 characters
> **A privacy-first AI interview coach combining LangGraph multi-turn orchestration, local RAG document grounding, deterministic scoring policies, and on-device Whisper speech-to-text with zero paid API dependencies.**

### Variant B: Short & Recruiter-Focused — 156 characters
> **Personalized AI interview coach with local RAG, LangGraph workflow, deterministic scoring, on-device Whisper voice answers, FastAPI, and Streamlit.**

### Variant C: Technical Architecture — 318 characters
> **Local-first GenAI interview coach: PyMuPDF CV name extraction, MiniLM dense embeddings, ChromaDB RAG, Gemini 2.5 Flash Lite structured evaluation, LangGraph state machine, deterministic adaptive difficulty, and faster-whisper CPU int8 transcription with an editable transcript review buffer.**

---

## 3. Safe Manual Publication Command Sequence

> [!IMPORTANT]
> Execute these commands manually in your terminal after reviewing the staged file list. Never force push and never publish without reviewing git status.

### Step 1: Initialize Git Repository (if not already initialized)
```powershell
# Navigate to repository root
cd d:\pythonn\AI-Interview-Coach

# Initialize git tracking
git init -b main
```

### Step 2: Verify .gitignore Effectiveness Before Staging
```powershell
# Inspect untracked files to verify no secrets or databases appear
git status --ignored
```
Confirm that:
- `.env` is **ignored**
- `data/*.db` and `data/interview_coach.db` are **ignored**
- `data/uploads/*` and `data/vector_store/*` are **ignored**
- `scratch/` is **ignored**
- `.venv/` is **ignored**
- `__pycache__/` and `*.pyc` are **ignored**

### Step 3: Stage Approved Files
```powershell
git add .
```

### Step 4: Pre-Commit Tracked File Verification
```powershell
# Check staged files
git status

# Verify staged file statistics
git diff --cached --stat
```
Verify that only:
- Source code (`app/`, `frontend/`)
- Test suites (`tests/`)
- Technical documentation (`docs/`, `README.md`)
- Official screenshots (`docs/assets/screenshots/`)
- Packaging configurations (`Dockerfile`, `.dockerignore`, `requirements.txt`, `start.sh`, `packages.txt`)
- Safe configurations (`.env.example`, `.gitignore`, `.gitkeep` files)
are staged for commit.

### Step 5: Create Initial Release Commit
```powershell
git commit -m "Initial portfolio release: AI-Powered Interview Coach with RAG, LangGraph & local Whisper"
```

### Step 6: Link to Remote GitHub Repository & Push
```powershell
# Check existing remotes
git remote -v

# Link to your personal GitHub repository (replace with your actual username/repo URL)
# Example: git remote add origin https://github.com/[your-username]/AI-Interview-Coach.git
git remote add origin https://github.com/[YOUR-USERNAME]/AI-Interview-Coach.git

# Push to main branch
git push -u origin main
```

---

## 4. Final Security & Pre-Push Checklist

- [x] No live Google Gemini API keys in tracked code or markdown files.
- [x] `.env` excluded by `.gitignore`.
- [x] `.env.example` contains only generic placeholder values (`your_gemini_api_key_here`).
- [x] Local SQLite databases (`data/*.db`, `data/interview_coach.db`) excluded.
- [x] Candidate CV uploads (`data/uploads/*`) excluded.
- [x] Vector store binary indexes (`data/vector_store/*`) excluded.
- [x] Ephemeral audio files (`*.wav`, `*.mp3`) excluded.
- [x] Acceptance test scripts and temporary browser profiles in `scratch/` excluded.
- [x] 8 visual screenshots verified free of personal email addresses, local absolute paths, or credentials.
- [x] 198 automated unit and integration tests passing.
