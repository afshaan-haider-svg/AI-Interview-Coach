# Portfolio Content: CV Entries, LinkedIn Presentation & Recruiter Summaries

This document provides ready-to-use project descriptions, bullet points, LinkedIn posts, and recruiter-focused presentation materials for the **AI-Powered Interview Coach**.

---

## 1. CV Project Entry (Standard 3–4 Bullet Points)

**AI-Powered Interview Coach** | Python, FastAPI, Streamlit, LangGraph, Gemini Flash Lite, ChromaDB, Whisper  
*AI / Generative AI Engineer Portfolio Project*

- Engineered an end-to-end adaptive technical interview coaching system using **LangGraph** for multi-turn conversational workflow orchestration, persisting session state and question turns in **SQLite** for fault-tolerant recovery.
- Built a document-grounded **local RAG pipeline** using PyMuPDF for automated candidate name extraction, `sentence-transformers/all-MiniLM-L6-v2` for dense embeddings, and **ChromaDB** with HNSW cosine search for personalized question grounding.
- Designed an objective **5-dimension evaluation engine** (*Relevance, Technical Correctness, Completeness, Clarity, Structure*) combining Gemini structured generation with **deterministic Python weighted formulas** and adaptive difficulty state machines.
- Implemented an on-device speech-to-text pipeline using **`faster-whisper` (int8 CPU)**, providing sub-3s local audio transcription with zero audio cloud persistence and an editable transcript buffer to prevent technical terminology misrecognition.
- Authored **198 automated unit and integration tests** across 13 test suites with zero failures, verified through comprehensive end-to-end acceptance testing.

---

## 2. One-Line CV Project Version (Space-Constrained)

**AI-Powered Interview Coach** | *Python, FastAPI, Streamlit, LangGraph, ChromaDB, Gemini, Whisper*  
- Developed an adaptive technical interview platform featuring local RAG document grounding, on-device Whisper speech-to-text, LangGraph multi-turn state orchestration, and deterministic 5-dimension scoring verified by 198 automated tests.

---

## 3. Recruiter-Friendly Project Summary (80–120 Words)

> **The AI-Powered Interview Coach is a full-stack, local-first generative AI application designed to practice technical interviews with authentic candidate personalization. Built with FastAPI and Streamlit, it combines local RAG (MiniLM + ChromaDB) to ground questions in the candidate's actual resume, LangGraph for resilient multi-turn state machine management, and Google Gemini for structured semantic feedback. To ensure reliability and privacy, all speech transcription runs locally via `faster-whisper` (CPU int8), and all scoring formulas and readiness tiers are computed deterministically in Python rather than hallucinated. The project is supported by comprehensive documentation, Docker packaging, and 198 passing automated tests.**

---

## 4. LinkedIn Project Section Entry

### Project Title
**AI-Powered Interview Coach — RAG, LangGraph & Voice AI**

### Association / Role
Personal Portfolio Project | AI / Machine Learning Engineer

### Skills to Associate
- Generative AI
- Retrieval-Augmented Generation (RAG)
- Large Language Models (LLMs)
- LangGraph
- FastAPI
- Streamlit
- Vector Databases (ChromaDB)
- Natural Language Processing (NLP)
- Speech Recognition (faster-whisper)
- Python

### Project Description
Developed an end-to-end personalized technical interview coach that pairs local RAG document grounding with stateful LangGraph orchestration and on-device speech-to-text.

**Key Engineering Highlights:**
- **Local RAG Document Grounding**: Extracts candidate names from PDF resumes and indexes technical background into ChromaDB via `all-MiniLM-L6-v2` dense embeddings ($0 cost).
- **Deterministic Evaluation & Adaptive Difficulty**: Separates qualitative language modeling from scoring logic; overall scores and adaptive difficulty shifts are calculated via deterministic Python formulas rather than arbitrary prompt grading.
- **On-Device Voice Answers**: Local `faster-whisper` (CPU int8) transcribes answers in seconds with zero audio cloud storage, coupled with an in-browser editable transcript review buffer.
- **Resilient Multi-Turn State Machine**: LangGraph state graph persists turn history, active questions, and evaluations in SQLite, allowing instant session recovery across server restarts.
- **Production QA Baseline**: 198 automated test cases across 13 suites, backed by a complete end-to-end acceptance run.

### Recommended Media to Attach
1. `docs/assets/screenshots/01-home.png` (Landing page)
2. `docs/assets/screenshots/06-answer-evaluation.png` (Turn evaluation view)
3. `docs/assets/screenshots/07-final-report.png` (Candidate coaching report)
4. Link to GitHub repository: `[GitHub Repository]`

---

## 5. LinkedIn Launch Post (Polished Long Version)

Most AI interview tools suffer from the same problem: they rely on generic prompts, hallucinate arbitrary scores without an objective rubric, and upload your confidential CV and voice data to external cloud APIs.

To tackle this, I built the **AI-Powered Interview Coach** — a local-first, privacy-conscious technical interview preparation platform.

Here is the technical architecture behind it:

🔹 **Local RAG Document Grounding**: Parses candidate resumes (PDF) and target job descriptions, automatically extracting candidate names and indexing dense embeddings into **ChromaDB** using `all-MiniLM-L6-v2` on CPU. Questions probe authentic project experience rather than generic templates.  
🔹 **LangGraph State Orchestration**: Implements an explicit state graph for multi-turn interview flow. Active questions, evaluations, and turn state persist directly into SQLite, making sessions resilient to server restarts.  
🔹 **Deterministic 5-Dimension Scoring**: Uses Google Gemini 2.5 Flash Lite for structured semantic feedback, but delegates all mathematical scoring to deterministic Python formulas across Relevance, Technical Correctness, Completeness, Clarity, and Structure.  
🔹 **On-Device Speech-to-Text**: Candidates can speak their answers. An on-device **`faster-whisper` (int8 CPU)** engine decodes audio locally in ~2 seconds with zero audio cloud persistence. Candidates can review and edit technical terminology in their transcript before submitting.  
🔹 **Adaptive Difficulty & Readiness Rubric**: Dynamically modulates challenge between Easy, Medium, and Hard using strict technical performance guardrails, concluding with an objective readiness assessment.  

💡 **An Interesting Engineering Challenge:**  
One key challenge was speech recognition of specialized developer terms (e.g. LangGraph, ChromaDB, FastAPI). Rather than passing imperfect speech straight to the LLM, I designed an editable transcript review buffer right inside Streamlit, allowing candidates to fix technical terms before evaluation while keeping inference 100% offline.

🧪 **Verification & Reliability:**  
The platform is backed by **198 automated unit and integration tests** across 13 suites, and fully verified with real end-to-end acceptance tests.

Check out the code, architecture diagrams, and test suite on GitHub:  
👉 [GitHub Repository]  
🎥 Demo Video Walkthrough: [Demo Video]  

I'd love to hear feedback from AI engineers, hiring managers, and developers preparing for technical interviews!

#GenerativeAI #RAG #MachineLearning #LangGraph #FastAPI #Streamlit #Python #SpeechToText #AIEngineering

---

## 6. Short LinkedIn Post (Compact Alternative)

Excited to share my latest portfolio project: the **AI-Powered Interview Coach**! 🎙️🤖

I wanted to build an interview platform that avoids generic prompt templates, hallucinated scoring, and cloud privacy issues.

Key features:
• **Local RAG Grounding**: Questions tailored to candidate CVs and target JDs using ChromaDB and MiniLM.  
• **Stateful LangGraph Workflow**: Multi-turn orchestration with durable SQLite turn recovery.  
• **Deterministic 5-Dimension Scoring**: Objective Python mathematical formulas instead of arbitrary LLM grades.  
• **Local Whisper Voice Input**: On-device speech recognition via `faster-whisper` (int8 CPU) with an editable transcript review buffer.  
• **Automated Testing**: 198 unit & integration tests passing across 13 test suites.  

Explore the open repository, architecture docs, and setup guide:  
🔗 GitHub: [GitHub Repository]  
📹 Demo: [Demo Video]  

#ArtificialIntelligence #GenerativeAI #Python #FastAPI #LangGraph #MachineLearning

---

## 7. GitHub Pinned-Repository Presentation

| Element | Recommended Content |
|---|---|
| **Repository Name** | `AI-Interview-Coach` |
| **One-Line Description** | Privacy-first AI interview coach with local RAG, LangGraph workflow, deterministic scoring, on-device Whisper STT, FastAPI, and Streamlit. |
| **Topics** | `python`, `generative-ai`, `rag`, `langgraph`, `fastapi`, `streamlit`, `chromadb`, `speech-to-text` |
| **Featured Screenshot** | `docs/assets/screenshots/01-home.png` or `docs/assets/screenshots/07-final-report.png` |
| **Demo Link** | `[Demo Video]` placeholder in README header |
