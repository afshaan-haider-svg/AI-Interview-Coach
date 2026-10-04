# Cloud Deployment Strategy & Architectural Decision Matrix

**Project:** AI-Powered Interview Coach  
**Research Date:** October 2026 (Verified against official provider documentation)  
**Status:** Hosting Eligibility Audit & Corrected Strategy (Phase 13C)

---

## 🎯 Non-Negotiable Deployment Constraints

This deployment architecture is governed by strict user criteria:
1. **$0.00 Mandatory Cost**: No initial charges, no recurring fees.
2. **Zero Payment Card Requirement**: No credit card, debit card, or billing verification hold allowed.
3. **No Paid Subscriptions**: Must run on standard free-tier infrastructure (e.g. no PRO, Team, or Enterprise subscriptions).
4. **Zero Billing Risk**: Impossible to incur surprise overdraft charges or usage overages.
5. **No Degradation of Local Functionality**: Local Windows development and complete feature set must remain fully operational.

---

## 🔍 Account Eligibility & Platform Evaluation Matrix

Every candidate provider was evaluated against two mandatory criteria:
1. **Account Eligibility**: Can a standard free personal account create and run the compute service without a payment card or paid subscription?
2. **Resource Viability**: Can the hardware run our measured stack (**785.51 MB peak backend RAM** + **~118 MB Streamlit RAM** = **~904 MB total**) without crashing?

| Hosting Provider | Genuinely $0? | Payment Card Required? | Paid Subscription Required? | Compute & RAM Available | Resource Viability (785 MB Stack) | Eligibility & Technical Verdict |
|---|:---:|:---:|:---:|:---:|:---:|---|
| **Hugging Face Spaces** | ❌ No for compute | ❌ Requires card for PRO | ❌ **PRO Plan ($9/mo) Required** | 2 vCPU / 16 GB RAM (PRO) | ✅ Hardware fits | ❌ **DISQUALIFIED**<br>Official HF documentation confirms that creating new compute-backed Spaces (Docker, Gradio, or CPU Basic) now mandates a paid PRO or Team subscription. Free accounts are restricted to static HTML/JS Spaces. |
| **Render Free Web Service** | ✅ Yes | ✅ **No Card Required** | ✅ No Subscription | 0.1 vCPU / **512 MB RAM** | ❌ **FAILS (OOM Kill)** | ❌ **DISQUALIFIED ON MEMORY**<br>FastAPI + PyTorch base import alone takes 532.89 MB. On 512 MB instances, the Linux kernel OOM-killer immediately terminates the process during boot. Free persistent disks do not exist. |
| **Streamlit Community Cloud** | ✅ Yes | ✅ **No Card Required** | ✅ No Subscription | 1 shared vCPU / **1 GB RAM** | ⚠️ **UNSTABLE / OOM RISK** | ⚠️ **UNSUITABLE FOR FULL STACK**<br>Only hosts Streamlit applications (no separate FastAPI backend service). Combined in-process execution consumes ~904 MB (88.3% of 1024 MB cap), causing unpredictable crashes during audio decoding or PDF parsing. |
| **Koyeb** | ❌ No | ❌ **Card Required** ($29 hold) | ⚠️ Starter plan | 0.1 vCPU / 512 MB RAM | ❌ FAILS (OOM) | ❌ **DISQUALIFIED**<br>Requires credit card with a $29 authorization hold during onboarding to prevent abuse. |
| **Fly.io** | ❌ No | ❌ **Card Required** | ❌ Pay-as-you-go | 256 MB RAM | ❌ FAILS (OOM) | ❌ **DISQUALIFIED**<br>Requires credit card for all new accounts; permanent free tier deprecated. |
| **Railway** | ⚠️ Trial | ⚠️ Trial limit | ❌ Paid after trial | Shared compute | ⚠️ Limited | ❌ **DISQUALIFIED**<br>New accounts receive only $1/month credit, exhausting within days. Not a viable permanent free host. |
| **PythonAnywhere** | ✅ Yes | ✅ **No Card Required** | ✅ No Subscription | 1 CPU / 512 MB RAM | ❌ FAILS | ❌ **DISQUALIFIED**<br>Free tier does not support ASGI/FastAPI, lacks WebSockets for Streamlit, and restricts outbound network access (preventing model downloads). |
| **Vercel** | ✅ Yes | ✅ **No Card Required** | ✅ No Subscription | Serverless (Lambda) | ❌ FAILS | ❌ **DISQUALIFIED**<br>15s execution timeout and 250 MB zip limit cannot run Streamlit WebSockets, PyTorch, or CTranslate2. |

---

## 🛑 Critical Engineering Finding

> [!IMPORTANT]
> **Under the strict non-negotiable constraints ($0 mandatory cost, NO credit/debit card, NO paid subscription), NO public cloud hosting provider currently exists that can host the full 785+ MB local machine-learning stack (FastAPI + Streamlit + PyTorch + sentence-transformers + ChromaDB + faster-whisper + SQLite).**
>
> 1. Providers that offer sufficient RAM (>1 GB) have gated compute creation behind paid subscriptions or credit card anti-fraud requirements (e.g., Hugging Face Spaces now requires PRO $9/mo for compute spaces; Fly.io/Koyeb mandate payment cards).
> 2. Providers that remain truly free without a card (Render, SnapDeploy) strictly cap free instances at **512 MB RAM**, which is physically insufficient to even import the PyTorch CPU runtime (532.89 MB measured baseline).

---

## 🏆 Recommended Realistic Portfolio Strategy

Rather than compromising application functionality, removing core features, or pretending that fragile 512 MB hosting can run PyTorch and Whisper, the following **three-pillar portfolio strategy** is recommended:

```
┌────────────────────────────────────────────────────────────────────────┐
│                   THREE-PILLAR PORTFOLIO STRATEGY                       │
├────────────────────────────────┬───────────────────────────────────────┤
│ 1. LOCAL FULL-FEATURE ENGINE   │ Complete, uncompromised system runs   │
│    (Authoritative Baseline)    │ locally: Voice, Whisper, MiniLM, RAG, │
│                                │ LangGraph, Gemini, SQLite, Streamlit  │
├────────────────────────────────┼───────────────────────────────────────┤
│ 2. PROFESSIONAL REPO SHOWCASE  │ Recruiter-grade documentation, deep   │
│    (Public GitHub Visibility)  │ architecture diagrams, resource logs, │
│                                │ 198 verified tests, & video demo      │
├────────────────────────────────┼───────────────────────────────────────┤
│ 3. CONTAINERIZED READINESS     │ Production Dockerfile & start.sh ready│
│    (Single-Command Deploy)     │ for any Docker host, evaluation VM,  │
│                                │ or future funded cloud environment    │
└────────────────────────────────┴───────────────────────────────────────┘
```

### Pillar 1: Local Full-Feature Engine (Zero Compromise)
- **100% of features remain active**: PDF resume upload, candidate name extraction, local MiniLM embeddings, persistent ChromaDB RAG, multi-turn LangGraph interview, local `faster-whisper` voice transcription, deterministic 5-dimension evaluation, adaptive difficulty, and personalized coaching reports.
- **198 / 198 automated tests** passing across 13 test suites.
- **Cost**: $0.00.
- **Stability**: Zero cold starts, zero cloud disconnects, zero sleeping downtime.

### Pillar 2: Recruiter-Grade Documentation & Public Media Showcase
- **Architectural Transparency**: Complete engineering documentation in [README.md](../README.md), [docs/ARCHITECTURE.md](ARCHITECTURE.md), [docs/API.md](API.md), [docs/TESTING.md](TESTING.md), and [docs/RESOURCE_PROFILE.md](RESOURCE_PROFILE.md).
- **Video Walkthrough (Planned for Phase 13E)**: A recorded demonstration showcasing:
  1. PDF resume upload and candidate name extraction (`Zain Ahmed`).
  2. Grounded question generation via Gemini and ChromaDB RAG.
  3. Live voice answer recording and local CPU Whisper transcription.
  4. Instant 5-dimension scoring and adaptive difficulty advancement.
  5. Final coaching analytics and interview readiness badge.
- **High-Resolution Screenshot Gallery**: Visual proof of all 8 core views in [docs/assets/screenshots/](assets/screenshots/README.md).

### Pillar 3: Production Docker Containerization (Packaging Ready)
- The repository includes a production-hardened [Dockerfile](../Dockerfile), [.dockerignore](../.dockerignore), [packages.txt](../packages.txt), and [start.sh](../start.sh).
- **Single-command portability**: Anyone with Docker installed can launch the complete platform locally or on their own infrastructure:
  ```bash
  docker build -t ai-interview-coach .
  docker run -p 8000:8000 -p 7860:7860 -e GEMINI_API_KEY="your-key-here" ai-interview-coach
  ```
- Excludes all candidate data, local databases, test audio, and secrets from container builds.

---

## 🛡️ Future Cloud Demo Path (If Live URL Is Strictly Required)

If a live public URL becomes mandatory in the future without a credit card, the only viable path is a **decoupled cloud-native demo architecture**:
1. Host the Streamlit UI on **Streamlit Community Cloud** (Free, no card, 1 GB RAM).
2. Utilize Gemini directly or via an ultra-lightweight cloud API gateway (under 50 MB, zero PyTorch).
3. Cloud demo would designate voice transcription and local vector search as "Local Full-Engine Features", allowing the web demo to focus on typed multi-turn interview evaluation.

*Note: Per user instructions, no local architecture or database code has been rewritten or downgraded.*

---

## 🔒 Security & Data Integrity Summary

- **Secrets**: `GEMINI_API_KEY` remains exclusively in `.env` (git-ignored) and is never baked into images, code, or documentation.
- **Candidate Privacy**: All candidate CVs (`data/uploads/*`) and database records (`data/*.db`) are excluded via `.gitignore` and `.dockerignore`.
- **Zero Paid Dependencies**: The platform remains 100% free-tier compliant.
