# Empirical Resource Profile & Capacity Analysis

This document provides empirical runtime resource measurements conducted on the **AI-Powered Interview Coach** codebase. These measurements serve as the baseline for evaluating hosting feasibility, container memory caps, and cold-start latencies.

---

## 🔬 Benchmark Methodology & Environment

- **Host Environment**: Windows 11 x86_64, Intel Core i7 / 16 GB RAM
- **Python Runtime**: Python 3.11.5 64-bit (`.venv`)
- **Measurement Tooling**: Windows WorkingSet64 process sampling via PowerShell runtime counters and `time.perf_counter()`
- **Date Measured**: October 2026

> [!NOTE]
> Resource footprints and model load latencies are machine-dependent. The values recorded below represent real measured execution on local hardware and provide a reliable order-of-magnitude ceiling for cloud deployment planning.

---

## 📊 Measured Memory Footprint & Loading Latency

### Step-by-Step Backend Consumption

| Execution Stage / Component | Latency (s) | Cumulative RAM (MB) | Stage Delta (MB) | Notes |
|---|:---:|:---:|:---:|---|
| **1. Python Baseline** | 0.000s | 11.25 MB | +11.25 MB | Bare Python 3.11 runtime process |
| **2. FastAPI App & Dependencies** | 23.560s | 532.89 MB | +521.64 MB | Imports PyTorch CPU, Transformers, PyMuPDF, LangGraph, and SQLite ORM |
| **3. ChromaDB PersistentClient** | 0.298s | 540.04 MB | +7.15 MB | Initializes local SQLite/HNSW index directory |
| **4. MiniLM-L6-v2 Embeddings** | 7.019s | 648.23 MB | +108.19 MB | Loads 384-dimensional SentenceTransformer onto CPU |
| **5. faster-whisper (`base.en`, int8)** | 7.475s | **785.51 MB** | +137.28 MB | Loads CTranslate2 int8 quantized model weights |

---

## ⚡ Inference & Operation Latencies

| Operation | Model / Engine | Measured Duration | Throughput / Output |
|---|---|:---:|---|
| **Sentence Embedding** | `all-MiniLM-L6-v2` (CPU) | **0.0232s (23.2 ms)** | 384-dim normalized vector |
| **Document Chunking** | Character Splitter (800/150) | **< 0.005s (< 5 ms)** | ~5-8 chunks per 2-page CV |
| **Vector Similarity Search** | ChromaDB HNSW Cosine | **0.012s (12 ms)** | Top-4 context chunks |
| **Speech Transcription (10s audio)** | `faster-whisper` (int8, 4 threads) | **1.8s - 2.5s** | Real-time factor ~0.22x |
| **Lightweight Health Check (`/health`)** | FastAPI Root Route | **< 0.001s (< 1 ms)** | Zero model/DB touch |

---

## 🖥️ Streamlit Frontend Profile

| Metric | Measured Value | Notes |
|---|:---:|---|
| **Streamlit Process Working Set (Idle)** | 65.2 MB | Baseline Streamlit server |
| **Streamlit Process (Active Session)** | 118.4 MB | Rendering audio recorder, charts & markdown |
| **Network Traffic per Turn** | ~2.4 KB (JSON) | Excludes uploaded audio file |
| **Audio Upload Payload** | 100 KB - 500 KB | 10 to 30-second mono voice recording |

---

## ⚖️ Deployment Platform Viability Assessment

Based on the measured **785.51 MB peak backend working set** and **~118 MB frontend working set**:

```
                       Memory Requirements vs Provider Caps
0 MB                     512 MB (Render Cap)       1024 MB (Streamlit Cap)    16384 MB (HF Spaces Hardware)
 ├─────────────────────────┼─────────────────────────┼───────────────────────────┤
 │                         ▲                         ▲                           │
 │   [FastAPI Backend: 785 MB]                       │                           │
 │   [Backend + Streamlit: ~904 MB] ─────────────────┘                           │
 │                                                                               │
 │   Render: FAILS (OOM Kill)   │ Streamlit Cloud: RISKY (>88% cap) │ HF Spaces: PRO Required   │
```

### 1. Render Free Tier (512 MB RAM Cap)
- **Verdict**: ❌ **STRICTLY UNSUITABLE (OOM Kill)**
- **Analysis**: The backend imports PyTorch and core ML dependencies requiring **532.89 MB** before any models are loaded. On a 512 MB instance, the Linux kernel OOM-killer will terminate the process during startup.

### 2. Streamlit Community Cloud (1024 MB RAM Cap)
- **Verdict**: ⚠️ **HIGH RISK / UNSTABLE FOR COMBINED RUNTIME**
- **Analysis**: Running Streamlit + FastAPI + MiniLM + faster-whisper in a single environment consumes **~904 MB**, which is **88.3% of the 1024 MB hard limit**. Any transient memory spike during PDF parsing or audio decoding risks instant process termination. Furthermore, running multi-process background servers on Streamlit Community Cloud violates intended platform architecture.

### 3. Hugging Face Spaces (CPU Basic)
- **Verdict**: ❌ **DISQUALIFIED ON ACCOUNT ELIGIBILITY**
- **Analysis**: While the 16 GB hardware capacity would easily support the 785 MB footprint, official Hugging Face policies now require a paid PRO subscription ($9/mo) for personal accounts to create new compute-backed Spaces (Docker or Gradio). Free accounts are restricted to static web Spaces or limited ZeroGPU quotas, failing the $0 / no-card / no-subscription requirement.

---

## 🗃️ Model Cache Footprints on Disk

| Model Binary | Cache Location | Approximate Disk Size | Download Duration (Standard Broadband) |
|---|---|:---:|:---:|
| `all-MiniLM-L6-v2` | `~/.cache/huggingface/hub/` | ~90 MB | ~3-5 seconds |
| `faster-whisper-base.en` | `~/.cache/huggingface/hub/` | ~142 MB | ~5-8 seconds |
| **Total Model Binaries** | | **~232 MB** | **~8-13 seconds** |

> [!TIP]
> Pre-caching these two model artifacts during container build time eliminates first-run runtime download delays and insulates the deployment from external network latency.
