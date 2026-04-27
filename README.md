# Local RAG Pipeline

> Production-ready Retrieval-Augmented Generation pipeline powered by **Gemma 3 4B**, **ChromaDB**, and **BAAI/bge-large-en-v1.5** — fully local, no cloud dependencies.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Server Setup — Step by Step](#server-setup--step-by-step)
- [API Reference](#api-reference)
- [CLI Usage](#cli-usage)
- [Adding Documents](#adding-documents)
- [Cache System](#cache-system)
- [Session Management](#session-management)
- [GPU Upgrade Path](#gpu-upgrade-path)
- [Troubleshooting](#troubleshooting)

---

## Overview

This pipeline lets you query your own documents using a fully local LLM stack. No data leaves your machine. It supports two query modes:

| Mode | Description |
|---|---|
| `session` | Answers from your uploaded documents using ChromaDB vector search |
| `general` | Searches the web via DuckDuckGo, summarised by Gemma 3 |

Every answer includes source citations, timing breakdowns, cache diagnostics, and session metadata — all returned as structured JSON.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    CLIENT / CURL / APP                   │
└──────────────────────────┬──────────────────────────────┘
                           │  POST /ai/response-gemma
                           ▼
┌─────────────────────────────────────────────────────────┐
│                     FastAPI  (api.py)                    │
│  • Validates inputs          • Manages file uploads      │
│  • Routes response_type      • Returns JSON response     │
└────────┬──────────────────────────────────┬─────────────┘
         │ session                           │ general
         ▼                                   ▼
┌─────────────────────┐         ┌────────────────────────┐
│   RAGPipeline       │         │   GeneralSearch         │
│   (pipeline.py)     │         │   (general_search.py)   │
└────────┬────────────┘         │   DuckDuckGo + Gemma 3  │
         │                      └────────────────────────┘
         ▼
┌─────────────────────────────────────────────────────────┐
│                   3-Layer Cache  (cache.py)              │
│                                                          │
│  Layer 3 ── LLM Response Cache  (checked first)         │
│             cosine sim ≥ 0.90 → return instantly         │
│                        │ MISS                            │
│  Layer 1 ── Embedding Cache                              │
│             exact query match → skip BGE recompute       │
│                        │ MISS                            │
│  Layer 2 ── Vector Search Cache                          │
│             cosine sim ≥ 0.90 → skip ChromaDB ANN        │
│                        │ MISS                            │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│               Retriever  (retriever.py)                  │
│   embed_query()  →  ChromaDB ANN search  →  top-K chunks │
│   BAAI/bge-large-en-v1.5  (1024-dim vectors)             │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                  LLM  (llm.py)                           │
│   Build prompt  →  Gemma 3 4B GGUF  →  stream tokens    │
│   System rules: cite sources · no hallucination          │
│   "I don't know" fallback if context is missing          │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
                   Store in LLM cache
                   Return JSON response
```

### Data Ingestion Flow

```
data/raw/  (PDF · DOCX · TXT · MD)
     │
     ▼
 ingest.py
 RecursiveCharacterTextSplitter
 chunk_size=512, overlap=64 (~12%)
 Split on: \n\n → \n → ". " → " "
     │
     ▼
 embedder.py
 BAAI/bge-large-en-v1.5
 1024-dimensional vectors
 batch_size=32, normalised
     │
     ▼
 ChromaDB  (vectorstore/)
 HNSW index, cosine similarity
 Persisted to disk
```

### Per-Session Isolation

```
Request: session_uuid=sess-001
              │
              ▼
   SessionManager.get_or_create()
              │
   ┌──────────┴──────────┐
   │  data/sessions/     │   ← metadata (JSON, survives restart)
   │    sess-001/        │
   │      meta.json      │
   └──────────┬──────────┘
              │
   ┌──────────┴──────────┐
   │  data/uploads/      │   ← uploaded files
   │    sess-001/        │       auto-deleted after N hours
   │      report.pdf     │       (0 = never delete)
   └──────────┬──────────┘
              │
   ┌──────────┴──────────┐
   │  vectorstore/       │   ← session-specific ChromaDB
   │    sessions/        │       collection: session_sess-001
   │      sess-001/      │
   └─────────────────────┘
```

---

## Features

- **Fully local** — Gemma 3 4B GGUF, BGE embeddings, ChromaDB, DuckDuckGo. Zero cloud API calls for `session` mode.
- **3-layer cache** — LLM response → embedding → vector search. Repeat queries return in ~3ms.
- **Per-session isolation** — each `session_uuid` gets its own ChromaDB collection, upload folder, and cache namespace.
- **Session persistence** — sessions survive server restarts, loaded from disk on startup.
- **Streaming generation** — tokens printed live to terminal; full answer assembled for API response.
- **Citation enforcement** — system prompt hard-instructs Gemma 3 to cite source + page for every claim.
- **Hallucination guard** — model returns "I don't know based on the provided documents" when context is missing.
- **Multi-format ingestion** — PDF, DOCX, TXT, MD all supported out of the box.
- **Configurable file TTL** — `UPLOAD_AUTO_DELETE_HOURS=0` keeps files forever; any positive value auto-deletes.
- **GPU-ready** — switch from CPU to CUDA in 3 config lines + one pip install.
- **Swagger UI** — interactive API docs at `http://localhost:8000/docs`.

---

## Project Structure

```
rag_pipeline/
├── data/
│   ├── raw/                ← drop your documents here
│   ├── processed/          ← chunked JSON (auto-generated)
│   ├── uploads/            ← per-session uploaded files
│   └── sessions/           ← session metadata (JSON)
├── cache/
│   ├── embeddings/         ← Layer 1 cache (diskcache)
│   ├── vector_search/      ← Layer 2 cache (diskcache)
│   └── llm_responses/      ← Layer 3 cache (diskcache)
├── vectorstore/
│   ├── (global)            ← ChromaDB for data/raw/ documents
│   └── sessions/           ← per-session ChromaDB collections
├── models/
│   └── gemma-3-4b-it-Q4_K_M.gguf   ← LLM model file (~2.8 GB)
├── src/
│   ├── __init__.py
│   ├── ingest.py           ← Step 2: document loading + chunking
│   ├── embedder.py         ← Step 3: BGE embeddings + ChromaDB
│   ├── cache.py            ← Step 4: 3-layer cache manager
│   ├── retriever.py        ← Step 5: cache-aware ANN retrieval
│   ├── llm.py              ← Step 6: Gemma 3 generation + prompt
│   ├── pipeline.py         ← Step 7: full pipeline orchestrator
│   ├── session_manager.py  ← API: per-session state management
│   └── general_search.py   ← API: DuckDuckGo + Gemma 3
├── api.py                  ← FastAPI application
├── main.py                 ← Interactive CLI
├── config.py               ← Central configuration
├── .env                    ← Environment variables (not in git)
├── .gitignore
└── README.md
```

---

## Requirements

### Hardware

| Component | Minimum | Recommended |
|---|---|---|
| RAM | 8 GB | 16 GB |
| Disk | 10 GB free | 20 GB free |
| CPU | 4 cores | 8+ cores |
| GPU | None (CPU only) | NVIDIA 8 GB VRAM+ |

### Software

- Python 3.10 or higher
- Windows 10/11, Ubuntu 20.04+, or macOS 12+
- Internet connection (first run only — model + embedding downloads)

---

## Installation

### 1. Clone / create the project folder

```bash
mkdir rag_pipeline
cd rag_pipeline
```

### 2. Create virtual environment

```bash
python -m venv venv

# Activate — Windows
venv\Scripts\activate

# Activate — Linux / macOS
source venv/bin/activate
```

### 3. Install dependencies

Install one package at a time so you can spot any failure:

```bash
pip install --upgrade pip
pip install langchain==0.2.16
pip install langchain-community==0.2.16
pip install langchain-huggingface==0.0.3
pip install chromadb==0.5.3
pip install sentence-transformers==3.0.1
pip install llama-cpp-python==0.3.8
pip install python-dotenv==1.0.1
pip install pymupdf --only-binary=:all:
pip install python-docx==1.1.2
pip install tiktoken==0.7.0
pip install numpy==1.26.4
pip install diskcache==5.6.3
pip install huggingface-hub
pip install fastapi==0.111.0
pip install uvicorn==0.30.1
pip install python-multipart==0.0.9
pip install duckduckgo-search --upgrade
pip install aiofiles==23.2.1
```

### 4. Create folder structure

```bash
# Windows
mkdir data\raw data\processed data\uploads data\sessions
mkdir cache\embeddings cache\vector_search cache\llm_responses
mkdir vectorstore\sessions models

# Linux / macOS
mkdir -p data/{raw,processed,uploads,sessions}
mkdir -p cache/{embeddings,vector_search,llm_responses}
mkdir -p vectorstore/sessions models
```

### 5. Download Gemma 3 4B model

```bash
python -c "
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id='unsloth/gemma-3-4b-it-GGUF',
    filename='gemma-3-4b-it-Q4_K_M.gguf',
    local_dir='./models'
)
print('Downloaded to:', path)
"
```

> Download is ~2.8 GB. Requires a free Hugging Face account.  
> Alternative: download manually from https://huggingface.co/unsloth/gemma-3-4b-it-GGUF

### 6. Verify installation

```bash
python -c "
import langchain, chromadb, sentence_transformers, llama_cpp, diskcache, fastapi
print('langchain          :', langchain.__version__)
print('chromadb           :', chromadb.__version__)
print('sentence-transformers:', sentence_transformers.__version__)
print('llama-cpp-python   :', llama_cpp.__version__)
print('diskcache          :', diskcache.__version__)
print('fastapi            :', fastapi.__version__)
print('All OK!')
"
```

---

## Configuration

All settings live in two files. **Never commit `.env` to git.**

### `.env`

```bash
# Application
APP_ENV=development
LOG_LEVEL=INFO

# API server
API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=true

# File uploads
MAX_FILE_SIZE_MB=20
MAX_FILES_PER_REQUEST=5
ALLOWED_EXTENSIONS=.pdf,.docx,.txt,.md

# File auto-delete (0 = never delete)
UPLOAD_AUTO_DELETE_HOURS=24

# Session TTL (0 = never expire)
SESSION_TTL_HOURS=24

# Web search (general mode)
DDG_MAX_RESULTS=5
DDG_REQUEST_DELAY_SECONDS=3
```

### `config.py` — key parameters

| Parameter | Default | Description |
|---|---|---|
| `CHUNK_SIZE` | 512 | Characters per chunk |
| `CHUNK_OVERLAP` | 64 | Overlap between chunks (~12%) |
| `EMBEDDING_MODEL` | `BAAI/bge-large-en-v1.5` | Embedding model |
| `DEVICE` | `cpu` | `cpu` or `cuda` |
| `TOP_K` | 5 | Chunks retrieved per query |
| `CACHE_SIM_THRESHOLD` | 0.90 | Cosine similarity for cache hit |
| `CACHE_TTL_SECONDS` | 86400 | Cache entry lifetime (24 hours) |
| `LLM_TEMPERATURE` | 1.0 | Gemma 3 recommended value |
| `LLM_TOP_K` | 64 | Gemma 3 recommended value |
| `LLM_N_GPU_LAYERS` | 0 | 0 = CPU; 35 = full GPU offload |
| `LLM_N_THREADS` | 8 | CPU threads for inference |

---

## Server Setup — Step by Step

Follow these steps in order on a fresh machine.

### Step 1 — Prepare the environment

```bash
# Confirm Python version (must be 3.10+)
python --version

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate          # Linux/macOS
# venv\Scripts\activate           # Windows

# Upgrade pip
pip install --upgrade pip
```

### Step 2 — Install all packages

```bash
# Run the install block from the Installation section above
# Then verify:
pip list | grep -E "langchain|chromadb|sentence|llama|diskcache|fastapi"
```

### Step 3 — Create folder structure

```bash
mkdir -p data/{raw,processed,uploads,sessions}
mkdir -p cache/{embeddings,vector_search,llm_responses}
mkdir -p vectorstore/sessions models
```

### Step 4 — Create config files

Create `.env` with the contents from the Configuration section above.

Ensure `config.py` exists with all parameters (see Step 1 of the build guide).

### Step 5 — Download the model

```bash
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='unsloth/gemma-3-4b-it-GGUF',
    filename='gemma-3-4b-it-Q4_K_M.gguf',
    local_dir='./models'
)
print('Model ready.')
"
```

### Step 6 — Index your documents (optional pre-load)

Drop files into `data/raw/` then run:

```bash
python src/ingest.py
python src/embedder.py
```

### Step 7 — Start the API server

```bash
python api.py
```

Expected output:
```
INFO — SessionManager ready — 0 active sessions
INFO — Loading Gemma 3 4B from: models/gemma-3-4b-it-Q4_K_M.gguf
INFO — Gemma 3 4B loaded successfully.
INFO — RAG API ready.
INFO — Uvicorn running on http://0.0.0.0:8000
```

### Step 8 — Verify the server

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "ok",
  "model_ready": true,
  "chromadb_docs": 0,
  "active_sessions": 0,
  "uptime_seconds": 4.2,
  "timestamp": "2025-04-22T10:00:00+00:00"
}
```

### Step 9 — Open Swagger UI

Navigate to `http://localhost:8000/docs` in your browser for the interactive API explorer.

### Step 10 — Run as background service (Linux production)

Create a systemd service file:

```bash
sudo nano /etc/systemd/system/rag-pipeline.service
```

```ini
[Unit]
Description=RAG Pipeline API
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/rag_pipeline
Environment=PATH=/home/ubuntu/rag_pipeline/venv/bin
ExecStart=/home/ubuntu/rag_pipeline/venv/bin/python api.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable rag-pipeline
sudo systemctl start rag-pipeline
sudo systemctl status rag-pipeline
```

View logs:
```bash
sudo journalctl -u rag-pipeline -f
```

---

## API Reference

### `POST /ai/response-gemma`

Main endpoint. Accepts `multipart/form-data`.

#### Request fields

| Field | Type | Required | Description |
|---|---|---|---|
| `session_uuid` | string | ✅ | Unique session identifier |
| `user_uuid` | string | ✅ | Unique user identifier |
| `question` | string | ✅ | Natural language question |
| `response_type` | string | ✅ | `session` or `general` |
| `files` | file(s) | ❌ | PDF, DOCX, TXT, MD (up to `MAX_FILES_PER_REQUEST`) |

#### Response fields

```json
{
  "session_uuid":         "sess-001",
  "user_uuid":            "user-abc",
  "question":             "What does the report say about revenue?",
  "response_type":        "session",

  "answer":               "Revenue grew 24% YoY. [source: report.pdf, page: 3]",
  "sources":              ["report.pdf"],

  "files_uploaded":       ["report.pdf"],
  "files_indexed":        ["report.pdf"],

  "model_name":           "gemma-3-4b-it-Q4_K_M.gguf",
  "embedding_model":      "BAAI/bge-large-en-v1.5",

  "cache_hit":            false,
  "cache_layer":          "none",

  "response_time_ms":     11240.5,
  "retrieval_ms":         38.2,
  "generation_ms":        9820.1,
  "file_processing_ms":   210.3,

  "session_query_count":  1,
  "session_created_at":   "2025-04-22T10:30:00+00:00",
  "session_files_total":  1,

  "chunks_retrieved":     3,
  "timestamp":            "2025-04-22T10:30:11+00:00",
  "api_version":          "1.0.0"
}
```

#### Example — session mode with file upload

```bash
curl -X POST http://localhost:8000/ai/response-gemma \
  -F "session_uuid=sess-001" \
  -F "user_uuid=user-abc" \
  -F "question=What is the main finding?" \
  -F "response_type=session" \
  -F "files=@data/raw/report.pdf"
```

#### Example — general mode (web search)

```bash
curl -X POST http://localhost:8000/ai/response-gemma \
  -F "session_uuid=sess-001" \
  -F "user_uuid=user-abc" \
  -F "question=What is the latest news about AI?" \
  -F "response_type=general"
```

#### Example — follow-up in same session (no file re-upload needed)

```bash
curl -X POST http://localhost:8000/ai/response-gemma \
  -F "session_uuid=sess-001" \
  -F "user_uuid=user-abc" \
  -F "question=Can you elaborate on that finding?" \
  -F "response_type=session"
```

---

### `GET /health`

Health check endpoint.

```bash
curl http://localhost:8000/health
```

---

## CLI Usage

Start the interactive terminal interface:

```bash
python main.py
```

### Available commands

| Command | Description |
|---|---|
| `<question>` | Ask anything about your documents |
| `/ingest` | Scan `data/raw/` and index new documents |
| `/status` | Show pipeline health, document count, model info |
| `/cache` | Show statistics for all 3 cache layers |
| `/clear` | Wipe all caches (forces fresh answers) |
| `/history` | Show questions asked this session |
| `/report` | Detailed report of the last answer |
| `/help` | Show command list |
| `/quit` | Exit |

---

## Adding Documents

### Via CLI

```bash
# 1. Copy files to data/raw/
cp my_report.pdf data/raw/

# 2. Start CLI and run /ingest
python main.py
> /ingest
```

### Via API (upload with question)

```bash
curl -X POST http://localhost:8000/ai/response-gemma \
  -F "session_uuid=my-session" \
  -F "user_uuid=my-user" \
  -F "question=Summarise this document" \
  -F "response_type=session" \
  -F "files=@my_report.pdf"
```

The file is indexed automatically before the question is answered.

### Supported formats

| Format | Extension |
|---|---|
| PDF | `.pdf` |
| Word document | `.docx` |
| Plain text | `.txt` |
| Markdown | `.md` |

### After adding new documents

If you asked questions before adding the new document and got cached answers, clear the cache to include new content:

```bash
# Via CLI
> /clear

# Via API (one query without cache)
-F "skip_cache=true"   # add to your next request
```

---

## Cache System

Three independent layers using `diskcache` — all persist to disk and survive restarts.

### Layer 1 — Embedding cache

- **Stores**: query text → 1024-dim vector
- **Hit condition**: exact query match (deterministic, no threshold needed)
- **Benefit**: skips BGE model inference (~300ms saved per hit)
- **Location**: `cache/embeddings/`

### Layer 2 — Vector search cache

- **Stores**: query vector → top-K retrieved chunks
- **Hit condition**: cosine similarity ≥ `CACHE_SIM_THRESHOLD` (0.90)
- **Benefit**: skips ChromaDB ANN search (~40ms saved per hit)
- **Location**: `cache/vector_search/`

### Layer 3 — LLM response cache (checked first)

- **Stores**: query vector → full answer + sources
- **Hit condition**: cosine similarity ≥ `CACHE_SIM_THRESHOLD` (0.90)
- **Benefit**: skips everything — embedding + retrieval + LLM (~10 seconds saved)
- **Location**: `cache/llm_responses/`

### Cache check order

```
Query arrives
    │
    ▼
Layer 3 (LLM cache)  ──── HIT ──► return answer in ~3ms
    │ MISS
    ▼
Layer 1 (Embedding)  ──── HIT ──► reuse vector, skip BGE
    │ MISS
    ▼
BGE model inference  (~300ms)
    │
    ▼
Layer 2 (Vector search) ─ HIT ──► return cached chunks
    │ MISS
    ▼
ChromaDB ANN search  (~40ms)
    │
    ▼
Gemma 3 generation   (~8–15s on CPU)
    │
    ▼
Store in all 3 layers
```

### Tuning the threshold

```python
# config.py
CACHE_SIM_THRESHOLD = 0.90  # default

# Higher (0.95) — more precise, fewer hits
# Lower (0.85)  — more hits, risk of stale answers
```

---

## Session Management

Sessions are identified by `session_uuid`. Each session has:

- Its own ChromaDB collection (uploaded files only searchable within that session)
- Its own upload directory
- Persistent metadata saved to `data/sessions/{session_uuid}/meta.json`
- Query counter, file list, indexed file list, timestamps

Sessions are automatically loaded on server startup. Expired sessions (older than `SESSION_TTL_HOURS`) are cleaned up.

### File retention

```bash
# .env

UPLOAD_AUTO_DELETE_HOURS=24   # delete uploaded files after 24 hours
UPLOAD_AUTO_DELETE_HOURS=0    # never delete uploaded files
SESSION_TTL_HOURS=24          # expire session metadata after 24 hours
SESSION_TTL_HOURS=0           # never expire sessions
```

---

## GPU Upgrade Path

When deploying to a GPU server, only three lines in `config.py` change:

```python
# config.py — GPU server settings
DEVICE           = "cuda"   # was "cpu"
LLM_N_GPU_LAYERS = 35       # was 0  (35 = full model on GPU)
LLM_N_THREADS    = 16       # match your server's CPU core count
```

Reinstall llama-cpp-python with CUDA support:

```bash
CMAKE_ARGS="-DLLAMA_CUDA=on" \
pip install llama-cpp-python --upgrade --force-reinstall
```

Expected latency improvement:

| Hardware | Generation time |
|---|---|
| CPU only (8 cores) | 8–15 seconds |
| NVIDIA RTX 3080 (10 GB) | ~1 second |
| NVIDIA A100 (40 GB) | ~0.3 seconds |

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'langchain'`

Virtual environment is not active.

```bash
# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate

# Confirm
which python   # should point inside venv/
```

### `PyMuPDF` fails to install on Windows

```bash
pip install pymupdf --only-binary=:all:
# or
pip install pymupdf4llm
```

### DuckDuckGo returns empty results

```bash
# Upgrade to latest version
pip install duckduckgo-search --upgrade

# Test directly
python -c "
from duckduckgo_search import DDGS
with DDGS(timeout=20) as ddgs:
    r = list(ddgs.text('python programming', max_results=3))
print(len(r), 'results')
"
```

If still empty, you are rate-limited. Increase the delay:
```bash
# .env
DDG_REQUEST_DELAY_SECONDS=5
```

### Model not found

```bash
ls models/
# Should show: gemma-3-4b-it-Q4_K_M.gguf

# If missing, re-download:
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='unsloth/gemma-3-4b-it-GGUF',
    filename='gemma-3-4b-it-Q4_K_M.gguf',
    local_dir='./models'
)
"
```

### ChromaDB is empty after ingestion

```bash
# Run both steps
python src/ingest.py
python src/embedder.py

# Verify
python -c "
import chromadb
c = chromadb.PersistentClient(path='vectorstore')
col = c.get_collection('rag_docs')
print('Docs in store:', col.count())
"
```

### Out of memory during embedding

Reduce batch size in `config.py`:
```python
# src/embedder.py — find this line and reduce
batch_size = 8   # was 32
```

### Slow responses on CPU

This is expected. Gemma 3 4B on CPU takes 8–15 seconds per generation. The 3-layer cache eliminates this for repeated or similar queries. For production speed, use a GPU server (see GPU Upgrade Path above).

---

## .gitignore

```
.env
venv/
models/*.gguf
data/uploads/
data/sessions/
data/processed/
cache/
vectorstore/
__pycache__/
*.pyc
.DS_Store
```

---

## License

MIT License. Model weights are subject to [Gemma Terms of Use](https://ai.google.dev/gemma/terms).
