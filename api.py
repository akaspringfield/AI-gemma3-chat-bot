# api.py

import os
import sys
import time
import uuid
import shutil
import logging
import asyncio
from pathlib import Path
from typing import List, Optional
from datetime import datetime, timezone

from fastapi import (
    FastAPI, File, Form, UploadFile,
    HTTPException, BackgroundTasks,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import (
    MAX_FILE_SIZE_MB,
    MAX_FILES_PER_REQUEST,
    ALLOWED_EXTENSIONS,
    UPLOAD_DIR,
    LLM_MODEL_PATH,
    EMBEDDING_MODEL,
    CHROMA_COLLECTION,
)
from src.pipeline        import RAGPipeline
from src.session_manager import SessionManager
from src.general_search  import answer_general
from src.ingest          import ingest_all
from src.embedder        import store_chunks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.makedirs(UPLOAD_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# APP INIT
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title       = "RAG Pipeline API",
    description = "Local RAG pipeline powered by Gemma 3 4B",
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# Singletons — loaded once on startup
_pipeline : Optional[RAGPipeline]    = None
_sessions : Optional[SessionManager] = None


@app.on_event("startup")
async def startup():
    global _pipeline, _sessions
    logger.info("Starting up RAG API...")
    _sessions = SessionManager()
    _sessions.cleanup_expired()
    _pipeline = RAGPipeline()
    logger.info("RAG API ready.")


def get_pipeline() -> RAGPipeline:
    if _pipeline is None:
        raise HTTPException(503, "Pipeline not ready yet.")
    return _pipeline

def get_sessions() -> SessionManager:
    if _sessions is None:
        raise HTTPException(503, "Session manager not ready.")
    return _sessions


# ══════════════════════════════════════════════════════════════════════════════
# RESPONSE SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════

class AIResponse(BaseModel):
    # Request echo
    session_uuid  : str
    user_uuid     : str
    question      : str
    response_type : str

    # Answer
    answer        : str
    sources       : List[str]

    # Files processed this request
    files_uploaded: List[str] = Field(default_factory=list)
    files_indexed : List[str] = Field(default_factory=list)

    # Model info
    model_name    : str
    embedding_model: str

    # Cache info
    cache_hit     : bool
    cache_layer   : str

    # Timing (all ms)
    response_time_ms  : float
    retrieval_ms      : float
    generation_ms     : float
    file_processing_ms: float

    # Session info
    session_query_count: int
    session_created_at : str
    session_files_total: int

    # Metadata
    timestamp     : str
    api_version   : str = "1.0.0"
    chunks_retrieved: int


class HealthResponse(BaseModel):
    status         : str
    model_ready    : bool
    chromadb_docs  : int
    active_sessions: int
    uptime_seconds : float
    timestamp      : str


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

_start_time = time.time()

MAX_FILE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


def validate_file(file: UploadFile) -> None:
    """Raise HTTPException if file fails validation."""
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            400,
            f"File type '{ext}' not allowed. "
            f"Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )


async def save_uploaded_files(
    files       : List[UploadFile],
    session_uuid: str,
) -> tuple[List[str], float]:
    """
    Save uploaded files to session upload dir.

    Returns:
        (list of saved filenames, processing time ms)
    """
    t0         = time.time()
    saved      = []
    upload_dir = os.path.join(UPLOAD_DIR, session_uuid)
    os.makedirs(upload_dir, exist_ok=True)

    for file in files:
        validate_file(file)
        content = await file.read()

        # Size check
        if len(content) > MAX_FILE_BYTES:
            raise HTTPException(
                413,
                f"File '{file.filename}' exceeds "
                f"{MAX_FILE_SIZE_MB}MB limit."
            )

        dest = os.path.join(upload_dir, file.filename)
        with open(dest, "wb") as f:
            f.write(content)

        saved.append(file.filename)
        logger.info(
            f"Saved {file.filename} "
            f"({len(content)/1024:.1f} KB) → {dest}"
        )

    elapsed = round((time.time() - t0) * 1000, 1)
    return saved, elapsed


def index_session_files(
    session_uuid    : str,
    filenames       : List[str],
    session_manager : SessionManager,
    pipeline        : RAGPipeline,
) -> List[str]:
    """
    Ingest and embed new files for a session into a
    session-specific ChromaDB collection.

    Returns:
        list of filenames that were actually indexed
    """
    import chromadb
    from chromadb.config import Settings
    from src.embedder import embed_texts
    from src.ingest   import load_document, chunk_documents

    upload_dir  = os.path.join(UPLOAD_DIR, session_uuid)
    session     = session_manager.get(session_uuid)
    collection_name = session["chroma_collection"]
    indexed     = []

    # Get or create session-specific ChromaDB collection
    client = chromadb.PersistentClient(
        path=os.path.join("vectorstore", "sessions"),
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_or_create_collection(
        name     = collection_name,
        metadata = {"hnsw:space": "cosine"},
    )

    for filename in filenames:
        # Skip already indexed
        if session_manager.is_file_indexed(session_uuid, filename):
            logger.info(f"Already indexed: {filename} — skipping")
            continue

        filepath = os.path.join(upload_dir, filename)
        if not os.path.exists(filepath):
            logger.warning(f"File not found: {filepath}")
            continue

        # Load + chunk
        docs   = load_document(filepath)
        chunks = chunk_documents(docs)

        if not chunks:
            continue

        # Embed + store
        texts      = [c["text"]     for c in chunks]
        embeddings = embed_texts(texts)

        collection.add(
            ids        = [c["chunk_id"] for c in chunks],
            embeddings = embeddings.tolist(),
            documents  = texts,
            metadatas  = [
                {
                    "source":     c["source"],
                    "page":       int(c["page"]),
                    "char_count": int(c["char_count"]),
                    "session":    session_uuid,
                }
                for c in chunks
            ],
        )

        session_manager.mark_indexed(session_uuid, filename)
        indexed.append(filename)
        logger.info(
            f"Indexed {filename} → "
            f"{len(chunks)} chunks into {collection_name}"
        )

    return indexed


def query_session_collection(
    session_uuid : str,
    session      : dict,
    query_vec    : any,
    top_k        : int = 5,
) -> List[dict]:
    """
    Query the session-specific ChromaDB collection.

    Returns:
        list of chunk dicts with text, source, page, score
    """
    import chromadb
    from chromadb.config import Settings
    import numpy as np

    client = chromadb.PersistentClient(
        path=os.path.join("vectorstore", "sessions"),
        settings=Settings(anonymized_telemetry=False),
    )

    try:
        collection = client.get_collection(session["chroma_collection"])
    except Exception:
        return []

    if collection.count() == 0:
        return []

    results = collection.query(
        query_embeddings = [query_vec.tolist()],
        n_results        = min(top_k, collection.count()),
        include          = ["documents", "metadatas", "distances"],
    )

    chunks = []
    docs      = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for text, meta, dist in zip(docs, metadatas, distances):
        score = round(1 - (dist / 2), 4)
        chunks.append({
            "text":   text,
            "source": meta.get("source", "unknown"),
            "page":   meta.get("page",   0),
            "score":  score,
        })

    chunks.sort(key=lambda x: x["score"], reverse=True)
    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENDPOINT
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/ai/response-gemma", response_model=AIResponse)
async def ai_response(
    background_tasks: BackgroundTasks,
    session_uuid : str          = Form(..., description="Unique session ID"),
    user_uuid    : str          = Form(..., description="Unique user ID"),
    question     : str          = Form(..., description="Question to ask"),
    response_type: str          = Form(
        "session",
        description="'session' = from uploaded docs | 'general' = web search"
    ),
    files        : List[UploadFile] = File(
        default=[],
        description=f"Upload up to {MAX_FILES_PER_REQUEST} files "
                    f"(max {MAX_FILE_SIZE_MB}MB each)"
    ),
):
    """
    Main AI endpoint.

    - **session_uuid** : groups context for a conversation
    - **user_uuid**    : identifies the caller
    - **question**     : natural language question
    - **response_type**: `session` (from docs) or `general` (web search)
    - **files**        : optional file uploads (PDF, DOCX, TXT, MD)
    """
    request_start = time.time()

    # ── Validate inputs ───────────────────────────────────────────────────────
    if not question.strip():
        raise HTTPException(400, "Question cannot be empty.")

    if response_type not in ("session", "general"):
        raise HTTPException(
            400,
            "response_type must be 'session' or 'general'."
        )

    # Filter out empty file slots (form sends empty file if none uploaded)
    valid_files = [
        f for f in files
        if f.filename and f.filename.strip()
    ]

    if len(valid_files) > MAX_FILES_PER_REQUEST:
        raise HTTPException(
            400,
            f"Maximum {MAX_FILES_PER_REQUEST} files per request."
        )

    pipeline = get_pipeline()
    sm       = get_sessions()

    # ── Session setup ─────────────────────────────────────────────────────────
    session = sm.get_or_create(session_uuid, user_uuid)

    # ── Handle file uploads ───────────────────────────────────────────────────
    files_uploaded = []
    files_indexed  = []
    file_ms        = 0.0

    if valid_files:
        files_uploaded, file_ms = await save_uploaded_files(
            valid_files, session_uuid
        )
        sm.update_files(session_uuid, files_uploaded)

        # Index new files synchronously so they're searchable immediately
        if response_type == "session":
            files_indexed = index_session_files(
                session_uuid  = session_uuid,
                filenames     = files_uploaded,
                session_manager = sm,
                pipeline      = pipeline,
            )
            # Clear session cache so new docs are included
            if files_indexed:
                pipeline.clear_cache()

    # ── Generate answer ───────────────────────────────────────────────────────
    retrieval_ms  = 0.0
    generation_ms = 0.0
    cache_hit     = False
    cache_layer   = "none"
    answer        = ""
    sources       = []
    chunks_count  = 0

    if response_type == "general":
        # ── General mode: DuckDuckGo + Gemma 3 ───────────────────────────────
        gen_result    = answer_general(question)
        answer        = gen_result["answer"]
        sources       = gen_result["sources"]
        generation_ms = gen_result["latency_breakdown"]["generation_ms"]
        retrieval_ms  = gen_result["latency_breakdown"]["search_ms"]

    else:
        # ── Session mode: query session ChromaDB collection ───────────────────
        session_refreshed = sm.get(session_uuid)

        # Check if session has any indexed files
        if not session_refreshed["indexed_files"]:
            # No docs yet — fall back to general pipeline
            result = pipeline.ask(
                query        = question,
                print_stream = False,
            )
        else:
            # Query session-specific collection
            from src.embedder import embed_query as _embed_query
            from src.llm      import generate    as _generate

            query_vec = _embed_query(question)
            chunks    = query_session_collection(
                session_uuid = session_uuid,
                session      = session_refreshed,
                query_vec    = query_vec,
                top_k        = 5,
            )

            if chunks:
                gen = _generate(
                    query        = question,
                    chunks       = chunks,
                    print_stream = False,
                )
                answer        = gen["answer"]
                sources       = gen["sources"]
                generation_ms = gen["latency_ms"]
                chunks_count  = len(chunks)

                # Cache the answer
                pipeline.cm.llm.set(
                    query           = question,
                    query_embedding = query_vec,
                    answer          = answer,
                    sources         = sources,
                )
            else:
                answer  = "I don't know based on the provided documents."
                sources = []

            result = None

        # Use global pipeline result if no session docs
        if result is not None:
            answer        = result.answer
            sources       = result.sources
            cache_hit     = result.cache_hit_layer != "none"
            cache_layer   = result.cache_hit_layer
            retrieval_ms  = result.latency.get("chroma_search_ms", 0)
            generation_ms = result.latency.get("llm_generate_ms", 0)
            chunks_count  = len(result.chunks)

    # ── Update session ────────────────────────────────────────────────────────
    sm.increment_query(session_uuid)
    session_final = sm.get(session_uuid)

    # ── Build response ────────────────────────────────────────────────────────
    total_ms = round((time.time() - request_start) * 1000, 1)

    return AIResponse(
        # Request echo
        session_uuid   = session_uuid,
        user_uuid      = user_uuid,
        question       = question,
        response_type  = response_type,

        # Answer
        answer         = answer,
        sources        = sources,

        # Files
        files_uploaded = files_uploaded,
        files_indexed  = files_indexed,

        # Model info
        model_name         = os.path.basename(LLM_MODEL_PATH),
        embedding_model    = EMBEDDING_MODEL,

        # Cache
        cache_hit    = cache_hit,
        cache_layer  = cache_layer,

        # Timing
        response_time_ms   = total_ms,
        retrieval_ms       = retrieval_ms,
        generation_ms      = generation_ms,
        file_processing_ms = file_ms,

        # Session
        session_query_count = session_final["query_count"],
        session_created_at  = datetime.fromtimestamp(
            session_final["created_at"],
            tz=timezone.utc,
        ).isoformat(),
        session_files_total = len(session_final["files"]),

        # Meta
        timestamp        = datetime.now(timezone.utc).isoformat(),
        chunks_retrieved = chunks_count,
    )


# ══════════════════════════════════════════════════════════════════════════════
# HEALTH CHECK
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health", response_model=HealthResponse)
async def health():
    """Pipeline health check."""
    pipeline = get_pipeline()
    sessions = get_sessions()
    status   = pipeline.status()

    return HealthResponse(
        status          = "ok",
        model_ready     = status["llm_model_ready"],
        chromadb_docs   = status["chromadb_docs"],
        active_sessions = len(sessions._sessions),
        uptime_seconds  = round(time.time() - _start_time, 1),
        timestamp       = datetime.now(timezone.utc).isoformat(),
    )


# ══════════════════════════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    from config import API_HOST, API_PORT, API_RELOAD

    uvicorn.run(
        "api:app",
        host    = API_HOST,
        port    = API_PORT,
        reload  = API_RELOAD,
    )