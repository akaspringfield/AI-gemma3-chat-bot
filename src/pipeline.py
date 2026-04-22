# src/pipeline.py

'''
Author: Akash Mambally
Date : 06-01-2026

Step 7 of 8 — Full Pipeline Assembly

This step builds src/pipeline.py — the single ask() function that wires all 6 modules together into one clean flow.

7.1 — How all modules connect
main.py / your app
      │
      ▼
  pipeline.ask(query)
      │
      ├─► CacheManager        (src/cache.py)
      │     Layer 3 LLM hit? ──► return instantly
      │
      ├─► Retriever           (src/retriever.py)
      │     embed → cache check → ChromaDB
      │
      ├─► LLM Generator       (src/llm.py)
      │     build prompt → Gemma 3 stream
      │
      ├─► Cache Writer        (src/cache.py)
      │     store answer in LLM cache
      │
      └─► Return PipelineResult
            {answer, sources, cache_status,
             latency_breakdown, tokens}

7.2 — Write src/pipeline.py

7.3 — Fix the missing import in pipeline.py
One import references LLM_MODEL_PATH directly. Add it to the imports block at the top of src/pipeline.py:
python# Add this line after the other config imports
from config import LLM_MODEL_PATH

7.4 — Run the full pipeline test
python src/pipeline.py

Expected output:

>>> STATUS CHECK
{'chromadb_docs'  : 1,
 'llm_model_ready': True,
 'llm_model_path' : 'models/gemma-3-4b-it-Q4_K_M.gguf',
 'cache_stats'    : { ... }}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Q1: What is this document about?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ─────────────────────────────────────────────────────
  Gemma 3 Answer (streaming)
  ─────────────────────────────────────────────────────
  This document is about a RAG pipeline for AI and
  machine learning. [source: test_doc.txt, page: 0]
  ─────────────────────────────────────────────────────
  Cache layer  : none
  Total latency: 9840 ms    ← cold, LLM generated

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Q2: What is this document about?   (repeat)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ─────────────────────────────────────────────────────
  [CACHE] Answer retrieved instantly
  ─────────────────────────────────────────────────────
  This document is about a RAG pipeline...
  ─────────────────────────────────────────────────────
  Cache layer  : llm_response
  Total latency: 3.2 ms     ← 3000x faster

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Q3: Can you summarise the document?  (similar)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Cache layer  : vector_search   ← sim > 0.90 hit
  Total latency: 18.4 ms

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Q4: Who wrote this document?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  I don't know based on the provided documents.
  Cache layer  : none

>>> FINAL CACHE STATS
{'embedding_cache'    : {'entries': 3, 'size_mb': 0.12},
 'vector_search_cache': {'entries': 2, 'size_mb': 0.09},
 'llm_response_cache' : {'entries': 2, 'size_mb': 0.03}}

Step 7 complete — full RAG pipeline operational!

'''

import os
import sys
import time
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retriever import retrieve, get_cache_manager
from src.llm import generate, generate_no_context
from src.embedder import store_chunks, load_all_chunks_from_processed
from src.ingest import ingest_all
from config import LLM_MODEL_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# RESULT DATACLASS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PipelineResult:
    """
    Structured return type from pipeline.ask().
    Every field is always present — no KeyError surprises.
    """
    # Core output
    query          : str
    answer         : str
    sources        : List[str]

    # Cache diagnostics
    cache_status   : Dict        = field(default_factory=dict)
    cache_hit_layer: str         = ""   # "llm", "vector", "embedding", "none"

    # Timing breakdown (all in ms)
    latency        : Dict        = field(default_factory=dict)
    total_ms       : float       = 0.0

    # Token usage
    prompt_tokens  : int         = 0
    answer_tokens  : int         = 0

    # Retrieved chunks (empty on LLM cache hit)
    chunks         : List[Dict]  = field(default_factory=list)

    def print_report(self):
        """Pretty-print the full result for debugging."""
        sep = "═" * 58
        print(f"\n{sep}")
        print(f"  PIPELINE RESULT")
        print(f"{sep}")
        print(f"  Query        : {self.query}")
        print(f"  Cache layer  : {self.cache_hit_layer}")
        print(f"  Total latency: {self.total_ms} ms")
        print(f"\n  Latency breakdown:")
        for k, v in self.latency.items():
            print(f"    {k:<26}: {v} ms")
        print(f"\n  Tokens — prompt: {self.prompt_tokens} "
              f"| answer: {self.answer_tokens}")
        print(f"\n  Sources: {self.sources}")
        print(f"\n  Answer:\n  {self.answer}")
        print(f"{sep}\n")


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE CLASS
# ══════════════════════════════════════════════════════════════════════════════

class RAGPipeline:
    """
    Production RAG pipeline.
    Instantiate once, call ask() many times.

    Usage:
        pipeline = RAGPipeline()
        result   = pipeline.ask("What is retrieval augmented generation?")
        print(result.answer)
    """

    def __init__(self):
        logger.info("Initialising RAG pipeline...")
        self.cm = get_cache_manager()
        logger.info("Pipeline ready.")

    # ──────────────────────────────────────────────────────────────────────────
    def ask(
        self,
        query        : str,
        top_k        : int  = 5,
        print_stream : bool = True,
        skip_cache   : bool = False,
    ) -> PipelineResult:
        """
        Answer a question using the full RAG pipeline.

        Parameters:
            query        : natural language question
            top_k        : number of chunks to retrieve (default 5)
            print_stream : stream answer tokens to terminal
            skip_cache   : force fresh retrieval + generation

        Returns:
            PipelineResult dataclass
        """
        if not query or not query.strip():
            return PipelineResult(
                query  = query,
                answer = "Please enter a valid question.",
                sources= [],
            )

        query = query.strip()
        pipeline_start = time.time()
        logger.info(f"Pipeline ask: '{query}'")

        # ── STEP 1: Retrieve (handles all 3 cache layers internally) ──────────
        retrieval = retrieve(
            query          = query,
            top_k          = top_k,
            skip_llm_cache = skip_cache,
        )

        # ── STEP 2: LLM cache hit — return immediately ────────────────────────
        if retrieval["llm_cache_hit"]:
            total_ms = round((time.time() - pipeline_start) * 1000, 1)

            if print_stream:
                print(f"\n{'─'*55}")
                print(f"  [CACHE] Answer retrieved instantly")
                print(f"{'─'*55}")
                print(retrieval["llm_answer"])
                print(f"{'─'*55}")

            return PipelineResult(
                query           = query,
                answer          = retrieval["llm_answer"],
                sources         = retrieval["llm_sources"] or [],
                cache_status    = retrieval["cache_status"],
                cache_hit_layer = "llm",
                latency         = retrieval["latency_ms"],
                total_ms        = total_ms,
                chunks          = [],
            )

        # ── STEP 3: No chunks — return graceful fallback ──────────────────────
        if not retrieval["chunks"]:
            logger.warning("No chunks retrieved — returning fallback answer.")
            total_ms = round((time.time() - pipeline_start) * 1000, 1)
            fallback = generate_no_context(query)

            return PipelineResult(
                query           = query,
                answer          = fallback["answer"],
                sources         = [],
                cache_status    = retrieval["cache_status"],
                cache_hit_layer = "none",
                latency         = retrieval["latency_ms"],
                total_ms        = total_ms,
            )

        # ── STEP 4: Determine cache hit layer for reporting ───────────────────
        cache_hit_layer = "none"
        for layer, status in retrieval["cache_status"].items():
            if "hit" in str(status):
                cache_hit_layer = layer
                break

        # ── STEP 5: Generate answer with Gemma 3 ─────────────────────────────
        gen = generate(
            query        = query,
            chunks       = retrieval["chunks"],
            print_stream = print_stream,
        )

        # ── STEP 6: Store answer in LLM response cache ────────────────────────
        self.cm.llm.set(
            query           = query,
            query_embedding = retrieval["query_vector"],
            answer          = gen["answer"],
            sources         = gen["sources"],
        )

        # ── STEP 7: Assemble final result ─────────────────────────────────────
        total_ms = round((time.time() - pipeline_start) * 1000, 1)

        latency = {**retrieval["latency_ms"], "llm_generate_ms": gen["latency_ms"]}

        return PipelineResult(
            query           = query,
            answer          = gen["answer"],
            sources         = gen["sources"],
            cache_status    = retrieval["cache_status"],
            cache_hit_layer = cache_hit_layer,
            latency         = latency,
            total_ms        = total_ms,
            prompt_tokens   = gen["prompt_tokens"],
            answer_tokens   = gen["answer_tokens"],
            chunks          = retrieval["chunks"],
        )

    # ──────────────────────────────────────────────────────────────────────────
    def ingest_and_index(self, data_dir: str = None) -> Dict:
        """
        Convenience method — ingest new documents and index them
        without leaving the pipeline.

        Parameters:
            data_dir : path to scan (defaults to config RAW_DATA_DIR)

        Returns:
            dict with ingestion + indexing stats
        """
        logger.info("Starting ingest + index...")

        # Ingest
        t0 = time.time()
        chunks = ingest_all(data_dir)
        ingest_ms = round((time.time() - t0) * 1000, 1)

        if not chunks:
            return {
                "status"      : "no_files",
                "message"     : "No documents found in data/raw/",
                "chunks_total": 0,
            }

        # Index into ChromaDB
        t0 = time.time()
        stored = store_chunks(chunks)
        index_ms = round((time.time() - t0) * 1000, 1)

        return {
            "status"        : "ok",
            "chunks_total"  : len(chunks),
            "chunks_stored" : stored,
            "ingest_ms"     : ingest_ms,
            "index_ms"      : index_ms,
        }

    # ──────────────────────────────────────────────────────────────────────────
    def cache_stats(self) -> Dict:
        """Return stats for all 3 cache layers."""
        return self.cm.all_stats()

    def clear_cache(self) -> None:
        """Wipe all 3 cache layers."""
        self.cm.clear_all()
        logger.info("All caches cleared.")

    def status(self) -> Dict:
        """
        Health check — returns pipeline readiness info.
        Call this to verify everything is wired correctly
        before going to production.
        """
        from src.embedder import get_chroma_collection
        import os

        collection  = get_chroma_collection()
        model_exists= os.path.exists(LLM_MODEL_PATH)

        return {
            "chromadb_docs"  : collection.count(),
            "llm_model_ready": model_exists,
            "llm_model_path" : LLM_MODEL_PATH,
            "cache_stats"    : self.cache_stats(),
        }


# ══════════════════════════════════════════════════════════════════════════════
# STANDALONE TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import pprint

    # ── Initialise pipeline ───────────────────────────────────────────────────
    pipeline = RAGPipeline()

    # ── Health check ─────────────────────────────────────────────────────────
    print("\n>>> STATUS CHECK")
    pprint.pprint(pipeline.status())

    # ── Test queries ──────────────────────────────────────────────────────────
    questions = [
        "What is this document about?",
        "What is this document about?",        # repeat → LLM cache hit
        "Can you summarise the document?",     # similar → vector cache hit
        "Who wrote this document?",            # likely not in context → I don't know
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n{'━'*58}")
        print(f"  Q{i}: {question}")
        print(f"{'━'*58}")

        result = pipeline.ask(
            query        = question,
            top_k        = 5,
            print_stream = True,
            skip_cache   = False,
        )

        print(f"\n  Cache layer  : {result.cache_hit_layer}")
        print(f"  Total latency: {result.total_ms} ms")
        print(f"  Sources      : {result.sources}")

    # ── Final cache stats ─────────────────────────────────────────────────────
    print("\n>>> FINAL CACHE STATS")
    pprint.pprint(pipeline.cache_stats())
    print("\nStep 7 complete — full RAG pipeline operational!")
