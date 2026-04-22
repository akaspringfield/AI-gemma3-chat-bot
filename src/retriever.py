# src/retriever.py

'''
Author: Akash Mambally
Date : 05-01-2026

Step 5 — Semantic Retriever
This step builds src/retriever.py — the module that takes a user query, 
runs it through the cache layers, 
and if needed performs the actual ANN search against ChromaDB to return the most relevant chunks.

5.1 — How this step fits in the pipeline

User Query (string)
       │
       ▼
  CacheManager
  Layer 3: LLM cache?  ──HIT──► return cached answer
       │ MISS
       ▼
  Layer 1: Embedding cache?  ──HIT──► reuse vector
       │ MISS                         │
       ▼                              │
  embed_query()                       │
  (BGE model)                         │
       └──────────────────────────────┘
       │ vector ready
       ▼
  Layer 2: Vector search cache?  ──HIT──► return cached chunks
       │ MISS
       ▼
  ChromaDB ANN search
  top-K=5 chunks
       │
       ▼
  Re-rank by relevance score
       │
       ▼
  Return: {chunks, query_vec, cache_status}
  (passed to Step 6 LLM)

5.2 — Write src/retriever.py

5.3 — Run the retriever test
python src/retriever.py

Expected output:
>>> RUN 1 — Cold start (no cache)
=======================================================
  RETRIEVAL REPORT
=======================================================
  Query       : What is this document about?

  Cache status:
    ✗ llm_response         : miss
    ✗ embedding            : miss
    ✗ vector_search        : miss → stored

  Latency breakdown:
    embedding_ms           : 312.4 ms
    llm_cache_ms           : 1.2 ms
    chroma_search_ms       : 8.7 ms

  Chunks retrieved : 1
  [1] score=0.7821 | source=test_doc.txt | page=0
      This is a test document for our RAG pipeline...

>>> RUN 2 — Repeat query (cache warm)
  Cache status:
    ✗ llm_response         : miss
    ✓ embedding            : hit
    ✓ vector_search        : hit (sim=1.0000)

  Latency breakdown:
    embedding_ms           : 0.3 ms    ← cache saved 312ms
    vec_cache_ms           : 1.1 ms    ← cache saved 8.7ms

>>> RUN 3 — Similar query
  Cache status:
    ✓ vector_search        : hit (sim=0.9234)  ← above 0.90 threshold

Notice Run 2 drops from ~320ms to ~1.5ms total — that's the cache working.

'''

import os
import logging
import time
from typing import List, Dict, Tuple, Optional

import numpy as np

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    TOP_K,
    CACHE_SIM_THRESHOLD,
)
from src.embedder import embed_query, get_chroma_collection
from src.cache import CacheManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Singleton CacheManager — shared across all retrieve() calls ───────────────
_cache_manager: Optional[CacheManager] = None

def get_cache_manager() -> CacheManager:
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager


# ══════════════════════════════════════════════════════════════════════════════
# CORE RETRIEVAL FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def retrieve(
    query: str,
    top_k: int = TOP_K,
    skip_llm_cache: bool = False,
) -> Dict:
    """
    Main retrieval function called by the pipeline.

    Checks all cache layers in order before hitting ChromaDB.
    Returns everything the LLM step needs to generate an answer.

    Parameters:
        query          : raw user question string
        top_k          : number of chunks to retrieve (default from config)
        skip_llm_cache : set True to force fresh retrieval + generation
                         (useful for testing or explicit refresh)

    Returns dict with keys:
        query          : original query string
        query_vector   : numpy array (1024,) — the query embedding
        chunks         : list of top-K chunk dicts, each containing:
                           text, source, page, chunk_id, score
        cache_status   : dict showing which layers hit/missed
        latency_ms     : dict of timing per stage
        llm_cache_hit  : bool — True means skip LLM entirely
        llm_answer     : str or None — pre-cached answer if hit
        llm_sources    : list or None — pre-cached sources if hit
    """
    cm = get_cache_manager()
    timings = {}
    cache_status = {
        "llm_response":  "skip" if skip_llm_cache else "miss",
        "embedding":     "miss",
        "vector_search": "miss",
    }

    # ── STAGE 1: embed query (or use embedding cache) ─────────────────────────
    t0 = time.time()
    query_vec = cm.embedding.get(query)

    if query_vec is None:
        cache_status["embedding"] = "miss"
        query_vec = embed_query(query)          # calls BGE model
        cm.embedding.set(query, query_vec)      # store for next time
    else:
        cache_status["embedding"] = "hit"

    timings["embedding_ms"] = round((time.time() - t0) * 1000, 1)

    # ── STAGE 2: check LLM response cache ────────────────────────────────────
    if not skip_llm_cache:
        t0 = time.time()
        cached_answer, cached_sources, sim = cm.llm.get(query_vec)
        timings["llm_cache_ms"] = round((time.time() - t0) * 1000, 1)

        if cached_answer is not None:
            cache_status["llm_response"] = f"hit (sim={sim:.4f})"
            logger.info(
                f"LLM cache HIT — returning answer without retrieval or inference"
            )
            return {
                "query":         query,
                "query_vector":  query_vec,
                "chunks":        [],
                "cache_status":  cache_status,
                "latency_ms":    timings,
                "llm_cache_hit": True,
                "llm_answer":    cached_answer,
                "llm_sources":   cached_sources,
            }

    # ── STAGE 3: check vector search cache ───────────────────────────────────
    t0 = time.time()
    cached_chunks, vec_sim = cm.vector_search.get(query_vec)
    timings["vec_cache_ms"] = round((time.time() - t0) * 1000, 1)

    if cached_chunks is not None:
        cache_status["vector_search"] = f"hit (sim={vec_sim:.4f})"
        logger.info(
            f"Vector cache HIT — {len(cached_chunks)} chunks served from cache"
        )
        return {
            "query":         query,
            "query_vector":  query_vec,
            "chunks":        cached_chunks,
            "cache_status":  cache_status,
            "latency_ms":    timings,
            "llm_cache_hit": False,
            "llm_answer":    None,
            "llm_sources":   None,
        }

    # ── STAGE 4: full ChromaDB ANN search ────────────────────────────────────
    t0 = time.time()
    logger.info(f"Cache miss — running ChromaDB ANN search (top_k={top_k})")

    collection = get_chroma_collection()

    if collection.count() == 0:
        logger.error("ChromaDB is empty — run ingest.py and embedder.py first")
        return _empty_result(query, query_vec, cache_status, timings)

    raw_results = collection.query(
        query_embeddings = [query_vec.tolist()],
        n_results        = min(top_k, collection.count()),
        include          = ["documents", "metadatas", "distances"],
    )

    timings["chroma_search_ms"] = round((time.time() - t0) * 1000, 1)

    # ── STAGE 5: format + score chunks ───────────────────────────────────────
    chunks = _format_results(raw_results)

    # ── STAGE 6: store in vector search cache ────────────────────────────────
    cm.vector_search.set(query, query_vec, chunks)
    cache_status["vector_search"] = "miss → stored"

    logger.info(
        f"Retrieved {len(chunks)} chunks | "
        f"top score: {chunks[0]['score']:.4f} | "
        f"timings: {timings}"
    )

    return {
        "query":         query,
        "query_vector":  query_vec,
        "chunks":        chunks,
        "cache_status":  cache_status,
        "latency_ms":    timings,
        "llm_cache_hit": False,
        "llm_answer":    None,
        "llm_sources":   None,
    }


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def _format_results(raw_results: Dict) -> List[Dict]:
    """
    Convert raw ChromaDB query output into clean chunk dicts.

    ChromaDB returns distances (lower = more similar for cosine).
    We convert to similarity scores (higher = more similar).

    Parameters:
        raw_results : dict returned by collection.query()

    Returns:
        list of chunk dicts sorted by score descending
    """
    chunks = []

    documents = raw_results.get("documents", [[]])[0]
    metadatas = raw_results.get("metadatas", [[]])[0]
    distances = raw_results.get("distances", [[]])[0]

    for text, meta, dist in zip(documents, metadatas, distances):
        # ChromaDB cosine distance → similarity score
        # distance=0 means identical, distance=2 means opposite
        score = round(1 - (dist / 2), 4)

        chunks.append({
            "text":       text,
            "source":     meta.get("source", "unknown"),
            "page":       meta.get("page", 0),
            "char_count": meta.get("char_count", len(text)),
            "score":      score,
        })

    # Sort highest score first
    chunks.sort(key=lambda x: x["score"], reverse=True)
    return chunks


def _empty_result(
    query: str,
    query_vec: np.ndarray,
    cache_status: Dict,
    timings: Dict,
) -> Dict:
    """Return a safe empty result when ChromaDB has no documents."""
    return {
        "query":         query,
        "query_vector":  query_vec,
        "chunks":        [],
        "cache_status":  cache_status,
        "latency_ms":    timings,
        "llm_cache_hit": False,
        "llm_answer":    None,
        "llm_sources":   None,
    }


def print_retrieval_report(result: Dict) -> None:
    """
    Pretty-print a retrieval result.
    Useful during development and debugging.

    Parameters:
        result : dict returned by retrieve()
    """
    sep = "=" * 55
    print(f"\n{sep}")
    print(f"  RETRIEVAL REPORT")
    print(f"{sep}")
    print(f"  Query       : {result['query']}")
    print(f"\n  Cache status:")
    for layer, status in result["cache_status"].items():
        icon = "✓" if "hit" in str(status) else "✗"
        print(f"    {icon} {layer:<20} : {status}")

    print(f"\n  Latency breakdown:")
    for stage, ms in result["latency_ms"].items():
        print(f"    {stage:<22} : {ms} ms")

    if result["llm_cache_hit"]:
        print(f"\n  [LLM CACHE HIT] Answer served from cache:")
        print(f"  {result['llm_answer'][:300]}...")
        print(f"  Sources : {result['llm_sources']}")
    else:
        print(f"\n  Chunks retrieved : {len(result['chunks'])}")
        for i, chunk in enumerate(result["chunks"], 1):
            print(f"\n  [{i}] score={chunk['score']:.4f} | "
                  f"source={chunk['source']} | page={chunk['page']}")
            print(f"      {chunk['text'][:200]}...")
    print(f"{sep}\n")


# ── Run standalone test ────────────────────────────────────────────────────────
if __name__ == "__main__":

    # Test query — change this to something relevant to your documents
    test_query = "What is this document about?"

    print(f"Testing retriever with query: '{test_query}'")
    print("First run will load the embedding model...\n")

    # ── Run 1: cold (all cache miss) ──────────────────────────────────────────
    print(">>> RUN 1 — Cold start (no cache)")
    result1 = retrieve(test_query)
    print_retrieval_report(result1)

    # ── Run 2: repeat query (embedding + vector cache should hit) ─────────────
    print(">>> RUN 2 — Repeat query (cache warm)")
    result2 = retrieve(test_query)
    print_retrieval_report(result2)

    # ── Run 3: similar query (vector cache should still hit at sim > 0.90) ────
    similar_query = "Can you summarise what this document covers?"
    print(f">>> RUN 3 — Similar query: '{similar_query}'")
    result3 = retrieve(similar_query)
    print_retrieval_report(result3)

    # ── Cache stats ───────────────────────────────────────────────────────────
    cm = get_cache_manager()
    import pprint
    print("Cache stats after 3 runs:")
    pprint.pprint(cm.all_stats())