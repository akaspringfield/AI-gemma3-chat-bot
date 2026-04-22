# src/cache.py

'''
Author: Akash Mambally
Date : 05-01-2026

Step 4 of 8 — Multi-Layer Cache Architecture
This step builds src/cache.py — the most performance-critical module in the pipeline. All 3 cache layers live here with cosine similarity threshold logic.

4.1 — How the 3 layers work together

User Query
    │
    ▼
┌─────────────────────────────────┐
│  Layer 3 — LLM Response Cache   │  ← Check FIRST (cheapest hit)
│  "Have I answered this before?" │
│  sim > 0.90 → return instantly  │
└──────────────┬──────────────────┘
               │ MISS
               ▼
┌─────────────────────────────────┐
│  Layer 1 — Embedding Cache      │  ← Check SECOND
│  "Have I embedded this before?" │
│  exact match → skip recompute   │
└──────────────┬──────────────────┘
               │ MISS
               ▼
┌─────────────────────────────────┐
│  Layer 2 — Vector Search Cache  │  ← Check THIRD
│  "Have I retrieved this before?"│
│  sim > 0.90 → skip ANN search   │
└──────────────┬──────────────────┘
               │ MISS
               ▼
         Full Pipeline
    (embed → search → LLM)

4.2 — Write src/cache.py

4.3 — Run the cache test
python src/cache.py

Expected output:
Initialising CacheManager...
Embedding cache ready  — 0 entries
Vector search cache ready — 0 entries
LLM response cache ready  — 0 entries

--- Simulating cache write ---
--- Simulating cache read (exact query) ---
LLM cache      : HIT (sim=1.0000)
  Answer       : RAG combines retrieval with generation...
  Sources      : ['doc.pdf']
Embedding cache: HIT
Vector cache   : HIT (sim=1.0000)

--- Cache stats ---
{'embedding_cache':     {'entries': 1, 'layer': 'embedding',     'size_mb': 0.04},
 'llm_response_cache':  {'entries': 1, 'layer': 'llm_response',  'size_mb': 0.01},
 'vector_search_cache': {'entries': 1, 'layer': 'vector_search', 'size_mb': 0.04}}

Step 4 complete — all 3 cache layers operational!

Notice

| Cache           | Recommended    |
| --------------- | -------------- |
| LLM cache       | 0.85–0.9       |
| vector cache    | 0.80–0.85      |
| embedding cache | exact match OK |

'''


import os
import json
import time
import hashlib
import logging
import pickle
from typing import List, Dict, Optional, Tuple

import numpy as np
import diskcache

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    CACHE_DIR,
    CACHE_SIM_THRESHOLD,
    CACHE_TTL_SECONDS,
    EMBEDDING_DIM,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Cache directory paths ──────────────────────────────────────────────────────
EMB_CACHE_DIR  = os.path.join(CACHE_DIR, "embeddings")
VEC_CACHE_DIR  = os.path.join(CACHE_DIR, "vector_search")
LLM_CACHE_DIR  = os.path.join(CACHE_DIR, "llm_responses")

for _dir in [EMB_CACHE_DIR, VEC_CACHE_DIR, LLM_CACHE_DIR]:
    os.makedirs(_dir, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY
# ══════════════════════════════════════════════════════════════════════════════

def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    Since BGE embeddings are already L2-normalised,
    this is just a dot product — very fast.

    Parameters:
        vec_a, vec_b : 1-D numpy arrays of same length

    Returns:
        float in range [-1.0, 1.0]
        1.0 = identical, 0.0 = unrelated, -1.0 = opposite
    """
    vec_a = np.array(vec_a, dtype=np.float32)
    vec_b = np.array(vec_b, dtype=np.float32)

    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def query_fingerprint(query: str) -> str:
    """
    Create a short stable key from a query string.
    Used as the primary lookup key in all caches.

    Parameters:
        query : raw user query string

    Returns:
        16-char hex string
    """
    return hashlib.md5(query.strip().lower().encode()).hexdigest()[:16]


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 1 — EMBEDDING CACHE
# ══════════════════════════════════════════════════════════════════════════════

class EmbeddingCache:
    """
    Caches query embeddings to avoid recomputing them.

    Storage format per entry:
        key   : query_fingerprint (16-char hex)
        value : {
            "query"     : original query string,
            "embedding" : list of 1024 floats,
            "timestamp" : unix epoch float
        }

    Hit condition:
        Exact fingerprint match (same query text).
        No similarity threshold needed — embeddings are
        deterministic so exact match is always correct.
    """

    def __init__(self):
        self.cache = diskcache.Cache(
            EMB_CACHE_DIR,
            size_limit=500 * 1024 * 1024,  # 500 MB max
        )
        logger.info(
            f"Embedding cache ready — "
            f"{len(self.cache)} entries at {EMB_CACHE_DIR}"
        )

    def get(self, query: str) -> Optional[np.ndarray]:
        """
        Look up embedding for a query.

        Parameters:
            query : raw query string

        Returns:
            numpy array (1024,) if found, else None
        """
        key = query_fingerprint(query)
        entry = self.cache.get(key)

        if entry is None:
            logger.debug(f"Embedding cache MISS: '{query[:50]}'")
            return None

        # Check TTL manually
        if time.time() - entry["timestamp"] > CACHE_TTL_SECONDS:
            logger.debug(f"Embedding cache EXPIRED: '{query[:50]}'")
            self.cache.delete(key)
            return None

        logger.info(f"Embedding cache HIT: '{query[:50]}'")
        return np.array(entry["embedding"], dtype=np.float32)

    def set(self, query: str, embedding: np.ndarray) -> None:
        """
        Store an embedding for a query.

        Parameters:
            query     : raw query string
            embedding : numpy array (1024,)
        """
        key = query_fingerprint(query)
        self.cache.set(key, {
            "query":     query,
            "embedding": embedding.tolist(),
            "timestamp": time.time(),
        })
        logger.debug(f"Embedding cached: '{query[:50]}'")

    def stats(self) -> Dict:
        return {
            "layer":   "embedding",
            "entries": len(self.cache),
            "size_mb": round(self.cache.volume() / 1024 / 1024, 2),
        }

    def clear(self) -> None:
        self.cache.clear()
        logger.info("Embedding cache cleared.")


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 2 — VECTOR SEARCH CACHE
# ══════════════════════════════════════════════════════════════════════════════

class VectorSearchCache:
    """
    Caches the top-K retrieved chunks for a query embedding.
    Uses cosine similarity to match NEW queries against
    CACHED query embeddings — serves cached results if
    the new query is semantically close enough.

    Storage format per entry:
        key   : query_fingerprint
        value : {
            "query"     : original query string,
            "embedding" : list of 1024 floats,
            "chunks"    : list of retrieved chunk dicts,
            "timestamp" : unix epoch float
        }

    Hit condition:
        cosine_similarity(new_query_vec, cached_query_vec)
        >= CACHE_SIM_THRESHOLD (0.90)
    """

    def __init__(self):
        self.cache = diskcache.Cache(
            VEC_CACHE_DIR,
            size_limit=500 * 1024 * 1024,  # 500 MB max
        )
        logger.info(
            f"Vector search cache ready — "
            f"{len(self.cache)} entries at {VEC_CACHE_DIR}"
        )

    def get(
        self,
        query_embedding: np.ndarray,
    ) -> Tuple[Optional[List[Dict]], float]:
        """
        Search cache for a semantically similar past query.

        Parameters:
            query_embedding : numpy array (1024,) of current query

        Returns:
            Tuple of:
              - list of chunk dicts if hit, else None
              - best similarity score found (0.0 if no entries)
        """
        best_sim   = 0.0
        best_entry = None

        for key in self.cache.iterkeys():
            entry = self.cache.get(key)
            if entry is None:
                continue

            # Skip expired entries
            if time.time() - entry["timestamp"] > CACHE_TTL_SECONDS:
                self.cache.delete(key)
                continue

            sim = cosine_similarity(
                query_embedding,
                entry["embedding"],
            )

            if sim > best_sim:
                best_sim   = sim
                best_entry = entry

        if best_sim >= CACHE_SIM_THRESHOLD and best_entry is not None:
            logger.info(
                f"Vector search cache HIT "
                f"(sim={best_sim:.4f} >= {CACHE_SIM_THRESHOLD})"
            )
            return best_entry["chunks"], best_sim

        logger.info(
            f"Vector search cache MISS "
            f"(best sim={best_sim:.4f} < {CACHE_SIM_THRESHOLD})"
        )
        return None, best_sim

    def set(
        self,
        query: str,
        query_embedding: np.ndarray,
        chunks: List[Dict],
    ) -> None:
        """
        Store retrieved chunks for a query.

        Parameters:
            query           : raw query string
            query_embedding : numpy array (1024,)
            chunks          : top-K retrieved chunk dicts
        """
        key = query_fingerprint(query)
        self.cache.set(key, {
            "query":     query,
            "embedding": query_embedding.tolist(),
            "chunks":    chunks,
            "timestamp": time.time(),
        })
        logger.debug(f"Vector search cached: '{query[:50]}'")

    def stats(self) -> Dict:
        return {
            "layer":     "vector_search",
            "entries":   len(self.cache),
            "size_mb":   round(self.cache.volume() / 1024 / 1024, 2),
            "threshold": CACHE_SIM_THRESHOLD,
        }

    def clear(self) -> None:
        self.cache.clear()
        logger.info("Vector search cache cleared.")


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 3 — LLM RESPONSE CACHE
# ══════════════════════════════════════════════════════════════════════════════

class LLMResponseCache:
    """
    Caches final generated answers.
    This is the highest-value cache — a hit here skips
    embedding, retrieval AND LLM inference entirely.

    Storage format per entry:
        key   : query_fingerprint
        value : {
            "query"     : original query string,
            "embedding" : list of 1024 floats,
            "answer"    : full answer string,
            "sources"   : list of source filenames cited,
            "timestamp" : unix epoch float
        }

    Hit condition:
        cosine_similarity(new_query_vec, cached_query_vec)
        >= CACHE_SIM_THRESHOLD (0.90)
    """

    def __init__(self):
        self.cache = diskcache.Cache(
            LLM_CACHE_DIR,
            size_limit=1024 * 1024 * 1024,  # 1 GB max
        )
        logger.info(
            f"LLM response cache ready — "
            f"{len(self.cache)} entries at {LLM_CACHE_DIR}"
        )

    def get(
        self,
        query_embedding: np.ndarray,
    ) -> Tuple[Optional[str], Optional[List[str]], float]:
        """
        Search for a semantically equivalent cached answer.

        Parameters:
            query_embedding : numpy array (1024,) of current query

        Returns:
            Tuple of:
              - answer string if hit, else None
              - list of source filenames if hit, else None
              - best similarity score found
        """
        best_sim   = 0.0
        best_entry = None

        for key in self.cache.iterkeys():
            entry = self.cache.get(key)
            if entry is None:
                continue

            # Skip expired
            if time.time() - entry["timestamp"] > CACHE_TTL_SECONDS:
                self.cache.delete(key)
                continue

            sim = cosine_similarity(
                query_embedding,
                entry["embedding"],
            )

            if sim > best_sim:
                best_sim   = sim
                best_entry = entry

        if best_sim >= CACHE_SIM_THRESHOLD and best_entry is not None:
            logger.info(
                f"LLM cache HIT "
                f"(sim={best_sim:.4f} >= {CACHE_SIM_THRESHOLD})"
            )
            return best_entry["answer"], best_entry["sources"], best_sim

        logger.info(
            f"LLM cache MISS "
            f"(best sim={best_sim:.4f} < {CACHE_SIM_THRESHOLD})"
        )
        return None, None, best_sim

    def set(
        self,
        query: str,
        query_embedding: np.ndarray,
        answer: str,
        sources: List[str],
    ) -> None:
        """
        Store a generated answer for a query.

        Parameters:
            query           : raw query string
            query_embedding : numpy array (1024,)
            answer          : full generated answer text
            sources         : list of source filenames cited
        """
        key = query_fingerprint(query)
        self.cache.set(key, {
            "query":     query,
            "embedding": query_embedding.tolist(),
            "answer":    answer,
            "sources":   sources,
            "timestamp": time.time(),
        })
        logger.info(f"LLM response cached: '{query[:50]}'")

    def stats(self) -> Dict:
        return {
            "layer":     "llm_response",
            "entries":   len(self.cache),
            "size_mb":   round(self.cache.volume() / 1024 / 1024, 2),
            "threshold": CACHE_SIM_THRESHOLD,
        }

    def clear(self) -> None:
        self.cache.clear()
        logger.info("LLM response cache cleared.")


# ══════════════════════════════════════════════════════════════════════════════
# CACHE MANAGER — single interface for the full pipeline
# ══════════════════════════════════════════════════════════════════════════════

class CacheManager:
    """
    Unified interface to all 3 cache layers.
    The pipeline imports only this class.

    Usage:
        cm = CacheManager()

        # Check LLM cache first
        answer, sources, sim = cm.llm.get(query_vec)
        if answer:
            return answer   # skip everything else

        # Check embedding cache
        vec = cm.embedding.get(query)
        if vec is None:
            vec = embed_query(query)
            cm.embedding.set(query, vec)

        # Check vector search cache
        chunks, sim = cm.vector_search.get(vec)
        if chunks is None:
            chunks = chroma_search(vec)
            cm.vector_search.set(query, vec, chunks)

        # Generate answer
        answer = llm.generate(query, chunks)
        cm.llm.set(query, vec, answer, sources)
    """

    def __init__(self):
        self.embedding     = EmbeddingCache()
        self.vector_search = VectorSearchCache()
        self.llm           = LLMResponseCache()

    def all_stats(self) -> Dict:
        """Print stats for all 3 layers."""
        return {
            "embedding_cache":     self.embedding.stats(),
            "vector_search_cache": self.vector_search.stats(),
            "llm_response_cache":  self.llm.stats(),
        }

    def clear_all(self) -> None:
        """Wipe all 3 caches — useful for testing."""
        self.embedding.clear()
        self.vector_search.clear()
        self.llm.clear()
        logger.info("All caches cleared.")


# ── Run standalone test ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import pprint

    print("\nInitialising CacheManager...")
    cm = CacheManager()

    # ── Simulate a cache write + read cycle ───────────────────────────────────
    print("\n--- Simulating cache write ---")
    fake_query   = "What is retrieval augmented generation?"
    fake_vec     = np.random.randn(1024).astype(np.float32)
    # Normalise — mimics real BGE output
    fake_vec    /= np.linalg.norm(fake_vec)
    fake_chunks  = [{"chunk_id": "abc123", "text": "RAG is...", "source": "doc.pdf"}]
    fake_answer  = "RAG combines retrieval with generation to answer questions."

    # Write to all 3 layers
    cm.embedding.set(fake_query, fake_vec)
    cm.vector_search.set(fake_query, fake_vec, fake_chunks)
    cm.llm.set(fake_query, fake_vec, fake_answer, ["doc.pdf"])

    print("\n--- Simulating cache read (exact query) ---")
    # Layer 3 check
    answer, sources, sim = cm.llm.get(fake_vec)
    print(f"LLM cache     : {'HIT' if answer else 'MISS'} (sim={sim:.4f})")
    if answer:
        print(f"  Answer      : {answer}")
        print(f"  Sources     : {sources}")

    # Layer 1 check
    vec = cm.embedding.get(fake_query)
    print(f"Embedding cache: {'HIT' if vec is not None else 'MISS'}")

    # Layer 2 check
    chunks, sim = cm.vector_search.get(fake_vec)
    print(f"Vector cache  : {'HIT' if chunks else 'MISS'} (sim={sim:.4f})")

    print("\n--- Cache stats ---")
    pprint.pprint(cm.all_stats())

    print("\nStep 4 complete — all 3 cache layers operational!")