# src/embedder.py

'''
Author: Akash Mambally
Date : 04-01-2026

Step 3 of 8 — Embedding & ChromaDB Storage
This step builds src/embedder.py — the module that takes your chunks 
from Step 2, converts them into vectors using BAAI/bge-large-en-v1.5, 
and stores them in ChromaDB with full metadata.

data/processed/*.json  (chunks from Step 2)
          │
          ▼
  BAAI/bge-large-en-v1.5
  (downloads ~1.3GB on first run)
          │
          ▼
  1024-dimensional vectors
          │
          ▼
  ChromaDB (vectorstore/)
  stored with metadata:
  {chunk_id, source, page, char_count}
          │
          ▼
  Ready for semantic search in Step 4

3.2 — Write src/embedder.py
3.3 — Run the embedder
What happens on first run:
INFO — Loading embedding model: BAAI/bge-large-en-v1.5
INFO — First run will download ~1.3GB — please wait...
INFO — Embedding model loaded successfully.
INFO — ChromaDB collection 'rag_docs' ready. Documents in store: 0
INFO — Embedding 1 new chunks...
INFO — Batch 1: embedded 1 chunks in 2.3s

==================================================
Status        : ok
Total in DB   : 1
Top result    : This is a test document for our RAG pipeline...
From source   : test_doc.txt
Distance      : 0.0821
==================================================

Step 3 complete — ChromaDB is ready for retrieval!

3.4 — Verify ChromaDB files on disk
# Windows
dir vectorstore\

# Linux/Mac
ls -lh vectorstore/

You should see ChromaDB's storage files:

vectorstore/
├── chroma.sqlite3      ← metadata index
└── <uuid>/
    ├── data_level0.bin ← HNSW vector index
    └── header.bin

    
3.5 — Important notes for when you add real documents
When you add real PDFs to data/raw/ and re-run the full pipeline:

# Run both steps together
python src/ingest.py && python src/embedder.py
Both scripts are idempotent — safe to re-run anytime:

Ingest skips unchanged files (MD5 hash check)
Embedder skips already-stored chunk IDs

'''
import os
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Tuple

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
    DEVICE,
    VECTORSTORE_DIR,
    CHROMA_COLLECTION,
    PROCESSED_DIR,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Singleton pattern — load model once, reuse everywhere ─────────────────────
_embedding_model = None

def get_embedding_model() -> SentenceTransformer:
    """
    Load the embedding model once and cache it in memory.
    Downloads ~1.3GB on first run, then loads from cache.

    Returns:
        SentenceTransformer model ready for inference
    """
    global _embedding_model
    if _embedding_model is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        logger.info("First run will download ~1.3GB — please wait...")
        _embedding_model = SentenceTransformer(
            EMBEDDING_MODEL,
            device=DEVICE,          # "cpu" now, "cuda" on server
        )
        logger.info("Embedding model loaded successfully.")
    return _embedding_model


def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Convert a list of text strings into embedding vectors.

    Parameters:
        texts : list of strings to embed

    Returns:
        numpy array of shape (len(texts), EMBEDDING_DIM)
        i.e. (N, 1024) for bge-large-en-v1.5

    Notes:
        - normalize_embeddings=True makes cosine similarity
          equivalent to dot product — faster cache comparison
        - batch_size=32 balances RAM vs speed on CPU
        - show_progress_bar=True for visibility on large batches
    """
    model = get_embedding_model()

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,   # critical for cosine similarity
        batch_size=32,               # reduce to 8 if RAM is tight
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return embeddings


def embed_query(query: str) -> np.ndarray:
    """
    Embed a single user query.
    BGE models need a special prefix for query embedding
    (different from document embedding).

    Parameters:
        query : raw user question string

    Returns:
        numpy array of shape (1024,)
    """
    model = get_embedding_model()

    # BGE-specific instruction prefix for queries
    # This is NOT needed for documents, only queries
    prefixed_query = f"Represent this sentence for searching relevant passages: {query}"

    embedding = model.encode(
        prefixed_query,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embedding


# ── ChromaDB client — singleton ───────────────────────────────────────────────
_chroma_client = None
_chroma_collection = None

def get_chroma_collection():
    """
    Get or create the ChromaDB collection.
    Persists to disk at vectorstore/.

    Returns:
        ChromaDB collection object
    """
    global _chroma_client, _chroma_collection

    if _chroma_collection is None:
        os.makedirs(VECTORSTORE_DIR, exist_ok=True)

        _chroma_client = chromadb.PersistentClient(
            path=VECTORSTORE_DIR,
            settings=Settings(
                anonymized_telemetry=False,  # no data sent to Chroma
            ),
        )

        # get_or_create — safe to call multiple times
        _chroma_collection = _chroma_client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            metadata={
                "hnsw:space": "cosine",      # cosine similarity for search
                "hnsw:construction_ef": 200, # higher = better index quality
                "hnsw:M": 16,                # connections per node
            },
        )
        logger.info(
            f"ChromaDB collection '{CHROMA_COLLECTION}' ready. "
            f"Documents in store: {_chroma_collection.count()}"
        )

    return _chroma_collection


def store_chunks(chunks: List[Dict]) -> int:
    """
    Embed all chunks and store them in ChromaDB.
    Skips chunks that are already stored (by chunk_id).

    Parameters:
        chunks : list of chunk dicts from ingest.py
                 each must have: chunk_id, text, source, page, char_count

    Returns:
        int : number of NEW chunks actually stored
    """
    collection = get_chroma_collection()

    # Find which chunk_ids are already in the store
    existing_ids = set()
    if collection.count() > 0:
        existing = collection.get(include=[])  # just IDs, no vectors
        existing_ids = set(existing["ids"])
        logger.info(f"Found {len(existing_ids)} existing chunks in ChromaDB")

    # Filter to only new chunks
    new_chunks = [c for c in chunks if c["chunk_id"] not in existing_ids]

    if not new_chunks:
        logger.info("All chunks already stored — nothing to add.")
        return 0

    logger.info(f"Embedding {len(new_chunks)} new chunks...")

    # Embed in batches to avoid OOM on CPU
    batch_size = 32
    total_stored = 0

    for i in range(0, len(new_chunks), batch_size):
        batch = new_chunks[i : i + batch_size]
        texts = [c["text"] for c in batch]

        start = time.time()
        embeddings = embed_texts(texts)
        elapsed = time.time() - start

        logger.info(
            f"Batch {i//batch_size + 1}: "
            f"embedded {len(batch)} chunks in {elapsed:.1f}s"
        )

        # ChromaDB expects lists, not numpy arrays
        collection.add(
            ids        = [c["chunk_id"] for c in batch],
            embeddings = embeddings.tolist(),
            documents  = texts,
            metadatas  = [
                {
                    "source":     c["source"],
                    "page":       int(c["page"]),
                    "char_count": int(c["char_count"]),
                }
                for c in batch
            ],
        )
        total_stored += len(batch)

    logger.info(
        f"Stored {total_stored} new chunks. "
        f"Total in ChromaDB: {collection.count()}"
    )
    return total_stored


def load_all_chunks_from_processed() -> List[Dict]:
    """
    Load all chunk JSON files from data/processed/.
    Used when running embedder standalone.

    Returns:
        flat list of all chunk dicts
    """
    all_chunks = []
    chunk_files = list(Path(PROCESSED_DIR).glob("*_chunks.json"))

    if not chunk_files:
        logger.warning(f"No chunk files found in {PROCESSED_DIR}")
        logger.warning("Run src/ingest.py first.")
        return []

    for chunk_file in chunk_files:
        with open(chunk_file, encoding="utf-8") as f:
            chunks = json.load(f)
            all_chunks.extend(chunks)
            logger.info(f"Loaded {len(chunks)} chunks from {chunk_file.name}")

    return all_chunks


def verify_storage(sample_query: str = "test") -> Dict:
    """
    Quick sanity check — embed a query and retrieve
    the closest chunk to confirm everything works.

    Parameters:
        sample_query : any short string to test with

    Returns:
        dict with verification results
    """
    collection = get_chroma_collection()

    if collection.count() == 0:
        return {"status": "empty", "message": "No chunks in ChromaDB yet."}

    query_vec = embed_query(sample_query)

    results = collection.query(
        query_embeddings=[query_vec.tolist()],
        n_results=1,
        include=["documents", "metadatas", "distances"],
    )

    return {
        "status":      "ok",
        "total_stored": collection.count(),
        "sample_query": sample_query,
        "top_result":  results["documents"][0][0][:200],
        "source":      results["metadatas"][0][0]["source"],
        "distance":    round(results["distances"][0][0], 4),
    }


# ── Run standalone ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Step 1: load chunks from processed dir
    chunks = load_all_chunks_from_processed()

    if not chunks:
        print("No chunks to embed. Run src/ingest.py first.")
        exit(1)

    print(f"\nTotal chunks to process : {len(chunks)}")

    # Step 2: embed + store
    stored = store_chunks(chunks)
    print(f"New chunks stored       : {stored}")

    # Step 3: verify
    print("\nRunning verification query...")
    result = verify_storage("What is this document about?")

    print(f"\n{'='*50}")
    print(f"Status        : {result['status']}")
    print(f"Total in DB   : {result['total_stored']}")
    print(f"Top result    : {result['top_result']}...")
    print(f"From source   : {result['source']}")
    print(f"Distance      : {result['distance']}")
    print(f"{'='*50}")
    print("\nStep 3 complete — ChromaDB is ready for retrieval!")