# src/ingest.py

'''
Author: Akash Mambally
Date : 04-01-2026


Step 2 of 8 — Data Ingestion & Chunking
This step builds src/ingest.py — the module that loads your raw 
documents (PDF, DOCX, TXT), splits them into chunks, 
and saves them ready for embedding.

2.1 — How chunking works in this pipeline

Raw Document (e.g. 10-page PDF)
        │
        ▼
 RecursiveCharacterTextSplitter
  chunk_size=512, overlap=64
        │
        ▼
 [chunk_1, chunk_2, chunk_3 ... chunk_N]
  each with metadata:
  {source, page, chunk_id, char_count}
        │
        ▼
 Saved to data/processed/
 (ready for Step 3 — Embedding)

2.2 — Code

2.3 — Add a test document
Before running, drop any PDF, DOCX or TXT file into data/raw/. If you don't have one handy, create a quick test file:
bash# Windows
echo This is a test document for our RAG pipeline. It contains sample text about artificial intelligence and machine learning. We will use this to verify our chunking pipeline works correctly before moving to embeddings. > data\raw\test_doc.txt


Expected output:
INFO — Processing: test_doc.txt
INFO — Loaded 1 page(s) from: test_doc.txt
INFO — Saved 1 chunks → data/processed/test_doc_chunks.json
INFO — Ingestion complete — 1 total chunks from 1 file(s)

==================================================
Total chunks : 1
==================================================

Sample chunk #1:
  chunk_id  : a3f1c2d4e5b6f7a8
  source    : test_doc.txt
  page      : 0
  char_count: 218
  text      : This is a test document...

2.5 — Verify the output file
# Windows
type data\processed\test_doc_chunks.json

# Linux/Mac
cat data/processed/test_doc_chunks.json

You should see clean JSON like:

[
  {
    "chunk_id": "a3f1c2d4e5b6f7a8",
    "text": "This is a test document for our RAG pipeline...",
    "source": "test_doc.txt",
    "page": 0,
    "char_count": 218
  }
]
  
'''

import os
import json
import hashlib
import logging
from pathlib import Path
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    Docx2txtLoader,
    TextLoader,
)

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    RAW_DATA_DIR,
    PROCESSED_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Supported file types ───────────────────────────────────────────────────────
LOADER_MAP = {
    ".pdf":  PyMuPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt":  TextLoader,
    ".md":   TextLoader,
}


def get_file_hash(filepath: str) -> str:
    """
    MD5 hash of file contents.
    Used to skip re-processing unchanged files.
    """
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def load_document(filepath: str) -> List:
    """
    Load a single document using the correct loader
    based on file extension.

    Returns a list of LangChain Document objects.
    """
    ext = Path(filepath).suffix.lower()

    if ext not in LOADER_MAP:
        logger.warning(f"Unsupported file type: {ext} — skipping {filepath}")
        return []

    loader_class = LOADER_MAP[ext]
    loader = loader_class(filepath)

    try:
        docs = loader.load()
        logger.info(f"Loaded {len(docs)} page(s) from: {Path(filepath).name}")
        return docs
    except Exception as e:
        logger.error(f"Failed to load {filepath}: {e}")
        return []


def chunk_documents(docs: List) -> List[Dict]:
    """
    Split loaded documents into chunks using
    RecursiveCharacterTextSplitter.

    Parameters:
        docs : list of LangChain Document objects

    Returns:
        list of dicts, each with keys:
          chunk_id   — unique hash id for this chunk
          text       — the chunk text
          source     — original filename
          page       — page number (if available)
          char_count — length of chunk text
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,          # 512 chars per chunk
        chunk_overlap=CHUNK_OVERLAP,    # 64 chars overlap (~12%)
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for doc in docs:
        splits = splitter.split_text(doc.page_content)

        for i, text in enumerate(splits):
            # Skip empty or whitespace-only chunks
            if not text.strip():
                continue

            # Unique ID = hash of (source + text)
            source = doc.metadata.get("source", "unknown")
            chunk_id = hashlib.md5(
                f"{source}_{i}_{text[:50]}".encode()
            ).hexdigest()[:16]

            chunks.append({
                "chunk_id":   chunk_id,
                "text":       text.strip(),
                "source":     Path(source).name,
                "page":       doc.metadata.get("page", 0),
                "char_count": len(text.strip()),
            })

    return chunks


def save_chunks(chunks: List[Dict], output_filename: str) -> str:
    """
    Save chunks to a JSON file in data/processed/.

    Parameters:
        chunks          : list of chunk dicts
        output_filename : name for the output JSON file

    Returns:
        full path to saved file
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    output_path = os.path.join(PROCESSED_DIR, output_filename)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(chunks)} chunks → {output_path}")
    return output_path


def ingest_all(data_dir: str = None) -> List[Dict]:
    """
    Main entry point.
    Scans data/raw/, loads all supported files,
    chunks them, saves to data/processed/,
    and returns the full chunk list.

    Parameters:
        data_dir : path to scan (defaults to RAW_DATA_DIR from config)

    Returns:
        all_chunks : flat list of all chunk dicts across all files
    """
    data_dir = data_dir or RAW_DATA_DIR
    all_chunks = []
    processed_log = os.path.join(PROCESSED_DIR, "processed_files.json")

    # Load record of already-processed files (skip unchanged)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    if os.path.exists(processed_log):
        with open(processed_log) as f:
            processed_files = json.load(f)
    else:
        processed_files = {}

    files = [
        f for f in Path(data_dir).iterdir()
        if f.suffix.lower() in LOADER_MAP
    ]

    if not files:
        logger.warning(f"No supported files found in {data_dir}")
        logger.warning("Drop PDF, DOCX or TXT files into data/raw/ and re-run.")
        return []

    for filepath in files:
        file_hash = get_file_hash(str(filepath))

        # Skip if file hasn't changed since last run
        if processed_files.get(filepath.name) == file_hash:
            logger.info(f"Skipping unchanged file: {filepath.name}")

            # Load existing chunks from processed dir
            chunk_file = os.path.join(
                PROCESSED_DIR, filepath.stem + "_chunks.json"
            )
            if os.path.exists(chunk_file):
                with open(chunk_file) as f:
                    all_chunks.extend(json.load(f))
            continue

        # Load → chunk → save
        logger.info(f"Processing: {filepath.name}")
        docs   = load_document(str(filepath))
        chunks = chunk_documents(docs)

        if chunks:
            save_chunks(chunks, filepath.stem + "_chunks.json")
            all_chunks.extend(chunks)

            # Record file hash so we skip it next run
            processed_files[filepath.name] = file_hash

    # Update processed files log
    with open(processed_log, "w") as f:
        json.dump(processed_files, f, indent=2)

    logger.info(
        f"Ingestion complete — {len(all_chunks)} total chunks "
        f"from {len(files)} file(s)"
    )
    return all_chunks


# ── Quick test when run directly ───────────────────────────────────────────────
if __name__ == "__main__":
    chunks = ingest_all()

    if chunks:
        print(f"\n{'='*50}")
        print(f"Total chunks : {len(chunks)}")
        print(f"{'='*50}")
        print(f"\nSample chunk #1:")
        print(f"  chunk_id  : {chunks[0]['chunk_id']}")
        print(f"  source    : {chunks[0]['source']}")
        print(f"  page      : {chunks[0]['page']}")
        print(f"  char_count: {chunks[0]['char_count']}")
        print(f"  text      : {chunks[0]['text'][:200]}...")