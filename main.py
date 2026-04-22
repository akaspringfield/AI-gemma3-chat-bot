'''
Author: Akash Mambally
Date : 04-01-2026

Step 8 of 8 — main.py Interactive CLI

This is the final step — the user-facing interface that ties everything together into a clean terminal application.

8.1 — What the CLI supports

Commands available at the prompt:
───────────────────────────────────────────────
  ask       → type any question (default mode)
  /ingest   → scan data/raw/ and index new docs
  /status   → show pipeline health + doc count
  /cache    → show all 3 cache layer stats
  /clear    → wipe all caches
  /history  → show questions asked this session
  /report   → detailed report of last answer
  /help     → show command list
  /quit     → exit
───────────────────────────────────────────────

8.2 — Write main.py

8.3 — Run the CLI
python main.py

Full session walkthrough:

╔══════════════════════════════════════════════════════╗
║          LOCAL RAG PIPELINE  —  Gemma 3 4B           ║
║      ChromaDB  ·  BGE Embeddings  ·  3-Layer Cache   ║
╚══════════════════════════════════════════════════════╝

  ✓ Ready — 1 chunks indexed | Gemma 3 4B loaded

  Available commands:
  <question>   Ask anything about your documents
  /ingest      Scan data/raw/ and index new documents
  /status      Pipeline health, doc count, model info
  /cache       Show all 3 cache layer statistics
  ...

  Ask > What is this document about?

  ───────────────────────────────────────────────────
  Gemma 3 Answer (streaming)
  ───────────────────────────────────────────────────
  This document is about a test RAG pipeline for
  artificial intelligence and machine learning.
  [source: test_doc.txt, page: 0]
  ───────────────────────────────────────────────────

  Cache  : Fresh generation
  Sources: test_doc.txt
  Latency: 9823 ms

  Ask > What is this document about?

  ───────────────────────────────────────────────────
  [CACHE] Answer retrieved instantly
  ───────────────────────────────────────────────────
  This document is about a test RAG pipeline...
  ───────────────────────────────────────────────────

  Cache  : LLM cache  (fastest)
  Latency: 3.1 ms

  Ask > /status
  Pipeline Status
  ────────────────────────────────────────
  ChromaDB documents : 1
  LLM model          : ✓ ready
  ...

  Ask > /ingest
  Scanning data/raw/ for new documents...
  ✓ Ingested  : 1 total chunks
  ✓ Indexed   : 0 new chunks stored  ← already indexed

  Ask > /history
  Session History (2 queries)
  ──────────────────────────────────────────────────
  1. [fresh]    9823ms  What is this document about?
  2. [llm]      3.1ms   What is this document about?

  Ask > /quit
  Goodbye!


8.4 — Add real documents and test
Now drop real documents into data/raw/ and run the full flow:
# 1. Copy your PDFs/DOCX/TXT into data/raw/
# 2. Start the CLI
python main.py

# 3. Inside the CLI — index your documents
Ask > /ingest

# 4. Ask questions
Ask > What are the main topics covered in the documents?
Ask > Summarise the key findings from the report.
Ask > Who are the authors mentioned?

8.5 — GPU server upgrade checklist

When you deploy to your server, only config.py changes:
# config.py — server settings (3 lines change)
DEVICE           = "cuda"    # was "cpu"
LLM_N_GPU_LAYERS = 35        # was 0
LLM_N_THREADS    = 16        # match your server core count


Then reinstall llama-cpp-python with CUDA:
CMAKE_ARGS="-DLLAMA_CUDA=on" \
pip install llama-cpp-python --upgrade --force-reinstall

'''

# main.py

import os
import sys
import time
import logging
from typing import List, Optional
from datetime import datetime

# Suppress noisy logs in CLI mode — only show warnings+
logging.basicConfig(level=logging.WARNING)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.pipeline import RAGPipeline, PipelineResult


# ══════════════════════════════════════════════════════════════════════════════
# CLI HELPERS
# ══════════════════════════════════════════════════════════════════════════════

CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

def c(text: str, colour: str) -> str:
    """Wrap text in ANSI colour code."""
    return f"{colour}{text}{RESET}"


def print_banner():
    print(c("""
╔══════════════════════════════════════════════════════╗
║          LOCAL RAG PIPELINE  —  Gemma 3 4B           ║
║      ChromaDB  ·  BGE Embeddings  ·  3-Layer Cache   ║
╚══════════════════════════════════════════════════════╝
""", CYAN))


def print_help():
    print(c("\n  Available commands:", BOLD))
    cmds = [
        ("  <question>", "Ask anything about your documents"),
        ("  /ingest   ", "Scan data/raw/ and index new documents"),
        ("  /status   ", "Pipeline health, doc count, model info"),
        ("  /cache    ", "Show all 3 cache layer statistics"),
        ("  /clear    ", "Wipe all caches (forces fresh answers)"),
        ("  /history  ", "Show questions asked this session"),
        ("  /report   ", "Detailed report of the last answer"),
        ("  /help     ", "Show this command list"),
        ("  /quit     ", "Exit the application"),
    ]
    for cmd, desc in cmds:
        print(f"  {c(cmd, CYAN)}  {c(desc, DIM)}")
    print()


def print_status(pipeline: RAGPipeline):
    status = pipeline.status()
    print(c("\n  Pipeline Status", BOLD))
    print(f"  {'─'*40}")

    # ChromaDB
    doc_count = status["chromadb_docs"]
    doc_colour = GREEN if doc_count > 0 else RED
    print(f"  ChromaDB documents : {c(str(doc_count), doc_colour)}")

    # LLM model
    model_ready = status["llm_model_ready"]
    model_icon  = c("✓ ready", GREEN) if model_ready else c("✗ not found", RED)
    print(f"  LLM model          : {model_icon}")
    print(f"  Model path         : {c(status['llm_model_path'], DIM)}")

    # Cache stats
    print(f"\n  {'─'*40}")
    print(c("  Cache layers:", BOLD))
    for layer, stats in status["cache_stats"].items():
        entries = stats["entries"]
        mb      = stats["size_mb"]
        print(
            f"  {layer:<26} "
            f"{c(str(entries) + ' entries', CYAN)}  "
            f"{c(str(mb) + ' MB', DIM)}"
        )
    print()


def print_cache_stats(pipeline: RAGPipeline):
    stats = pipeline.cache_stats()
    print(c("\n  Cache Statistics", BOLD))
    print(f"  {'─'*50}")
    for layer, data in stats.items():
        print(f"\n  {c(layer, CYAN)}")
        for k, v in data.items():
            print(f"    {k:<16} : {v}")
    print()


def print_history(history: List[dict]):
    if not history:
        print(c("  No questions asked yet this session.\n", DIM))
        return

    print(c(f"\n  Session History ({len(history)} queries)", BOLD))
    print(f"  {'─'*50}")
    for i, entry in enumerate(history, 1):
        hit   = c(f"[{entry['cache_layer']}]", GREEN) \
                if entry["cache_layer"] != "none" \
                else c("[fresh]", YELLOW)
        print(
            f"  {c(str(i), DIM)}. {hit} "
            f"{c(str(entry['total_ms']) + 'ms', DIM)}  "
            f"{entry['query'][:70]}"
        )
    print()


def print_last_report(last_result: Optional[PipelineResult]):
    if last_result is None:
        print(c("  No answer yet this session.\n", DIM))
        return
    last_result.print_report()


def run_ingest(pipeline: RAGPipeline):
    print(c("\n  Scanning data/raw/ for new documents...", YELLOW))
    result = pipeline.ingest_and_index()

    if result["status"] == "no_files":
        print(c(
            "  No supported files found in data/raw/\n"
            "  Drop PDF, DOCX, or TXT files there and try again.\n",
            RED
        ))
        return

    print(
        f"  {c('✓', GREEN)} Ingested  : "
        f"{c(str(result['chunks_total']), CYAN)} total chunks"
    )
    print(
        f"  {c('✓', GREEN)} Indexed   : "
        f"{c(str(result['chunks_stored']), CYAN)} new chunks stored"
    )
    print(
        f"  {c('✓', GREEN)} Timings   : "
        f"ingest {result['ingest_ms']}ms  |  "
        f"index {result['index_ms']}ms\n"
    )


def format_sources(sources: List[str]) -> str:
    if not sources:
        return c("none", DIM)
    return "  ".join(c(s, CYAN) for s in sources)


def format_cache_hit(layer: str) -> str:
    if layer == "llm":
        return c("LLM cache  (fastest)", GREEN)
    elif layer == "vector_search":
        return c("Vector cache", GREEN)
    elif layer == "embedding":
        return c("Embedding cache", GREEN)
    else:
        return c("Fresh generation", YELLOW)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print_banner()

    # ── Init pipeline ─────────────────────────────────────────────────────────
    print(c("  Initialising pipeline...", DIM))
    try:
        pipeline = RAGPipeline()
    except Exception as e:
        print(c(f"\n  Failed to initialise pipeline:\n  {e}\n", RED))
        sys.exit(1)

    # ── Quick startup status ──────────────────────────────────────────────────
    status     = pipeline.status()
    doc_count  = status["chromadb_docs"]
    model_ready= status["llm_model_ready"]

    if not model_ready:
        print(c(
            f"  ✗ LLM model not found at:\n"
            f"    {status['llm_model_path']}\n"
            f"  Download it first — see Step 6 instructions.\n",
            RED
        ))
        sys.exit(1)

    if doc_count == 0:
        print(c(
            "  ✗ No documents indexed yet.\n"
            "  Drop files into data/raw/ then type /ingest\n",
            YELLOW
        ))
    else:
        print(c(
            f"  ✓ Ready — {doc_count} chunks indexed | "
            f"Gemma 3 4B loaded\n",
            GREEN
        ))

    print_help()

    # ── Session state ─────────────────────────────────────────────────────────
    session_history : List[dict]             = []
    last_result     : Optional[PipelineResult] = None

    # ── Main input loop ───────────────────────────────────────────────────────
    while True:
        try:
            raw = input(c("  Ask > ", BOLD + CYAN)).strip()
        except (EOFError, KeyboardInterrupt):
            print(c("\n\n  Goodbye!\n", CYAN))
            break

        if not raw:
            continue

        # ── Commands ──────────────────────────────────────────────────────────
        cmd = raw.lower()

        if cmd in ("/quit", "/exit", "/q"):
            print(c("\n  Goodbye!\n", CYAN))
            break

        elif cmd == "/help":
            print_help()

        elif cmd == "/status":
            print_status(pipeline)

        elif cmd == "/cache":
            print_cache_stats(pipeline)

        elif cmd == "/clear":
            pipeline.clear_cache()
            print(c("  ✓ All caches cleared.\n", GREEN))

        elif cmd == "/history":
            print_history(session_history)

        elif cmd == "/report":
            print_last_report(last_result)

        elif cmd == "/ingest":
            run_ingest(pipeline)

        elif raw.startswith("/"):
            print(c(f"  Unknown command: {raw}  (type /help)\n", RED))

        # ── Question → pipeline ───────────────────────────────────────────────
        else:
            print()
            try:
                result = pipeline.ask(
                    query        = raw,
                    top_k        = 5,
                    print_stream = True,
                    skip_cache   = False,
                )

                # Post-answer summary line
                print(
                    f"\n  {c('Cache', DIM)}  : {format_cache_hit(result.cache_hit_layer)}\n"
                    f"  {c('Sources', DIM)}: {format_sources(result.sources)}\n"
                    f"  {c('Latency', DIM)}: {c(str(result.total_ms) + ' ms', DIM)}\n"
                )

                # Store for /history and /report
                last_result = result
                session_history.append({
                    "query"      : raw,
                    "cache_layer": result.cache_hit_layer,
                    "total_ms"   : result.total_ms,
                    "timestamp"  : datetime.now().strftime("%H:%M:%S"),
                })

            except FileNotFoundError as e:
                print(c(f"\n  Model not found:\n  {e}\n", RED))

            except Exception as e:
                print(c(f"\n  Error during generation:\n  {e}\n", RED))
                import traceback
                traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()
