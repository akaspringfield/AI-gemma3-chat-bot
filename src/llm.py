# src/llm.py

'''
Author: Akash Mambally
Date : 06-01-2026

Step 6 of 8 — LLM Integration
This step builds src/llm.py — loads Gemma 3 4B, 
builds the RAG system prompt with citation + hallucination-prevention rules, 
and streams the answer token by token.

6.1 — How this step fits in the pipeline

retrieve() result dict  (from Step 5)
          │
          ▼
   Build context block
   chunk_1 [source: doc.pdf, page: 2]
   chunk_2 [source: doc.pdf, page: 5]
   ...chunk_N
          │
          ▼
   Gemma 3 prompt template
   <start_of_turn>system
     - Answer ONLY from context
     - Cite source + page
     - Say "I don't know" if missing
   <start_of_turn>user
     {query}
   <start_of_turn>model
          │
          ▼
   Gemma 3 4B (llama-cpp-python)
   streaming tokens → printed live
          │
          ▼
   Full answer assembled
          │
          ▼
   Store in LLM response cache
          │
          ▼
   Return {answer, sources, tokens, latency}

6.2 — Write src/llm.py

6.3 — Run the LLM test
python src/llm.py

Expected output:

>>> TEST 1 — Prompt builder (no model load)
<bos><start_of_turn>user
You are a precise document assistant.
...
--- CONTEXT PASSAGES ---
[1] Source: rag_intro.pdf | Page: 1 | Relevance: 0.91
RAG stands for Retrieval Augmented Generation...
--- END OF CONTEXT ---

Question: What is RAG?
<end_of_turn>
<start_of_turn>model

Estimated prompt tokens: 187

>>> TEST 2 — Full RAG generation with Gemma 3
Loading model (20-60 seconds on CPU)...

───────────────────────────────────────────────────────
  Gemma 3 Answer (streaming)
───────────────────────────────────────────────────────
This document is about a test RAG pipeline for
artificial intelligence and machine learning.
[source: test_doc.txt, page: 0]
───────────────────────────────────────────────────────

  Sources       : ['test_doc.txt']
  Prompt tokens : 142
  Answer tokens : 38
  Latency       : 8420 ms

Step 6 complete — Gemma 3 is generating cited answers!   
   
'''

import os
import sys
import time
import logging
from typing import List, Dict, Optional, Generator

from llama_cpp import Llama

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    LLM_MODEL_PATH,
    LLM_CONTEXT_LEN,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    LLM_TOP_K,
    LLM_TOP_P,
    LLM_MIN_P,
    LLM_REPEAT_PENALTY,
    LLM_N_THREADS,
    LLM_N_GPU_LAYERS,
    LLM_STREAM,
    TOP_K,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADER — singleton, loads once
# ══════════════════════════════════════════════════════════════════════════════

_llm: Optional[Llama] = None

def get_llm() -> Llama:
    """
    Load Gemma 3 4B GGUF model once and reuse.
    First call takes 20-60 seconds on CPU.
    Subsequent calls return the cached instance instantly.

    Returns:
        Llama instance ready for inference
    """
    global _llm
    if _llm is None:
        if not os.path.exists(LLM_MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at: {LLM_MODEL_PATH}\n"
                f"Download it by running:\n"
                f"  python -c \"from huggingface_hub import hf_hub_download; "
                f"hf_hub_download(repo_id='unsloth/gemma-3-4b-it-GGUF', "
                f"filename='gemma-3-4b-it-Q4_K_M.gguf', local_dir='./models')\""
            )

        logger.info(f"Loading Gemma 3 4B from: {LLM_MODEL_PATH}")
        logger.info("This takes 20-60 seconds on first load — please wait...")

        _llm = Llama(
            model_path    = LLM_MODEL_PATH,
            n_ctx         = LLM_CONTEXT_LEN,   # 8192 context window
            n_threads     = LLM_N_THREADS,      # CPU threads
            n_gpu_layers  = LLM_N_GPU_LAYERS,   # 0=CPU, 35=GPU server
            verbose       = False,              # suppress llama.cpp logs
            use_mmap      = True,               # memory-map model file
            use_mlock     = False,              # don't lock RAM on CPU
        )
        logger.info("Gemma 3 4B loaded successfully.")
    return _llm


# ══════════════════════════════════════════════════════════════════════════════
# PROMPT BUILDER
# ══════════════════════════════════════════════════════════════════════════════

# Gemma 3 special tokens
BOS  = "<bos>"
USER = "<start_of_turn>user"
MODEL= "<start_of_turn>model"
END  = "<end_of_turn>"

# System instructions embedded in first user turn
# (Gemma 3 has no dedicated system role — instructions go here)
RAG_SYSTEM_INSTRUCTIONS = """You are a precise document assistant.
Answer the user's question using ONLY the context passages provided below.

Rules you must follow:
1. Base your answer strictly on the provided context. Do not use outside knowledge.
2. After each factual statement, cite the source like this: [source: filename, page: N]
3. If the answer is not found in the context, respond exactly with:
   "I don't know based on the provided documents."
4. Never guess, invent, or extrapolate beyond what the context states.
5. Keep your answer clear, concise, and well-structured."""


def build_context_block(chunks: List[Dict]) -> str:
    """
    Format retrieved chunks into a numbered context block
    that the LLM can reference and cite.

    Parameters:
        chunks : list of chunk dicts from retriever
                 each has: text, source, page, score

    Returns:
        formatted string with all chunks numbered and labelled
    """
    if not chunks:
        return "No relevant context found in the documents."

    lines = ["--- CONTEXT PASSAGES ---\n"]
    for i, chunk in enumerate(chunks, 1):
        lines.append(
            f"[{i}] Source: {chunk['source']} | Page: {chunk['page']} "
            f"| Relevance: {chunk['score']:.2f}"
        )
        lines.append(chunk["text"])
        lines.append("")  # blank line between chunks

    lines.append("--- END OF CONTEXT ---")
    return "\n".join(lines)


def build_prompt(query: str, chunks: List[Dict]) -> str:
    """
    Build the full Gemma 3 instruction-tuned prompt.

    Gemma 3 chat format:
        <bos><start_of_turn>user
        {instructions + context + question}
        <end_of_turn>
        <start_of_turn>model

    The model then generates from the <start_of_turn>model token.

    Parameters:
        query  : raw user question
        chunks : retrieved context chunks

    Returns:
        complete prompt string ready for inference
    """
    context = build_context_block(chunks)

    prompt = (
        f"{BOS}{USER}\n"
        f"{RAG_SYSTEM_INSTRUCTIONS}\n\n"
        f"{context}\n\n"
        f"Question: {query}\n"
        f"{END}\n"
        f"{MODEL}\n"
    )
    return prompt


def estimate_token_count(text: str) -> int:
    """
    Rough token estimate (1 token ≈ 4 chars for English).
    Used to warn if context is too large for the context window.

    Parameters:
        text : any string

    Returns:
        estimated token count
    """
    return len(text) // 4


# ══════════════════════════════════════════════════════════════════════════════
# GENERATION FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def generate_streaming(
    prompt: str,
) -> Generator[str, None, None]:
    """
    Stream Gemma 3 output token by token.
    Yields each token text as it is generated.

    Parameters:
        prompt : complete formatted prompt string

    Yields:
        str token pieces as they are generated
    """
    llm = get_llm()

    stream = llm(
        prompt,
        max_tokens     = LLM_MAX_TOKENS,
        temperature    = LLM_TEMPERATURE,    # 1.0 for Gemma 3
        top_k          = LLM_TOP_K,          # 64
        top_p          = LLM_TOP_P,          # 0.95
        min_p          = LLM_MIN_P,          # 0.0
        repeat_penalty = LLM_REPEAT_PENALTY, # 1.0
        stream         = True,
        echo           = False,              # don't repeat the prompt
    )

    for token_chunk in stream:
        token_text = token_chunk["choices"][0]["text"]
        yield token_text


def generate(
    query: str,
    chunks: List[Dict],
    print_stream: bool = True,
) -> Dict:
    """
    Full generation pipeline:
    build prompt → stream tokens → assemble answer → extract sources.

    Parameters:
        query        : raw user question
        chunks       : retrieved context chunks from retriever
        print_stream : if True, print tokens to terminal as generated

    Returns dict with keys:
        answer       : complete generated answer string
        sources      : list of unique source filenames cited
        prompt_tokens: estimated prompt token count
        answer_tokens: estimated answer token count
        latency_ms   : total generation time in milliseconds
    """
    # Build prompt
    prompt = build_prompt(query, chunks)
    prompt_tokens = estimate_token_count(prompt)

    # Warn if approaching context limit
    if prompt_tokens > LLM_CONTEXT_LEN * 0.85:
        logger.warning(
            f"Prompt is ~{prompt_tokens} tokens — close to "
            f"context limit of {LLM_CONTEXT_LEN}. "
            f"Consider reducing TOP_K in config.py."
        )

    logger.info(f"Generating answer | ~{prompt_tokens} prompt tokens")

    # Stream + collect
    t0 = time.time()
    answer_parts = []

    if print_stream:
        print("\n" + "─" * 55)
        print("  Gemma 3 Answer (streaming)")
        print("─" * 55)

    for token in generate_streaming(prompt):
        answer_parts.append(token)
        if print_stream:
            print(token, end="", flush=True)

    if print_stream:
        print("\n" + "─" * 55)

    latency_ms = round((time.time() - t0) * 1000, 1)
    answer = "".join(answer_parts).strip()

    # Extract sources from chunks (only those likely cited)
    sources = list({c["source"] for c in chunks})

    answer_tokens = estimate_token_count(answer)

    logger.info(
        f"Generation complete | "
        f"{answer_tokens} answer tokens | "
        f"{latency_ms}ms"
    )

    return {
        "answer":        answer,
        "sources":       sources,
        "prompt_tokens": prompt_tokens,
        "answer_tokens": answer_tokens,
        "latency_ms":    latency_ms,
    }


def generate_no_context(query: str) -> Dict:
    """
    Called when ChromaDB is empty or retrieval returns nothing.
    Returns the standard 'I don't know' response without
    hitting the LLM at all.

    Parameters:
        query : raw user question

    Returns:
        same dict structure as generate()
    """
    return {
        "answer":        "I don't know based on the provided documents.",
        "sources":       [],
        "prompt_tokens": 0,
        "answer_tokens": 0,
        "latency_ms":    0,
    }


# ══════════════════════════════════════════════════════════════════════════════
# STANDALONE TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from src.retriever import retrieve

    # ── Test 1: verify prompt builder (no LLM needed) ────────────────────────
    print("\n>>> TEST 1 — Prompt builder (no model load)")
    fake_chunks = [
        {
            "text":   "RAG stands for Retrieval Augmented Generation. "
                      "It combines a retriever with a language model.",
            "source": "rag_intro.pdf",
            "page":   1,
            "score":  0.91,
        },
        {
            "text":   "The retriever searches a vector database "
                      "for semantically relevant passages.",
            "source": "rag_intro.pdf",
            "page":   2,
            "score":  0.87,
        },
    ]
    prompt = build_prompt("What is RAG?", fake_chunks)
    print(prompt)
    print(f"\nEstimated prompt tokens: {estimate_token_count(prompt)}")

    # ── Test 2: full retrieval + generation ───────────────────────────────────
    print("\n>>> TEST 2 — Full RAG generation with Gemma 3")
    print("Loading model (20-60 seconds on CPU)...\n")

    test_query = "What is this document about?"
    result = retrieve(test_query)

    if result["llm_cache_hit"]:
        print(f"LLM cache hit — answer: {result['llm_answer']}")
    elif not result["chunks"]:
        print("No chunks retrieved — check ingest and embedder steps.")
    else:
        gen_result = generate(
            query       = test_query,
            chunks      = result["chunks"],
            print_stream= True,
        )

        print(f"\n  Sources      : {gen_result['sources']}")
        print(f"  Prompt tokens: {gen_result['prompt_tokens']}")
        print(f"  Answer tokens: {gen_result['answer_tokens']}")
        print(f"  Latency      : {gen_result['latency_ms']} ms")
        print("\nStep 6 complete — Gemma 3 is generating cited answers!")