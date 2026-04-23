# src/general_search.py

import os
import sys
import time
import logging
from typing import List, Dict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DDG_MAX_RESULTS
from src.llm import get_llm, BOS, USER, MODEL, END
from config import (
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    LLM_TOP_K,
    LLM_TOP_P,
    LLM_MIN_P,
    LLM_REPEAT_PENALTY,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import os
import requests
import time
import logging
from typing import List, Dict

from dotenv import load_dotenv
load_dotenv()

BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")

logger = logging.getLogger(__name__)


def search_web(query: str, max_results: int = 5) -> List[Dict]:
    """
    Web search using Brave Search API (stable replacement for DuckDuckGo).
    """

    if not BRAVE_API_KEY:
        logger.error("BRAVE_API_KEY not set in environment")
        return []

    url = "https://api.search.brave.com/res/v1/web/search"

    headers = {
        "X-Subscription-Token": BRAVE_API_KEY,
        "Accept": "application/json"
    }

    params = {
        "q": query,
        "count": max_results,
        "search_lang": "en"
    }

    for attempt in range(1, 4):
        try:
            logger.info(f"Brave search attempt {attempt}: {query}")

            resp = requests.get(url, headers=headers, params=params, timeout=20)

            if resp.status_code != 200:
                logger.warning(f"Brave error {resp.status_code}: {resp.text}")
                time.sleep(attempt * 2)
                continue

            data = resp.json()

            results = []
            web_results = data.get("web", {}).get("results", [])

            for r in web_results:
                results.append({
                    "title": r.get("title", ""),
                    "href": r.get("url", ""),
                    "body": r.get("description", "")
                })

            if results:
                logger.info(f"Brave returned {len(results)} results")
                return results

            time.sleep(attempt * 2)

        except Exception as e:
            logger.warning(f"Brave search failed attempt {attempt}: {e}")
            time.sleep(attempt * 2)

    logger.error("Brave search failed completely")
    return []

# def search_web(query: str, max_results: int = DDG_MAX_RESULTS) -> List[Dict]:
#     """
#     Search DuckDuckGo with retry logic.
#     Compatible with duckduckgo-search v7+/v8+.

#     Parameters:
#         query       : search query string
#         max_results : number of results to return

#     Returns:
#         list of dicts with keys: title, href, body
#         empty list if all attempts fail
#     """
#     from duckduckgo_search import DDGS
#     from duckduckgo_search.exceptions import DuckDuckGoSearchException

#     # ── Strategy 1: standard text search with retries ─────────────────────
#     for attempt in range(1, 4):
#         try:
#             logger.info(
#                 f"DDG search attempt {attempt}/3: '{query[:60]}'"
#             )
#             with DDGS(timeout=20) as ddgs:
#                 results = list(
#                     ddgs.text(
#                         keywords   = query,
#                         max_results= max_results,
#                         safesearch = "off",
#                         timelimit  = None,
#                     )
#                 )

#             if results:
#                 logger.info(f"DDG returned {len(results)} results")
#                 return results

#             logger.warning(f"DDG attempt {attempt}: empty results")
#             time.sleep(attempt * 2)   # back-off: 2s, 4s, 6s

#         except DuckDuckGoSearchException as e:
#             logger.warning(f"DDG attempt {attempt} failed: {e}")
#             time.sleep(attempt * 2)

#         except Exception as e:
#             logger.warning(f"DDG attempt {attempt} unexpected error: {e}")
#             time.sleep(attempt * 2)

#     # ── Strategy 2: fallback to news search ───────────────────────────────
#     logger.info("Falling back to DDG news search...")
#     try:
#         with DDGS(timeout=20) as ddgs:
#             results = list(
#                 ddgs.news(
#                     keywords   = query,
#                     max_results= max_results,
#                 )
#             )
#         if results:
#             # News results use 'url' not 'href' — normalise
#             normalised = []
#             for r in results:
#                 normalised.append({
#                     "title": r.get("title", ""),
#                     "href":  r.get("url", r.get("href", "")),
#                     "body":  r.get("body", r.get("excerpt", "")),
#                 })
#             logger.info(f"DDG news fallback: {len(normalised)} results")
#             return normalised

#     except Exception as e:
#         logger.error(f"DDG news fallback also failed: {e}")

#     # ── All strategies failed ─────────────────────────────────────────────
#     logger.error("All DDG search strategies failed — returning empty list")
#     return []


def build_general_prompt(query: str, search_results: List[Dict]) -> str:
    """
    Build Gemma 3 prompt for general web queries.

    Parameters:
        query          : user question
        search_results : list of DDG result dicts

    Returns:
        formatted prompt string
    """
    if not search_results:
        context = (
            "No web search results are available at this time.\n"
            "Answer based on your general knowledge if possible, "
            "and clearly state that no live results were retrieved."
        )
    else:
        lines = ["--- WEB SEARCH RESULTS ---\n"]
        for i, r in enumerate(search_results, 1):
            title = r.get("title", "No title")
            href  = r.get("href",  "")
            body  = r.get("body",  "No description")
            lines.append(f"[{i}] {title}")
            if href:
                lines.append(f"    Source: {href}")
            lines.append(f"    {body}\n")
        lines.append("--- END OF RESULTS ---")
        context = "\n".join(lines)

    instructions = """You are a helpful assistant answering questions \
using web search results.

Rules:
1. Answer clearly and concisely using the search results provided.
2. Reference results by number like [1], [2] when citing.
3. If the results don't fully answer the question, say so and \
share what you do know.
4. Never fabricate URLs or facts not present in the results."""

    return (
        f"{BOS}{USER}\n"
        f"{instructions}\n\n"
        f"{context}\n\n"
        f"Question: {query}\n"
        f"{END}\n"
        f"{MODEL}\n"
    )


def answer_general(query: str) -> Dict:
    """
    Full general-mode pipeline:
    DuckDuckGo search → Gemma 3 generation.

    Parameters:
        query : user question string

    Returns dict with keys:
        answer, sources, search_results_count,
        latency_breakdown, total_ms
    """
    t0 = time.time()

    # ── Search ────────────────────────────────────────────────────────────
    search_start = time.time()
    results      = search_web(query)
    search_ms    = round((time.time() - search_start) * 1000, 1)
    logger.info(f"Search completed in {search_ms}ms — {len(results)} results")

    # ── Build prompt ──────────────────────────────────────────────────────
    prompt = build_general_prompt(query, results)

    # ── Generate ──────────────────────────────────────────────────────────
    llm = get_llm()
    gen_start = time.time()

    output = llm(
        prompt,
        max_tokens     = LLM_MAX_TOKENS,
        temperature    = LLM_TEMPERATURE,
        top_k          = LLM_TOP_K,
        top_p          = LLM_TOP_P,
        min_p          = LLM_MIN_P,
        repeat_penalty = LLM_REPEAT_PENALTY,
        stream         = False,
        echo           = False,
    )

    gen_ms = round((time.time() - gen_start) * 1000, 1)
    answer = output["choices"][0]["text"].strip()

    # ── Extract sources ───────────────────────────────────────────────────
    sources  = [
        r.get("href", "")
        for r in results
        if r.get("href", "")
    ]
    total_ms = round((time.time() - t0) * 1000, 1)

    return {
        "answer":               answer,
        "sources":              sources,
        "search_results_count": len(results),
        "latency_breakdown": {
            "search_ms":     search_ms,
            "generation_ms": gen_ms,
        },
        "total_ms": total_ms,
    }


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import pprint

    print("Testing DDG search directly...\n")

    # ── Test 1: raw search ────────────────────────────────────────────────
    results = search_web("latest AI news 2025", max_results=3)
    print(f"Raw results count: {len(results)}")
    if results:
        print(f"First result title: {results[0].get('title')}")
        print(f"First result body : {results[0].get('body','')[:200]}")
    else:
        print("WARNING: No results returned — check internet connection")

    print("\n" + "="*55)

    # ── Test 2: full answer_general ───────────────────────────────────────
    print("Testing full general answer pipeline...\n")
    result = answer_general("What is retrieval augmented generation?")
    print(f"Answer     : {result['answer'][:400]}")
    print(f"Sources    : {result['sources'][:2]}")
    print(f"Search ms  : {result['latency_breakdown']['search_ms']}")
    print(f"Generate ms: {result['latency_breakdown']['generation_ms']}")
    print(f"Total ms   : {result['total_ms']}")