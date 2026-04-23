# config.py
'''
1.4 — Set up config.py
This is your single source of truth for all parameters. 
Changing hardware later means editing only this file.
'''

import os
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR    = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR   = os.path.join(BASE_DIR, "data", "processed")
VECTORSTORE_DIR = os.path.join(BASE_DIR, "vectorstore")
MODELS_DIR      = os.path.join(BASE_DIR, "models")
CACHE_DIR       = os.path.join(BASE_DIR, "cache")

# ── Chunking ───────────────────────────────────────────
CHUNK_SIZE      = 512       # characters per chunk
CHUNK_OVERLAP   = 64        # ~12% overlap to preserve context

# ── Embedding model ────────────────────────────────────
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
EMBEDDING_DIM   = 1024      # output dimension of bge-large
DEVICE          = "cpu"     # change to "cuda" on GPU server

# ── ChromaDB ───────────────────────────────────────────
CHROMA_COLLECTION = "rag_docs"

# ── Cache settings ─────────────────────────────────────
CACHE_SIM_THRESHOLD = 0.90  # cosine similarity floor for cache hit
CACHE_TTL_SECONDS   = 86400 # 24 hours

# ── Retrieval ──────────────────────────────────────────
TOP_K           = 5         # number of chunks to retrieve


# # ── LLM (Mistral-7B GGUF) ─────────────────────────────
# LLM_MODEL_PATH  = os.path.join(MODELS_DIR, "mistral-7b-instruct-v0.3.Q4_K_M.gguf")
# LLM_CONTEXT_LEN = 4096      # context window
# LLM_MAX_TOKENS  = 512       # max tokens to generate per answer
# LLM_TEMPERATURE = 0.1       # low = factual, less hallucination
# LLM_N_THREADS   = 8         # CPU threads — set to your core count
# LLM_N_GPU_LAYERS= 0         # 0 = CPU only; set to 35 on GPU server
# LLM_STREAM      = True      # token streaming


# ── LLM (Gemma 3 4B GGUF) ─────────────────────────────
LLM_MODEL_PATH   = os.path.join(MODELS_DIR, "gemma-3-4b-it-Q4_K_M.gguf")
LLM_CONTEXT_LEN  = 8192      # Gemma 3 supports up to 128K; 8K is safe for CPU
LLM_MAX_TOKENS   = 512
LLM_TEMPERATURE  = 1.0       # Gemma 3 team recommends temp=1.0 (not 0.1 like Mistral)
LLM_TOP_K        = 64        # Gemma 3 recommended setting
LLM_TOP_P        = 0.95      # Gemma 3 recommended setting
LLM_MIN_P        = 0.0
LLM_REPEAT_PENALTY = 1.0     # Gemma 3 recommended — do not change
LLM_N_THREADS    = 8
LLM_N_GPU_LAYERS = 0         # 0 = CPU only now; raise to 35 on GPU server
LLM_STREAM       = True
# ── API settings ───────────────────────────────────────
API_HOST     = os.getenv("API_HOST", "0.0.0.0")
API_PORT     = int(os.getenv("API_PORT", 8000))
API_RELOAD   = os.getenv("API_RELOAD", "true").lower() == "true"

# ── File upload ────────────────────────────────────────
MAX_FILE_SIZE_MB          = int(os.getenv("MAX_FILE_SIZE_MB", 20))
MAX_FILES_PER_REQUEST     = int(os.getenv("MAX_FILES_PER_REQUEST", 5))
ALLOWED_EXTENSIONS        = os.getenv(
    "ALLOWED_EXTENSIONS", ".pdf,.docx,.txt,.md"
).split(",")
UPLOAD_DIR                = os.path.join(BASE_DIR, "data", "uploads")
UPLOAD_AUTO_DELETE_HOURS  = int(os.getenv("UPLOAD_AUTO_DELETE_HOURS", 24))

# ── Session ────────────────────────────────────────────
SESSION_DIR       = os.path.join(BASE_DIR, "data", "sessions")
SESSION_TTL_HOURS = int(os.getenv("SESSION_TTL_HOURS", 24))

# ── General mode ───────────────────────────────────────
DDG_MAX_RESULTS = int(os.getenv("DDG_MAX_RESULTS", 5))