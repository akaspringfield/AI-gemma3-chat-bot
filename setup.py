#!/usr/bin/env python3
"""
RAG Pipeline — One-shot setup script
Supports: Windows, Linux, macOS
Usage:  python setup.py
        python setup.py --skip-model       (skip Gemma 3 download)
        python setup.py --cpu-only         (force CPU llama-cpp-python)
        python setup.py --gpu cuda         (install CUDA llama-cpp-python)
"""

import os
import sys
import json
import time
import shutil
import platform
import argparse
import subprocess
from pathlib import Path

# ── Colour helpers (graceful fallback on Windows without colorama) ─────────────
try:
    import ctypes
    if platform.system() == "Windows":
        ctypes.windll.kernel32.SetConsoleMode(
            ctypes.windll.kernel32.GetStdHandle(-11), 7
        )
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    CYAN   = "\033[96m"
    BOLD   = "\033[1m"
    RESET  = "\033[0m"
except Exception:
    GREEN = YELLOW = RED = CYAN = BOLD = RESET = ""

def ok(msg):   print(f"{GREEN}  ✓  {msg}{RESET}")
def warn(msg): print(f"{YELLOW}  ⚠  {msg}{RESET}")
def err(msg):  print(f"{RED}  ✗  {msg}{RESET}")
def info(msg): print(f"{CYAN}  →  {msg}{RESET}")
def head(msg): print(f"\n{BOLD}{msg}{RESET}\n{'─'*56}")


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

BASE_DIR = Path(__file__).parent.resolve()

# Folders to create
FOLDERS = [
    "data/raw",
    "data/processed",
    "data/uploads",
    "data/sessions",
    "cache/embeddings",
    "cache/vector_search",
    "cache/llm_responses",
    "vectorstore/sessions",
    "models",
    "src",
]

# Files to create if missing (path → default content)
DEFAULT_FILES = {
    ".env": """\
# Application
APP_ENV=development
LOG_LEVEL=INFO

# API server
API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=true

# File uploads
MAX_FILE_SIZE_MB=20
MAX_FILES_PER_REQUEST=5
ALLOWED_EXTENSIONS=.pdf,.docx,.txt,.md

# File auto-delete (0 = never delete)
UPLOAD_AUTO_DELETE_HOURS=24

# Session TTL (0 = never expire)
SESSION_TTL_HOURS=24

# Web search (general mode)
DDG_MAX_RESULTS=5
DDG_REQUEST_DELAY_SECONDS=3
""",
    ".gitignore": """\
.env
venv/
models/*.gguf
data/uploads/
data/sessions/
data/processed/
cache/
vectorstore/
__pycache__/
*.pyc
.DS_Store
""",
    "src/__init__.py": "# RAG Pipeline source package\n",
}

# pip packages — (package_name, install_name, version_pin)
PACKAGES = [
    ("langchain",              "langchain==0.2.16",              "0.2.16"),
    ("langchain_community",    "langchain-community==0.2.16",    "0.2.16"),
    ("langchain_huggingface",  "langchain-huggingface==0.0.3",   "0.0.3"),
    ("chromadb",               "chromadb==0.5.3",                "0.5.3"),
    ("sentence_transformers",  "sentence-transformers==3.0.1",   "3.0.1"),
    ("dotenv",                 "python-dotenv==1.0.1",           "1.0.1"),
    ("docx",                   "python-docx==1.1.2",             "1.1.2"),
    ("tiktoken",               "tiktoken==0.7.0",                "0.7.0"),
    ("numpy",                  "numpy==1.26.4",                  "1.26.4"),
    ("diskcache",              "diskcache==5.6.3",               "5.6.3"),
    ("huggingface_hub",        "huggingface-hub",                None),
    ("fastapi",                "fastapi==0.111.0",               "0.111.0"),
    ("uvicorn",                "uvicorn==0.30.1",                "0.30.1"),
    ("multipart",              "python-multipart==0.0.9",        "0.0.9"),
    ("duckduckgo_search",      "duckduckgo-search",              None),
    ("aiofiles",               "aiofiles==23.2.1",               "23.2.1"),
]

# PyMuPDF — special handling (Windows needs --only-binary)
PYMUPDF_INSTALL = "pymupdf --only-binary=:all:"

# llama-cpp-python variants
LLAMA_VARIANTS = {
    "cpu":      "llama-cpp-python==0.3.8",
    "cuda":     "llama-cpp-python==0.3.8",   # rebuilt with CMAKE_ARGS
    "metal":    "llama-cpp-python==0.3.8",   # rebuilt with CMAKE_ARGS
    "rocm":     "llama-cpp-python==0.3.8",
}

# Gemma 3 model
MODEL_REPO     = "unsloth/gemma-3-4b-it-GGUF"
MODEL_FILENAME = "gemma-3-4b-it-Q4_K_M.gguf"
MODEL_PATH     = BASE_DIR / "models" / MODEL_FILENAME
MODEL_SIZE_GB  = 2.8


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def run(cmd, check=True, capture=False):
    """Run a shell command, optionally capturing output."""
    result = subprocess.run(
        cmd,
        shell=True,
        check=check,
        capture_output=capture,
        text=True,
    )
    return result


def pip(package, extra_args=""):
    """Install a pip package using the current venv Python."""
    cmd = f'"{PYTHON}" -m pip install {package} {extra_args} --quiet'
    result = run(cmd, check=False)
    if result.returncode != 0:
        err(f"pip install failed: {package}")
        if hasattr(result, "stderr") and result.stderr:
            print(f"      {result.stderr.strip()[:300]}")
        return False
    return True


def is_installed(import_name):
    """Check if a Python package is importable."""
    result = run(
        f'"{PYTHON}" -c "import {import_name}"',
        check=False, capture=True
    )
    return result.returncode == 0


def python_version_ok():
    """Confirm Python 3.10+."""
    v = sys.version_info
    return v.major == 3 and v.minor >= 10


def get_venv_python():
    """Return path to venv Python executable."""
    if platform.system() == "Windows":
        return BASE_DIR / "venv" / "Scripts" / "python.exe"
    return BASE_DIR / "venv" / "bin" / "python"


def get_free_disk_gb():
    """Return free disk space in GB."""
    try:
        total, used, free = shutil.disk_usage(BASE_DIR)
        return free / (1024 ** 3)
    except Exception:
        return 999


# ══════════════════════════════════════════════════════════════════════════════
# STEPS
# ══════════════════════════════════════════════════════════════════════════════

def step_check_python():
    head("Step 1 — Checking Python version")
    v = sys.version_info
    info(f"Python {v.major}.{v.minor}.{v.micro} detected")

    if not python_version_ok():
        err("Python 3.10 or higher is required.")
        err(f"You have Python {v.major}.{v.minor}.")
        err("Download from: https://www.python.org/downloads/")
        sys.exit(1)

    ok(f"Python {v.major}.{v.minor}.{v.micro} — OK")


def step_create_venv():
    head("Step 2 — Creating virtual environment")
    venv_dir = BASE_DIR / "venv"

    if venv_dir.exists():
        ok("Virtual environment already exists — skipping")
        return

    info("Creating venv/ ...")
    run(f'"{sys.executable}" -m venv "{venv_dir}"')
    ok("Virtual environment created at venv/")


def step_create_folders():
    head("Step 3 — Creating project folders")
    created = 0
    for folder in FOLDERS:
        path = BASE_DIR / folder
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            ok(f"Created  {folder}/")
            created += 1
        else:
            info(f"Exists   {folder}/")

    if created == 0:
        ok("All folders already present")
    else:
        ok(f"{created} new folder(s) created")


def step_create_files():
    head("Step 4 — Creating default config files")
    created = 0
    for filepath, content in DEFAULT_FILES.items():
        path = BASE_DIR / filepath
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            ok(f"Created  {filepath}")
            created += 1
        else:
            info(f"Exists   {filepath}  (not overwritten)")

    if created == 0:
        ok("All config files already present")


def step_upgrade_pip():
    head("Step 5 — Upgrading pip")
    info("Upgrading pip in venv ...")
    result = run(
        f'"{PYTHON}" -m pip install --upgrade pip --quiet',
        check=False
    )
    if result.returncode == 0:
        ok("pip upgraded")
    else:
        warn("pip upgrade failed — continuing anyway")


def step_install_pymupdf():
    head("Step 6 — Installing PyMuPDF")

    if is_installed("pymupdf") or is_installed("fitz"):
        ok("PyMuPDF already installed — skipping")
        return

    info("Installing PyMuPDF (pre-built binary) ...")
    success = pip(PYMUPDF_INSTALL)

    if not success:
        warn("PyMuPDF binary install failed — trying pymupdf4llm fallback ...")
        success = pip("pymupdf4llm")
        if success:
            ok("pymupdf4llm installed as fallback")
        else:
            err("PyMuPDF installation failed.")
            err("You can install Visual Studio Build Tools from:")
            err("  https://visualstudio.microsoft.com/visual-cpp-build-tools/")
            warn("Continuing — PDF ingestion may not work until fixed.")
    else:
        ok("PyMuPDF installed")


def step_install_packages():
    head("Step 7 — Installing Python packages")

    failed = []
    for import_name, install_spec, _ in PACKAGES:
        if is_installed(import_name):
            info(f"Already installed  {install_spec.split('==')[0]}")
            continue

        info(f"Installing  {install_spec} ...")
        success = pip(install_spec)
        if success:
            ok(f"Installed  {install_spec}")
        else:
            failed.append(install_spec)

    if failed:
        err(f"\n{len(failed)} package(s) failed to install:")
        for f in failed:
            err(f"  {f}")
        warn("Fix the failures above before running the server.")
    else:
        ok("All packages installed successfully")


def step_install_llama(gpu_target):
    head("Step 8 — Installing llama-cpp-python")

    if is_installed("llama_cpp"):
        info("llama-cpp-python already installed")

        if gpu_target == "cpu":
            ok("Keeping existing install (CPU mode)")
            return

        warn(f"Reinstalling with {gpu_target.upper()} support ...")

    if gpu_target == "cuda":
        info("Building llama-cpp-python with CUDA support ...")
        info("This may take 5–10 minutes ...")
        env = f'CMAKE_ARGS="-DLLAMA_CUDA=on"'
        if platform.system() == "Windows":
            cmd = (
                f'set CMAKE_ARGS=-DLLAMA_CUDA=on && '
                f'"{PYTHON}" -m pip install llama-cpp-python==0.3.8 '
                f'--upgrade --force-reinstall --quiet'
            )
        else:
            cmd = (
                f'CMAKE_ARGS="-DLLAMA_CUDA=on" '
                f'"{PYTHON}" -m pip install llama-cpp-python==0.3.8 '
                f'--upgrade --force-reinstall --quiet'
            )
        result = run(cmd, check=False)
        if result.returncode == 0:
            ok("llama-cpp-python installed with CUDA support")
        else:
            err("CUDA build failed — falling back to CPU build")
            pip("llama-cpp-python==0.3.8")

    elif gpu_target == "metal":
        info("Building llama-cpp-python with Metal (Apple Silicon) support ...")
        cmd = (
            f'CMAKE_ARGS="-DLLAMA_METAL=on" '
            f'"{PYTHON}" -m pip install llama-cpp-python==0.3.8 '
            f'--upgrade --force-reinstall --quiet'
        )
        result = run(cmd, check=False)
        if result.returncode == 0:
            ok("llama-cpp-python installed with Metal support")
        else:
            err("Metal build failed — falling back to CPU build")
            pip("llama-cpp-python==0.3.8")

    else:
        info("Installing llama-cpp-python (CPU build) ...")
        success = pip("llama-cpp-python==0.3.8")
        if success:
            ok("llama-cpp-python installed (CPU mode)")
        else:
            err("llama-cpp-python installation failed.")
            err("On Windows, ensure Visual Studio Build Tools are installed.")
            warn("See: https://visualstudio.microsoft.com/visual-cpp-build-tools/")


def step_download_model(skip_model):
    head("Step 9 — Downloading Gemma 3 4B model")

    if MODEL_PATH.exists():
        size_gb = MODEL_PATH.stat().st_size / (1024 ** 3)
        ok(f"Model already present ({size_gb:.1f} GB) — skipping download")
        return

    if skip_model:
        warn("--skip-model flag set — skipping model download")
        warn(f"Download manually and place at: {MODEL_PATH}")
        return

    free_gb = get_free_disk_gb()
    if free_gb < MODEL_SIZE_GB + 1:
        err(f"Insufficient disk space.")
        err(f"  Required : {MODEL_SIZE_GB + 1:.1f} GB")
        err(f"  Available: {free_gb:.1f} GB")
        sys.exit(1)

    info(f"Downloading {MODEL_FILENAME} (~{MODEL_SIZE_GB} GB) ...")
    info("This may take several minutes depending on your connection ...")
    info("From: https://huggingface.co/unsloth/gemma-3-4b-it-GGUF")

    # Use huggingface_hub which we just installed
    download_script = f"""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent / 'venv' / 'Lib' / 'site-packages'))
from huggingface_hub import hf_hub_download
import os

path = hf_hub_download(
    repo_id="{MODEL_REPO}",
    filename="{MODEL_FILENAME}",
    local_dir="{(BASE_DIR / 'models').as_posix()}",
)
print("Downloaded to:", path)
"""
    tmp = BASE_DIR / "_download_model_tmp.py"
    tmp.write_text(download_script, encoding="utf-8")

    result = run(f'"{PYTHON}" "{tmp}"', check=False)
    tmp.unlink(missing_ok=True)

    if result.returncode == 0 and MODEL_PATH.exists():
        size_gb = MODEL_PATH.stat().st_size / (1024 ** 3)
        ok(f"Model downloaded successfully ({size_gb:.1f} GB)")
    else:
        err("Model download failed.")
        err("Try downloading manually from:")
        err(f"  https://huggingface.co/unsloth/gemma-3-4b-it-GGUF")
        err(f"  → save as: {MODEL_PATH}")
        warn("The server will not start until the model file is present.")


def step_verify():
    head("Step 10 — Verifying installation")

    checks = [
        ("langchain",             "LangChain"),
        ("chromadb",              "ChromaDB"),
        ("sentence_transformers", "sentence-transformers"),
        ("llama_cpp",             "llama-cpp-python"),
        ("diskcache",             "diskcache"),
        ("fastapi",               "FastAPI"),
        ("uvicorn",               "uvicorn"),
        ("duckduckgo_search",     "duckduckgo-search"),
    ]

    all_ok = True
    for import_name, display_name in checks:
        if is_installed(import_name):
            ok(f"{display_name}")
        else:
            err(f"{display_name} — NOT FOUND")
            all_ok = False

    # Model file
    if MODEL_PATH.exists():
        ok(f"Gemma 3 model file present")
    else:
        warn(f"Gemma 3 model not found at models/{MODEL_FILENAME}")
        all_ok = False

    # .env file
    if (BASE_DIR / ".env").exists():
        ok(".env config file present")
    else:
        err(".env file missing")
        all_ok = False

    return all_ok


def step_print_summary(all_ok, gpu_target):
    head("Setup Summary")

    if all_ok:
        print(f"{GREEN}{BOLD}")
        print("  ╔══════════════════════════════════════════════════════╗")
        print("  ║           Setup completed successfully!              ║")
        print("  ╚══════════════════════════════════════════════════════╝")
        print(RESET)
    else:
        print(f"{YELLOW}{BOLD}")
        print("  ╔══════════════════════════════════════════════════════╗")
        print("  ║     Setup completed with warnings — check above      ║")
        print("  ╚══════════════════════════════════════════════════════╝")
        print(RESET)

    # Activation command
    if platform.system() == "Windows":
        activate_cmd = r"venv\Scripts\activate"
    else:
        activate_cmd = "source venv/bin/activate"

    print(f"\n{BOLD}  How to run:{RESET}\n")

    print(f"  1. Activate virtual environment:")
    print(f"     {CYAN}{activate_cmd}{RESET}\n")

    print(f"  2. Start the FastAPI server:")
    print(f"     {CYAN}python api.py{RESET}\n")

    print(f"  3. Open Swagger UI in your browser:")
    print(f"     {CYAN}http://localhost:8000/docs{RESET}\n")

    print(f"  4. Or use the interactive CLI instead:")
    print(f"     {CYAN}python main.py{RESET}\n")

    print(f"  5. Add documents and index them (CLI):")
    print(f"     {CYAN}# drop files into data/raw/ then type /ingest{RESET}\n")

    print(f"  6. Test the API with curl:")
    print(f"""     {CYAN}curl -X POST http://localhost:8000/ai/response-gemma \\
       -F "session_uuid=test-session" \\
       -F "user_uuid=test-user" \\
       -F "question=Hello, what can you do?" \\
       -F "response_type=general"{RESET}\n""")

    if gpu_target != "cpu":
        print(f"  {YELLOW}GPU mode ({gpu_target}) selected — ensure drivers are installed.{RESET}\n")

    print(f"  {BOLD}Useful files:{RESET}")
    print(f"    .env          ← all tunable settings")
    print(f"    config.py     ← model + pipeline parameters")
    print(f"    data/raw/     ← drop documents here")
    print(f"    models/       ← GGUF model file lives here")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="RAG Pipeline setup script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python setup.py                   # standard CPU setup
  python setup.py --skip-model      # skip Gemma 3 download
  python setup.py --gpu cuda        # install with CUDA support
  python setup.py --gpu metal       # install with Metal (Apple Silicon)
        """
    )
    parser.add_argument(
        "--skip-model",
        action="store_true",
        help="Skip downloading the Gemma 3 model file",
    )
    parser.add_argument(
        "--gpu",
        choices=["cpu", "cuda", "metal", "rocm"],
        default="cpu",
        help="GPU backend for llama-cpp-python (default: cpu)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"\n{BOLD}{CYAN}")
    print("  ╔══════════════════════════════════════════════════════╗")
    print("  ║        RAG Pipeline — Automated Setup                ║")
    print("  ║   Gemma 3 4B · ChromaDB · BGE Embeddings · FastAPI  ║")
    print("  ╚══════════════════════════════════════════════════════╝")
    print(RESET)
    print(f"  Platform : {platform.system()} {platform.machine()}")
    print(f"  Python   : {sys.version.split()[0]}")
    print(f"  GPU mode : {args.gpu}")
    print(f"  Directory: {BASE_DIR}")
    print()

    # Step 1 — Python version check
    step_check_python()

    # Step 2 — Create virtual environment
    step_create_venv()

    # Now switch to venv Python for all subsequent pip calls
    global PYTHON
    venv_python = get_venv_python()
    if venv_python.exists():
        PYTHON = str(venv_python)
        ok(f"Using venv Python: {PYTHON}")
    else:
        warn("venv Python not found — using system Python")
        PYTHON = sys.executable

    # Step 3 — Folders
    step_create_folders()

    # Step 4 — Default files
    step_create_files()

    # Step 5 — Upgrade pip
    step_upgrade_pip()

    # Step 6 — PyMuPDF (special handling)
    step_install_pymupdf()

    # Step 7 — All other packages
    step_install_packages()

    # Step 8 — llama-cpp-python (GPU-aware)
    step_install_llama(args.gpu)

    # Step 9 — Download Gemma 3
    step_download_model(args.skip_model)

    # Step 10 — Verify
    all_ok = step_verify()

    # Summary + run instructions
    step_print_summary(all_ok, args.gpu)

    sys.exit(0 if all_ok else 1)


# Global — set properly in main() after venv is created
PYTHON = sys.executable

if __name__ == "__main__":
    main()
