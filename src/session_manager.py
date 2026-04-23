# src/session_manager.py

import os
import json
import time
import shutil
import logging
from typing import Dict, List, Optional
from pathlib import Path

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    SESSION_DIR,
    SESSION_TTL_HOURS,
    UPLOAD_DIR,
    UPLOAD_AUTO_DELETE_HOURS,
    VECTORSTORE_DIR,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.makedirs(SESSION_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR,  exist_ok=True)


class SessionManager:
    """
    Manages per-user-session state including:
      - uploaded file tracking
      - per-session ChromaDB collection name
      - per-session cache keys
      - TTL expiry and cleanup

    Each session is identified by:
      session_uuid : groups a conversation context
      user_uuid    : identifies the user

    Storage layout:
      data/sessions/{session_uuid}/meta.json
      data/uploads/{session_uuid}/{filename}
    """

    def __init__(self):
        # In-memory index of active sessions
        # { session_uuid: session_meta_dict }
        self._sessions: Dict[str, dict] = {}
        self._load_all_sessions()
        logger.info(
            f"SessionManager ready — "
            f"{len(self._sessions)} active sessions"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # SESSION CRUD
    # ──────────────────────────────────────────────────────────────────────────

    def get_or_create(
        self,
        session_uuid: str,
        user_uuid:    str,
    ) -> dict:
        """
        Get existing session or create a new one.

        Parameters:
            session_uuid : unique session identifier
            user_uuid    : unique user identifier

        Returns:
            session meta dict with keys:
              session_uuid, user_uuid, created_at,
              last_active, files, chroma_collection,
              query_count
        """
        # Refresh TTL on access
        if session_uuid in self._sessions:
            session = self._sessions[session_uuid]
            session["last_active"] = time.time()
            self._save_session(session_uuid)
            logger.info(
                f"Session resumed: {session_uuid[:8]}... "
                f"(user={user_uuid[:8]}...)"
            )
            return session

        # Create new session
        session = {
            "session_uuid":      session_uuid,
            "user_uuid":         user_uuid,
            "created_at":        time.time(),
            "last_active":       time.time(),
            "files":             [],   # list of uploaded filenames
            "chroma_collection": f"session_{session_uuid[:16]}",
            "query_count":       0,
            "indexed_files":     [],   # files already embedded
        }

        self._sessions[session_uuid] = session
        self._save_session(session_uuid)

        # Create upload dir for this session
        session_upload_dir = os.path.join(UPLOAD_DIR, session_uuid)
        os.makedirs(session_upload_dir, exist_ok=True)

        logger.info(
            f"Session created: {session_uuid[:8]}... "
            f"(user={user_uuid[:8]}...)"
        )
        return session

    def get(self, session_uuid: str) -> Optional[dict]:
        """Get session by UUID. Returns None if not found."""
        return self._sessions.get(session_uuid)

    def update_files(
        self,
        session_uuid: str,
        new_files:    List[str],
    ) -> None:
        """
        Add newly uploaded filenames to a session.

        Parameters:
            session_uuid : session to update
            new_files    : list of filenames saved to disk
        """
        session = self._sessions.get(session_uuid)
        if not session:
            return

        for f in new_files:
            if f not in session["files"]:
                session["files"].append(f)

        session["last_active"] = time.time()
        self._save_session(session_uuid)

    def mark_indexed(
        self,
        session_uuid: str,
        filename:     str,
    ) -> None:
        """Mark a file as embedded into ChromaDB for this session."""
        session = self._sessions.get(session_uuid)
        if not session:
            return
        if filename not in session["indexed_files"]:
            session["indexed_files"].append(filename)
        self._save_session(session_uuid)

    def increment_query(self, session_uuid: str) -> None:
        """Increment query counter for this session."""
        session = self._sessions.get(session_uuid)
        if session:
            session["query_count"] += 1
            session["last_active"] = time.time()
            self._save_session(session_uuid)

    def get_upload_dir(self, session_uuid: str) -> str:
        """Return upload directory path for a session."""
        path = os.path.join(UPLOAD_DIR, session_uuid)
        os.makedirs(path, exist_ok=True)
        return path

    def is_file_indexed(
        self,
        session_uuid: str,
        filename:     str,
    ) -> bool:
        """Check if a file has already been embedded for this session."""
        session = self._sessions.get(session_uuid)
        if not session:
            return False
        return filename in session["indexed_files"]

    # ──────────────────────────────────────────────────────────────────────────
    # PERSISTENCE
    # ──────────────────────────────────────────────────────────────────────────

    def _save_session(self, session_uuid: str) -> None:
        """Persist session metadata to disk."""
        session_dir = os.path.join(SESSION_DIR, session_uuid)
        os.makedirs(session_dir, exist_ok=True)
        meta_path = os.path.join(session_dir, "meta.json")
        with open(meta_path, "w") as f:
            json.dump(self._sessions[session_uuid], f, indent=2)

    def _load_all_sessions(self) -> None:
        """Load all persisted sessions from disk on startup."""
        ttl_cutoff = time.time() - (SESSION_TTL_HOURS * 3600)

        for session_dir in Path(SESSION_DIR).iterdir():
            if not session_dir.is_dir():
                continue
            meta_path = session_dir / "meta.json"
            if not meta_path.exists():
                continue
            try:
                with open(meta_path) as f:
                    session = json.load(f)

                # Skip expired sessions
                if session.get("last_active", 0) < ttl_cutoff:
                    logger.info(
                        f"Skipping expired session: "
                        f"{session.get('session_uuid','?')[:8]}..."
                    )
                    continue

                self._sessions[session["session_uuid"]] = session

            except Exception as e:
                logger.warning(f"Could not load session {session_dir}: {e}")

    # ──────────────────────────────────────────────────────────────────────────
    # CLEANUP
    # ──────────────────────────────────────────────────────────────────────────

    # src/session_manager.py
    # Find the cleanup_expired() method and replace the file_cutoff line:

    def cleanup_expired(self) -> int:
        ttl_cutoff  = time.time() - (SESSION_TTL_HOURS * 3600)
        cleaned     = 0

        expired = [
            sid for sid, s in self._sessions.items()
            if s.get("last_active", 0) < ttl_cutoff
        ]

        for session_uuid in expired:
            upload_dir = os.path.join(UPLOAD_DIR, session_uuid)

            # ── 0 = never delete files ────────────────────────
            if UPLOAD_AUTO_DELETE_HOURS == 0:
                logger.info(
                    f"UPLOAD_AUTO_DELETE_HOURS=0 — "
                    f"keeping files for {session_uuid[:8]}..."
                )
            else:
                if os.path.exists(upload_dir):
                    shutil.rmtree(upload_dir)
                    logger.info(
                        f"Deleted uploads for "
                        f"session {session_uuid[:8]}..."
                    )

            # Always clean session metadata
            session_dir = os.path.join(SESSION_DIR, session_uuid)
            if os.path.exists(session_dir):
                shutil.rmtree(session_dir)

            del self._sessions[session_uuid]
            cleaned += 1

        return cleaned

    # def cleanup_expired(self) -> int:
    #     """
    #     Remove expired sessions and their uploaded files.
    #     Call periodically (e.g. on startup or via scheduler).

    #     Returns:
    #         number of sessions cleaned up
    #     """
    #     ttl_cutoff   = time.time() - (SESSION_TTL_HOURS * 3600)
    #     file_cutoff  = time.time() - (UPLOAD_AUTO_DELETE_HOURS * 3600)
    #     cleaned      = 0

    #     expired = [
    #         sid for sid, s in self._sessions.items()
    #         if s.get("last_active", 0) < ttl_cutoff
    #     ]

    #     for session_uuid in expired:
    #         # Delete uploaded files
    #         upload_dir = os.path.join(UPLOAD_DIR, session_uuid)
    #         if os.path.exists(upload_dir):
    #             shutil.rmtree(upload_dir)

    #         # Delete session meta
    #         session_dir = os.path.join(SESSION_DIR, session_uuid)
    #         if os.path.exists(session_dir):
    #             shutil.rmtree(session_dir)

    #         del self._sessions[session_uuid]
    #         cleaned += 1
    #         logger.info(f"Cleaned expired session: {session_uuid[:8]}...")

    #     if cleaned:
    #         logger.info(f"Cleanup complete — removed {cleaned} expired sessions")

    #     return cleaned







    def summary(self) -> dict:
        """Return summary stats for all active sessions."""
        return {
            "active_sessions": len(self._sessions),
            "sessions": [
                {
                    "session_uuid": s["session_uuid"][:8] + "...",
                    "user_uuid":    s["user_uuid"][:8] + "...",
                    "files":        len(s["files"]),
                    "queries":      s["query_count"],
                    "last_active":  s["last_active"],
                }
                for s in self._sessions.values()
            ],
        }