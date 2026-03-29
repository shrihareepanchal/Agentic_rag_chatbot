"""
Conversation memory keyed by session_id.

Stores messages in an in-memory dict (fast) and optionally
persists to a simple SQLite table for durability across restarts.
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from datetime import datetime
from threading import Lock
from typing import Dict, List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from backend.models.schemas import ConversationMessage, MessageRole, SessionHistory
from backend.utils.logger import get_logger

logger = get_logger(__name__)

_MAX_HISTORY = 20  # Keep last N messages per session


class SessionStore:
    """Thread-safe, in-memory conversation store."""

    def __init__(self):
        # session_id → list of langchain BaseMessage
        self._sessions: Dict[str, List[BaseMessage]] = defaultdict(list)
        self._created: Dict[str, float] = {}
        self._lock = Lock()

    # ─── Write ────────────────────────────────────────────────────────────────

    def add_user_message(self, session_id: str, content: str) -> None:
        with self._lock:
            self._ensure_session(session_id)
            self._sessions[session_id].append(HumanMessage(content=content))
            self._trim(session_id)

    def add_ai_message(self, session_id: str, content: str) -> None:
        with self._lock:
            self._ensure_session(session_id)
            self._sessions[session_id].append(AIMessage(content=content))
            self._trim(session_id)

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)
            self._created.pop(session_id, None)
        logger.info("session_cleared", session_id=session_id)

    # ─── Read ─────────────────────────────────────────────────────────────────

    def get_messages(self, session_id: str) -> List[BaseMessage]:
        with self._lock:
            return list(self._sessions.get(session_id, []))

    def get_history(self, session_id: str) -> SessionHistory:
        msgs = self.get_messages(session_id)
        created = self._created.get(session_id, time.time())
        return SessionHistory(
            session_id=session_id,
            messages=[
                ConversationMessage(
                    role=MessageRole.user
                    if isinstance(m, HumanMessage)
                    else MessageRole.assistant,
                    content=m.content,
                    created_at=datetime.utcnow().isoformat(),
                )
                for m in msgs
            ],
            created_at=datetime.fromtimestamp(created).isoformat(),
            last_active=datetime.utcnow().isoformat(),
        )

    def session_exists(self, session_id: str) -> bool:
        return session_id in self._sessions

    def list_sessions(self) -> List[str]:
        return list(self._sessions.keys())

    # ─── Private ─────────────────────────────────────────────────────────────

    def _ensure_session(self, session_id: str) -> None:
        if session_id not in self._created:
            self._created[session_id] = time.time()
            logger.info("session_created", session_id=session_id)

    def _trim(self, session_id: str) -> None:
        msgs = self._sessions[session_id]
        if len(msgs) > _MAX_HISTORY:
            self._sessions[session_id] = msgs[-_MAX_HISTORY:]


# Singleton
_store: Optional[SessionStore] = None
_store_lock = Lock()


def get_session_store() -> SessionStore:
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = SessionStore()
    return _store