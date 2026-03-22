"""WebSocket session state for the inference server."""

from dataclasses import dataclass
from typing import Dict, Optional

from src.protocol import InferChunkMessage, SessionSettings


@dataclass
class SessionState:
    session_id: str
    settings: SessionSettings
    pending_infer: Optional[InferChunkMessage] = None


class SessionManager:
    def __init__(self):
        self._sessions: Dict[str, SessionState] = {}

    def create(self, session_id: str, settings: Optional[SessionSettings] = None) -> SessionState:
        state = SessionState(session_id=session_id, settings=settings or SessionSettings())
        self._sessions[session_id] = state
        return state

    def get(self, session_id: str) -> Optional[SessionState]:
        return self._sessions.get(session_id)

    def remove(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def set_pending_infer(self, session_id: str, message: InferChunkMessage) -> None:
        state = self._sessions[session_id]
        state.pending_infer = message

    def pop_pending_infer(self, session_id: str) -> Optional[InferChunkMessage]:
        state = self._sessions[session_id]
        pending = state.pending_infer
        state.pending_infer = None
        return pending