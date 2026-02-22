"""Runtime context helpers shared across CLI/daemon.

Used to persist lightweight session metadata so commands like `remember`
can auto-fill session_id without requiring explicit user flags.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def _session_file(config) -> Path:
    return config.root / ".know" / "current_session"


def get_active_session_id(config) -> Optional[str]:
    """Get the last active session ID if available."""
    # Environment override for external orchestrators.
    env_sid = os.environ.get("KNOW_SESSION_ID", "").strip()
    if env_sid:
        return env_sid

    path = _session_file(config)
    if not path.exists():
        return None

    try:
        sid = path.read_text(encoding="utf-8").strip()
    except Exception:
        return None

    if not sid:
        return None
    # Keep this conservative to avoid persisting huge/untrusted payloads.
    if len(sid) > 128:
        return None
    return sid


def set_active_session_id(config, session_id: str) -> None:
    """Persist current session ID for later command auto-fill."""
    sid = (session_id or "").strip()
    if not sid:
        return

    path = _session_file(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{sid}\n", encoding="utf-8")


def infer_agent_name(default: str = "know-cli") -> str:
    """Best-effort agent identity for memory metadata."""
    return (
        os.environ.get("KNOW_AGENT")
        or os.environ.get("AGENT_NAME")
        or os.environ.get("CLAUDECODE_AGENT")
        or default
    )
