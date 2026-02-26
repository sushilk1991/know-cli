"""Helpers for auto-capturing high-signal workflow memories."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from know.logger import get_logger

logger = get_logger()
AUTO_WORKFLOW_DECISION_TTL_HOURS = 24 * 30  # 30 days


def capture_workflow_decision(
    config,
    query: str,
    workflow_result: Dict[str, Any],
    *,
    session_id: Optional[str] = None,
    source: str = "auto-workflow",
    agent: str = "know-cli",
) -> Optional[int]:
    """Persist a structured decision memory from a workflow execution.

    Returns the created memory integer ID, or None when there is nothing useful
    to store.
    """
    selected = workflow_result.get("selected_deep_target")
    deep = workflow_result.get("deep") or {}
    target = deep.get("target") or {}
    if not selected and not target:
        return None

    context = workflow_result.get("context") or {}
    source_files = context.get("source_files") or []
    focus = ", ".join(source_files[:3]) if source_files else target.get("file", "")
    if not focus and target.get("file"):
        focus = target["file"]

    evidence = ""
    if target.get("file"):
        evidence = f"{target['file']}:{target.get('line_start', 0)}"
    confidence = float(context.get("confidence", 0.7) or 0.7)

    text = (
        f"Workflow decision: query='{query}' -> deep_target='{selected or target.get('name', '')}'. "
        f"Focus files: {focus}"
    )
    tags = "workflow,decision"
    target_name = selected or target.get("name", "")
    if target_name:
        tags = f"{tags},{target_name}"

    try:
        from know.knowledge_base import KnowledgeBase

        kb = KnowledgeBase(config)
        # Explicit dedup check avoids duplicate auto-memories when workflow is
        # retried with the same resolved target.
        active = kb.list_all(source=source, memory_type="decision", decision_status="active")
        for mem in active:
            if mem.text == text:
                return mem.id

        expires_at = time.time() + (AUTO_WORKFLOW_DECISION_TTL_HOURS * 3600)
        return kb.remember(
            text=text,
            source=source,
            tags=tags,
            memory_type="decision",
            decision_status="active",
            confidence=confidence,
            evidence=evidence,
            session_id=session_id or workflow_result.get("session_id", "") or "",
            agent=agent,
            trust_level="local_verified",
            expires_at=expires_at,
        )
    except Exception as e:
        logger.debug(f"Workflow decision capture skipped: {e}")
        return None
