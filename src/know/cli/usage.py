"""Helpers for post-command usage telemetry rendering."""

from __future__ import annotations

from typing import Any

from know.cli import console


def build_usage_payload(
    *,
    source: str,
    tokens_used: int | None = None,
    elapsed_ms: int | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a normalized usage payload for JSON and rich output."""
    payload: dict[str, Any] = {"source": source}
    if tokens_used is not None:
        payload["tokens_used"] = int(tokens_used)
    if elapsed_ms is not None:
        payload["elapsed_ms"] = int(elapsed_ms)
    if details:
        payload["details"] = details
    return payload


def attach_usage(payload: dict[str, Any], usage: dict[str, Any]) -> dict[str, Any]:
    """Attach usage telemetry to an outgoing JSON payload."""
    payload["usage"] = usage
    return payload


def render_usage(ctx, usage: dict[str, Any]) -> None:
    """Render one-line usage telemetry for human output modes."""
    if ctx.obj.get("json") or ctx.obj.get("quiet"):
        return

    parts = [f"source={usage.get('source', 'unknown')}"]
    if usage.get("tokens_used") is not None:
        parts.append(f"tokens={int(usage['tokens_used']):,}")
    if usage.get("elapsed_ms") is not None:
        parts.append(f"time={int(usage['elapsed_ms'])}ms")

    details = usage.get("details") or {}
    if isinstance(details, dict):
        for key in ("files_read", "files_matched", "terms"):
            if details.get(key) is not None:
                parts.append(f"{key}={details[key]}")

    console.print(f"[dim]usage: {' | '.join(parts)}[/dim]", highlight=False)
