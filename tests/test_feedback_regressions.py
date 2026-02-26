"""Regression tests for reported field feedback on map/deep behavior."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from know.config import Config
from know.context_engine import ContextEngine


class _FakeDB:
    def __init__(self, chunks):
        self._chunks = list(chunks)

    def get_chunks_by_name(self, name, limit=20):
        return [c for c in self._chunks if c.get("chunk_name") == name][:limit]

    def get_method_chunks_by_suffix(self, method_name, limit=20):
        return [c for c in self._chunks if c.get("chunk_name", "").endswith(f".{method_name}")][:limit]

    def search_chunks(self, query, limit=20):
        q = (query or "").lower()
        out = []
        for c in self._chunks:
            cname = (c.get("chunk_name") or "").lower()
            sig = (c.get("signature") or "").lower()
            if q and (q in cname or q in sig):
                out.append(c)
            if len(out) >= limit:
                break
        return out


def _mk_config(tmp_path: Path) -> Config:
    (tmp_path / ".know").mkdir()
    cfg = Config.create_default(tmp_path)
    cfg.root = tmp_path
    cfg.save(tmp_path / ".know" / "config.yaml")
    return cfg


def test_map_accepts_session_flag_and_emits_json(tmp_path):
    cfg = _mk_config(tmp_path)
    from know.cli import cli

    fake_client = MagicMock()
    fake_client.call_sync.return_value = {
        "results": [
            {
                "file_path": "src/main.py",
                "chunk_name": "hello",
                "score": 0.9,
            }
        ]
    }

    runner = CliRunner()
    with patch("know.cli.agent._get_daemon_client", return_value=fake_client):
        result = runner.invoke(
            cli,
            [
                "--config",
                str(cfg.root / ".know" / "config.yaml"),
                "--json",
                "map",
                "hello",
                "--session",
                "auto",
            ],
        )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["count"] == 1
    assert payload["session_id"] != "auto"
    assert len(payload["session_id"]) == 8


def test_deep_resolves_leaf_symbol_when_index_stores_fully_qualified_names(tmp_path):
    cfg = _mk_config(tmp_path)
    engine = ContextEngine(cfg)
    db = _FakeDB(
        chunks=[
            {
                "file_path": "src/agent/mod.rs",
                "chunk_name": "crate::agent::spawn_stream",
                "chunk_type": "function",
                "signature": "pub async fn spawn_stream(...)",
                "start_line": 10,
                "end_line": 40,
                "body": "pub async fn spawn_stream() {}",
                "token_count": 12,
            }
        ]
    )

    candidates = engine._resolve_function(db, "spawn_stream", include_tests=True)
    assert len(candidates) == 1
    assert candidates[0]["chunk_name"] == "crate::agent::spawn_stream"


def test_workflow_daemon_path_does_not_double_capture_memory(tmp_path):
    cfg = _mk_config(tmp_path)
    from know.cli import cli

    fake_client = MagicMock()
    fake_client.call_sync.return_value = {
        "query": "workflow test",
        "session_id": "abcd1234",
        "daemon_api_version": 2,
        "workflow_mode": "implement",
        "latency_budget_ms": 6000,
        "selected_deep_target": "hello",
        "map": {"results": [], "count": 0, "truncated": False, "tokens": 0},
        "context": {
            "query": "workflow test",
            "budget": 4000,
            "used_tokens": 10,
            "indexing_status": "complete",
            "code": [],
            "warnings": [],
            "confidence": 0.7,
        },
        "deep": {
            "target": {"name": "hello", "file": "src/main.py", "line_start": 1, "line_end": 2},
            "callers": [],
            "callees": [],
            "budget_used": 5,
            "call_graph_available": True,
            "call_graph_reason": None,
        },
        "total_tokens": 15,
        "latency_ms": {"map": 1, "context": 2, "deep": 1, "total": 4},
        "degraded_by_latency": False,
    }

    runner = CliRunner()
    with patch("know.cli.agent._get_daemon_client", return_value=fake_client):
        with patch("know.memory_capture.capture_workflow_decision") as capture:
            result = runner.invoke(
                cli,
                [
                    "--config",
                    str(cfg.root / ".know" / "config.yaml"),
                    "--json",
                    "workflow",
                    "workflow test",
                    "--json-compact",
                ],
            )

    assert result.exit_code == 0
    assert capture.call_count == 0
