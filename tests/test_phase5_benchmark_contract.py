"""Phase 5 TDD: benchmark contract and reporting metrics."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_dual_repo_module():
    module_path = Path(__file__).parent.parent / "benchmark" / "bench_dual_repo_parallel.py"
    spec = importlib.util.spec_from_file_location("bench_dual_repo_parallel", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_common_queries_cover_ten_scenarios():
    mod = _load_dual_repo_module()
    assert len(mod.COMMON_QUERIES) == 10


def test_run_know_agent_reports_payload_and_fallback(monkeypatch, tmp_path):
    mod = _load_dual_repo_module()

    class _FakeCmdResult:
        def __init__(self, stdout: str):
            self.returncode = 0
            self.stdout = stdout
            self.stderr = ""

    fake_json = (
        '{"query":"q","map":{"results":[],"count":0,"tokens":0},'
        '"context":{"used_tokens":10},"deep":{"error":"no_target","call_graph_available":false},'
        '"total_tokens":10}'
    )

    monkeypatch.setattr(mod, "run_cmd", lambda *_args, **_kwargs: _FakeCmdResult(fake_json))
    row = mod.run_know_agent(tmp_path, "q")
    assert "payload_bytes" in row
    assert "fallback_triggered" in row


def test_summary_contains_latency_percentiles():
    mod = _load_dual_repo_module()

    data = {
        "generated_at": "2026-01-01T00:00:00Z",
        "repos": [
            {
                "repo": "/tmp/repo",
                "queries": [],
                "summary": {
                    "know_warm_p50_s": 0.2,
                    "know_warm_p95_s": 0.4,
                    "know_cold_start_s": 1.5,
                    "fallback_rate_pct": 10.0,
                    "token_reduction_pct": 90.0,
                    "latency_ratio_know_over_grep": 1.8,
                    "tool_call_reduction_pct": 90.0,
                    "deep_call_graph_available_rate": 100.0,
                    "deep_non_empty_edges_rate": 60.0,
                },
            }
        ],
    }
    md = mod.render_markdown(data)
    assert "Warm p50/p95" in md
    assert "Cold start" in md
    assert "Fallback rate" in md
