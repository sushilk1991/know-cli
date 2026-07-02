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
    commands = []

    class _FakeCmdResult:
        def __init__(self, stdout: str):
            self.returncode = 0
            self.stdout = stdout
            self.stderr = ""

    fake_json = (
        '{"query":"q","context":{"snippets":[]},'
        '"deep":{"error":"no_target","call_graph_available":false,"callers":0,"callees":0},'
        '"metrics":{"profile":"compact","mode":"thorough","latency_budget_ms":15000,'
        '"total_tokens":10,"map_tokens":2,"context_tokens":8,"deep_tokens":0}}'
    )

    def _fake_run_cmd(cmd, *_args, **_kwargs):
        commands.append(cmd)
        return _FakeCmdResult(fake_json)

    monkeypatch.setattr(mod, "run_cmd", _fake_run_cmd)
    row = mod.run_know_agent(tmp_path, "q")
    assert commands
    assert "--read-only" in commands[0]
    assert commands[0][commands[0].index("--mode") + 1] == "thorough"
    assert commands[0][commands[0].index("--max-latency-ms") + 1] == "15000"
    assert "payload_bytes" in row
    assert "payload_tokens" in row
    assert "retrieval_tokens" in row
    assert row["retrieval_tokens"] == 10
    assert row["map_tokens"] == 2
    assert row["workflow_mode"] == "thorough"
    assert row["latency_budget_ms"] == 15000
    assert row["read_only"] is True
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
                    "quality_gate_passed": False,
                    "token_reduction_pct": 90.0,
                    "latency_ratio_know_over_grep": 1.8,
                    "tool_call_reduction_pct": 90.0,
                    "lookup_call_reduction_pct": 90.0,
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
    assert "Quality gate" in md
    assert "Lookup-call reduction" in md
