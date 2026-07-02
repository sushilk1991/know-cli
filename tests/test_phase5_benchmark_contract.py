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


def _load_workflow_relevance_module():
    module_path = Path(__file__).parent.parent / "benchmark" / "bench_workflow_relevance.py"
    spec = importlib.util.spec_from_file_location("bench_workflow_relevance", module_path)
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


def test_workflow_defaults_follow_selected_mode():
    mod = _load_dual_repo_module()

    explore = mod.resolve_workflow_defaults(
        mode="explore",
        max_latency_ms=None,
        map_limit=None,
        context_budget=None,
        deep_budget=None,
    )
    implement = mod.resolve_workflow_defaults(
        mode="implement",
        max_latency_ms=None,
        map_limit=None,
        context_budget=None,
        deep_budget=None,
    )

    assert explore["max_latency_ms"] == 2500
    assert explore["deep_budget"] == 0
    assert implement["max_latency_ms"] == 6000
    assert implement["deep_budget"] == 3000


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


def test_workflow_relevance_collects_files_from_compact_payload():
    mod = _load_workflow_relevance_module()

    payload = {
        "targets": {
            "selected_file_path": "src/know/cli/agent.py",
            "candidates": [{"file_path": "src/know/config.py"}],
        },
        "context": {
            "source_files": ["src/know/stats.py"],
            "snippets": [{"file": "src/know/daemon.py"}],
        },
        "deep": {
            "target": {"file_path": "src/know/context_engine.py"},
            "callers": 2,
            "callees": 3,
            "caller_examples": [{"file": "src/know/cli/search.py"}],
            "callee_examples": [{"file_path": "src/know/token_counter.py"}],
        },
    }

    files = mod.collect_workflow_files(payload)

    assert "src/know/cli/agent.py" in files
    assert "src/know/config.py" in files
    assert "src/know/context_engine.py" in files
    assert "src/know/token_counter.py" in files


def test_workflow_relevance_summary_separates_selected_accuracy(monkeypatch, tmp_path):
    mod = _load_workflow_relevance_module()

    cases = [
        {"id": "context", "query": "q1", "expected_files": ["expected.py"]},
        {"id": "selected", "query": "q2", "expected_files": ["target.py"]},
    ]

    def _fake_run_workflow(_repo, query, **_kwargs):
        if query == "q1":
            return {
                "ok": True,
                "elapsed_s": 0.1,
                "payload_tokens": 100,
                "retrieval_tokens": 50,
                "selected_file": "other.py",
                "files": ["expected.py", "other.py"],
            }
        return {
            "ok": True,
            "elapsed_s": 0.2,
            "payload_tokens": 120,
            "retrieval_tokens": 60,
            "selected_file": "target.py",
            "files": ["target.py"],
        }

    monkeypatch.setattr(mod, "run_workflow", _fake_run_workflow)

    result = mod.run_suite(
        tmp_path,
        mode="implement",
        max_latency_ms=2000,
        config_path=None,
        refresh_index=False,
        cases=cases,
    )

    assert result["summary"]["relevance_rate_pct"] == 100.0
    assert result["summary"]["selected_rate_pct"] == 50.0
    assert result["summary"]["payload_to_retrieval_ratio"] == 2.0
