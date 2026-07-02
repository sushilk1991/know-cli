# Context Intelligence Upgrade Checklist

Date: 2026-03-05
Owner: know-cli core
Status: Active
Scope: retrieval/context identity only (`map/context/deep/workflow`), hook-first suggest mode, retrieval telemetry/analytics.

## Non-Goals

1. No non-retrieval command optimization expansion.
2. No automatic command rewriting or `updatedInput` mutation payloads.
3. No Rust migration in this cycle.
4. No top-level `know confidence` feature family.

## Phase Gate Tracker

| Phase | Gate | Type | Status | Pass Criteria | Evidence |
|---|---|---|---|---|---|
| P0 | P0-G1 | Automated | Done | `know commands --all` excludes `confidence` | `tests/test_context_intelligence_upgrade.py::test_confidence_command_removed_from_registry` |
| P0 | P0-G2 | Automated | Done | README + CLI help have no `know confidence` refs | `README.md`, `src/know/cli/__init__.py` |
| P0 | P0-G3 | Automated | Done | Retrieval commands still functional | `tests/test_context_intelligence_upgrade.py::test_retrieval_events_recorded_and_summary_contract` |
| P0 | P0-G4 | Manual | Done | Baseline benchmark artifact linked | `benchmark/results/DUAL_REPO_BENCHMARK.md` |
| P1 | P1-G1 | Automated | Done | Existing hooks install/uninstall/status behavior unchanged | `tests/test_hooks_cli_behavior.py` |
| P1 | P1-G2 | Automated | Done | `hooks suggest --agent claude` JSON/plain schemas valid | `tests/test_context_intelligence_upgrade.py::test_hooks_suggest_claude_json_contract` |
| P1 | P1-G3 | Automated | Done | Suggest payload does not include `updatedInput` | `tests/test_context_intelligence_upgrade.py::test_hooks_suggest_claude_json_contract` |
| P1 | P1-G4 | Manual | Pending | Claude dry-run adoption validated | Manual canary runbook |
| P2 | P2-G1 | Automated | Done | `map/context/deep/workflow` each record events | `tests/test_context_intelligence_upgrade.py::test_retrieval_events_recorded_and_summary_contract` |
| P2 | P2-G2 | Automated | Done | Legacy `get_summary()` keys preserved | `tests/test_context_intelligence_upgrade.py::test_retrieval_events_recorded_and_summary_contract` |
| P2 | P2-G3 | Automated | Done | Workflow metadata includes mode/degradation/call-graph signals | `src/know/stats.py`, workflow telemetry tests |
| P2 | P2-G4 | Automated | Done | Single workflow invocation records one event | `tests/test_context_intelligence_upgrade.py::test_workflow_records_single_event_per_invocation` |
| P3 | P3-G1 | Automated | Done | `know --json stats` includes `retrieval` block | `tests/test_context_intelligence_upgrade.py::test_stats_json_includes_retrieval_block` |
| P3 | P3-G2 | Automated | Done | Human `know stats` prints retrieval KPI section | `tests/test_context_intelligence_upgrade.py::test_stats_human_output_shows_retrieval_kpi_section` |
| P3 | P3-G3 | Automated | Done | README + skill guidance updated | `README.md`, `KNOW_SKILL.md`, `src/know/resources/KNOW_SKILL.md` |
| P3 | P3-G4 | Manual | Pending | Retrieval-only docs consistency review | Manual documentation review |
| P4 | P4-G1 | Automated | Done | Min token reduction >= 90% | `benchmark/results/DUAL_REPO_BENCHMARK.md` (2026-03-05 run: min 93.4%) |
| P4 | P4-G2 | Automated | Done | Min tool-call reduction >= 90% | `benchmark/results/DUAL_REPO_BENCHMARK.md` (2026-03-05 run: min 93.8%) |
| P4 | P4-G3 | Automated | Done | Max latency ratio <= 2.5x | `benchmark/results/DUAL_REPO_BENCHMARK.md` (2026-03-05 run: max 2.27x) |
| P4 | P4-G4 | Automated | Done | Min non-empty deep edges >= 70% | `benchmark/results/DUAL_REPO_BENCHMARK.md` (2026-03-05 run: min 70.0%) |
| P4 | P4-G5 | Manual | Pending | 1-week canary clean | Manual canary checklist |

## TDD Evidence

1. RED: Added failing tests in `tests/test_context_intelligence_upgrade.py`.
2. GREEN: Implemented hooks suggest + retrieval telemetry APIs + CLI instrumentation + stats retrieval block.
3. REFACTOR: Removed confidence detour modules/tests/docs and aligned README/skills to retrieval identity.

## Benchmark Snapshot (2026-03-05)

1. Repo `know-cli`: token reduction 93.4%, latency ratio 2.27x, tool-call reduction 93.8%, non-empty edges 90.0%.
2. Repo `farfield`: token reduction 95.5%, latency ratio 1.54x, tool-call reduction 93.8%, non-empty edges 70.0%.
3. Gate outcomes: P4-G1 to P4-G4 pass.

## Release Notes (Scope)

1. `know confidence` removed.
2. Added `know hooks suggest --agent claude|codex` (suggestion-only guidance).
3. Added retrieval telemetry APIs and retrieval analytics output in `know stats`.
4. No command-rewrite behavior introduced.
