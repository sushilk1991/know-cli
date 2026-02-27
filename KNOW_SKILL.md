---
name: know-cli
description: Default retrieval and memory workflow for coding agents. Use for code search, edit-target selection, dependency impact checks, and cross-session decision recall with token budgets.
metadata:
  short-description: Agent-native code retrieval + memory workflow
---

# know-cli Skill (Agent-Optimized)

Use [know-cli](https://github.com/sushilk1991/know-cli) as the default retrieval and memory layer for coding agents.

Goals:
- find the right edit target quickly
- reduce token waste
- preserve cross-session and cross-agent decisions

This skill is designed for Codex, Claude, and Gemini style coding loops.

## Non-Negotiable Defaults

- Start with one call: `know workflow`.
- Always begin with a compatibility-safe one-liner:
  - `know workflow "<query>" --session auto`
- Before using advanced workflow flags, probe capabilities once:
  - `know workflow --help`
  - use `--mode`, `--max-latency-ms`, `--json-compact`, `--json-full` only if they appear in help.
- Use `--json` for machine mode only when needed.
- For agent execution, prefer `--json-compact` to avoid oversized payloads.
- Use `--json-full` only for strict parser compatibility.
- Use explicit mode for intent:
  - `--mode explore` for fast discovery
  - `--mode implement` for balanced coding
  - `--mode thorough` for deep refactors
- Keep a strict latency budget with `--max-latency-ms` to avoid long stalls.
- Start with small budgets, then escalate.
- Keep a stable session id (`--session auto` or a persisted session id) for dedup.
- `--session auto` now resolves to a concrete ID across workflow/context/map/deep and is persisted.
- Use fallback ladder only when confidence is low (<0.55) or deep target is missing.

## Use-Case Command Matrix

| Use-case | Best first command | Why |
|---|---|---|
| Find likely edit file quickly | `know next-file "<query>" --json` | lowest-latency target hint |
| Get actionable context for coding task (works on all versions) | `know workflow "<query>" --session auto` | guaranteed single-call retrieval without flag mismatch |
| Get actionable context for coding task (newer CLI) | `know --json workflow "<query>" --mode implement --json-compact --session auto` | balanced quality + speed in one call |
| Get strict full JSON for automation | `know --json workflow "<query>" --json-full --session auto` | stable full schema for scripts/MCP |
| Fast exploration | `know --json workflow "<query>" --mode explore --max-latency-ms 2500 --json-compact --session auto` | fast target discovery without deep stall |
| Understand one symbol deeply | `know deep "file.py:function_name" --budget 3000 --json` | callers/callees + focused body |
| Broad exploration | `know map "<area>" --limit 30 --json --session auto` | cheap orientation before bigger context |
| Dependency impact | `know graph <file_path>` then `know related <file_path> --json` | import/dependent blast radius |
| Capture important decision | `know decide "<decision>" --why "<rationale>"` | structured long-lived memory |
| Recover prior design choices | `know recall "<query>" --type decision --status active` | stable memory retrieval |

## Primary Flow (Default)

Compatibility-safe default:

```bash
know workflow "<task or bug description>" --session auto
```

Advanced default (newer CLI with workflow flags):

```bash
know --json workflow "<task or bug description>" \
  --mode implement \
  --max-latency-ms 6000 \
  --json-compact \
  --map-limit 20 \
  --context-budget 4000 \
  --deep-budget 3000 \
  --session auto
```

When strict machine parsing is required:

```bash
know --json workflow "<task or bug description>" --json-full --session auto
```

Interpretation:
- `map`: quick candidate symbols/files
- `context`: token-budgeted multi-file context
- `deep`: focused internals for final edit target

## Fallback Ladder (Strict Order)

1. `know map "<query>" --json --limit 30 --session auto`
2. `know next-file "<query>" --json`
3. `know --json workflow "<query>" --mode explore --max-latency-ms 2500 --json-compact --session auto`
4. `know context "<query>" --budget 4000 --session auto --json`
5. `know deep "<symbol or file:symbol>" --budget 3000 --json`
6. `know related <file_path> --json`
7. `know graph <file_path>`

Budget escalation:
- `4000 -> 6000 -> 8000` only if specific missing context remains.

## Deep Use Cases

### 1) Multi-file refactor

```bash
know --json workflow "replace legacy auth adapter with service token provider" --mode thorough --max-latency-ms 15000 --json-compact --session auto
know related src/auth/adapter.py --json
know graph src/auth/adapter.py
```

### 2) Production bug trace

```bash
know --json workflow "500 on checkout after coupon apply" --mode implement --max-latency-ms 6000 --json-compact --session auto
know deep "payments.py:apply_coupon" --budget 3000 --json
know callers "apply_coupon" --json
```

### 3) API contract safety

```bash
know map "request validation and response schema for create_order" --json --limit 30
know context "where create_order input/output contracts are enforced" --budget 6000 --json --quiet
```

### 4) Cross-agent handoff (Codex -> Claude -> Gemini)

```bash
know decide "Normalize invoice totals in domain layer before persistence" \
  --why "prevents API/controller duplication and rounding drift" \
  --evidence src/billing/domain.py:118
know remember "checkout flow entrypoint is src/checkout/handler.ts"
know recall "invoice normalization decision" --type decision --status active
```

## Memory Policy (Cross-Agent)

- Prefer `know decide` for architecture and policy choices.
- Prefer `know remember` for concrete facts.
- Always attach evidence (`file:line`) for key decisions.
- Resolve stale decisions after migration:
  - `know memories resolve <id> --status resolved`
  - or shortcut: `know done <id>`
- Use `agent` and `session_id` fields (auto-filled where available) to preserve provenance across Codex/Claude/Gemini.

## Simplified Surface (Human-Friendly, Backward-Compatible)

- `know ask "<query>"` wraps workflow with sensible defaults.
- `know docs` refreshes digest/API/diagram.
- `know recall "<query>"` fetches relevant memories.
- `know decide "<decision>" --why "<rationale>"` stores structured decisions.
- `know done <id>` resolves a memory quickly.
- `know commands --all` shows advanced and legacy commands.

Legacy commands remain supported (`know context`, `know deep`, `know map`, `know search`, etc.).

## Background Automation Defaults

- Daemon incremental refresh is enabled by default.
- Use `know warm` after install/upgrade or when first-call latency is high.
- Full indexing auto-purges out-of-scope noise from older indexes (`.venv*`, `site-packages`, `dist`, `build`, cache trees).
- Active workflow session is persisted at `.know/current_session`.
- `know remember` and `know decide` auto-fill `session_id` from current session when available.
- For very large repos, daemon auto-refresh may self-suspend; force-enable with:
  - `KNOW_DAEMON_AUTO_REFRESH=1`

## Anti-Patterns (Avoid)

- Running `know context` with huge budget first (`12000+`) without map/workflow.
- Re-running broad context without session dedup.
- Editing before deep/context returns concrete files and symbols.
- Storing architectural decisions as plain notes without `--why` and evidence.
- Ignoring dependency checks on shared modules.

## Recovery Playbook

If retrieval quality looks wrong:
1. `know doctor --repair --reindex`
2. `know reindex`
3. Retry workflow with explicit query and `--session auto`
4. Use fallback ladder and narrow with `file:symbol`

If command surface seems missing:
1. `know --version`
2. `know workflow --help`
3. `know commands --all | grep workflow`
4. If help does not show `--mode`/`--json-compact`, use:
   - `know workflow "<query>" --session auto`
   - avoid advanced flags for this run
5. `python -m pip uninstall -y know know-cli`
6. `python -m pip install -U know-cli`

## Context Commands (Direct Use)

```bash
know context "<task>" --budget 4000 --session auto --json
know context "<task>" --budget 8000 --session auto --json
know map "<feature area>" --type function --json
know deep "service.py:process_payment" --budget 3000 --json
know graph src/auth/middleware.py
know search "token rotation" --json
```

## Token Discipline

- Use `map`/`next-file` before broad context for narrow tasks.
- Keep first pass near `--context-budget 4000`.
- Reuse session ids for dedup across follow-ups.
- Escalate only after identifying missing facts.

Examples:

```bash
know --json workflow "fix task creation validation regression" --mode implement --max-latency-ms 6000 --json-compact --session auto
know ask "where is task creation validation"
know decide "Validate task payload in service layer only" \
  --why "single source of truth" \
  --evidence src/tasks/service.py:44
know recall "task validation decision" --type decision --status active
```
