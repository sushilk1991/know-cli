# know-cli Skill (Agent-Optimized)

Use [know-cli](https://github.com/sushilk1991/know-cli) as the default code-retrieval layer for agents.
Goal: minimize token waste, pick the right edit target quickly, and keep context quality high.

## Non-Negotiable Defaults

- Prefer a single-call retrieval first: `know workflow`.
- Prefer machine output for agents: `--json --quiet`.
- Use small budgets first, then escalate only if needed.
- Use session dedup across follow-ups (`--session auto` or persisted session id).
- Use fallback ladder when confidence is low.

## Primary Flow (Default)

Run this first for most coding tasks:

```bash
know --json workflow "<task or bug description>" \
  --map-limit 20 \
  --context-budget 4000 \
  --deep-budget 3000 \
  --session auto
```

This gives map + context + deep in one call with fewer tool round trips.

## Fallback Ladder (When Workflow Is Not Enough)

Use this exact fallback order:

1. `know map "<query>" --json --limit 30`
2. `know next-file "<query>" --json`
3. `know context "<query>" --budget 4000 --session auto --json --quiet`
4. `know deep "<symbol or file:symbol>" --budget 3000 --json`
5. `know related <file_path> --json`

If still ambiguous, increase budget in small steps: `4000 -> 6000 -> 8000`.

## Edit-Targeting Rules

- Start edits only after `workflow` or `context` returns concrete file paths and symbols.
- Prefer deep targets using `file:symbol` when names collide.
- If `deep` is ambiguous, retry with the candidate file hint from map/context.
- For dependency-sensitive changes, run `know graph <file_path>` before editing.

## Search and Recall

Use semantic search only when map/context misses:

```bash
know search "<query>" --json
know search "<query>" --chunk --json
```

Recall and persist project-specific knowledge:

```bash
know recall "<query>"
know remember "<insight>"
know decide "<decision>" --why "<rationale>" --evidence <file:line>
know recall "<query>" --type decision --status active
know memories resolve <id> --status resolved
```

Examples:

```bash
know remember "Auth middleware validates JWT then hydrates request.user"
know recall "where is token validation and refresh logic"
know decide "Use RRF-fused retrieval lanes for memory recall" \
  --why "more robust than single-lane fallback" \
  --evidence src/know/knowledge_base.py:190
```

## Structured Memory Policy

- Store architectural choices as `decision` memories, not generic notes.
- Include evidence (`file:line`) for every decision memory whenever possible.
- Use `session_id` for workflow/deep runs so follow-up recall can bind to the same thread.
- Keep trust strict:
  - `local_verified` for direct code observations
  - `imported_unverified` for imported memory sets
  - `blocked` for poisoned/unsafe notes (excluded by default recall)

## Context Commands (Direct Use)

```bash
know context "<task>" --budget 4000 --session auto --json --quiet
know context "<task>" --budget 8000 --session auto --json --quiet
know map "<feature area>" --type function --json
know deep "service.py:process_payment" --budget 3000 --json
know graph src/auth/middleware.py
```

## Token Discipline

- Use `map`/`next-file` before broad context when task is narrow.
- Keep `context-budget` near 4000 for first pass.
- Reuse session ids to enforce dedup and avoid re-sending the same chunks.
- Only escalate to wider context after specific gaps are identified.
- Run `know reindex` only after major refactors or when stale-index symptoms appear.
