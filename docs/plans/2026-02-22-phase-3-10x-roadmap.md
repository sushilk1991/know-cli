# know-cli: 10x Roadmap (Post Phase 2)

Date: 2026-02-22

## Current State (Measured)

Using `benchmark/bench_dual_repo_parallel.py` on `know-cli` and `farfield`:

- Token reduction vs grep+read: **93.1%** (`know-cli`) and **94.9%** (`farfield`)
- Tool-call reduction: **81.2%** on both repos
- Latency ratio (`know/grep`): **7.65x** (`know-cli`) and **4.63x** (`farfield`)
- Deep call-graph availability: **100%** on both repos in this run
- Deep non-empty caller/callee edges: **100%** (`know-cli`) and **25%** (`farfield`)

Interpretation:
- The product is already excellent on token efficiency.
- The biggest gaps are **latency** and **call-graph completeness on larger mixed-language repos**.

## Audit Findings

1. CLI process overhead dominates multi-step workflows.
- The 3-tier workflow executes separate CLI processes (`map`, `context`, `deep`), each with startup + config + daemon handshake overhead.

2. `context` does not use daemon-side full pipeline.
- `know context` runs `ContextEngine` in the CLI process via direct DB access.
- Daemon has a lightweight `_handle_context`, but not parity with the richer v3 pipeline used by CLI.

3. `related` still has expensive fallback behavior.
- It refreshes stale target file (good), but then does language-agnostic fallback through full scanner traversal when daemon data is incomplete.

4. Call graph still under-extracts in large TSX/Python codebases.
- We improved extraction and freshness, but farfield still returns few non-empty deep edges for many queries.
- Regex fallback parsers (or unresolved dynamic dispatch) limit symbol-ref recall.

5. Ranking lane weighting bug was present and is now fixed.
- `search_chunks` previously tied lane weight to append order; if OR lane was empty, AND lane could lose intended weight.
- Fixed by storing explicit `(lane, weight)` tuples and added regression coverage.

## 10x Plan

## Phase 3A (P0): Latency First

1. Add a single daemon RPC for the full 3-tier workflow.
- New RPC: `workflow(query, context_budget, deep_budget, session_id)` returning `map+context+deep`.
- Benefit: eliminate 2 extra Python process startups and 2 extra socket round-trips.

2. Move `context` to daemon-side parity with `ContextEngine._build_context_v3_inner`.
- Keep DB connection warm and reuse parser/query objects in daemon memory.
- Target: p50 warm `context` <150ms on mid-size repos.

3. Add short-lived query-result cache in daemon.
- Key by `(query, budgets, include/exclude filters, index_version)`.
- Cache TTL 30-120s; invalidate on file-index updates.

4. Replace full-scan fallback in `related` with index-first expansion.
- Use indexed import graph + symbol refs first, scanner fallback only when confidence is low.

## Phase 3B (P0/P1): Call Graph Completeness

1. Prioritize function/method chunks for `deep` target resolution by default.
- Avoid constants being selected as deep targets unless explicitly requested.

2. Increase TS/TSX/Python call-ref extraction coverage.
- Add more AST node patterns for JSX handlers, member calls, decorators, async wrappers, and factory exports.
- Add bounded type-hint-assisted resolution for Python method calls where receiver type is obvious.

3. Add a confidence score for call-graph completeness.
- Report per-result signals: parser type (tree-sitter vs regex), symbol-ref density, unresolved reference rate.
- Let agents decide when to trust `deep` vs fallback grep/read.

## Phase 3C (P1): Retrieval Quality and Cost Control

1. Add selective retrieval gate before heavy context generation.
- If top map signals are high-confidence and task appears local, skip deep/context expansion.
- This is directly aligned with selective retrieval literature.

2. Train a lightweight reranker from real agent traces.
- Learn from accepted edits, opened files, and final patch success.
- Use listwise reranking over candidate chunks from FTS lanes.

3. Add budget-aware verification rerank stage.
- Use a cheap verifier signal (tests touched files, static analysis hints, symbol overlap) before spending deep budget.

## Phase 3D (P1): Benchmark and Evaluation

1. Add execution-based benchmark track (not token-only).
- For each task: evaluate retrieval quality (`Recall@k` for gold touched files), fix success, and end-to-end elapsed time.

2. Add multilingual benchmark matrix per language.
- Python, TS/TSX/JS/JSX, Go, Rust, Swift separately.
- Track deep non-empty edge rate and false-positive rate.

3. Add contamination-resistant benchmark generation.
- Use recent PR-derived tasks and stronger test augmentation.

## Research-Backed Ideas to Productize

1. Iterative retrieval-generation loop (RepoCoder, EMNLP 2023).
- Idea: after first retrieval, use model draft/partial hypothesis to issue a second focused retrieval pass.
- Source: https://aclanthology.org/2023.emnlp-main.151/

2. Selective retrieval gate (Repoformer, ICML 2024).
- Idea: skip retrieval when unnecessary; improves speed with little/no quality loss.
- Reported online inference speedup up to 70%.
- Source: https://proceedings.mlr.press/v235/wu24a.html

3. Learned retriever with stop signal (RLCoder, 2024).
- Idea: learn when retrieval helps and when to stop adding context.
- Reported +12.2 EM on RepoEval/CrossCodeEval over prior methods.
- Source: https://arxiv.org/abs/2407.19487

4. Graph-based retrieval expansion (RepoHyper, 2024).
- Idea: search-expand-refine on repository semantic graph (imports/calls/types), then rerank.
- Source: https://arxiv.org/abs/2403.06095

5. Code-edit retriever tuned for dependencies (CoRet, 2025).
- Idea: include call-graph and repository structure directly in retrieval objective.
- Reported +15 recall points on bug-localization benchmarks.
- Source: https://arxiv.org/abs/2505.24715

6. Stronger evaluation rigor (UTBoost 2025; SWE-Bench++ 2025; FeatureBench 2026).
- Idea: test augmentation and larger multilingual execution-based benchmarks to prevent metric inflation.
- Sources:
  - https://aclanthology.org/2025.acl-long.189/
  - https://arxiv.org/abs/2512.17419
  - https://arxiv.org/abs/2602.10975

## Suggested Success Targets (next release cycle)

- Warm `map`: p50 <80ms, p95 <200ms
- Warm `context`: p50 <150ms, p95 <350ms
- Warm `deep`: p50 <250ms, p95 <600ms
- Deep non-empty edge rate on farfield-like repos: from ~25% -> **>=70%**
- Keep token reduction vs grep+read above **90%**
