#!/usr/bin/env python3
"""Suite 3+4: End-to-End Agent Benchmark + Quality Scoring.

Three agents answer 3 questions from benchmark_task.md:
  - v0.7.0: know_map + know_context (session) + know_deep
  - v0.6.0: know_context (no session)
  - grep+read: grep_search + read_file

Uses Anthropic SDK with tool-use, temperature=0.
LLM-as-judge scores answers against factual checklists.

Requires ANTHROPIC_API_KEY environment variable.
"""

import json
import os
import subprocess
import time
from pathlib import Path

from conftest import (
    AGENT_QUESTIONS, AGENT_MODEL, AGENT_MAX_TURNS, AGENT_TEMPERATURE,
    FARFIELD_DIR, ANSWER_KEYS_DIR,
    get_know_version, get_know_status, save_results, has_api_key,
)

try:
    import anthropic
except ImportError:
    anthropic = None


# ── Tool Definitions ───────────────────────────────────────────────────────────

TOOLS_V7 = [
    {
        "name": "know_map",
        "description": "Lightweight signature search. Returns function/class signatures matching a query (no bodies). Use to orient before reading code.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query for function/class signatures"},
                "limit": {"type": "integer", "description": "Max results (default 20)", "default": 20},
            },
            "required": ["query"],
        },
    },
    {
        "name": "know_context",
        "description": "Build LLM-optimized context for a query. Returns ranked code chunks with bodies. Uses session-based deduplication.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural language query about the codebase"},
                "budget": {"type": "integer", "description": "Token budget (default 4000)", "default": 4000},
            },
            "required": ["query"],
        },
    },
    {
        "name": "know_deep",
        "description": "Deep context for a specific function: body + callers + callees. Use after know_context to dive deeper.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Function name (e.g. 'check_access', 'Class.method')"},
                "budget": {"type": "integer", "description": "Token budget (default 3000)", "default": 3000},
            },
            "required": ["name"],
        },
    },
]

TOOLS_V6 = [
    {
        "name": "know_context",
        "description": "Build LLM-optimized context for a query. Returns ranked code chunks with bodies.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural language query about the codebase"},
                "budget": {"type": "integer", "description": "Token budget (default 8000)", "default": 8000},
            },
            "required": ["query"],
        },
    },
]

TOOLS_GREP = [
    {
        "name": "grep_search",
        "description": "Search for a pattern in the codebase using grep. Returns matching lines with file paths.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Search pattern (case-insensitive)"},
                "file_types": {"type": "string", "description": "Comma-separated extensions (e.g. 'py,ts')", "default": "py,ts,tsx,js,jsx"},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "read_file",
        "description": "Read the contents of a file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative file path from project root"},
                "start_line": {"type": "integer", "description": "Start line (optional)"},
                "end_line": {"type": "integer", "description": "End line (optional)"},
            },
            "required": ["path"],
        },
    },
]


# ── Tool Execution ─────────────────────────────────────────────────────────────

def execute_tool(name: str, params: dict, session_id: str | None = None) -> str:
    """Execute a tool call by shelling out to real CLI."""

    if name == "know_map":
        cmd = ["know", "--json", "map", params["query"]]
        if params.get("limit"):
            cmd += ["--limit", str(params["limit"])]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(FARFIELD_DIR), timeout=60)
        return result.stdout or result.stderr

    elif name == "know_context":
        budget = params.get("budget", 8000)
        cmd = ["know", "--json", "context", params["query"], "--budget", str(budget)]
        if session_id:
            cmd += ["--session", session_id]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(FARFIELD_DIR), timeout=60)
        return result.stdout or result.stderr

    elif name == "know_deep":
        budget = params.get("budget", 3000)
        cmd = ["know", "--json", "deep", params["name"], "--budget", str(budget)]
        if session_id:
            cmd += ["--session", session_id]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(FARFIELD_DIR), timeout=60)
        return result.stdout or result.stderr

    elif name == "grep_search":
        pattern = params["pattern"]
        file_types = params.get("file_types", "py,ts,tsx,js,jsx")
        includes = []
        for ext in file_types.split(","):
            includes += ["--include", f"*.{ext.strip()}"]
        result = subprocess.run(
            ["grep", "-rn", "-i"] + includes + [pattern],
            capture_output=True, text=True, cwd=str(FARFIELD_DIR), timeout=30,
        )
        lines = result.stdout.strip().split("\n")
        # Truncate to avoid huge outputs
        if len(lines) > 50:
            return "\n".join(lines[:50]) + f"\n... ({len(lines)} total matches)"
        return result.stdout or "(no matches)"

    elif name == "read_file":
        fpath = FARFIELD_DIR / params["path"]
        if not fpath.exists():
            return f"Error: file not found: {params['path']}"
        content = fpath.read_text(errors="replace")
        lines = content.split("\n")
        start = params.get("start_line", 1) - 1
        end = params.get("end_line", len(lines))
        selected = lines[max(0, start):end]
        # Truncate
        if len(selected) > 200:
            selected = selected[:200] + [f"... (truncated, {len(lines)} total lines)"]
        return "\n".join(f"{i+start+1}: {line}" for i, line in enumerate(selected))

    return f"Unknown tool: {name}"


# ── Agent Loop ─────────────────────────────────────────────────────────────────

def run_agent(
    client: "anthropic.Anthropic",
    question: str,
    tools: list[dict],
    agent_name: str,
    session_id: str | None = None,
) -> dict:
    """Run an agent loop: send question, handle tool calls, collect metrics."""
    system = (
        "You are investigating the farfield codebase. "
        "Answer the question using the available tools. Be thorough but concise. "
        "Include specific file paths, function names, and how components connect. "
        "When you have enough information, provide your final answer."
    )

    messages = [{"role": "user", "content": question}]
    tool_calls = 0
    total_input_tokens = 0
    total_output_tokens = 0
    turns = 0
    final_answer = ""

    for turn in range(AGENT_MAX_TURNS):
        turns += 1
        try:
            response = client.messages.create(
                model=AGENT_MODEL,
                max_tokens=4096,
                temperature=AGENT_TEMPERATURE,
                system=system,
                tools=tools,
                messages=messages,
            )
        except Exception as e:
            final_answer = f"[ERROR: {e}]"
            break

        total_input_tokens += response.usage.input_tokens
        total_output_tokens += response.usage.output_tokens

        # Check if done (no tool use, just text)
        if response.stop_reason == "end_turn":
            for block in response.content:
                if block.type == "text":
                    final_answer += block.text
            break

        # Process response content
        assistant_content = []
        tool_results = []

        for block in response.content:
            if block.type == "text":
                assistant_content.append({"type": "text", "text": block.text})
                final_answer += block.text
            elif block.type == "tool_use":
                tool_calls += 1
                assistant_content.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })
                result = execute_tool(block.name, block.input, session_id)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })

        messages.append({"role": "assistant", "content": assistant_content})
        if tool_results:
            messages.append({"role": "user", "content": tool_results})

    return {
        "agent": agent_name,
        "tool_calls": tool_calls,
        "turns": turns,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens,
        "answer": final_answer,
    }


# ── Quality Scoring ────────────────────────────────────────────────────────────

def score_answer(client: "anthropic.Anthropic", answer: str, answer_key: dict) -> dict:
    """Use LLM-as-judge to score an answer against factual checklist."""
    facts_text = "\n".join(
        f"- [{f['id']}] (weight={f['weight']}): {f['description']}"
        for f in answer_key["facts"]
    )

    prompt = f"""Score this answer about "{answer_key['topic']}" against the following factual checklist.

For each fact, respond with PRESENT (full weight), PARTIAL (half weight), or ABSENT (0).

Facts to check:
{facts_text}

Answer to evaluate:
{answer}

Respond with ONLY a JSON object in this format:
{{
  "scores": {{
    "f1": {{"verdict": "PRESENT|PARTIAL|ABSENT", "evidence": "brief quote or reason"}},
    "f2": ...
  }},
  "total_score": <number>,
  "max_score": {answer_key['max_score']}
}}"""

    try:
        response = client.messages.create(
            model=AGENT_MODEL,
            max_tokens=2048,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as e:
        return {"error": f"API call failed: {e}", "total_score": 0, "max_score": answer_key["max_score"]}

    text = response.content[0].text
    # Extract JSON from response
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        return json.loads(text[start:end])
    except (ValueError, json.JSONDecodeError):
        return {"error": "Failed to parse judge response", "raw": text[:500]}


# ── Main Suite ─────────────────────────────────────────────────────────────────

def run_suite():
    """Run end-to-end agent benchmark with quality scoring."""
    if not has_api_key():
        print("Suite 3+4: Skipped (no ANTHROPIC_API_KEY)")
        return None

    if anthropic is None:
        print("Suite 3+4: Skipped (anthropic package not installed)")
        print("  Install with: pip install anthropic")
        return None

    print("=" * 60)
    print("Suite 3+4: End-to-End Agent Benchmark + Quality Scoring")
    print("=" * 60)

    client = anthropic.Anthropic()
    version = get_know_version()

    agents = [
        {"name": "v7_3tier", "tools": TOOLS_V7, "session": True},
        {"name": "v6_context", "tools": TOOLS_V6, "session": False},
        {"name": "grep_read", "tools": TOOLS_GREP, "session": False},
    ]

    # Load answer keys
    answer_keys = {}
    for q in AGENT_QUESTIONS:
        key_path = ANSWER_KEYS_DIR / f"{q['id']}.json"
        if key_path.exists():
            answer_keys[q["id"]] = json.loads(key_path.read_text())

    all_results = []

    for agent_cfg in agents:
        agent_name = agent_cfg["name"]
        print(f"\n  Agent: {agent_name}")

        # Create session for v7 agent
        import uuid
        session_id = f"bench-{uuid.uuid4().hex[:8]}" if agent_cfg["session"] else None

        agent_results = []
        for q in AGENT_QUESTIONS:
            print(f"    Q: {q['id']}...", end=" ", flush=True)
            t0 = time.monotonic()
            result = run_agent(
                client, q["question"], agent_cfg["tools"], agent_name, session_id,
            )
            elapsed = time.monotonic() - t0
            result["elapsed_s"] = round(elapsed, 1)
            result["question_id"] = q["id"]

            # Quality scoring
            if q["id"] in answer_keys:
                score = score_answer(client, result["answer"], answer_keys[q["id"]])
                result["quality"] = score
                score_str = f", quality={score.get('total_score', '?')}/{score.get('max_score', '?')}"
            else:
                score_str = ""

            print(f"{result['tool_calls']} calls, {result['total_tokens']:,} tok, {elapsed:.1f}s{score_str}")
            agent_results.append(result)

        all_results.append({
            "agent": agent_name,
            "questions": agent_results,
            "summary": {
                "avg_tool_calls": round(sum(r["tool_calls"] for r in agent_results) / len(agent_results), 1),
                "avg_total_tokens": round(sum(r["total_tokens"] for r in agent_results) / len(agent_results)),
                "total_tool_calls": sum(r["tool_calls"] for r in agent_results),
                "total_tokens": sum(r["total_tokens"] for r in agent_results),
                "avg_quality": round(
                    sum(r.get("quality", {}).get("total_score", 0) for r in agent_results) / len(agent_results), 1
                ) if any(r.get("quality") for r in agent_results) else None,
            },
        })

    save_results("agent_e2e.json", {
        "suite": "agent_e2e",
        "version": version,
        "model": AGENT_MODEL,
        "agents": all_results,
    })

    return all_results


if __name__ == "__main__":
    run_suite()
