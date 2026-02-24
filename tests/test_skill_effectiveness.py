"""Quality checks for KNOW_SKILL.md agent guidance."""

from pathlib import Path


def _skill_text() -> str:
    skill_path = Path(__file__).parent.parent / "KNOW_SKILL.md"
    return skill_path.read_text(encoding="utf-8")


def test_skill_prefers_single_call_workflow():
    skill = _skill_text().lower()
    assert "know workflow" in skill
    assert "--context-budget" in skill
    assert "--deep-budget" in skill


def test_skill_mentions_session_dedup():
    skill = _skill_text().lower()
    assert "--session auto" in skill or "--session" in skill
    assert "dedup" in skill


def test_skill_has_fallback_playbook():
    skill = _skill_text().lower()
    assert "fallback" in skill
    assert "know map" in skill
    assert "know deep" in skill
    assert "know next-file" in skill


def test_skill_has_use_case_matrix():
    skill = _skill_text().lower()
    assert "use-case" in skill or "use case" in skill
    assert "workflow" in skill
    assert "map" in skill
    assert "context" in skill
    assert "deep" in skill


def test_skill_mentions_cross_agent_memory():
    skill = _skill_text().lower()
    assert "cross-agent" in skill or "cross agent" in skill
    assert "codex" in skill
    assert "claude" in skill
    assert "gemini" in skill


def test_skill_has_anti_patterns_and_recovery():
    skill = _skill_text().lower()
    assert "anti-pattern" in skill or "anti pattern" in skill
    assert "know doctor --repair --reindex" in skill
