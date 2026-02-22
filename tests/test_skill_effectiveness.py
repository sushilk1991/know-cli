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

