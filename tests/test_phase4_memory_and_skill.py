"""Phase 4 TDD: decision memory discipline and skill bootstrap updates."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from know.config import Config


@pytest.fixture
def tmp_project(tmp_path):
    (tmp_path / ".know").mkdir()
    (tmp_path / ".know" / "cache").mkdir()

    config = Config.create_default(tmp_path)
    config.root = tmp_path
    config.project.name = "phase4-project"
    config.save(tmp_path / ".know" / "config.yaml")
    return config


def _workflow_result():
    return {
        "query": "billing routing",
        "session_id": "sess1234",
        "selected_deep_target": "check_cloud_access",
        "context": {
            "confidence": 0.82,
            "source_files": ["src/billing/service.py", "src/api/router.py"],
        },
        "deep": {
            "target": {
                "name": "check_cloud_access",
                "file": "src/billing/service.py",
                "line_start": 11,
            }
        },
    }


class TestDecisionMemoryPolicy:
    def test_workflow_decision_capture_dedups_and_sets_ttl(self, tmp_project):
        from know.knowledge_base import KnowledgeBase
        from know.memory_capture import capture_workflow_decision

        first_id = capture_workflow_decision(
            tmp_project,
            "billing routing",
            _workflow_result(),
            session_id="sess1234",
        )
        second_id = capture_workflow_decision(
            tmp_project,
            "billing routing",
            _workflow_result(),
            session_id="sess1234",
        )

        assert first_id is not None
        assert second_id == first_id

        kb = KnowledgeBase(tmp_project)
        memories = kb.list_all(source="auto-workflow")
        assert len(memories) == 1
        assert memories[0].expires_at != ""

    def test_workflow_decision_dedups_when_focus_order_changes(self, tmp_project):
        from know.knowledge_base import KnowledgeBase
        from know.memory_capture import capture_workflow_decision

        first = _workflow_result()
        second = _workflow_result()
        second["context"]["source_files"] = [
            "src/api/router.py",
            "src/billing/service.py",
            "src/extra.py",
        ]

        first_id = capture_workflow_decision(
            tmp_project,
            "Billing   routing",
            first,
            session_id="sess1234",
        )
        second_id = capture_workflow_decision(
            tmp_project,
            "billing routing",
            second,
            session_id="sess1234",
        )

        assert first_id is not None
        assert second_id == first_id

        kb = KnowledgeBase(tmp_project)
        memories = kb.list_all(source="auto-workflow")
        assert len(memories) == 1
        tags = set((memories[0].tags or "").split(","))
        assert any(tag.startswith("q:") for tag in tags)
        assert any(tag.startswith("t:") for tag in tags)


class TestSkillBootstrapUpdate:
    def test_install_skill_updates_stale_managed_skill(self, tmp_path):
        from know.skill_installer import install_skill_file, skill_target_paths

        home = tmp_path / "home"
        home.mkdir(parents=True, exist_ok=True)
        targets = skill_target_paths(home=home)
        codex_skill = targets["codex"]
        codex_skill.parent.mkdir(parents=True, exist_ok=True)

        # Simulate an old managed skill file from a previous version.
        codex_skill.write_text(
            "---\nname: know-cli\ndescription: old\n---\n# know-cli Skill (Agent-Optimized)\nOLD\n",
            encoding="utf-8",
        )

        result = install_skill_file(home=home, force=False, targets=["codex"])
        assert result["error_count"] == 0
        assert result["installed_count"] == 1
        assert "OLD" not in codex_skill.read_text(encoding="utf-8")
