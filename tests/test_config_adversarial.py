"""Adversarial configuration-loading regressions."""

from __future__ import annotations

import pytest

from know.config import Config


def test_empty_and_comments_only_config_load_defaults(tmp_path):
    config_path = tmp_path / ".know" / "config.yaml"
    config_path.parent.mkdir()

    for content in ("", "# intentionally empty\n"):
        config_path.write_text(content)
        config = Config.load(config_path)
        assert config.root == tmp_path.resolve()
        assert config.ai.provider == "anthropic"
        assert config.output.directory == "docs"


def test_unknown_nested_keys_are_ignored_for_forward_compatibility(tmp_path):
    config_path = tmp_path / ".know" / "config.yaml"
    config_path.parent.mkdir()
    config_path.write_text(
        """
future_top_level: true
project:
  name: forward-compatible
  future_project_option: value
ai:
  model: model-id
  future_ai_option: value
  generate:
    summaries: false
output:
  directory: generated
  future_output_option: value
  git:
    auto_commit: true
    future_git_option: value
  watch:
    debounce_seconds: 2
    future_watch_option: value
diagrams:
  max_depth: 7
  future_diagram_option: value
api:
  include_examples: false
  future_api_option: value
"""
    )

    config = Config.load(config_path)

    assert config.project.name == "forward-compatible"
    assert config.ai.model == "model-id"
    assert config.ai.generate == {"summaries": False}
    assert config.output.directory == "generated"
    assert config.output.git.auto_commit is True
    assert config.output.watch.debounce_seconds == 2
    assert config.diagrams.max_depth == 7
    assert config.api.include_examples is False


@pytest.mark.parametrize("content", ["- python\n- javascript\n", "false\n", "42\n"])
def test_non_mapping_config_root_is_rejected_instead_of_silently_defaulted(
    tmp_path,
    content,
):
    config_path = tmp_path / ".know" / "config.yaml"
    config_path.parent.mkdir()
    config_path.write_text(content)

    with pytest.raises(ValueError, match="root must be a YAML mapping"):
        Config.load(config_path)


@pytest.mark.parametrize(
    ("content", "field"),
    [
        ("project: nope\n", "project"),
        ("ai: 42\n", "ai"),
        ("languages: python\n", "languages"),
        ("include: [src/, 7]\n", "include"),
        ("ai:\n  generate: true\n", "ai.generate"),
        ("output:\n  git: false\n", "output.git"),
        ("output:\n  watch: []\n", "output.watch"),
    ],
)
def test_malformed_known_sections_are_rejected_instead_of_silently_ignored(
    tmp_path, content, field
):
    config_path = tmp_path / ".know" / "config.yaml"
    config_path.parent.mkdir()
    config_path.write_text(content)

    with pytest.raises(ValueError, match=field.replace(".", r"\.")):
        Config.load(config_path)
