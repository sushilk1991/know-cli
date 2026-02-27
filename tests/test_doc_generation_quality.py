"""Quality/regression tests for generated architecture docs."""

import re

from know.config import Config
from know.generator import DocGenerator


def test_system_doc_is_evidence_based(tmp_path):
    """arc.md should be grounded in scanned facts, not free-form project-name guesses."""
    config = Config.create_default(tmp_path)
    config.project.name = "farfield"
    config.project.description = ""
    config.output.directory = "docs"

    generator = DocGenerator(config)
    structure = {
        "modules": [
            {
                "name": "apps.api.main",
                "path": "apps/api/main.py",
                "description": "",
                "function_count": 2,
                "class_count": 1,
            },
            {
                "name": "apps.web.page",
                "path": "apps/web/page.tsx",
                "description": "",
                "function_count": 1,
                "class_count": 0,
            },
        ],
        "key_files": ["apps/api/main.py"],
        "file_count": 2,
        "module_count": 2,
        "function_count": 3,
        "class_count": 1,
    }

    path = generator.generate_system_doc(structure)
    content = path.read_text(encoding="utf-8")

    assert "Generated from static code scan" in content
    assert "Primary languages:" in content
    assert "apps/" in content
    assert "electromagnetic" not in content.lower()


def test_architecture_diagram_uses_unique_node_ids_and_known_edges_only(tmp_path, monkeypatch):
    """architecture.md should avoid node-id collisions and edges to unknown nodes."""
    config = Config.create_default(tmp_path)
    config.output.directory = "docs"
    generator = DocGenerator(config)

    structure = {
        "modules": [
            {
                "name": "apps.api.__init__",
                "path": "apps/api/__init__.py",
                "function_count": 1,
                "class_count": 0,
            },
            {
                "name": "apps.web.__init__",
                "path": "apps/web/__init__.py",
                "function_count": 1,
                "class_count": 0,
            },
            {
                "name": "apps.api.router",
                "path": "apps/api/router.py",
                "function_count": 3,
                "class_count": 0,
            },
        ]
    }

    monkeypatch.setattr(
        generator,
        "_resolve_imports",
        lambda _structure: [
            ("apps.api.router", "apps.api.__init__"),
            ("apps.api.router", "apps.web.__init__"),
            ("apps.api.router", "unknown.module"),
        ],
    )

    path = generator.generate_c4_diagram(structure)
    content = path.read_text(encoding="utf-8")

    node_ids = set(re.findall(r"^\s+(m\d+)\[", content, flags=re.MULTILINE))
    edge_pairs = re.findall(r"^\s+(m\d+)\s+-->\s+(m\d+)", content, flags=re.MULTILINE)

    # Both colliding "__init__" modules should be represented without ID collision.
    assert len(node_ids) >= 3
    assert len(edge_pairs) == 2
    assert all(src in node_ids and dst in node_ids for src, dst in edge_pairs)
    assert "unknown.module" not in content
    assert "__init__ -->" not in content


def test_resolve_imports_keeps_fully_qualified_targets_when_short_names_collide(tmp_path):
    """`import a.utils` should resolve to a.utils even when b.utils also exists."""
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "x").mkdir()
    (tmp_path / "a" / "utils.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "b" / "utils.py").write_text("VALUE = 2\n", encoding="utf-8")
    (tmp_path / "x" / "main.py").write_text("import a.utils\n", encoding="utf-8")

    config = Config.create_default(tmp_path)
    generator = DocGenerator(config)
    structure = {
        "modules": [
            {"name": "a.utils", "path": "a/utils.py"},
            {"name": "b.utils", "path": "b/utils.py"},
            {"name": "x.main", "path": "x/main.py"},
        ]
    }

    edges = generator._resolve_imports(structure)
    assert ("x.main", "a.utils") in edges
    assert ("x.main", "b.utils") not in edges


def test_resolve_imports_avoids_symbol_module_false_positive_from_import(tmp_path):
    """`from pkg.settings import config` should not map to unrelated `*.config` modules."""
    (tmp_path / "pkg").mkdir()
    (tmp_path / "app").mkdir()
    (tmp_path / "pkg" / "settings.py").write_text("config = {'env': 'dev'}\n", encoding="utf-8")
    (tmp_path / "app" / "config.py").write_text("ENV = 'prod'\n", encoding="utf-8")
    (tmp_path / "app" / "main.py").write_text("from pkg.settings import config\n", encoding="utf-8")

    config = Config.create_default(tmp_path)
    generator = DocGenerator(config)
    structure = {
        "modules": [
            {"name": "pkg.settings", "path": "pkg/settings.py"},
            {"name": "app.config", "path": "app/config.py"},
            {"name": "app.main", "path": "app/main.py"},
        ]
    }

    edges = generator._resolve_imports(structure)
    assert ("app.main", "pkg.settings") in edges
    assert ("app.main", "app.config") not in edges


def test_system_doc_tolerates_non_numeric_counts(tmp_path):
    """Loose/non-numeric structure counters should not crash generation."""
    config = Config.create_default(tmp_path)
    config.output.directory = "docs"
    generator = DocGenerator(config)

    structure = {
        "modules": [{"name": "mod.one", "path": "mod/one.py", "function_count": "N/A", "class_count": None}],
        "key_files": ["mod/one.py"],
        "file_count": "N/A",
        "module_count": None,
        "function_count": "unknown",
        "class_count": "unknown",
    }

    path = generator.generate_system_doc(structure)
    content = path.read_text(encoding="utf-8")
    assert path.exists()
    assert "This repository contains 1 source files across 1 modules." in content


def test_dependency_graph_keeps_at_least_one_real_edge_when_truncated(tmp_path, monkeypatch):
    """When module count exceeds cap, connected modules should still be shown."""
    config = Config.create_default(tmp_path)
    config.output.directory = "docs"
    generator = DocGenerator(config)

    modules = [
        {"name": f"pkg.m{i}", "path": f"pkg/m{i}.py", "function_count": 0, "class_count": 0}
        for i in range(20)
    ]
    structure = {"modules": modules}

    monkeypatch.setattr(
        generator,
        "_resolve_imports",
        lambda _structure: [("pkg.m18", "pkg.m19")],
    )

    path = generator.generate_dependency_graph(structure)
    content = path.read_text(encoding="utf-8")

    edge_pairs = re.findall(r"^\s+(d\d+)\s+-->\s+(d\d+)", content, flags=re.MULTILINE)
    assert len(edge_pairs) >= 1
    assert "prioritizing connected nodes" in content


def test_generate_onboarding_uses_fallback_when_api_key_missing(tmp_path, monkeypatch):
    config = Config.create_default(tmp_path)
    config.output.directory = "docs"
    generator = DocGenerator(config)
    monkeypatch.delenv(config.ai.api_key_env, raising=False)

    path = generator.generate_onboarding({"modules": [], "key_files": []})
    content = path.read_text(encoding="utf-8")
    assert "Onboarding Guide" in content


def test_generate_onboarding_uses_ai_branch_when_api_key_present(tmp_path, monkeypatch):
    config = Config.create_default(tmp_path)
    config.output.directory = "docs"
    generator = DocGenerator(config)
    monkeypatch.setenv(config.ai.api_key_env, "test-key")

    class FakeAISummarizer:
        def __init__(self, _config):
            pass

        def generate_onboarding_guide(self, _structure, _audience):
            return "AI GUIDE"

    import know.ai as ai_module

    monkeypatch.setattr(ai_module, "AISummarizer", FakeAISummarizer)
    path = generator.generate_onboarding({"modules": [], "key_files": []})
    content = path.read_text(encoding="utf-8")
    assert "AI GUIDE" in content
