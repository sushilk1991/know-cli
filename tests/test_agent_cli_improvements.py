"""Focused tests for agent command quality improvements."""

from pathlib import Path

from know.cli.agent import _file_intent_boost, _query_domain_intent
from know.import_graph import ImportGraph, _extract_import_target, _resolve_relative_import
from know.parsers import PythonParser, TypeScriptRegexParser


def test_typescript_regex_parser_extracts_arrow_component_with_body(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    src = root / "src"
    src.mkdir()
    file_path = src / "sidebar.tsx"
    file_path.write_text(
        'import { useState } from "react"\n'
        "export const AppSidebar = () => {\n"
        "  const [open, setOpen] = useState(false)\n"
        "  return <div>{open ? 'open' : 'closed'}</div>\n"
        "}\n",
        encoding="utf-8",
    )

    parser = TypeScriptRegexParser()
    mod = parser.parse(file_path, root)
    funcs = [f for f in mod.functions if f.name == "AppSidebar"]
    assert funcs, "Expected AppSidebar to be parsed as a function/component"
    assert funcs[0].end_line >= funcs[0].line_number + 2


def test_related_files_from_modules_handles_tsx_relative_imports():
    modules = [
        {
            "path": "apps/web/components/sidebar.tsx",
            "name": "apps.web.components.sidebar",
            "imports": ['import { SettingsPage } from "../pages/settings/page"'],
        },
        {
            "path": "apps/web/pages/settings/page.tsx",
            "name": "apps.web.pages.settings.page",
            "imports": [],
        },
    ]

    imports, imported_by = ImportGraph.related_files_from_modules(
        "apps/web/components/sidebar.tsx",
        modules,
    )
    assert "apps/web/pages/settings/page.tsx" in imports
    assert imported_by == []

    _, reverse_imported_by = ImportGraph.related_files_from_modules(
        "apps/web/pages/settings/page.tsx",
        modules,
    )
    assert "apps/web/components/sidebar.tsx" in reverse_imported_by


def test_query_intent_detects_frontend_and_boosts_frontend_paths():
    intent = _query_domain_intent("fix sidebar redirect in react page")
    assert intent == "frontend"
    assert _file_intent_boost("apps/web/components/sidebar.tsx", intent) > 0
    assert _file_intent_boost("apps/api/src/core/config.py", intent) < 0


def test_typescript_parser_captures_export_from_as_import(tmp_path):
    root = tmp_path / "proj2"
    root.mkdir()
    src = root / "src"
    src.mkdir()
    idx = src / "index.ts"
    idx.write_text(
        'export { AppSidebar, type SettingsTab } from "./sidebar"\n',
        encoding="utf-8",
    )
    parser = TypeScriptRegexParser()
    mod = parser.parse(idx, root)
    assert mod.imports
    assert "./sidebar" in mod.imports[0]


def test_typescript_parser_captures_multiline_import_target(tmp_path):
    root = tmp_path / "proj3"
    root.mkdir()
    src = root / "src"
    src.mkdir()
    file_path = src / "view.tsx"
    file_path.write_text(
        "import {\n"
        "  Sidebar,\n"
        "  SidebarInset,\n"
        '} from "./ui/sidebar"\n',
        encoding="utf-8",
    )
    parser = TypeScriptRegexParser()
    mod = parser.parse(file_path, root)
    assert any("./ui/sidebar" in imp for imp in mod.imports)


def test_extract_import_target_supports_python_relative_imports():
    assert _extract_import_target("from .utils import parse") == ".utils"
    assert _extract_import_target("from ..core.types import User") == "..core.types"
    assert _extract_import_target(".utils") == ".utils"


def test_resolve_relative_import_normalizes_windows_style_paths():
    module_to_file = {"apps.web.pages.settings.page": "apps\\web\\pages\\settings\\page.tsx"}
    resolved = _resolve_relative_import(
        "apps/web/components/sidebar.tsx",
        "../pages/settings/page",
        module_to_file,
    )
    assert resolved == "apps/web/pages/settings/page.tsx"


def test_resolve_relative_import_supports_python_dot_levels():
    module_to_file = {
        "pkg.mod.utils": "pkg/mod/utils.py",
        "pkg.core.types": "pkg/core/types.py",
    }
    same_pkg = _resolve_relative_import("pkg/mod/service.py", ".utils", module_to_file)
    parent_pkg = _resolve_relative_import("pkg/mod/service.py", "..core.types", module_to_file)
    assert same_pkg == "pkg/mod/utils.py"
    assert parent_pkg == "pkg/core/types.py"


def test_python_parser_preserves_relative_import_levels(tmp_path):
    root = tmp_path / "pyproj"
    root.mkdir()
    pkg = root / "pkg"
    pkg.mkdir()
    f = pkg / "service.py"
    f.write_text(
        "from .utils import parse\nfrom ..core.types import User\n",
        encoding="utf-8",
    )
    parser = PythonParser()
    mod = parser.parse(f, root)
    assert ".utils" in mod.imports
    assert "..core.types" in mod.imports


def test_related_files_from_modules_resolves_directory_index_imports():
    modules = [
        {
            "path": "apps/web/src/components/layout.tsx",
            "name": "apps.web.src.components.layout",
            "imports": ['import Sidebar from "apps/web/src/components/sidebar"'],
        },
        {
            "path": "apps/web/src/components/sidebar/index.tsx",
            "name": "apps.web.src.components.sidebar.index",
            "imports": [],
        },
    ]

    imports, _ = ImportGraph.related_files_from_modules(
        "apps/web/src/components/layout.tsx",
        modules,
    )
    assert "apps/web/src/components/sidebar/index.tsx" in imports
