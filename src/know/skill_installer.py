"""Install and bootstrap the know-cli agent skill into common agent homes."""

from __future__ import annotations

import json
import os
from datetime import datetime, UTC
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

from know import __version__

SKILL_DIR_NAME = "know-cli"
BOOTSTRAP_SCHEMA = 1


def _normalize_home(home: Optional[Path]) -> Path:
    return (home or Path.home()).expanduser()


def _skill_template_candidates() -> list[Path]:
    here = Path(__file__).resolve()
    return [
        here.parent / "resources" / "KNOW_SKILL.md",  # packaged resource
        here.parents[2] / "KNOW_SKILL.md",  # repo checkout
        here.parents[1] / "KNOW_SKILL.md",  # installed wheel include
    ]


def _load_skill_template() -> Tuple[Optional[str], Optional[Path]]:
    for candidate in _skill_template_candidates():
        try:
            if candidate.exists():
                text = candidate.read_text(encoding="utf-8")
                if text.strip():
                    return text, candidate
        except Exception:
            continue
    return None, None


def skill_target_paths(home: Optional[Path] = None) -> Dict[str, Path]:
    """Return default target skill file paths for common coding agents."""
    home_dir = _normalize_home(home)
    codex_home = Path(os.environ.get("CODEX_HOME", str(home_dir / ".codex"))).expanduser()

    return {
        "codex": codex_home / "skills" / SKILL_DIR_NAME / "SKILL.md",
        "claude": home_dir / ".claude" / "skills" / SKILL_DIR_NAME / "SKILL.md",
        "agents": home_dir / ".agents" / "skills" / SKILL_DIR_NAME / "SKILL.md",
    }


def skill_bootstrap_marker_path(home: Optional[Path] = None) -> Path:
    home_dir = _normalize_home(home)
    return home_dir / ".cache" / "know-cli" / "skill_bootstrap.json"


def install_skill_file(
    *,
    home: Optional[Path] = None,
    force: bool = False,
    targets: Optional[Iterable[str]] = None,
) -> dict:
    """Install know skill file into configured target agent directories."""
    template_text, template_path = _load_skill_template()
    target_map = skill_target_paths(home=home)
    selected_targets = set(targets or target_map.keys())

    result = {
        "template_available": bool(template_text),
        "template_path": str(template_path) if template_path else None,
        "installed": [],
        "skipped": [],
        "errors": [],
    }

    if not template_text:
        result["errors"].append({"target": "all", "error": "skill_template_missing"})
        result["installed_count"] = 0
        result["skipped_count"] = 0
        result["error_count"] = len(result["errors"])
        return result

    for target_name, target_path in target_map.items():
        if target_name not in selected_targets:
            continue

        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            if target_path.exists() and not force:
                current = target_path.read_text(encoding="utf-8")
                reason = "already_current" if current == template_text else "exists"
                result["skipped"].append(
                    {"target": target_name, "path": str(target_path), "reason": reason}
                )
                continue

            target_path.write_text(template_text, encoding="utf-8")
            result["installed"].append({"target": target_name, "path": str(target_path)})
        except Exception as e:
            result["errors"].append(
                {"target": target_name, "path": str(target_path), "error": str(e)}
            )

    result["installed_count"] = len(result["installed"])
    result["skipped_count"] = len(result["skipped"])
    result["error_count"] = len(result["errors"])
    return result


def skill_install_status(home: Optional[Path] = None) -> dict:
    """Return current skill installation status for diagnostics."""
    _, template_path = _load_skill_template()
    targets = skill_target_paths(home=home)

    return {
        "template_available": template_path is not None,
        "template_path": str(template_path) if template_path else None,
        "targets": {
            name: {"path": str(path), "exists": path.exists()}
            for name, path in targets.items()
        },
    }


def auto_bootstrap_skill_install(home: Optional[Path] = None) -> dict:
    """Install skill file on first run for the current know-cli version."""
    flag = os.environ.get("KNOW_AUTO_INSTALL_SKILL", "1").strip().lower()
    if flag in {"0", "false", "no", "off"}:
        return {"attempted": False, "reason": "disabled"}

    marker = skill_bootstrap_marker_path(home=home)
    if marker.exists():
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
            if (
                payload.get("schema") == BOOTSTRAP_SCHEMA
                and payload.get("version") == __version__
            ):
                return {"attempted": False, "reason": "already_bootstrapped"}
        except Exception:
            pass

    result = install_skill_file(home=home, force=False)

    try:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(
            json.dumps(
                {
                    "schema": BOOTSTRAP_SCHEMA,
                    "version": __version__,
                    "installed_count": result.get("installed_count", 0),
                    "skipped_count": result.get("skipped_count", 0),
                    "error_count": result.get("error_count", 0),
                    "updated_at": datetime.now(UTC).isoformat(),
                }
            ),
            encoding="utf-8",
        )
    except Exception:
        # Best effort only; missing marker should never break CLI usage.
        pass

    return {
        "attempted": True,
        "reason": "installed",
        "marker": str(marker),
        "result": result,
    }
