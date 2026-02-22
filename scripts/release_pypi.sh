#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required" >&2
  exit 1
fi

if [[ "${ALLOW_DIRTY:-0}" != "1" ]]; then
  if [[ -n "$(git status --porcelain)" ]]; then
    echo "Working tree is dirty. Commit/stash changes or set ALLOW_DIRTY=1." >&2
    exit 1
  fi
fi

VERSION="$(python3 - <<'PY'
import pathlib, tomllib
data = tomllib.loads(pathlib.Path("pyproject.toml").read_text(encoding="utf-8"))
print(data["project"]["version"])
PY
)"

echo "Preparing release for know-cli $VERSION"

if python3 - <<'PY'
import json, pathlib, tomllib, urllib.request, sys

name = "know-cli"
data = tomllib.loads(pathlib.Path("pyproject.toml").read_text(encoding="utf-8"))
version = data["project"]["version"]
url = f"https://pypi.org/pypi/{name}/json"
with urllib.request.urlopen(url, timeout=20) as r:
    payload = json.load(r)
exists = version in payload.get("releases", {}) and len(payload["releases"][version]) > 0
if exists:
    print(f"{name} {version} already exists on PyPI. Nothing to publish.")
    sys.exit(2)
print(f"{name} {version} not found on PyPI; continuing.")
PY
then
  :
else
  status="$?"
  if [[ "$status" == "2" ]]; then
    exit 0
  fi
  exit "$status"
fi

if [[ -z "${PYPI_API_TOKEN:-}" && -z "${TWINE_PASSWORD:-}" ]]; then
  echo "Missing auth token. Set PYPI_API_TOKEN (preferred) or TWINE_PASSWORD." >&2
  exit 1
fi

TOOL_PY="python3"
if ! python3 -m build --version >/dev/null 2>&1 || ! python3 -m twine --version >/dev/null 2>&1; then
  TOOL_VENV="${TOOL_VENV:-.venv-release}"
  python3 -m venv "$TOOL_VENV"
  "$TOOL_VENV/bin/python" -m pip install --upgrade pip build twine >/dev/null
  TOOL_PY="$TOOL_VENV/bin/python"
fi

rm -rf dist
"$TOOL_PY" -m build
"$TOOL_PY" -m twine check dist/*

export TWINE_USERNAME="__token__"
export TWINE_PASSWORD="${TWINE_PASSWORD:-${PYPI_API_TOKEN:-}}"

"$TOOL_PY" -m twine upload --non-interactive --skip-existing dist/*

python3 - <<'PY'
import json, pathlib, tomllib, urllib.request, sys

name = "know-cli"
data = tomllib.loads(pathlib.Path("pyproject.toml").read_text(encoding="utf-8"))
version = data["project"]["version"]
url = f"https://pypi.org/pypi/{name}/json"
with urllib.request.urlopen(url, timeout=20) as r:
    payload = json.load(r)
exists = version in payload.get("releases", {}) and len(payload["releases"][version]) > 0
if not exists:
    print(f"Publish verification failed: {name} {version} not visible on PyPI yet.")
    sys.exit(1)
print(f"Published and verified: {name}=={version}")
PY
