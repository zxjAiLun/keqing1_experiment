"""Static import-boundary gate for the future repository split.

Enforces (on source, via AST; lazy imports included because the scan walks
every ``Import``/``ImportFrom`` node):

- ``src/**`` never imports ``training/**``
- ``training/**`` never imports Workbench packages (``workbench``,
  ``replay_ui``, ``replay``, ``gateway``, ``participants``, ``convert``,
  ``tools``, ``scripts``)
- ``workbench/**`` never imports ``training/**``

Violations fail before the repos can be split; data files are the interface
between Training and Workbench, not Python imports.
"""
from __future__ import annotations

import ast
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]

TRAINING_TOPS = {"training"}
WORKBENCH_TOPS = {
    "workbench",
    "replay_ui",
    "replay",
    "gateway",
    "participants",
    "convert",
    "tools",
    "scripts",
}


def _first_party_tops(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tops: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            tops.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            tops.add(node.module.split(".")[0])
    return tops


def _scan(root: str) -> list[tuple[Path, set[str]]]:
    results: list[tuple[Path, set[str]]] = []
    base = _REPO_ROOT / root
    for py in sorted(base.rglob("*.py")):
        if "__pycache__" in py.parts:
            continue
        results.append((py.relative_to(_REPO_ROOT), _first_party_tops(py)))
    return results


def test_src_never_imports_training() -> None:
    offenders = [
        str(path)
        for path, tops in _scan("src")
        if tops & TRAINING_TOPS
    ]
    assert not offenders, f"src/ must not import training/: {offenders}"


def test_training_never_imports_workbench() -> None:
    offenders = [
        str(path)
        for path, tops in _scan("training")
        if tops & WORKBENCH_TOPS
    ]
    assert not offenders, f"training/ must not import workbench/: {offenders}"


def test_workbench_never_imports_training() -> None:
    offenders = [
        str(path)
        for path, tops in _scan("workbench")
        if tops & TRAINING_TOPS
    ]
    assert not offenders, f"workbench/ must not import training/: {offenders}"
