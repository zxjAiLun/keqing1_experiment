#!/usr/bin/env python3
"""Lightweight doc-layout validator for training/docs/mortal.

Adopted 2026-09-13. Validates ONLY the new canonical layout
(``training/docs/mortal/YYYY-MM/*.md``). Legacy documents -- the loose ``*.md``
files at the top level and ``experiments_zh/`` -- are deliberately NOT
validated: they keep their historical paths and are not rewritten just to
satisfy a check. That keeps this validator cheap and stops it from turning a
readability fix into a repository-wide migration.

Rules (mechanical only):
 1. every ``YYYY-MM/*.md`` has a front-matter block with exactly the four
    fields ``experiment``, ``date``, ``last_updated``, ``status``;
 2. ``date`` is ``YYYY-MM-DD`` and equals the filename date prefix;
 3. ``last_updated`` is ``YYYY-MM-DD`` and ``>= date``.

Deliberately NOT checked: git-history immutability of ``date`` (renames and
shallow checkouts make that unreliable, and it buys nothing here), and the
contents of legacy documents.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

DATE_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
FILENAME_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})_(.+)$")
MONTH_DIR_RE = re.compile(r"^\d{4}-\d{2}$")
REQUIRED_FIELDS = ("experiment", "date", "last_updated", "status")


def parse_front_matter(text: str) -> dict[str, str] | None:
    """Return the front-matter mapping, or None when there is no block."""
    if not text.startswith("---\n"):
        return None
    end = text.find("\n---", 3)
    if end == -1:
        return None
    fields: dict[str, str] = {}
    for line in text[4:end].splitlines():
        if not line.strip():
            continue
        key, separator, value = line.partition(":")
        if not separator:
            return None
        fields[key.strip()] = value.strip()
    return fields


def check_document(path: Path, root: Path) -> list[str]:
    errors: list[str] = []
    relative = path.relative_to(root).as_posix()

    filename_match = FILENAME_RE.match(path.name)
    if not filename_match:
        errors.append(
            f"{relative}: filename must be YYYY-MM-DD_<experiment>_<topic>.md"
        )
    fields = parse_front_matter(path.read_text(encoding="utf-8"))
    if fields is None:
        errors.append(f"{relative}: missing front-matter block")
        return errors

    missing = [name for name in REQUIRED_FIELDS if name not in fields]
    if missing:
        errors.append(f"{relative}: front-matter missing {', '.join(missing)}")
        return errors
    extra = [name for name in fields if name not in REQUIRED_FIELDS]
    if extra:
        errors.append(f"{relative}: front-matter has unexpected {', '.join(extra)}")

    date = fields["date"]
    if not DATE_RE.match(date):
        errors.append(f"{relative}: date must be YYYY-MM-DD, got {date!r}")
    if filename_match and date != filename_match.group(1):
        errors.append(
            f"{relative}: filename date {filename_match.group(1)} != front-matter date {date}"
        )

    last_updated = fields["last_updated"]
    if not DATE_RE.match(last_updated):
        errors.append(
            f"{relative}: last_updated must be YYYY-MM-DD, got {last_updated!r}"
        )
    if DATE_RE.match(date) and DATE_RE.match(last_updated) and last_updated < date:
        errors.append(
            f"{relative}: last_updated {last_updated} is before date {date}"
        )
    return errors


def validate(root: Path) -> list[str]:
    if not root.is_dir():
        return [f"docs root not found: {root}"]

    month_dirs = sorted(
        child
        for child in root.iterdir()
        if child.is_dir() and MONTH_DIR_RE.match(child.name)
    )
    errors: list[str] = []
    for month_dir in month_dirs:
        for doc in sorted(month_dir.glob("*.md")):
            errors.extend(check_document(doc, root))
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("training/docs/mortal"),
        help="directory holding the YYYY-MM/ canonical documents",
    )
    args = parser.parse_args()

    errors = validate(args.root)
    repo_root = Path.cwd()
    errors.extend(
        validate_registry_reports(repo_root / "training/docs/mortal/research_registry.json", repo_root)
    )
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        print(f"doc layout check FAILED ({len(errors)} problem(s))", file=sys.stderr)
        return 1
    print("doc layout check passed")
    return 0


def validate_registry_reports(registry_path: Path, repo_root: Path) -> list[str]:
    """Every registry report_path must resolve (rule 5)."""
    if not registry_path.exists():
        return [f"registry not found: {registry_path}"]
    try:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"registry is not valid JSON: {exc}"]
    errors: list[str] = []
    for record in registry.get("records", []):
        for raw in record.get("report_paths") or []:
            if not (repo_root / raw).exists():
                errors.append(
                    f"registry {record['experiment_id']}: report_path missing: {raw}"
                )
    return errors


if __name__ == "__main__":
    raise SystemExit(main())
