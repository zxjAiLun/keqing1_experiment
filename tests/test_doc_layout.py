"""Tests for the doc-layout validator (adopted 2026-09-13).

The validator exists to stop the repo drifting back into "reports live under
artifacts/". These pin the mechanical rules and, importantly, that it does NOT
touch legacy documents.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "training"))

from training.mortal.check_doc_layout import (
    check_document,
)


def write(tmp: Path, name: str, body: str) -> Path:
    tmp.mkdir(parents=True, exist_ok=True)
    path = tmp / name
    path.write_text(body, encoding="utf-8")
    return path


GOOD = """---\nexperiment: P4-MX\ndate: 2026-09-12\nlast_updated: 2026-09-13\nstatus: closed\n---\n\n# body\n"""


def test_missing_front_matter_is_rejected(tmp_path: Path) -> None:
    doc = write(tmp_path, "2026-09-12_P4-MX_topic.md", "# no front-matter\n")
    assert check_document(doc, tmp_path)


def test_good_front_matter_passes(tmp_path: Path) -> None:
    doc = write(tmp_path, "2026-09-12_P4-MX_topic.md", GOOD)
    assert check_document(doc, tmp_path) == []


def test_filename_date_must_match_front_matter(tmp_path: Path) -> None:
    doc = write(tmp_path, "2026-09-11_P4-MX_topic.md", GOOD)
    assert any("date" in e for e in check_document(doc, tmp_path))


def test_last_updated_must_not_precede_date(tmp_path: Path) -> None:
    body = GOOD.replace("last_updated: 2026-09-13", "last_updated: 2026-09-01")
    doc = write(tmp_path, "2026-09-12_P4-MX_topic.md", body)
    assert any("last_updated" in e for e in check_document(doc, tmp_path))


def test_extra_front_matter_fields_are_rejected(tmp_path: Path) -> None:
    body = GOOD.replace("status: closed", "status: closed\nowner: me")
    doc = write(tmp_path, "2026-09-12_P4-MX_topic.md", body)
    assert check_document(doc, tmp_path)
