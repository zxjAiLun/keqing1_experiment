from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from training.mortal.render_research_overview_zh import (
    ALLOWED_STATUSES,
    load_registry,
    render_block,
    status_text,
)

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "training/docs/mortal/research_registry.json"
C1_ID = "C1_corpus_cql_interaction_2026_08"


def test_current_registry_loads_and_c1_status_is_allowed() -> None:
    registry = load_registry(REGISTRY_PATH)
    c1 = next(record for record in registry["records"] if record["experiment_id"] == C1_ID)

    assert c1["status"] == "preregistered_frozen"
    assert c1["status"] in ALLOWED_STATUSES
    assert registry["current_state"]["next_experiment"] == C1_ID
    assert registry["current_state"]["next_experiment_status"] == "preregistered_frozen"


def test_frozen_status_text_is_not_started_semantics() -> None:
    rendered_status = status_text("preregistered_frozen")

    assert "冻结" in rendered_status
    assert "未启动" in rendered_status
    assert "运行中" not in rendered_status
    assert "通过" not in rendered_status


def test_render_block_contains_registered_c1_without_promoting_k1() -> None:
    block = render_block(load_registry(REGISTRY_PATH))

    assert C1_ID in block
    assert f"下一实验提案：`{C1_ID}`（preregistered_frozen）" in block
    assert "K1：`尚未产生`" in block
    assert "K1 =" not in block


def test_unknown_record_status_fails_closed(tmp_path: Path) -> None:
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    invalid_registry = copy.deepcopy(registry)
    invalid_registry["records"][0]["status"] = "unknown_status"
    invalid_path = tmp_path / "invalid_registry.json"
    invalid_path.write_text(
        json.dumps(invalid_registry, ensure_ascii=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported status"):
        load_registry(invalid_path)
