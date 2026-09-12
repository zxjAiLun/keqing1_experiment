"""Tests for game-event-level arena log comparison.

The tool exists to answer "did these two runs play the same game?" without
resorting to gzip container bytes or aggregate rank counts, so the tests pin the
three declared field tiers: game fields are decisive, float inference outputs are
reported only, and runtime bookkeeping fields are never a verdict.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from training.mortal.compare_arena_logs import compare_run_dirs

SEED_START = 710000
SEED_KEY = 8192
SPLITS = "a"


def _base_records() -> list[dict]:
    return [
        {"type": "start_game", "names": ["student50k", "ext_mortal", "ext_mortal", "ext_mortal"],
         "seed": [SEED_START, SEED_KEY]},
        {"type": "start_kyoku", "bakaze": "E", "dora_marker": "3s", "kyoku": 1,
         "honba": 0, "kyotaku": 0, "oya": 0, "scores": [25000, 25000, 25000, 25000],
         "tehais": [["1m"], ["2m"], ["3m"], ["4m"]]},
        {"type": "tsumo", "actor": 0, "pai": "9p"},
        {"type": "dahai", "actor": 0, "pai": "9p", "tsumogiri": True,
         "meta": {"q_values": [-1.0, -2.0, -3.0], "mask_bits": 12345, "is_greedy": True,
                  "shanten": 3, "at_furiten": False, "eval_time_ns": 111,
                  "batch_size": 8}},
        {"type": "hora", "actor": 1, "target": 0, "deltas": [0, 0, 0, 0],
         "ura_markers": ["1p"]},
        {"type": "end_kyoku"},
        {"type": "end_game"},
    ]


def _write(directory: Path, name: str, records: list[dict]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def _compare(tmp_path: Path, baseline: list[dict], candidate: list[dict],
             name: str | None = None) -> dict:
    name = name or f"{SEED_START}_{SEED_KEY}_a.json.gz"
    base_dir = tmp_path / "base"
    cand_dir = tmp_path / "cand"
    _write(base_dir, name, baseline)
    _write(cand_dir, name, candidate)
    return compare_run_dirs(base_dir, cand_dir, SEED_START, 1, SEED_KEY, SPLITS)


def test_identical_logs_are_reproduced(tmp_path: Path) -> None:
    document = _compare(tmp_path, _base_records(), _base_records())
    assert document["verdict"] == "REPRODUCED"
    assert document["game_field_mismatches"] == 0
    assert document["first_divergence"] is None
    assert document["records_compared"] == len(_base_records())


def test_game_event_divergence_is_located_and_scan_stops(tmp_path: Path) -> None:
    candidate = _base_records()
    candidate[2]["pai"] = "1s"  # a different drawn tile
    document = _compare(tmp_path, _base_records(), candidate)
    assert document["verdict"] == "DIVERGED"
    divergence = document["first_divergence"]
    assert divergence is not None
    assert divergence["kind"] == "game_event"
    assert divergence["record_index"] == 2
    assert divergence["baseline"]["pai"] == "9p"
    assert divergence["candidate"]["pai"] == "1s"
    # first divergence is the answer: no further files are scanned
    assert document["files_compared"] == 1


def test_legal_action_mask_difference_is_a_game_divergence(tmp_path: Path) -> None:
    candidate = _base_records()
    candidate[3]["meta"]["mask_bits"] = 999
    document = _compare(tmp_path, _base_records(), candidate)
    assert document["verdict"] == "DIVERGED"
    assert document["first_divergence"]["record_index"] == 3


def test_float_inference_noise_alone_is_not_a_divergence(tmp_path: Path) -> None:
    candidate = _base_records()
    candidate[3]["meta"]["q_values"] = [-1.0 + 1e-7, -2.0, -3.0]
    document = _compare(tmp_path, _base_records(), candidate)
    assert document["verdict"] == "REPRODUCED"
    report = document["float_field_report"]["q_values"]
    assert report["values_compared"] == 3
    assert report["max_abs_diff"] > 0
    assert report["argmax_mismatches"] == 0


def test_identical_floats_are_reported_as_compared_not_absent(tmp_path: Path) -> None:
    """An all-zero float diff must still show that values WERE compared."""
    document = _compare(tmp_path, _base_records(), _base_records())
    report = document["float_field_report"]["q_values"]
    assert report["values_compared"] == 3
    assert report["max_abs_diff"] == 0.0
    assert report["argmax_mismatches"] == 0


def test_float_argmax_mismatch_is_reported(tmp_path: Path) -> None:
    candidate = _base_records()
    # same chosen pai in the event stream, but the decoded value ordering flips
    candidate[3]["meta"]["q_values"] = [-3.0, -2.0, -1.0]
    document = _compare(tmp_path, _base_records(), candidate)
    assert document["verdict"] == "REPRODUCED"
    assert document["float_field_report"]["q_values"]["argmax_mismatches"] == 1


def test_runtime_bookkeeping_fields_are_reported_not_decisive(tmp_path: Path) -> None:
    candidate = _base_records()
    candidate[3]["meta"]["eval_time_ns"] = 999999
    candidate[3]["meta"]["batch_size"] = 2
    document = _compare(tmp_path, _base_records(), candidate)
    assert document["verdict"] == "REPRODUCED"
    assert document["excluded_field_diffs"]["eval_time_ns"] == 1
    assert document["excluded_field_diffs"]["batch_size"] == 1


def test_record_count_mismatch_diverges(tmp_path: Path) -> None:
    document = _compare(tmp_path, _base_records(), _base_records()[:-1])
    assert document["verdict"] == "DIVERGED"
    assert document["first_divergence"]["kind"] == "record_count"


def test_missing_candidate_file_diverges(tmp_path: Path) -> None:
    base_dir = tmp_path / "base"
    cand_dir = tmp_path / "cand"
    _write(base_dir, f"{SEED_START}_{SEED_KEY}_a.json.gz", _base_records())
    cand_dir.mkdir(parents=True, exist_ok=True)
    document = compare_run_dirs(base_dir, cand_dir, SEED_START, 1, SEED_KEY, SPLITS)
    assert document["verdict"] == "DIVERGED"
    assert document["missing_files"] == [f"{SEED_START}_{SEED_KEY}_a.json.gz"]


def test_declared_field_tiers_are_recorded_in_output(tmp_path: Path) -> None:
    document = _compare(tmp_path, _base_records(), _base_records())
    declared = document["declared_fields"]
    assert "mask_bits" in declared["game_meta_fields"]
    assert "q_values" in declared["float_meta_fields"]
    assert "eval_time_ns" in declared["excluded_meta_fields"]
    assert "batch_size" in declared["excluded_meta_fields"]
