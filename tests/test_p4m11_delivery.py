"""Focused tests for the P4-M11 delivery tooling.

These cover the tools that make the P4-M11 result *reproducible and readable*
rather than the training recipe itself:

* ``summarize_ovt_gate_stats`` -- the Chinese bidirectional summary.  The report
  reader has to tell the solo seat from the trio seats by itself, so most of
  these are refusal paths; the averaging itself is trivial.
* ``stat_report.write_stat_report`` -- the missing-game guard.  libriichi returns
  an empty Stat instead of failing when a log dir or a player name is wrong,
  which silently produces a report full of zeros.
* ``export_eval_weights`` / ``publish_authoritative_bundle`` /
  ``adjudicate_ovt_gate`` -- the promoted reproduction scripts.  Each must refuse
  to touch an existing artifact it disagrees with.

Like the other P4-M9/M10/M11 test files this one is not in the pyproject
``python_files`` allowlist, so the default suite does not collect it.  Run it
explicitly:

    python -m pytest tests/test_p4m11_delivery.py -q
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.mortal import summarize_ovt_gate_stats as summary  # noqa: E402
from training.mortal.adjudicate_ovt_gate import (  # noqa: E402
    artifact_id,
    band,
    identity_of,
    percentile,
    seed_pt,
)
from training.mortal.adjudicate_ovt_gate import main as adjudicate_main  # noqa: E402
from training.mortal.publish_authoritative_bundle import (  # noqa: E402
    parse_provenance,
    publish,
)

CANDIDATE = "p4m11_u32"


# --------------------------------------------------------------------------
# summarize_ovt_gate_stats
# --------------------------------------------------------------------------


def _derived(**overrides: float) -> dict[str, float]:
    base = {key: 0.0 for _title, key, _kind in summary.SUMMARY_ROWS}
    base.update(overrides)
    return base


def _write_direction(
    directory: Path,
    solo_label: str,
    trio_label: str,
    solo_values: dict[str, float] | None = None,
    trio_values: dict[str, float] | None = None,
) -> None:
    """Write a synthetic detailed_stats.json with the 1-seat / 3-seat basis."""
    directory.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "keqing.mortal.libriichi.stat.v1",
        "players": {
            solo_label: {
                "player_name": solo_label,
                "raw": {"game": 1024, "round": 10873},
                "derived": _derived(**(solo_values or {})),
            },
            trio_label: {
                "player_name": trio_label,
                "raw": {"game": 3072, "round": 32619},
                "derived": _derived(**(trio_values or {})),
            },
        },
    }
    (directory / "detailed_stats.json").write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")


def test_gate_label_survives_and_both_sides_are_averaged(tmp_path: Path):
    """The gate label must not be overwritten by the averaging loop."""
    _write_direction(tmp_path / "solo", CANDIDATE, "k0_70k", solo_values={"avg_rank": 2.4})
    _write_direction(tmp_path / "mirror", "k0_70k", CANDIDATE, trio_values={"avg_rank": 2.6})
    gate = summary.summarize_gate("Gate A：U32 vs K0_70k", tmp_path / "solo", tmp_path / "mirror", CANDIDATE)
    assert gate["label"] == "Gate A：U32 vs K0_70k"
    assert gate["candidate_label"] == CANDIDATE
    assert gate["opponent_label"] == "k0_70k"
    # candidate: 2.4 where it sat solo, 2.6 where it sat in the trio -> 2.5
    assert gate["averaged"][CANDIDATE]["avg_rank"] == pytest.approx(2.5)
    assert gate["averaged"]["k0_70k"]["avg_rank"] == pytest.approx(0.0)


def test_markdown_keeps_the_gate_label_and_states_the_sample_basis(tmp_path: Path):
    _write_direction(tmp_path / "solo", CANDIDATE, "k0_70k")
    _write_direction(tmp_path / "mirror", "k0_70k", CANDIDATE)
    gate = summary.summarize_gate("Gate A", tmp_path / "solo", tmp_path / "mirror", CANDIDATE)
    text = summary.format_markdown([gate])
    assert "## Gate A" in text
    assert "单挑方每半庄占 **1 座**" in text
    assert "只比较逐座比率，不比较计数" in text
    assert "p4m11_u32（逐座均值）" in text


def test_refuses_a_direction_that_is_not_one_vs_three(tmp_path: Path):
    directory = tmp_path / "solo"
    directory.mkdir()
    report = {
        "players": {
            CANDIDATE: {"player_name": CANDIDATE, "raw": {"game": 1024, "round": 1}, "derived": _derived()},
            "k0_70k": {"player_name": "k0_70k", "raw": {"game": 2048, "round": 1}, "derived": _derived()},
        }
    }
    (directory / "detailed_stats.json").write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(summary.GateSpecError, match="not a one-vs-three"):
        summary.read_direction(directory, CANDIDATE)


def test_refuses_directions_that_are_not_mirror_images(tmp_path: Path):
    _write_direction(tmp_path / "solo", CANDIDATE, "k0_70k")
    _write_direction(tmp_path / "mirror", CANDIDATE, "k0_70k")
    with pytest.raises(summary.GateSpecError, match="not mirror images"):
        summary.summarize_gate("Gate A", tmp_path / "solo", tmp_path / "mirror", CANDIDATE)


def test_refuses_when_the_candidate_is_absent(tmp_path: Path):
    _write_direction(tmp_path / "solo", "someone_else", "k0_70k")
    with pytest.raises(summary.GateSpecError, match="does not describe"):
        summary.read_direction(tmp_path / "solo", CANDIDATE)


def test_refuses_zero_games_instead_of_reporting_zeros(tmp_path: Path):
    directory = tmp_path / "solo"
    directory.mkdir()
    report = {
        "players": {
            CANDIDATE: {"player_name": CANDIDATE, "raw": {"game": 0, "round": 0}, "derived": _derived()},
            "k0_70k": {"player_name": "k0_70k", "raw": {"game": 0, "round": 0}, "derived": _derived()},
        }
    }
    (directory / "detailed_stats.json").write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(summary.GateSpecError, match="0 games"):
        summary.read_direction(directory, CANDIDATE)


def test_refuses_a_missing_report_with_a_useful_message(tmp_path: Path):
    with pytest.raises(summary.GateSpecError, match="stat_report.py"):
        summary.read_direction(tmp_path / "absent", CANDIDATE)


def test_parse_gate_spec():
    label, solo, mirror = summary.parse_gate_spec("Gate A=a|b")
    assert (label, str(solo), str(mirror)) == ("Gate A", "a", "b")
    for bad in ("Gate A", "Gate A=a", "=a|b", "Gate A=a|"):
        with pytest.raises(summary.GateSpecError):
            summary.parse_gate_spec(bad)


# --------------------------------------------------------------------------
# stat_report: no games is an error, not a report of zeros
# --------------------------------------------------------------------------


def _require_stat_class():
    from training.mortal import stat_report  # noqa: PLC0415

    try:
        stat_report.import_stat_class(REPO_ROOT / "third_party" / "Mortal")
    except Exception as exc:  # pragma: no cover - depends on a built extension
        pytest.skip(f"libriichi Stat is unavailable: {exc}")
    return stat_report


def test_stat_report_refuses_an_empty_log_dir_when_games_are_required(tmp_path: Path):
    stat_report = _require_stat_class()
    empty = tmp_path / "logs"
    empty.mkdir()
    with pytest.raises(RuntimeError, match="no games found"):
        stat_report.build_stat_report(
            log_dir=empty, players={"a": "a"}, require_games=True
        )


def test_stat_report_default_keeps_the_old_tolerant_behaviour(tmp_path: Path):
    """Callers that use the report to decide whether to resume rely on zeros."""
    stat_report = _require_stat_class()
    empty = tmp_path / "logs"
    empty.mkdir()
    report = stat_report.build_stat_report(log_dir=empty, players={"a": "a"})
    assert report["players"]["a"]["raw"]["game"] == 0


# --------------------------------------------------------------------------
# export_eval_weights
# --------------------------------------------------------------------------


def test_state_identity_names_the_first_difference():
    torch = pytest.importorskip("torch")
    from training.mortal.export_eval_weights import state_identity  # noqa: PLC0415

    a = {"x": torch.zeros(2), "y": torch.ones(2)}
    ok, detail = state_identity(a, {"x": torch.zeros(2), "y": torch.ones(2)})
    assert ok and "2 tensors identical" in detail
    ok, detail = state_identity(a, {"x": torch.zeros(2), "y": torch.full((2,), 2.0)})
    assert not ok and detail.startswith("y:")
    ok, detail = state_identity(a, {"x": torch.zeros(3), "y": torch.ones(2)})
    assert not ok and "shape/dtype" in detail
    ok, detail = state_identity(a, {"x": torch.zeros(2)})
    assert not ok and "key sets differ" in detail


def test_export_refuses_a_source_that_is_not_the_expected_artifact(tmp_path: Path):
    from training.mortal.export_eval_weights import export_eval_weights  # noqa: PLC0415

    checkpoint = tmp_path / "U32.pth"
    checkpoint.write_bytes(b"not really a checkpoint")
    with pytest.raises(SystemExit, match="refusing to export"):
        export_eval_weights(checkpoint=checkpoint, expect_source_sha256="0" * 64)


def test_export_refuses_a_checkpoint_without_the_student_contract(tmp_path: Path):
    torch = pytest.importorskip("torch")
    from training.mortal.export_eval_weights import export_eval_weights  # noqa: PLC0415

    checkpoint = tmp_path / "weird.pth"
    torch.save({"mortal": {}, "current_dqn": {}, "config": {}}, checkpoint)
    with pytest.raises(SystemExit, match="does not carry"):
        export_eval_weights(checkpoint=checkpoint)


def test_export_round_trips_and_refuses_a_disagreeing_existing_export(tmp_path: Path):
    torch = pytest.importorskip("torch")
    from training.mortal.export_eval_weights import export_eval_weights  # noqa: PLC0415

    checkpoint = tmp_path / "C.pth"
    torch.save(
        {
            "mortal": {"w": torch.ones(2)},
            "current_dqn": {"b": torch.zeros(1)},
            "training_contract": {"schema": "keqing.mortal.student_policy_v1"},
            "optimizer_state": {"dropped": True},
            "cycle": 32,
        },
        checkpoint,
    )
    out = tmp_path / "C_eval_weights.pth"
    record = export_eval_weights(checkpoint=checkpoint, out=out, verify_loader=False)
    assert record["export"]["sha256"]
    reloaded = torch.load(out, weights_only=True, map_location="cpu")
    assert set(reloaded) == {"mortal", "current_dqn", "training_contract"}

    # an existing export that disagrees must not be overwritten
    torch.save(
        {"mortal": {"w": torch.zeros(2)}, "current_dqn": {"b": torch.zeros(1)}, "training_contract": {}},
        out,
    )
    with pytest.raises(SystemExit, match="refusing to touch it"):
        export_eval_weights(checkpoint=checkpoint, out=out, verify_loader=False)


# --------------------------------------------------------------------------
# publish_authoritative_bundle
# --------------------------------------------------------------------------


def test_parse_provenance_accepts_json_and_plain_strings():
    parsed = parse_provenance(['gates={"a": "supported"}', "lineage=x -> y", "step=36"])
    assert parsed["gates"] == {"a": "supported"}
    assert parsed["lineage"] == "x -> y"
    assert parsed["step"] == 36
    with pytest.raises(SystemExit):
        parse_provenance(["nope"])


def test_publish_writes_a_manifest_and_is_idempotent(tmp_path: Path):
    source = tmp_path / "U32_eval_weights.pth"
    source.write_bytes(b"frozen weights")
    data_root = tmp_path / "keqing-data"
    kwargs = dict(
        data_root=data_root,
        bundle_id="SMOKE_2026_09",
        family="SMOKE_FAM",
        source=source,
        provenance={"gates": {"practical": "supported"}},
    )
    publish(**kwargs)
    manifest_path = data_root / "mortal" / "authoritative" / "SMOKE_2026_09" / "manifest.json"
    first = manifest_path.read_text(encoding="utf-8")
    assert json.loads(first)["artifacts"]["SMOKE_FAM"]["sha256"]
    publish(**kwargs)
    assert manifest_path.read_text(encoding="utf-8") == first, "a second publish rewrote the manifest"


def test_publish_refuses_a_source_that_does_not_match_the_expected_sha(tmp_path: Path):
    source = tmp_path / "U32_eval_weights.pth"
    source.write_bytes(b"frozen weights")
    with pytest.raises(SystemExit, match="!= expected"):
        publish(
            data_root=tmp_path / "keqing-data",
            bundle_id="SMOKE",
            family="FAM",
            source=source,
            expect_source_sha256="0" * 64,
        )


def test_publish_refuses_to_rewrite_a_manifest_for_different_content(tmp_path: Path):
    source = tmp_path / "a.pth"
    source.write_bytes(b"one")
    data_root = tmp_path / "keqing-data"
    publish(data_root=data_root, bundle_id="B", family="FAM", source=source)
    manifest_path = data_root / "mortal" / "authoritative" / "B" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["FAM"]["sha256"] = "deadbeef"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    other = tmp_path / "b.pth"
    other.write_bytes(b"two")
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        publish(data_root=data_root, bundle_id="B", family="FAM", source=other)


# --------------------------------------------------------------------------
# adjudicate_ovt_gate
# --------------------------------------------------------------------------


def test_percentile_interpolates():
    assert percentile([0.0, 1.0], 0.5) == pytest.approx(0.5)
    assert percentile([0.0, 1.0], 0.0) == pytest.approx(0.0)
    assert percentile([0.0, 1.0], 1.0) == pytest.approx(1.0)


def _metrics(
    *,
    challenger: str,
    champion: str,
    seed_start: int,
    seed_count: int,
    challenger_ranks: dict[int, list[int]],
) -> dict:
    return {
        "challenger": {"label": challenger, "checkpoint": f"{challenger}.pth"},
        "champion": {"label": champion, "checkpoint": f"{champion}.pth"},
        "seeds": {"seed_start": seed_start, "seed_count": seed_count, "seed_key": 8192},
        "hanchans": seed_count * 4,
        "batch_seeds": 8,
        "enable_amp": False,
        "device": "cuda",
        "rank_points_values": [90.0, 45.0, 0.0, -135.0],
        "challenger_result": {"rank_counts": [0, 0, 0, 0]},
        "cluster_statistics": {"challenger_avg_pt": {"seed_cluster_bootstrap_ci95": [0, 0]}},
        "integrity": {
            "per_seed_ranks": {str(seed): ranks for seed, ranks in challenger_ranks.items()},
            "native_batches": [],
        },
    }


def _pair(tmp_path: Path, *, labels_ok: bool = True) -> tuple[Path, Path]:
    """Candidate 1st on every seed solo, reference 4th on every seed mirrored."""
    solo_ranks = {s: [1, 1, 1, 1] for s in range(740000, 740004)}
    mirror_ranks = {s: [4, 4, 4, 4] for s in range(740000, 740004)}
    solo_challenger = CANDIDATE if labels_ok else "wrong"
    solo = tmp_path / "solo"
    mirror = tmp_path / "mirror"
    solo.mkdir()
    mirror.mkdir()
    (solo / "metrics.json").write_text(
        json.dumps(_metrics(challenger=solo_challenger, champion="k0_70k", seed_start=740000, seed_count=4, challenger_ranks=solo_ranks)),
        encoding="utf-8",
    )
    (mirror / "metrics.json").write_text(
        json.dumps(_metrics(challenger="k0_70k", champion=CANDIDATE, seed_start=740000, seed_count=4, challenger_ranks=mirror_ranks)),
        encoding="utf-8",
    )
    return solo / "metrics.json", mirror / "metrics.json"


def test_seed_pt_and_band():
    document = _metrics(challenger="c", champion="r", seed_start=740000, seed_count=2, challenger_ranks={740000: [1, 1, 1, 1]})
    assert seed_pt(document) == {740000: 90.0}
    assert band(document) == ("740000", 2, 8192)


def test_identity_prefers_the_run_identity_sidecar(tmp_path: Path):
    metrics = tmp_path / "metrics.json"
    document = {"challenger": {"label": "c", "checkpoint": "from-cli.pth"}}
    (tmp_path / "run_identity.json").write_text(
        json.dumps({"challenger": {"path": "actually-ran.pth", "sha256": "abc"}}), encoding="utf-8"
    )
    metrics.write_text("{}", encoding="utf-8")
    identity = identity_of(metrics, document, "challenger")
    assert identity == {"path": "actually-ran.pth", "sha256": "abc"}
    assert artifact_id(metrics, document, "challenger") == ("actually-ran.pth", "abc")


def test_adjudicate_produces_a_verdict_and_is_symmetric(tmp_path: Path):
    solo, mirror = _pair(tmp_path)
    out = tmp_path / "verdict.json"
    assert adjudicate_main([
        "--solo-metrics", str(solo), "--mirror-metrics", str(mirror),
        "--candidate-label", CANDIDATE, "--reference-label", "k0_70k",
        "--expect-seeds", "4", "--expect-hanchans", "16", "--expect-seed-start", "740000",
        "--reps", "200", "--output", str(out),
    ]) == 0
    verdict = json.loads(out.read_text(encoding="utf-8"))
    # candidate 1st on every solo seed, reference 4th on every mirrored seed
    assert verdict["S_solo_candidate"]["mean"] == pytest.approx(90.0)
    assert verdict["O_mirror_reference"]["mean"] == pytest.approx(-135.0)
    assert verdict["verdict"] == "supported"
    assert verdict["G"]["mean"] == pytest.approx((90.0 + 135.0 / 3.0) / 2.0)
    assert all(verdict["conditions"].values())


def test_adjudicate_refuses_a_csv_that_is_not_the_frozen_pair(tmp_path: Path):
    solo, mirror = _pair(tmp_path, labels_ok=False)
    with pytest.raises(SystemExit, match="refusing to adjudicate"):
        adjudicate_main([
            "--solo-metrics", str(solo), "--mirror-metrics", str(mirror),
            "--candidate-label", CANDIDATE, "--reference-label", "k0_70k",
            "--expect-seeds", "4", "--expect-hanchans", "16",
        ])


def test_adjudicate_refuses_a_wrong_seed_band(tmp_path: Path):
    solo, mirror = _pair(tmp_path)
    with pytest.raises(SystemExit, match="refusing to adjudicate"):
        adjudicate_main([
            "--solo-metrics", str(solo), "--mirror-metrics", str(mirror),
            "--candidate-label", CANDIDATE, "--reference-label", "k0_70k",
            "--expect-seeds", "4", "--expect-hanchans", "16", "--expect-seed-start", "999999",
        ])
