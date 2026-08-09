"""Regression fixture for the shard_005 audit defect (ryukyoku guard).

The authoritative smoke and the first B250 gate never hit the exhaustive-draw
edge where the POV can act but cannot vote for ryukyoku; shard_005's game
1801256 did. The unguarded ryukyoku label consumed one loader row that the
GameplayLoader never emits, shifting the rest of the game by +1.

This test replays the actual immutable shard_005 log when libriichi and the
artifact are available (target machine), and skips otherwise. It must show:
180/180 loader rows consumed, 0 label mismatches, 0 row exhaustion.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from training.mortal.d3_native_scene import read_log_events, reconstruct_native_scenes

libriichi_state = pytest.importorskip("libriichi.state")

SHARD_005_LOG = Path(
    r"E:\AUbuntuProject\project\keqing1_experiment\artifacts\experiments\model_pool_2026_07"
    r"\D3_uncertainty_guided_exploration_2026_08\generation_continuation"
    r"\shard_005_1801250_1801499\logs\1801256_8192_c.json.gz"
)


def _loader_rows(records: list, flags: list[bool]) -> list[dict]:
    import numpy as np  # noqa: PLC0415

    rows = []
    for record, is_primary in zip(records, flags, strict=True):
        if not is_primary:
            continue
        rows.append(
            {
                "action": int(record.action),
                "legal_count": int(np.asarray(record.mask, dtype=np.bool_).sum()),
                "kyoku": int(record.kyoku),
            }
        )
    return rows


def test_shard_005_1801256_native_scene_reconstruction_is_exact() -> None:
    if not SHARD_005_LOG.is_file():
        pytest.skip("immutable shard_005 artifact not present on this machine")
    import numpy as np  # noqa: PLC0415
    from libriichi.dataset import GameplayLoader  # noqa: PLC0415
    from training.mortal.audit_replay_distribution import records_from_game  # noqa: PLC0415
    from training.mortal.d3_production_audit_core import primary_row_flags  # noqa: PLC0415
    from training.mortal.d3_production_contract import RANK_POINTS  # noqa: PLC0415

    events_raw = read_log_events(SHARD_005_LOG)
    seat = events_raw[0]["names"].index("K0_70k")
    loader = GameplayLoader(
        version=4,
        oracle=False,
        player_names=["K0_70k"],
        always_include_kan_select=True,
        augmented=False,
    )
    pts = np.asarray(RANK_POINTS, dtype=np.float64)
    loaded = loader.load_gz_log_files([str(SHARD_005_LOG)])
    records = list(records_from_game(loaded[0][0], pts))
    flags = primary_row_flags(r.action for r in records)
    recon = reconstruct_native_scenes(SHARD_005_LOG, seat, _loader_rows(records, flags))
    assert recon["label_mismatches"] == 0
    assert recon["row_exhausted"] is False
    assert recon["leftover_rows"] == 0
    consumed = sum(1 for entry in recon["scenes"] if entry["label"] is not None)
    assert consumed == 180
