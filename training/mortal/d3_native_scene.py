"""Deterministic native-scene reconstruction for the D3 auditor v2.

The native arena (libriichi, with the D3 decision-context patch) consults the
model once per decision round: ``Game::poll`` returns when any player
``can_act()`` and every can-act player receives one ``set_scene_with_context``
with an arena-owned ``decision_index``. Generation-time exploration events carry
those arena contexts directly.

Auditor v1 instead derived per-kyoku decision indices from GameplayLoader row
order and a ``primary_row_flags`` heuristic. The authoritative 25h smoke proved
that derivation diverges from the native scene sequence in two systematic ways:

* the arena never consults a post-riichi forced discard (the D3 engine's
  ``not_eligible_finite_action_count`` is always zero), so loader rows for
  post-riichi draws with a single legal action are NOT arena scenes; and
* the loader skips rows for reactions pre-empted by another player's ron, even
  though the arena consulted the model (``has_any_ron`` label gap).

This module reproduces the arena consultation sequence deterministically from
the mjai log with the same ``libriichi.state.PlayerState`` the arena and the
loader share, and mirrors the GameplayLoader label logic (``gameplay.rs``)
exactly so each arena scene is paired with its loader row (or marked as a gap).

Validation (authoritative 25h smoke, same frozen runtime):
  arena scenes reconstructed == D3 ``states`` counter (4063)
  loader rows consumed == GameplayLoader primary rows (4194), no leftover
  label/row action mismatches == 0
  all 703 generation events map exactly once, behavior action mismatch == 0
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

HONOR_IDS = {"E": 27, "S": 28, "W": 29, "N": 30, "P": 31, "F": 32, "C": 33}
AKA_IDS = {"5mr": 34, "5pr": 35, "5sr": 36}
CHI_LABELS = {(1, 2): 38, (-1, 1): 39, (-2, -1): 40}
SKIP_NEXT_TYPES = ("reach_accepted", "dora")


def tile_label(pai: str) -> int:
    if pai in HONOR_IDS:
        return HONOR_IDS[pai]
    if pai in AKA_IDS:
        return AKA_IDS[pai]
    pai = pai.rstrip("r")
    return "mps".index(pai[-1]) * 9 + int(pai[0]) - 1


def chi_label(consumed: list[str], pai: str) -> int | None:
    if not consumed or pai in HONOR_IDS:
        return None
    pai = pai.rstrip("r")
    base = int(pai[0])
    rel = tuple(
        sorted(
            int(tile.rstrip("r")[0]) - base
            for tile in consumed
            if tile.rstrip("r")[-1] == pai[-1]
        )
    )
    return CHI_LABELS.get(rel)


def read_log_events(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [
            json.loads(line)
            for line in handle
            if line.strip()
        ]


def expected_label(
    events_raw: list[dict[str, Any]],
    index: int,
    seat: int,
    state: PlayerState,
) -> int | None:
    """Mirror gameplay.rs ``extend_from_event_window`` label for ``events_raw[index]``.

    Requires ``state`` to already have processed ``events_raw[index]``.
    ``None`` means the loader emits no row for this scene (interrupted-ron /
    tsumo-win final / ryukyoku-no-vote edges), which the arena still consulted.
    """
    n1 = events_raw[index + 1] if index + 1 < len(events_raw) else None
    n2 = events_raw[index + 2] if index + 2 < len(events_raw) else None
    nxt = n1
    if nxt is not None and nxt["type"] in SKIP_NEXT_TYPES:
        nxt = n2
    has_any_ron = bool(n1 and n1["type"] == "hora")
    label: int | None = None
    if nxt is not None:
        nt = nxt["type"]
        if nt == "dahai":
            label = tile_label(nxt["pai"])
        elif nt == "reach" and nxt.get("actor") == seat:
            label = 37
        elif nt == "chi" and nxt.get("actor") == seat:
            label = chi_label(nxt.get("consumed", []), nxt.get("pai", ""))
        elif nt == "pon" and nxt.get("actor") == seat:
            label = 41
        elif nt in ("daiminkan", "kakan", "ankan") and nxt.get("actor") == seat:
            label = 42
        elif nt == "ryukyoku" and state.last_cans.can_ryukyoku:
            label = 44
        else:
            ron_by_pov = False
            if has_any_ron:
                for ev in events_raw[index + 1 : index + 4]:
                    if ev["type"] == "end_kyoku":
                        break
                    if ev["type"] == "hora" and ev.get("actor") == seat:
                        ron_by_pov = True
                        break
            if ron_by_pov:
                label = 43
            elif (
                (state.last_cans.can_chi and nxt["type"] == "tsumo")
                or (
                    (state.last_cans.can_pon
                     or state.last_cans.can_daiminkan
                     or state.last_cans.can_ron_agari)
                    and not has_any_ron
                )
            ):
                label = 45
    return label


def reconstruct_native_scenes(
    log_path: Path,
    seat: int,
    loader_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Reconstruct the native arena scene sequence for one game.

    ``loader_rows`` must be the GameplayLoader primary rows for this game in
    loader order, each with keys ``action``, ``legal_count``, ``kyoku``.

    Returns:
      scenes: list in log order of can-act events with fields
              ``kyoku``, ``own_riichi``, ``label``, ``loader_row_index``
              (None when the loader emits no row), ``row_action``,
              ``row_legal``, ``arena_consulted``.
      scene_by_arena: {(kyoku, arena_index): entry} over arena-consulted events.
      loader_row_scene: {(kyoku, loader_row_index): arena_index} for rows that
              belong to arena-consulted scenes.
      label_mismatches, row_exhausted: reconstruction integrity counters.
    """
    from libriichi.state import PlayerState  # noqa: PLC0415

    events_raw = read_log_events(log_path)
    state = PlayerState(seat)
    kyoku = -1
    scenes: list[dict[str, Any]] = []
    row_pos = 0
    label_mismatches = 0
    row_exhausted = False
    per_kyoku_loader_idx: dict[int, int] = {}
    per_kyoku_arena_idx: dict[int, int] = {}

    for index, ev in enumerate(events_raw):
        if ev["type"] == "start_kyoku":
            kyoku += 1
        state.update(json.dumps(ev, ensure_ascii=False))
        if not state.last_cans.can_act:
            continue
        own_riichi = bool(state.self_riichi_declared or state.self_riichi_accepted)
        label = expected_label(events_raw, index, seat, state)
        loader_row_index: int | None = None
        row_action: int | None = None
        row_legal: int | None = None
        if label is not None:
            if row_pos >= len(loader_rows):
                row_exhausted = True
                row_pos += 1
            else:
                row = loader_rows[row_pos]
                row_pos += 1
                loader_row_index = per_kyoku_loader_idx.get(row["kyoku"], 0)
                per_kyoku_loader_idx[row["kyoku"]] = loader_row_index + 1
                row_action = row["action"]
                row_legal = row["legal_count"]
                if row_action != label:
                    label_mismatches += 1
        # The arena never consults a post-riichi forced discard (single legal
        # action): its D3 engine keeps not_eligible_finite_action_count == 0.
        forced = bool(own_riichi and row_legal == 1)
        arena_index: int | None = None
        if not forced:
            arena_index = per_kyoku_arena_idx.get(kyoku, 0)
            per_kyoku_arena_idx[kyoku] = arena_index + 1
        scenes.append(
            {
                "kyoku": kyoku,
                "own_riichi": own_riichi,
                "label": label,
                "loader_row_index": loader_row_index,
                "row_action": row_action,
                "row_legal": row_legal,
                "arena_consulted": not forced,
                "arena_index": arena_index,
            }
        )
    leftover = len(loader_rows) - min(row_pos, len(loader_rows))
    return {
        "scenes": scenes,
        "label_mismatches": label_mismatches,
        "row_exhausted": row_exhausted,
        "leftover_rows": max(leftover, 0),
        "total_arena_scenes": sum(
            1 for entry in scenes if entry["arena_consulted"]
        ),
        "total_loader_rows": sum(1 for entry in scenes if entry["label"] is not None),
    }
