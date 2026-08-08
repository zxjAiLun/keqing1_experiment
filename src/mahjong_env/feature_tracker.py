from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from mahjong_env.tiles import tile_to_34 as _to_34, tile_is_aka as _is_aka


def _count_tiles(tiles: Sequence[str]) -> List[int]:
    counts = [0] * 34
    for tile in tiles:
        idx = _to_34(tile)
        if 0 <= idx < 34:
            counts[idx] += 1
    return counts


def _tiles_from_melds(melds: Sequence[dict]) -> List[str]:
    meld_tiles: List[str] = []
    for meld in melds:
        meld_tiles.extend(meld.get("consumed", []))
        if meld.get("pai"):
            meld_tiles.append(meld["pai"])
    return meld_tiles


def _suit_bucket_from_tile34(tile34: int) -> int:
    if 0 <= tile34 <= 8:
        return 0
    if 9 <= tile34 <= 17:
        return 1
    if 18 <= tile34 <= 26:
        return 2
    return 3


def _apply_hand_count_delta(player: "PlayerRoundTracker", tile34: int, delta: int) -> None:
    before = player.hand_counts34[tile34]
    after = before + delta
    player.pair_count += int(after >= 2) - int(before >= 2)
    player.ankoutsu_count += int(after >= 3) - int(before >= 3)
    player.hand_counts34[tile34] = after


def _apply_overall_tile_delta(player: "PlayerRoundTracker", tile: str, delta: int) -> None:
    tile34 = _to_34(tile)
    if tile34 < 0:
        return
    player.suit_counts[_suit_bucket_from_tile34(tile34)] += delta
    if _is_aka(tile):
        player.aka_counts[0] += delta
        if "m" in tile:
            player.aka_counts[1] += delta
        elif "p" in tile:
            player.aka_counts[2] += delta
        elif "s" in tile:
            player.aka_counts[3] += delta


@dataclass
class PlayerRoundTracker:
    hand_tiles: List[str] = field(default_factory=list)
    meld_tiles: List[str] = field(default_factory=list)
    hand_counts34: List[int] = field(default_factory=lambda: [0] * 34)
    meld_counts34: List[int] = field(default_factory=lambda: [0] * 34)
    visible_counts34: List[int] = field(default_factory=lambda: [0] * 34)
    discards_count: int = 0
    meld_count: int = 0
    pair_count: int = 0
    ankoutsu_count: int = 0
    suit_counts: List[int] = field(default_factory=lambda: [0] * 4)
    aka_counts: List[int] = field(default_factory=lambda: [0] * 4)


@dataclass
class RoundFeatureTracker:
    players: List[PlayerRoundTracker] = field(
        default_factory=lambda: [PlayerRoundTracker() for _ in range(4)]
    )

    @classmethod
    def from_start_kyoku(cls, tehais: Sequence[Sequence[str]], dora_markers: Sequence[str]) -> "RoundFeatureTracker":
        tracker = cls()
        for actor in range(4):
            player = tracker.players[actor]
            player.hand_tiles = list(tehais[actor])
            for tile in tehais[actor]:
                idx = _to_34(tile)
                if idx >= 0:
                    _apply_hand_count_delta(player, idx, 1)
                    player.visible_counts34[idx] += 1
                _apply_overall_tile_delta(player, tile, 1)
        for marker in dora_markers:
            idx = _to_34(marker)
            if idx >= 0:
                for player in tracker.players:
                    player.visible_counts34[idx] += 1
        return tracker

    def on_tsumo(self, actor: int, pai: str) -> None:
        idx = _to_34(pai)
        if idx < 0:
            return
        player = self.players[actor]
        player.hand_tiles.append(pai)
        _apply_hand_count_delta(player, idx, 1)
        _apply_overall_tile_delta(player, pai, 1)
        player.visible_counts34[idx] += 1

    def on_dahai(self, actor: int, pai: str) -> None:
        idx = _to_34(pai)
        if idx < 0:
            return
        player = self.players[actor]
        try:
            player.hand_tiles.remove(pai)
        except ValueError:
            pass
        _apply_hand_count_delta(player, idx, -1)
        _apply_overall_tile_delta(player, pai, -1)
        player.discards_count += 1
        for player in self.players:
            player.visible_counts34[idx] += 1

    def on_open_meld(self, actor: int, consumed: Sequence[str], pai: str | None) -> None:
        player = self.players[actor]
        for tile in consumed:
            idx = _to_34(tile)
            if idx >= 0:
                _apply_hand_count_delta(player, idx, -1)
                player.meld_counts34[idx] += 1
            try:
                player.hand_tiles.remove(tile)
            except ValueError:
                pass
        player.meld_count += 1

        meld_tiles = list(consumed)
        if pai:
            meld_tiles.append(pai)
            _apply_overall_tile_delta(player, pai, 1)
        if pai:
            idx = _to_34(pai)
            if idx >= 0:
                player.meld_counts34[idx] += 1
        player.meld_tiles.extend(meld_tiles)
        for tile in meld_tiles:
            idx = _to_34(tile)
            if idx < 0:
                continue
            for p in self.players:
                p.visible_counts34[idx] += 1

    def on_ankan(self, actor: int, consumed: Sequence[str], pai: str | None) -> None:
        player = self.players[actor]
        for tile in consumed:
            idx = _to_34(tile)
            if idx >= 0:
                _apply_hand_count_delta(player, idx, -1)
                player.meld_counts34[idx] += 1
            try:
                player.hand_tiles.remove(tile)
            except ValueError:
                pass
        player.meld_count += 1

        meld_tiles = list(consumed)
        if pai:
            meld_tiles.append(pai)
            idx = _to_34(pai)
            if idx >= 0:
                player.meld_counts34[idx] += 1
        player.meld_tiles.extend(meld_tiles)
        for tile in meld_tiles:
            idx = _to_34(tile)
            if idx < 0:
                continue
            for p in self.players:
                p.visible_counts34[idx] += 1

    def on_kakan_accepted(self, actor: int, added_tile: str, pai: str | None) -> None:
        player = self.players[actor]
        idx = _to_34(added_tile)
        if idx >= 0:
            _apply_hand_count_delta(player, idx, -1)
            player.meld_counts34[idx] += 1
        try:
            player.hand_tiles.remove(added_tile)
        except ValueError:
            pass
        player.meld_tiles.append(added_tile)
        # 从 pon(3) 升到 kakan(5) 的当前 feature 语义，等价于再增加 2 份可见计数。
        for tile in [added_tile, pai]:
            if not tile:
                continue
            t34 = _to_34(tile)
            if t34 < 0:
                continue
            for p in self.players:
                p.visible_counts34[t34] += 1

    def on_dora(self, marker: str) -> None:
        idx = _to_34(marker)
        if idx < 0:
            return
        for player in self.players:
            player.visible_counts34[idx] += 1

    def snapshot_for_actor(
        self,
        actor: int,
        tsumo_pai: str | None = None,
    ) -> Dict:
        player = self.players[actor]
        hand_tiles = list(player.hand_tiles)
        meld_tiles = list(player.meld_tiles)

        visible = list(player.visible_counts34)
        if tsumo_pai:
            idx = _to_34(tsumo_pai)
            if idx >= 0:
                visible[idx] += 1

        hand_counts = list(player.hand_counts34)

        return {
            "actor": actor,
            "hand_tiles": hand_tiles,
            "meld_tiles": meld_tiles,
            "hand_counts34": tuple(hand_counts),
            "meld_counts34": tuple(player.meld_counts34),
            "visible_counts34": tuple(visible),
            "discards_count": player.discards_count,
            "meld_count": player.meld_count,
            "pair_count": player.pair_count,
            "ankoutsu_count": player.ankoutsu_count,
            "suit_counts": tuple(player.suit_counts),
            "aka_counts": tuple(player.aka_counts),
        }


@dataclass(frozen=True)
class SnapshotFeatureTracker:
    actor: int
    hand_tiles: List[str]
    meld_tiles: List[str]
    hand_counts34: tuple[int, ...]
    meld_counts34: tuple[int, ...]
    visible_counts34: tuple[int, ...]
    discards_count: int
    meld_count: int
    pair_count: int
    ankoutsu_count: int
    suit_counts: tuple[int, int, int, int]
    aka_counts: tuple[int, int, int, int]

    @classmethod
    def from_state(cls, state: Dict, actor: int) -> "SnapshotFeatureTracker":
        tracker_state = state.get("feature_tracker")
        if tracker_state is not None:
            return cls(
                actor=int(tracker_state["actor"]),
                hand_tiles=list(tracker_state["hand_tiles"]),
                meld_tiles=list(tracker_state["meld_tiles"]),
                hand_counts34=tuple(tracker_state["hand_counts34"]),
                meld_counts34=tuple(tracker_state["meld_counts34"]),
                visible_counts34=tuple(tracker_state["visible_counts34"]),
                discards_count=int(tracker_state["discards_count"]),
                meld_count=int(tracker_state["meld_count"]),
                pair_count=int(tracker_state["pair_count"]),
                ankoutsu_count=int(tracker_state["ankoutsu_count"]),
                suit_counts=tuple(tracker_state["suit_counts"]),
                aka_counts=tuple(tracker_state["aka_counts"]),
            )

        hand_tiles = list(state.get("hand", []))
        hand_counts = _count_tiles(hand_tiles)

        melds = (state.get("melds") or [[], [], [], []])[actor]
        meld_tiles = _tiles_from_melds(melds)
        meld_counts = _count_tiles(meld_tiles)

        visible = hand_counts[:]
        for meld_group in state.get("melds", [[], [], [], []]):
            for meld in meld_group:
                for p in meld.get("consumed", []):
                    idx = _to_34(p)
                    if idx >= 0:
                        visible[idx] += 1
                if meld.get("pai"):
                    idx = _to_34(meld["pai"])
                    if idx >= 0:
                        visible[idx] += 1
        for disc_group in state.get("discards", [[], [], [], []]):
            for disc in disc_group:
                pai = disc["pai"] if isinstance(disc, dict) else disc
                idx = _to_34(pai)
                if idx >= 0:
                    visible[idx] += 1
        for dm in state.get("dora_markers", []):
            idx = _to_34(dm)
            if idx >= 0:
                visible[idx] += 1
        tsumo_pai = state.get("tsumo_pai")
        if tsumo_pai:
            idx = _to_34(tsumo_pai)
            if idx >= 0:
                visible[idx] += 1

        pair_count = sum(1 for c in hand_counts if c >= 2)
        ankoutsu_count = sum(1 for c in hand_counts if c >= 3)

        all_actor_tiles = hand_tiles + meld_tiles
        man_cnt = sum(1 for t in all_actor_tiles if 0 <= _to_34(t) <= 8)
        pin_cnt = sum(1 for t in all_actor_tiles if 9 <= _to_34(t) <= 17)
        sou_cnt = sum(1 for t in all_actor_tiles if 18 <= _to_34(t) <= 26)
        honor_cnt = sum(1 for t in all_actor_tiles if 27 <= _to_34(t) <= 33)

        aka_m = sum(1 for t in all_actor_tiles if _is_aka(t) and "m" in t)
        aka_p = sum(1 for t in all_actor_tiles if _is_aka(t) and "p" in t)
        aka_s = sum(1 for t in all_actor_tiles if _is_aka(t) and "s" in t)

        return cls(
            actor=actor,
            hand_tiles=hand_tiles,
            meld_tiles=meld_tiles,
            hand_counts34=tuple(hand_counts),
            meld_counts34=tuple(meld_counts),
            visible_counts34=tuple(visible),
            discards_count=len((state.get("discards") or [[], [], [], []])[actor]),
            meld_count=len(melds),
            pair_count=pair_count,
            ankoutsu_count=ankoutsu_count,
            suit_counts=(man_cnt, pin_cnt, sou_cnt, honor_cnt),
            aka_counts=(aka_m + aka_p + aka_s, aka_m, aka_p, aka_s),
        )
