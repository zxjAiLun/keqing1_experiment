from __future__ import annotations

from collections import Counter
from typing import Dict, List

import keqing_core
from keqing_core import calc_standard_shanten as _calc_standard_shanten
from mahjong_env.scoring import can_hora_from_snapshot
from mahjong_env.tiles import normalize_tile, AKA_DORA_TILES, tile_to_34 as _tile_to_34
from mahjong_env.types import Action, ActionSpec


def _chi_patterns(tile: str) -> List[List[str]]:
    if tile in AKA_DORA_TILES:
        tile = normalize_tile(tile)
    if len(tile) != 2:
        return []
    num = int(tile[0])
    suit = tile[1]
    if suit not in ("m", "p", "s"):
        return []
    patterns: List[List[str]] = []
    for seq in ((num - 2, num - 1), (num - 1, num + 1), (num + 1, num + 2)):
        a, b = seq
        if 1 <= a <= 9 and 1 <= b <= 9:
            patterns.append([f"{a}{suit}", f"{b}{suit}"])
    return patterns


def _can_pon(hand: Counter, tile: str) -> bool:
    normalized = normalize_tile(tile)
    if not normalized:
        return False
    if len(normalized) == 1:
        return hand[normalized] >= 2
    aka_tile = normalized + "r"
    if aka_tile in AKA_DORA_TILES:
        return hand[normalized] + hand.get(aka_tile, 0) >= 2
    return hand[normalized] >= 2


def _can_daiminkan(hand: Counter, tile: str) -> bool:
    normalized = normalize_tile(tile)
    if not normalized:
        return False
    if len(normalized) == 1:
        return hand[normalized] >= 3
    aka_tile = normalized + "r"
    if aka_tile in AKA_DORA_TILES:
        return hand[normalized] + hand.get(aka_tile, 0) >= 3
    return hand[normalized] >= 3


def _hand_has_tile(hand: Counter, t: str) -> bool:
    """检查手牌中是否有牌 t，同时考虑赤宝牌（5m/5p/5s ↔ 5mr/5pr/5sr）。"""
    if hand[t] >= 1:
        return True
    if len(t) == 2 and t[0] == "5" and t[1] in ("m", "p", "s"):
        return hand[t + "r"] >= 1
    return False


def _pick_chi_tile(hand: Counter, t: str) -> str:
    """从手牌中取牌 t，优先返回赤宝牌版本。"""
    t = normalize_tile(t)
    if len(t) == 2 and t[0] == "5" and t[1] in ("m", "p", "s") and hand[t + "r"] >= 1:
        return t + "r"
    return t


def _pick_consumed(hand: Counter, normalized: str, n: int) -> List[str]:
    """从手牌中选出 n 张 normalized 牌，优先用赤宝牌版本。"""
    normalized = normalize_tile(normalized)
    aka = normalized[0] + normalized[1] + "r" if len(normalized) == 2 and normalized[0] == "5" and normalized[1] in ("m", "p", "s") else None
    result: List[str] = []
    # 先取赤宝牌
    if aka and aka in AKA_DORA_TILES:
        for _ in range(min(hand.get(aka, 0), n - len(result))):
            result.append(aka)
    # 再补普通牌
    while len(result) < n:
        result.append(normalized)
    return result


def _ankan_candidates(hand: Counter) -> List[tuple[str, tuple[str, ...]]]:
    result: List[tuple[str, tuple[str, ...]]] = []
    seen: set[str] = set()
    for tile in hand.keys():
        normalized = normalize_tile(tile)
        if normalized in seen:
            continue
        seen.add(normalized)
        consumed = tuple(_pick_consumed(hand, normalized, 4))
        if len(consumed) == 4 and Counter(consumed) <= hand:
            result.append((consumed[0], consumed))
    return result


def _remove_tile_once(hand: List[str], tile: str) -> bool:
    try:
        hand.remove(tile)
        return True
    except ValueError:
        normalized = normalize_tile(tile)
        if normalized != tile:
            try:
                hand.remove(normalized)
                return True
            except ValueError:
                return False
        return False


def _ankan_allowed_after_reach(
    hand: Counter,
    melds: List[Dict],
    consumed: tuple[str, ...],
    last_tsumo: str | None,
    last_tsumo_raw: str | None,
) -> bool:
    from mahjong_env.replay import _calc_shanten_waits

    full_hand = list(hand.elements())
    if last_tsumo is not None and len(full_hand) % 3 == 1:
        full_hand.append(last_tsumo_raw or last_tsumo)

    before_hand = list(full_hand)
    draw_tile = last_tsumo_raw or last_tsumo
    if draw_tile is not None:
        _remove_tile_once(before_hand, draw_tile)

    before_shanten, _before_wait_count, before_waits, _ = _calc_shanten_waits(before_hand, melds)
    if before_shanten != 0:
        return False

    after_hand = list(full_hand)
    for tile in consumed:
        if not _remove_tile_once(after_hand, tile):
            return False

    ankan_tile = consumed[0] if consumed else None
    after_melds = [
        *melds,
        {
            "type": "ankan",
            "pai": ankan_tile,
            "pai_raw": ankan_tile,
            "consumed": list(consumed),
            "target": 0,
        },
    ]
    after_shanten, _after_wait_count, after_waits, _ = _calc_shanten_waits(after_hand, after_melds)
    return after_shanten == 0 and list(before_waits) == list(after_waits)


def _reach_discard_candidates(
    hand: Counter,
    last_tsumo: str | None,
    last_tsumo_raw: str | None,
) -> List[tuple[str, bool]]:
    candidates: List[tuple[str, bool]] = []
    seen: set[tuple[str, bool]] = set()
    counts34 = [0] * 34
    for tile, cnt in hand.items():
        t34 = _tile_to_34(tile)
        if 0 <= t34 < 34:
            counts34[t34] += cnt

    for tile in list(hand.keys()):
        pai_out = tile
        tsumogiri = last_tsumo == tile
        if tsumogiri and last_tsumo_raw is not None:
            pai_out = last_tsumo_raw
        t34 = _tile_to_34(tile)
        if not (0 <= t34 < 34) or counts34[t34] <= 0:
            continue
        counts34[t34] -= 1
        if _calc_standard_shanten(tuple(counts34)) == 0:
            key = (pai_out, tsumogiri)
            if key not in seen:
                seen.add(key)
                candidates.append(key)
        counts34[t34] += 1
    return candidates


def _can_declare_reach(
    hand: Counter,
    melds: List[Dict],
    reached: bool,
    last_tsumo: str | None = None,
    last_tsumo_raw: str | None = None,
    shanten: int | None = 0,
) -> bool:
    if shanten is None or shanten != 0:
        return False
    if reached:
        return False
    if any(meld.get("type") != "ankan" for meld in melds):
        return False
    concealed_kan_count = sum(1 for meld in melds if meld.get("type") == "ankan")
    if sum(hand.values()) + 3 * concealed_kan_count != 14:
        return False
    if last_tsumo is None and last_tsumo_raw is None:
        return True
    return bool(_reach_discard_candidates(hand, last_tsumo, last_tsumo_raw))


def _call_can_hora(
    state_snapshot: Dict,
    *,
    actor: int,
    target: int,
    pai: str,
    is_tsumo: bool,
    **flags,
) -> bool:
    filtered_flags = {key: value for key, value in flags.items() if value is not None}
    try:
        return can_hora_from_snapshot(
            state_snapshot,
            actor=actor,
            target=target,
            pai=pai,
            is_tsumo=is_tsumo,
            **filtered_flags,
        )
    except TypeError:
        # Some tests monkeypatch can_hora_from_snapshot with an older/minimal signature.
        return can_hora_from_snapshot(
            state_snapshot,
            actor=actor,
            target=target,
            pai=pai,
            is_tsumo=is_tsumo,
        )


def _rust_public_action_specs(state_snapshot: Dict, actor: int) -> List[ActionSpec] | None:
    if not keqing_core.is_enabled():
        return None
    try:
        raw_specs = keqing_core.enumerate_public_legal_action_specs(state_snapshot, actor)
    except RuntimeError as exc:
        if keqing_core.is_missing_rust_capability_error(exc):
            return None
        raise
    specs: List[ActionSpec] = []
    for item in raw_specs:
        specs.append(
            ActionSpec(
                type=item["type"],
                actor=item.get("actor"),
                pai=item.get("pai"),
                consumed=tuple(item.get("consumed") or ()),
                target=item.get("target"),
                tsumogiri=item.get("tsumogiri"),
            )
        )
    return specs


def _enumerate_legal_action_specs_python(state_snapshot: Dict, actor: int) -> List[ActionSpec]:
    legal: List[ActionSpec] = []
    hand = Counter(state_snapshot["hand"])
    last_discard = state_snapshot["last_discard"]
    last_kakan = state_snapshot.get("last_kakan")
    actor_to_move = state_snapshot["actor_to_move"]
    last_tsumo = state_snapshot.get("last_tsumo", [None, None, None, None])[actor]
    last_tsumo_raw = state_snapshot.get("last_tsumo_raw", [None, None, None, None])[actor]
    reached = state_snapshot["reached"][actor]
    hora_is_haitei = state_snapshot.get("_hora_is_haitei")
    hora_is_houtei = state_snapshot.get("_hora_is_houtei")
    hora_is_rinshan = state_snapshot.get("_hora_is_rinshan")
    hora_is_chankan = state_snapshot.get("_hora_is_chankan")

    # If there is a pending `last_discard` from someone else, the player is in
    # a "call reaction" stage. Even if `actor_to_move` was advanced
    # simplistically, we must not generate `dahai` here.
    if last_kakan and last_kakan.get("actor") != actor:
        kakan_actor = last_kakan["actor"]
        tile_raw = last_kakan.get("pai_raw", last_kakan.get("pai"))
        furiten_list = state_snapshot.get("furiten", [False] * 4)
        actor_furiten = furiten_list[actor] if actor < len(furiten_list) else False
        if not actor_furiten and _call_can_hora(
            state_snapshot,
            actor=actor,
            target=kakan_actor,
            pai=tile_raw,
            is_tsumo=False,
            is_chankan=bool(hora_is_chankan) if hora_is_chankan is not None else True,
        ):
            legal.append(
                ActionSpec(type="hora", actor=actor, target=kakan_actor, pai=tile_raw)
            )
        legal.append(ActionSpec(type="none"))
        return legal

    if last_discard and last_discard.get("actor") != actor:
        discarder = last_discard["actor"]
        tile_norm = normalize_tile(last_discard["pai"])
        tile_raw = last_discard.get("pai_raw", tile_norm)
        # 荣和：检查 waits_tiles（由 bot.py/replay.py 注入），振听时不能荣和
        furiten_list = state_snapshot.get("furiten", [False] * 4)
        actor_furiten = furiten_list[actor] if actor < len(furiten_list) else False
        if not actor_furiten and _call_can_hora(
            state_snapshot,
            actor=actor,
            target=discarder,
            pai=tile_raw,
            is_tsumo=False,
            is_houtei=hora_is_houtei,
        ):
            legal.append(ActionSpec(type="hora", actor=actor, target=discarder, pai=tile_raw))
        if not reached:
            if _can_pon(hand, tile_raw):
                consumed_pon = _pick_consumed(hand, tile_norm, 2)
                legal.append(ActionSpec(type="pon", actor=actor, target=discarder, pai=tile_raw, consumed=tuple(consumed_pon)))
            if _can_daiminkan(hand, tile_raw):
                consumed_kan = _pick_consumed(hand, tile_norm, 3)
                legal.append(ActionSpec(type="daiminkan", actor=actor, target=discarder, pai=tile_raw, consumed=tuple(consumed_kan)))

            next_player = (discarder + 1) % 4
            if actor == next_player:
                for c in _chi_patterns(tile_raw):
                    if _hand_has_tile(hand, c[0]) and _hand_has_tile(hand, c[1]):
                        actual_c = [_pick_chi_tile(hand, c[0]), _pick_chi_tile(hand, c[1])]
                        legal.append(ActionSpec(type="chi", actor=actor, target=discarder, pai=tile_raw, consumed=tuple(actual_c)))

        legal.append(ActionSpec(type="none"))
        return legal

    if actor_to_move == actor:
        pending_reach = state_snapshot.get("pending_reach", [False, False, False, False])[actor]

        # 自摸：直接以真实和牌判定为准。不能依赖 waits_tiles，因为和了后的 14 张手
        # 可能 shanten=-1，waits_tiles 为空，若先看 waits 会把合法自摸错误过滤掉。
        if last_tsumo is not None:
            tsumo_pai_out = last_tsumo_raw if last_tsumo_raw is not None else last_tsumo
            if _call_can_hora(
                state_snapshot,
                actor=actor,
                target=actor,
                pai=tsumo_pai_out,
                is_tsumo=True,
                is_rinshan=hora_is_rinshan,
                is_haitei=hora_is_haitei,
            ):
                legal.append(ActionSpec(type="hora", actor=actor, target=actor, pai=tsumo_pai_out))

        if reached:
            # 立直已被接受：只能摸切（tsumogiri），或暗杠。
            if last_tsumo_raw is not None:
                legal.append(ActionSpec(type="dahai", actor=actor, pai=last_tsumo_raw, tsumogiri=True))
            elif last_tsumo is not None:
                legal.append(ActionSpec(type="dahai", actor=actor, pai=last_tsumo, tsumogiri=True))
            actor_melds = state_snapshot.get("melds", [[]])[actor]
            for pai, consumed in _ankan_candidates(hand):
                if _ankan_allowed_after_reach(hand, actor_melds, consumed, last_tsumo, last_tsumo_raw):
                    legal.append(ActionSpec(type="ankan", actor=actor, pai=pai, consumed=consumed))
            return legal

        if pending_reach:
            # 已宣告立直、等待打出立直宣言牌：只能打出让手牌保持听牌的牌。
            # 对每张候选牌，临时移除后检验 shanten==0（听牌）。
            for pai_out, tsumogiri in _reach_discard_candidates(hand, last_tsumo, last_tsumo_raw):
                legal.append(ActionSpec(type="dahai", actor=actor, pai=pai_out, tsumogiri=tsumogiri))
            if not legal:
                # fallback：shanten 计算失败时退化为全打法
                for tile in hand.keys():
                    pai_out = tile
                    if last_tsumo == tile and last_tsumo_raw is not None:
                        pai_out = last_tsumo_raw
                    legal.append(ActionSpec(type="dahai", actor=actor, pai=pai_out, tsumogiri=(last_tsumo == tile)))
            return legal

        # 普通打牌阶段
        # Note: mjai 会用 last_self_tsumo 的"牌面字符串"来判断 tsumogiri。
        # 我们的简化状态同样用 tile 字符串进行匹配。
        for tile in hand.keys():
            pai_out = tile
            if last_tsumo == tile and last_tsumo_raw is not None:
                pai_out = last_tsumo_raw
            legal.append(
                ActionSpec(
                    type="dahai",
                    actor=actor,
                    pai=pai_out,
                    tsumogiri=(last_tsumo == tile),
                )
            )
        # Reach 只能在"打牌选择"阶段宣告，此时手牌应为14张（已含自摸牌）且必须听牌。
        # 这里用更严格的条件，减少reach动作在不合时机时被加入legal set。
        # shanten 信息由 libriichi 在 mjai_bot.py 的 react() 中计算并添加到 snapshot。
        # 只有 shanten == 0 (听牌) 时才允许立直，防止模型错误选择导致 mjai_simulator 报错。
        actor_melds = state_snapshot.get("melds", [[]])[actor]
        if _can_declare_reach(hand, actor_melds, reached, last_tsumo, last_tsumo_raw):
            legal.append(ActionSpec(type="reach", actor=actor))
        for pai, consumed in _ankan_candidates(hand):
            legal.append(ActionSpec(type="ankan", actor=actor, pai=pai, consumed=consumed))
        # kakan（对已碰/明杠再杠）在"打牌选择"阶段可能出现。
        # 基于已有的 pon meld 枚举升级可能性；已正确处理 5m/5mr 等价，
        # _hand_has_tile 和 _pick_chi_tile 会优先返回手里实际存在的赤宝牌版本。
        for meld in state_snapshot.get("melds", [[]])[actor]:
            if meld.get("type") != "pon":
                continue
            meld_pai = meld.get("pai")
            if not meld_pai:
                continue
            # 需要手里还剩1张"杠牌"(考虑赤宝牌等价)。
            meld_norm = normalize_tile(meld_pai)
            if _hand_has_tile(hand, meld_norm):
                kakan_pai = _pick_chi_tile(hand, meld_norm)
                legal.append(
                    ActionSpec(
                        type="kakan",
                        actor=actor,
                        pai=kakan_pai,
                        consumed=tuple([
                            *(meld.get("consumed") or []),
                            meld_pai,
                        ]),
                    )
                )
        return legal

    if not last_discard:
        legal.append(ActionSpec(type="none"))
        return legal

    discarder = last_discard["actor"]
    tile = last_discard["pai"]
    tile_norm = normalize_tile(tile)
    tile_raw = last_discard.get("pai_raw", tile)
    if discarder == actor:
        legal.append(ActionSpec(type="none"))
        return legal

    if _can_pon(hand, tile_raw):
        legal.append(
            ActionSpec(
                type="pon",
                actor=actor,
                target=discarder,
                pai=tile_raw,
                consumed=tuple(_pick_consumed(hand, tile_norm, 2)),
            )
        )
    if _can_daiminkan(hand, tile_raw):
        legal.append(
            ActionSpec(
                type="daiminkan",
                actor=actor,
                target=discarder,
                pai=tile_raw,
                consumed=tuple(_pick_consumed(hand, tile_norm, 3)),
            )
        )

    next_player = (discarder + 1) % 4
    if actor == next_player:
        for c in _chi_patterns(tile_raw):
            if _hand_has_tile(hand, c[0]) and _hand_has_tile(hand, c[1]):
                actual_c = [_pick_chi_tile(hand, c[0]), _pick_chi_tile(hand, c[1])]
                legal.append(
                    ActionSpec(
                        type="chi",
                        actor=actor,
                        target=discarder,
                        pai=tile_raw,
                        consumed=tuple(actual_c),
                    )
                )

    legal.append(ActionSpec(type="none"))
    return legal


def enumerate_legal_action_specs(state_snapshot: Dict, actor: int) -> List[ActionSpec]:
    rust_public_specs = _rust_public_action_specs(state_snapshot, actor)
    if rust_public_specs is not None:
        return rust_public_specs

    # Keep Python legal assembly only as the missing-capability fallback.
    return _enumerate_legal_action_specs_python(state_snapshot, actor)


def enumerate_legal_actions(state_snapshot: Dict, actor: int) -> List[Action]:
    return [spec.to_action() for spec in enumerate_legal_action_specs(state_snapshot, actor)]
