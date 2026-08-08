from mahjong_env.replay import _calc_normal_progress, _calc_shanten_waits


def _parse_hand(s: str):
    out = []
    digits = ""
    honors = {"1": "E", "2": "S", "3": "W", "4": "N", "5": "P", "6": "F", "7": "C"}
    for ch in s:
        if ch.isdigit():
            digits += ch
        else:
            if ch in "mps":
                for d in digits:
                    out.append((d if d != "0" else "5") + ch)
            elif ch == "z":
                out.extend(honors[d] for d in digits)
            digits = ""
    return out


def test_normal_progress_one_shanten_has_ukeire_without_shape_improvement_metric():
    hand = ["1m", "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "2p", "3p", "7s", "7s"]
    progress = _calc_normal_progress(hand, [])

    assert progress.shanten == 1
    assert progress.ukeire_type_count > 0
    assert progress.ukeire_live_count >= progress.ukeire_type_count
    assert progress.improvement_type_count == 0
    assert progress.improvement_live_count == 0


def test_normal_progress_tenpai_has_waits_and_may_have_improvements():
    hand = ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "4p", "5p", "7s", "7s"]
    progress = _calc_normal_progress(hand, [])

    assert progress.shanten == 0
    assert progress.waits_count > 0
    assert any(progress.waits_tiles)
    assert progress.ukeire_type_count == 0
    assert progress.improvement_type_count == 0
    assert progress.improvement_live_count == 0


def test_normal_progress_reference_case_zeroes_good_shape_metrics():
    hand14 = _parse_hand("13458m34678p668s4m")
    hand13 = list(hand14)
    hand13.remove("1m")
    progress = _calc_normal_progress(hand13, [])

    assert progress.shanten == 2
    assert progress.ukeire_live_count == 48
    assert progress.good_shape_ukeire_live_count == 0
    assert progress.good_shape_ukeire_type_count == 0


def test_one_shanten_reference_case_zeroes_good_shape_metrics():
    hand14 = _parse_hand("4m2067p4s5z0s")
    hand13 = list(hand14)
    hand13.remove("4m")
    progress = _calc_normal_progress(hand13, [])

    assert progress.shanten == 1
    assert progress.ukeire_live_count == 14
    assert progress.good_shape_ukeire_live_count == 0
    assert progress.good_shape_ukeire_type_count == 0


def test_one_shanten_reference_case_matches_all_equal_discards():
    hand14 = _parse_hand("4m2067p4s5z0s")
    for discard in ("4m", "2p", "P"):
        hand13 = list(hand14)
        hand13.remove(discard)
        progress = _calc_normal_progress(hand13, [])
        assert progress.shanten == 1
        assert progress.ukeire_live_count == 14
        assert progress.good_shape_ukeire_live_count == 0


def test_two_shanten_reference_case_matches_16_live_ukeire():
    hand14 = _parse_hand("233479m58p2278s33z")
    hand13 = list(hand14)
    hand13.remove("3m")
    progress = _calc_normal_progress(hand13, [])

    assert progress.shanten == 2
    assert progress.ukeire_live_count == 16


def test_calc_shanten_waits_includes_chiitoi_waits_without_melds():
    hand = _parse_hand("1122334455667m")
    shanten, waits_count, waits_tiles, _ = _calc_shanten_waits(hand, [])

    assert shanten == 0
    assert waits_tiles[6] is True  # 7m
    assert waits_count >= 1


def test_calc_shanten_waits_includes_kokushi_waits_without_melds():
    hand = _parse_hand("19m19p19s1234566z")
    shanten, waits_count, waits_tiles, _ = _calc_shanten_waits(hand, [])

    assert shanten == 0
    assert waits_count == 1
    assert waits_tiles[33] is True  # 7z / C
