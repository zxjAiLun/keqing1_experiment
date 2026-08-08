from __future__ import annotations

from training.mortal import eval_metrics
from training.mortal import stat_report


def test_summarize_rank_counts_exports_rank_and_pt() -> None:
    summary = eval_metrics.summarize_rank_counts([1, 2, 1, 0])

    assert summary["games"] == 4
    assert summary["rank_counts"] == [1, 2, 1, 0]
    assert summary["avg_rank"] == 2.0
    assert summary["avg_rank_pt"] == 45.0


def test_rank_point_profiles_and_custom_values() -> None:
    profile, points = eval_metrics.resolve_rank_points(profile="avoid4_norm")
    assert profile == "avoid4_norm"
    assert points == (15.0 / 7.0, 9.0 / 7.0, 3.0 / 7.0, -27.0 / 7.0)
    assert eval_metrics.resolve_rank_points(profile="mortal_default")[1] == (6.0, 4.0, 2.0, 0.0)
    assert "zero_sum_balanced" not in eval_metrics.RANK_POINT_PROFILES
    assert eval_metrics.resolve_rank_points(rank_points="1,2,3,4") == ("custom", (1.0, 2.0, 3.0, 4.0))

    try:
        eval_metrics.parse_rank_points("1,2,3")
    except ValueError as exc:
        assert "length 4" in str(exc)
    else:
        raise AssertionError("expected invalid rank point list to fail")

    try:
        eval_metrics.resolve_rank_points(profile="custom")
    except ValueError as exc:
        assert "requires --rank-points" in str(exc)
    else:
        raise AssertionError("expected custom profile without values to fail")


def test_build_metrics_document_records_rank_point_metadata() -> None:
    document = eval_metrics.build_metrics_document(
        run={"kind": "unit"},
        metrics={},
        rank_points_profile="custom",
        rank_points_values=(1, 2, 3, 4),
    )

    assert document["rank_points_profile"] == "custom"
    assert document["rank_points_values"] == [1.0, 2.0, 3.0, 4.0]


def test_stat_report_normalizes_and_formats_markdown() -> None:
    class FakeStat:
        game = 4
        round = 40
        oya = 10
        point = 1200
        rank_1 = 1
        rank_2 = 1
        rank_3 = 1
        rank_4 = 1
        tobi = 0
        agari = 8
        houjuu = 4
        fuuro = 12
        fuuro_num = 18
        riichi = 6
        ryukyoku = 2
        yakuman = 0
        nagashi_mangan = 0

        rank_1_rate = 0.25
        rank_2_rate = 0.25
        rank_3_rate = 0.25
        rank_4_rate = 0.25
        tobi_rate = 0.0
        avg_rank = 2.5
        avg_point_per_game = 300.0
        avg_point_per_round = 30.0
        agari_rate = 0.2
        houjuu_rate = 0.1
        fuuro_rate = 0.3
        riichi_rate = 0.15
        ryukyoku_rate = 0.05
        avg_point_per_agari = 6000.0
        avg_point_per_oya_agari = float("nan")
        avg_point_per_ko_agari = 5200.0
        avg_point_per_riichi_agari = 7800.0
        avg_point_per_fuuro_agari = 4300.0
        avg_point_per_dama_agari = 6200.0
        avg_point_per_ryukyoku = 0.0
        avg_agari_jun = 11.0
        avg_riichi_agari_jun = 11.5
        avg_fuuro_agari_jun = 10.8
        avg_dama_agari_jun = 12.0
        avg_houjuu_jun = 11.2
        avg_point_per_houjuu = -5200.0
        avg_point_per_houjuu_to_oya = -7200.0
        avg_point_per_houjuu_to_ko = -4500.0
        chasing_riichi_rate = 0.1
        riichi_chased_rate = 0.2
        agari_rate_after_riichi = 0.5
        houjuu_rate_after_riichi = 0.16
        avg_riichi_jun = 7.8
        avg_riichi_point = 3000.0
        avg_fuuro_num = 1.5
        agari_rate_after_fuuro = 0.3
        houjuu_rate_after_fuuro = 0.14
        avg_fuuro_point = 800.0
        agari_rate_as_oya = 0.2
        agari_as_oya_rate = 0.25
        houjuu_to_oya_rate = 0.25
        yakuman_rate = 0.0
        nagashi_mangan_rate = 0.0

        def total_pt(self, pts):
            return sum(pts)

        def avg_pt(self, pts):
            return sum(pts) / self.game

        def __str__(self):
            return "fake stat"

    metrics = stat_report.stat_to_metrics(FakeStat(), rank_pts=[2.5, 1.5, 0.5, -4.5])
    report = {
        "backend": "fake",
        "log_dir": "logs",
        "rank_pts": [90, 45, 0, -135],
        "players": {"A (x1)": {"player_name": "challenger", **metrics}},
    }

    markdown = stat_report.format_markdown_report(report)

    assert metrics["derived"]["avg_point_per_oya_agari"] is None
    assert metrics["derived"]["total_rank_pt"] == 0.0
    assert metrics["derived"]["avg_rank_pt"] == 0.0
    assert "| Games | 4 |" in markdown
    assert "- Rank point profile: `custom`" in markdown
    assert "| 1st (rate) | 1 (0.250000) |" in markdown
