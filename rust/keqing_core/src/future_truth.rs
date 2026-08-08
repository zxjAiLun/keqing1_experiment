use serde_json::Value;

use crate::counts::TILE_COUNT;
use crate::hora_truth::{evaluate_hora_truth_from_prepared, HoraTruth};
use crate::keqingv4_summary::{
    counts34_from_tiles, hand_tiles_from_snapshot, progress_key, project_keqingv4_discard_snapshot,
    project_keqingv4_rinshan_draw_snapshot, snapshot_with_after_hand, strip_aka, TILE34_STRINGS,
};
use crate::legal_actions::prepare_hora_evaluation_from_snapshot;
use crate::progress_summary::{summarize_like_python, Summary3n1};
use crate::value_proxy::{hora_truth_value_proxy_from_parts, hora_truth_yaku_flags_from_names};

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FutureTruthMetrics {
    pub max_hand_value_proxy: f32,
    pub reach_hand_value_proxy: f32,
    pub tanyao_route: f32,
    pub yakuhai_route: f32,
    pub chiitoi_route: f32,
    pub iipeiko_route: f32,
    pub pinfu_route: f32,
}

impl FutureTruthMetrics {
    fn weighted_add(&mut self, other: Self, weight: f32) {
        self.max_hand_value_proxy =
            (self.max_hand_value_proxy + other.max_hand_value_proxy * weight).clamp(0.0, 1.0);
        self.reach_hand_value_proxy =
            (self.reach_hand_value_proxy + other.reach_hand_value_proxy * weight).clamp(0.0, 1.0);
        self.tanyao_route = (self.tanyao_route + other.tanyao_route * weight).clamp(0.0, 1.0);
        self.yakuhai_route = (self.yakuhai_route + other.yakuhai_route * weight).clamp(0.0, 1.0);
        self.chiitoi_route = (self.chiitoi_route + other.chiitoi_route * weight).clamp(0.0, 1.0);
        self.iipeiko_route = (self.iipeiko_route + other.iipeiko_route * weight).clamp(0.0, 1.0);
        self.pinfu_route = (self.pinfu_route + other.pinfu_route * weight).clamp(0.0, 1.0);
    }
}

fn future_truth_depth_for_shanten(shanten: i8) -> Option<u8> {
    match shanten {
        0 => Some(0),
        1 => Some(1),
        2 => Some(2),
        _ => None,
    }
}

fn truth_metrics_from_hora(truth: &HoraTruth) -> FutureTruthMetrics {
    let cost_total = truth.cost.get("total").and_then(Value::as_i64).unwrap_or(0);
    let yaku_flags = hora_truth_yaku_flags_from_names(truth.yaku.iter().map(String::as_str));
    FutureTruthMetrics {
        max_hand_value_proxy: hora_truth_value_proxy_from_parts(truth.han, cost_total),
        reach_hand_value_proxy: 0.0,
        tanyao_route: yaku_flags.tanyao,
        yakuhai_route: yaku_flags.yakuhai,
        chiitoi_route: yaku_flags.chiitoi,
        iipeiko_route: yaku_flags.iipeiko,
        pinfu_route: yaku_flags.pinfu,
    }
}

fn exact_tenpai_truth_metrics(
    snapshot: &Value,
    actor: usize,
    after_hand: &[String],
    summary: &Summary3n1,
    visible_counts34: &[u8; TILE_COUNT],
    open_hand_flag: f32,
) -> FutureTruthMetrics {
    if summary.shanten != 0 {
        return FutureTruthMetrics::default();
    }

    let base_snapshot = snapshot_with_after_hand(snapshot, actor, after_hand);
    let mut metrics = FutureTruthMetrics::default();

    for tile34 in 0..TILE_COUNT {
        if !summary.waits_tiles[tile34] {
            continue;
        }
        let live = 4u8.saturating_sub(visible_counts34[tile34]);
        if live == 0 {
            continue;
        }
        let draw_tile = TILE34_STRINGS[tile34].to_string();
        let draw_snapshot =
            project_keqingv4_rinshan_draw_snapshot(&base_snapshot, actor, &draw_tile);
        let Ok(prepared) = prepare_hora_evaluation_from_snapshot(
            &draw_snapshot,
            actor,
            &draw_tile,
            true,
            false,
            None,
            None,
            None,
        ) else {
            continue;
        };
        let Ok(truth) = evaluate_hora_truth_from_prepared(&prepared) else {
            continue;
        };
        let truth_metrics = truth_metrics_from_hora(&truth);
        metrics.max_hand_value_proxy = metrics
            .max_hand_value_proxy
            .max(truth_metrics.max_hand_value_proxy);
        metrics.tanyao_route = metrics.tanyao_route.max(truth_metrics.tanyao_route);
        metrics.yakuhai_route = metrics.yakuhai_route.max(truth_metrics.yakuhai_route);
        metrics.chiitoi_route = metrics.chiitoi_route.max(truth_metrics.chiitoi_route);
        metrics.iipeiko_route = metrics.iipeiko_route.max(truth_metrics.iipeiko_route);
        metrics.pinfu_route = metrics.pinfu_route.max(truth_metrics.pinfu_route);
        if open_hand_flag > 0.5 {
            continue;
        }
        let mut reached_snapshot = draw_snapshot.clone();
        if let Some(obj) = reached_snapshot.as_object_mut() {
            let mut reached = obj
                .get("reached")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_else(|| vec![Value::Bool(false); 4]);
            while reached.len() <= actor {
                reached.push(Value::Bool(false));
            }
            reached[actor] = Value::Bool(true);
            obj.insert("reached".to_string(), Value::Array(reached));
            let mut pending_reach = obj
                .get("pending_reach")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_else(|| vec![Value::Bool(false); 4]);
            while pending_reach.len() <= actor {
                pending_reach.push(Value::Bool(false));
            }
            pending_reach[actor] = Value::Bool(false);
            obj.insert("pending_reach".to_string(), Value::Array(pending_reach));
            let mut ippatsu = obj
                .get("ippatsu_eligible")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_else(|| vec![Value::Bool(false); 4]);
            while ippatsu.len() <= actor {
                ippatsu.push(Value::Bool(false));
            }
            ippatsu[actor] = Value::Bool(false);
            obj.insert("ippatsu_eligible".to_string(), Value::Array(ippatsu));
        }
        let Ok(reached_prepared) = prepare_hora_evaluation_from_snapshot(
            &reached_snapshot,
            actor,
            &draw_tile,
            true,
            false,
            None,
            None,
            None,
        ) else {
            continue;
        };
        let Ok(reached_truth) = evaluate_hora_truth_from_prepared(&reached_prepared) else {
            continue;
        };
        let reached_metrics = truth_metrics_from_hora(&reached_truth);
        metrics.reach_hand_value_proxy = metrics
            .reach_hand_value_proxy
            .max(reached_metrics.max_hand_value_proxy);
    }

    metrics
}

fn best_child_metrics_after_draw(
    draw_snapshot: &Value,
    actor: usize,
    visible_after_draw: &[u8; TILE_COUNT],
    depth_remaining: u8,
    open_hand_flag: f32,
) -> FutureTruthMetrics {
    let draw_hand = hand_tiles_from_snapshot(draw_snapshot);
    let mut best_key: Option<(i32, i32, i32, i32)> = None;
    let mut best_metrics = FutureTruthMetrics::default();
    let mut seen = std::collections::BTreeSet::<String>::new();
    for discard_tile in &draw_hand {
        let discard_key = strip_aka(discard_tile);
        if !seen.insert(discard_key) {
            continue;
        }
        let discard_snapshot =
            project_keqingv4_discard_snapshot(draw_snapshot, actor, discard_tile);
        let discard_hand = hand_tiles_from_snapshot(&discard_snapshot);
        let discard_counts = counts34_from_tiles(&discard_hand);
        let discard_summary = summarize_like_python(&discard_counts, visible_after_draw);
        let key = progress_key(&discard_summary);
        let candidate_metrics = future_truth_metrics_with_depth(
            &discard_snapshot,
            actor,
            &discard_hand,
            &discard_summary,
            visible_after_draw,
            open_hand_flag,
            depth_remaining.saturating_sub(1),
        );
        let should_take = best_key.as_ref().is_none_or(|current| {
            key > *current
                || (key == *current
                    && candidate_metrics.max_hand_value_proxy > best_metrics.max_hand_value_proxy)
        });
        if should_take {
            best_key = Some(key);
            best_metrics = candidate_metrics;
        }
    }
    best_metrics
}

fn future_truth_metrics_with_depth(
    snapshot: &Value,
    actor: usize,
    after_hand: &[String],
    summary: &Summary3n1,
    visible_counts34: &[u8; TILE_COUNT],
    open_hand_flag: f32,
    depth_remaining: u8,
) -> FutureTruthMetrics {
    if depth_remaining == 0 {
        return exact_tenpai_truth_metrics(
            snapshot,
            actor,
            after_hand,
            summary,
            visible_counts34,
            open_hand_flag,
        );
    }

    let base_snapshot = snapshot_with_after_hand(snapshot, actor, after_hand);
    let counts34 = counts34_from_tiles(after_hand);
    let mut metrics = FutureTruthMetrics::default();

    for tile34 in 0..TILE_COUNT {
        let live = 4u8.saturating_sub(visible_counts34[tile34]);
        if live == 0 || counts34[tile34] >= 4 {
            continue;
        }

        let draw_tile = TILE34_STRINGS[tile34].to_string();
        let draw_snapshot =
            project_keqingv4_rinshan_draw_snapshot(&base_snapshot, actor, &draw_tile);
        let mut visible_after_draw = *visible_counts34;
        visible_after_draw[tile34] = visible_after_draw[tile34].saturating_add(1);
        let best_metrics = best_child_metrics_after_draw(
            &draw_snapshot,
            actor,
            &visible_after_draw,
            depth_remaining,
            open_hand_flag,
        );
        metrics.weighted_add(best_metrics, live as f32 / 34.0);
    }

    metrics
}

pub fn future_truth_metrics(
    snapshot: &Value,
    actor: usize,
    after_hand: &[String],
    summary: &Summary3n1,
    visible_counts34: &[u8; TILE_COUNT],
    open_hand_flag: f32,
) -> FutureTruthMetrics {
    let Some(depth) = future_truth_depth_for_shanten(summary.shanten) else {
        return FutureTruthMetrics::default();
    };
    future_truth_metrics_with_depth(
        snapshot,
        actor,
        after_hand,
        summary,
        visible_counts34,
        open_hand_flag,
        depth,
    )
}

#[cfg(test)]
mod tests {
    use super::{future_truth_metrics, FutureTruthMetrics};
    use crate::progress_summary::Summary3n1;

    #[test]
    fn future_truth_metrics_returns_default_above_train_ready_depth() {
        let summary = Summary3n1 {
            shanten: 3,
            waits_count: 0,
            waits_tiles: [false; crate::counts::TILE_COUNT],
            tehai_count: 13,
            ukeire_type_count: 0,
            ukeire_live_count: 0,
            ukeire_tiles: [false; crate::counts::TILE_COUNT],
        };
        let snapshot = serde_json::json!({
            "hand": ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p"],
            "melds": [[], [], [], []],
            "discards": [[], [], [], []],
            "reached": [false, false, false, false],
        });
        let metrics = future_truth_metrics(
            &snapshot,
            0,
            &[
                "1m".to_string(),
                "2m".to_string(),
                "3m".to_string(),
                "4m".to_string(),
                "5m".to_string(),
                "6m".to_string(),
                "7m".to_string(),
                "8m".to_string(),
                "9m".to_string(),
                "1p".to_string(),
                "2p".to_string(),
                "3p".to_string(),
                "4p".to_string(),
            ],
            &summary,
            &[0; crate::counts::TILE_COUNT],
            0.0,
        );
        assert_eq!(metrics, FutureTruthMetrics::default());
    }
}
