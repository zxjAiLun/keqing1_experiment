use crate::state_core::{GameStateCore, SnapshotCore};

pub fn snapshot_for_actor(state: &GameStateCore, actor: usize) -> SnapshotCore {
    let player = &state.players[actor];
    let mut hand = Vec::new();
    for (tile, count) in player.hand.iter() {
        for _ in 0..*count {
            hand.push(tile.clone());
        }
    }
    SnapshotCore {
        bakaze: state.bakaze.clone(),
        kyoku: state.kyoku,
        honba: state.honba,
        kyotaku: state.kyotaku,
        oya: state.oya,
        scores: state.scores.clone(),
        dora_markers: state.dora_markers.clone(),
        ura_dora_markers: state.ura_dora_markers.clone(),
        actor,
        hand,
        tsumo_pai: None,
        discards: state.players.iter().map(|p| p.discards.clone()).collect(),
        melds: state.players.iter().map(|p| p.melds.clone()).collect(),
        reached: state.players.iter().map(|p| p.reached).collect(),
        pending_reach: state.players.iter().map(|p| p.pending_reach).collect(),
        actor_to_move: state.actor_to_move,
        last_discard: state.last_discard.clone(),
        last_kakan: state.last_kakan.clone(),
        last_tsumo: state.last_tsumo.clone(),
        last_tsumo_raw: state.last_tsumo_raw.clone(),
        remaining_wall: state.remaining_wall,
        pending_rinshan_actor: state.pending_rinshan_actor,
        ryukyoku_tenpai_players: state.ryukyoku_tenpai_players.clone(),
        furiten: state.players.iter().map(|p| p.furiten).collect(),
        sutehai_furiten: state.players.iter().map(|p| p.sutehai_furiten).collect(),
        riichi_furiten: state.players.iter().map(|p| p.riichi_furiten).collect(),
        doujun_furiten: state.players.iter().map(|p| p.doujun_furiten).collect(),
        ippatsu_eligible: state.players.iter().map(|p| p.ippatsu_eligible).collect(),
        rinshan_tsumo: state.players.iter().map(|p| p.rinshan_tsumo).collect(),
    }
}
