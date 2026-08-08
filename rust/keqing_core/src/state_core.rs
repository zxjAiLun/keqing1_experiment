use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::types::{DiscardEntry, LastDiscard};

#[derive(Clone, Debug, Default)]
pub struct PlayerStateCore {
    pub hand: BTreeMap<String, u8>,
    pub discards: Vec<DiscardEntry>,
    pub melds: Vec<serde_json::Value>,
    pub reached: bool,
    pub pending_reach: bool,
    pub furiten: bool,
    pub sutehai_furiten: bool,
    pub riichi_furiten: bool,
    pub doujun_furiten: bool,
    pub ippatsu_eligible: bool,
    pub rinshan_tsumo: bool,
}

#[derive(Clone, Debug)]
pub struct GameStateCore {
    pub bakaze: String,
    pub kyoku: i32,
    pub honba: i32,
    pub kyotaku: i32,
    pub oya: usize,
    pub dora_markers: Vec<String>,
    pub ura_dora_markers: Vec<String>,
    pub scores: Vec<i32>,
    pub players: Vec<PlayerStateCore>,
    pub actor_to_move: Option<usize>,
    pub last_discard: Option<LastDiscard>,
    pub last_kakan: Option<serde_json::Value>,
    pub last_tsumo: Vec<Option<String>>,
    pub last_tsumo_raw: Vec<Option<String>>,
    pub remaining_wall: Option<i32>,
    pub pending_rinshan_actor: Option<usize>,
    pub ryukyoku_tenpai_players: Vec<usize>,
    pub in_game: bool,
}

impl Default for GameStateCore {
    fn default() -> Self {
        Self {
            bakaze: "E".to_string(),
            kyoku: 1,
            honba: 0,
            kyotaku: 0,
            oya: 0,
            dora_markers: Vec::new(),
            ura_dora_markers: Vec::new(),
            scores: vec![25000, 25000, 25000, 25000],
            players: vec![
                PlayerStateCore::default(),
                PlayerStateCore::default(),
                PlayerStateCore::default(),
                PlayerStateCore::default(),
            ],
            actor_to_move: None,
            last_discard: None,
            last_kakan: None,
            last_tsumo: vec![None, None, None, None],
            last_tsumo_raw: vec![None, None, None, None],
            remaining_wall: None,
            pending_rinshan_actor: None,
            ryukyoku_tenpai_players: Vec::new(),
            in_game: false,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SnapshotCore {
    pub bakaze: String,
    pub kyoku: i32,
    pub honba: i32,
    pub kyotaku: i32,
    pub oya: usize,
    pub scores: Vec<i32>,
    pub dora_markers: Vec<String>,
    pub ura_dora_markers: Vec<String>,
    pub actor: usize,
    pub hand: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tsumo_pai: Option<String>,
    pub discards: Vec<Vec<DiscardEntry>>,
    pub melds: Vec<Vec<serde_json::Value>>,
    pub reached: Vec<bool>,
    pub pending_reach: Vec<bool>,
    pub actor_to_move: Option<usize>,
    pub last_discard: Option<LastDiscard>,
    pub last_kakan: Option<serde_json::Value>,
    pub last_tsumo: Vec<Option<String>>,
    pub last_tsumo_raw: Vec<Option<String>>,
    pub remaining_wall: Option<i32>,
    pub pending_rinshan_actor: Option<usize>,
    pub ryukyoku_tenpai_players: Vec<usize>,
    pub furiten: Vec<bool>,
    pub sutehai_furiten: Vec<bool>,
    pub riichi_furiten: Vec<bool>,
    pub doujun_furiten: Vec<bool>,
    pub ippatsu_eligible: Vec<bool>,
    pub rinshan_tsumo: Vec<bool>,
}
