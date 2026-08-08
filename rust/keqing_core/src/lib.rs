//! Keqing Core - Mahjong AI computation library
//!
//! This crate provides optimized Rust implementations of hot-path
//! computation functions for the Keqing Mahjong AI system.

pub mod action_features;
pub mod action_identity;
pub mod boundary_contract;
pub mod continuation_scenarios;
pub mod continuation_scoring;
pub mod counts;
pub mod eval_gate;
pub mod event_apply;
pub mod export_common;
pub mod future_truth;
pub mod hora_truth;
pub mod keqingv4_export;
pub mod keqingv4_summary;
pub mod legal_actions;
pub mod native_scoring;
pub mod progress_batch;
pub mod progress_delta;
pub mod progress_summary;
pub mod py_module;
pub mod replay_export_core;
pub mod replay_samples;
pub mod rulebase;
pub mod score_rules;
pub mod scoring_pool;
pub mod shanten;
pub mod shanten_table;
pub mod snapshot;
pub mod standard;
pub mod state_core;
pub mod terminal_resolver;
pub mod types;
pub mod value_proxy;
pub mod xmodel1_export;
pub mod xmodel1_schema;

pub use counts::{Counts34, TILE_COUNT};
pub use progress_delta::{
    calc_discard_deltas, calc_draw_deltas, calc_required_tiles, DiscardDelta, DrawDelta,
    RequiredTile,
};
pub use progress_summary::{summarize_3n1, Summary3n1};
pub use shanten_table::{calc_shanten_all, calc_shanten_normal};
pub use standard::counts34_to_ids;
pub use xmodel1_schema::{
    XMODEL1_CANDIDATE_FEATURE_DIM, XMODEL1_CANDIDATE_FLAG_DIM, XMODEL1_MAX_CANDIDATES,
    XMODEL1_SCHEMA_NAME, XMODEL1_SCHEMA_VERSION,
};
