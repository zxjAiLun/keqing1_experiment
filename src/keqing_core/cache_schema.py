from __future__ import annotations

BASE_CACHE_FIELDS = ("tile_feat", "scalar", "mask", "action_idx", "value")

KEQINGV4_SUMMARY_DIM = 28
KEQINGV4_CALL_SUMMARY_SLOTS = 8
KEQINGV4_SPECIAL_SUMMARY_SLOTS = 3
KEQINGV4_EVENT_HISTORY_LEN = 48
KEQINGV4_EVENT_HISTORY_DIM = 5
KEQINGV4_OPPORTUNITY_DIM = 3
KEQINGV4_RULE_CONTEXT_DIM = 6
KEQINGV4_EXTRA_FIELDS = (
    "score_delta_target",
    "win_target",
    "dealin_target",
    "pts_given_win_target",
    "pts_given_dealin_target",
    "opp_tenpai_target",
    "final_rank_target",
    "final_score_delta_points_target",
    "event_history",
    "v4_opportunity",
    "v4_discard_summary",
    "v4_call_summary",
    "v4_special_summary",
    "rule_context",
)

XMODEL2_FINAL_RANK_DIM = 4
XMODEL2_EXTRA_FIELDS = (
    "score_delta_target",
    "win_target",
    "dealin_target",
    "pts_given_win_target",
    "pts_given_dealin_target",
    "opp_tenpai_target",
    "final_rank_target",
    "final_score_delta_points_target",
)

# Xmodel1 discard cache schema (response-only + placement auxiliary)
XMODEL1_SCHEMA_NAME = "xmodel1_discard_v6"
XMODEL1_SCHEMA_VERSION = 6
XMODEL1_MAX_CANDIDATES = 14
XMODEL1_CANDIDATE_FEATURE_DIM = 22
XMODEL1_CANDIDATE_FLAG_DIM = 8
XMODEL1_MAX_SPECIAL_CANDIDATES = 12
XMODEL1_SPECIAL_CANDIDATE_FEATURE_DIM = 19
XMODEL1_HISTORY_SUMMARY_DIM = 20
XMODEL1_MAX_RESPONSE_CANDIDATES = 8
XMODEL1_RULE_CONTEXT_DIM = 6

XMODEL1_BASE_FIELDS = (
    "state_tile_feat",
    "state_scalar",
    "candidate_feat",
    "candidate_tile_id",
    "candidate_mask",
    "candidate_flags",
    "chosen_candidate_idx",
)

XMODEL1_TEACHER_FIELDS = (
    "candidate_quality_score",
    "candidate_hard_bad_flag",
)

XMODEL1_AUX_TARGET_FIELDS = (
    "win_target",
    "dealin_target",
    "pts_given_win_target",
    "pts_given_dealin_target",
    "opp_tenpai_target",
    "final_rank_target",
    "final_score_delta_points_target",
    "history_summary",
)

XMODEL1_METADATA_FIELDS = (
    "sample_type",
    "action_idx_target",
    "actor",
    "event_index",
    "kyoku",
    "honba",
    "is_open_hand",
)

XMODEL1_SPECIAL_FIELDS = ()

XMODEL1_RESPONSE_FIELDS = (
    "response_action_idx",
    "response_action_mask",
    "chosen_response_action_idx",
    "response_post_candidate_feat",
    "response_post_candidate_tile_id",
    "response_post_candidate_mask",
    "response_post_candidate_flags",
    "response_post_candidate_quality_score",
    "response_post_candidate_hard_bad_flag",
    "response_teacher_discard_idx",
)

__all__ = [
    "BASE_CACHE_FIELDS",
    "KEQINGV4_SUMMARY_DIM",
    "KEQINGV4_CALL_SUMMARY_SLOTS",
    "KEQINGV4_SPECIAL_SUMMARY_SLOTS",
    "KEQINGV4_EVENT_HISTORY_LEN",
    "KEQINGV4_EVENT_HISTORY_DIM",
    "KEQINGV4_OPPORTUNITY_DIM",
    "KEQINGV4_RULE_CONTEXT_DIM",
    "KEQINGV4_EXTRA_FIELDS",
    "XMODEL2_FINAL_RANK_DIM",
    "XMODEL2_EXTRA_FIELDS",
    "XMODEL1_SCHEMA_NAME",
    "XMODEL1_SCHEMA_VERSION",
    "XMODEL1_MAX_CANDIDATES",
    "XMODEL1_CANDIDATE_FEATURE_DIM",
    "XMODEL1_CANDIDATE_FLAG_DIM",
    "XMODEL1_MAX_SPECIAL_CANDIDATES",
    "XMODEL1_SPECIAL_CANDIDATE_FEATURE_DIM",
    "XMODEL1_HISTORY_SUMMARY_DIM",
    "XMODEL1_MAX_RESPONSE_CANDIDATES",
    "XMODEL1_RULE_CONTEXT_DIM",
    "XMODEL1_BASE_FIELDS",
    "XMODEL1_TEACHER_FIELDS",
    "XMODEL1_AUX_TARGET_FIELDS",
    "XMODEL1_METADATA_FIELDS",
    "XMODEL1_SPECIAL_FIELDS",
    "XMODEL1_RESPONSE_FIELDS",
]
