use serde_json::{json, Value};

pub const NATIVE_BOUNDARY_SCHEMA_NAME: &str = "keqingrl_native_boundary";
pub const NATIVE_BOUNDARY_SCHEMA_VERSION: u32 = 1;
pub const ACTION_IDENTITY_VERSION: u32 = 1;
pub const LEGAL_ENUMERATION_VERSION: u32 = 1;
pub const TERMINAL_RESOLVER_VERSION: u32 = 1;
pub const RUNTIME_FALLBACK_POLICY: &str = "fail_closed_no_silent_python_fallback";

pub fn native_schema_info() -> Value {
    json!({
        "schema_name": NATIVE_BOUNDARY_SCHEMA_NAME,
        "schema_version": NATIVE_BOUNDARY_SCHEMA_VERSION,
        "action_identity_version": ACTION_IDENTITY_VERSION,
        "legal_enumeration_version": LEGAL_ENUMERATION_VERSION,
        "terminal_resolver_version": TERMINAL_RESOLVER_VERSION,
        "runtime_fallback_policy": RUNTIME_FALLBACK_POLICY,
        "capabilities": {
            "action_identity_v1": "supported",
            "action_id_decode_v1": "supported",
            "mjai_event_expansion_v1": "supported",
            "legal_enumeration_v1": "supported",
            "terminal_resolver_v1": "supported",
            "state_apply_validation_v1": "supported",
            "action_feature_parity_v1": "supported",
            "fixed_seed_eval_gate_v1": "supported",
            "typed_action_feature_api_v1": "supported",
            "nuki": "unsupported",
            "typed_hot_path_api": "planned",
            "json_bridge": "golden_debug_only"
        },
        "supported_action_types": [
            "DISCARD",
            "REACH_DISCARD",
            "TSUMO",
            "RON",
            "CHI",
            "PON",
            "DAIMINKAN",
            "ANKAN",
            "KAKAN",
            "PASS",
            "RYUKYOKU"
        ],
        "unsupported_action_types": ["NUKI"]
    })
}
