use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct DiscardEntry {
    pub pai: String,
    pub tsumogiri: bool,
    pub reach_declared: bool,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct LastDiscard {
    pub actor: usize,
    pub pai: String,
    pub pai_raw: String,
}
