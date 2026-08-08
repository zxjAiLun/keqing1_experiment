use _native::xmodel1_export::{validate_xmodel1_discard_record, xmodel1_schema_info};
use _native::xmodel1_schema::{
    XMODEL1_CANDIDATE_FEATURE_DIM, XMODEL1_CANDIDATE_FLAG_DIM, XMODEL1_MAX_CANDIDATES,
    XMODEL1_SCHEMA_NAME, XMODEL1_SCHEMA_VERSION,
};

#[test]
fn xmodel1_schema_info_matches_constants() {
    let (name, version, max_candidates, candidate_dim, flag_dim) = xmodel1_schema_info();
    assert_eq!(name, XMODEL1_SCHEMA_NAME);
    assert_eq!(version, XMODEL1_SCHEMA_VERSION);
    assert_eq!(max_candidates, XMODEL1_MAX_CANDIDATES);
    assert_eq!(candidate_dim, XMODEL1_CANDIDATE_FEATURE_DIM);
    assert_eq!(flag_dim, XMODEL1_CANDIDATE_FLAG_DIM);
}

#[test]
fn validate_accepts_basic_valid_record() {
    let mut candidate_mask = vec![0u8; XMODEL1_MAX_CANDIDATES];
    let mut candidate_tile_id = vec![-1i16; XMODEL1_MAX_CANDIDATES];
    candidate_mask[0] = 1;
    candidate_mask[1] = 1;
    candidate_tile_id[0] = 4;
    candidate_tile_id[1] = 27;
    validate_xmodel1_discard_record(0, &candidate_mask, &candidate_tile_id).unwrap();
}

#[test]
fn validate_rejects_invalid_padding_tile_id() {
    let mut candidate_mask = vec![0u8; XMODEL1_MAX_CANDIDATES];
    let mut candidate_tile_id = vec![-1i16; XMODEL1_MAX_CANDIDATES];
    candidate_mask[0] = 1;
    candidate_tile_id[0] = 4;
    candidate_tile_id[1] = 3;
    let err = validate_xmodel1_discard_record(0, &candidate_mask, &candidate_tile_id).unwrap_err();
    assert!(err.contains("padding candidate"));
}
