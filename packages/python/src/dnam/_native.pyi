"""PEP 561 Static Type Stub for `dnam._native` (`ref[sdk.python.layout.curated-exports]`)."""

from __future__ import annotations

from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

class TelemetryDict(TypedDict):
    active_tier: int
    backend: str
    sample_rate_hz: int
    stereo_out: bool
    bypassed: bool
    trails_enabled: bool
    is_frozen: bool
    supports_freeze: bool
    stereo_width: float
    current_bpm: float
    active_clamp_mask: int
    control_decimation_host: int
    max_op_n_drive: float
    max_pole_radius: float
    max_effective_loop_gain: float
    tail_time_ms: float
    z_head: tuple[float, float, float, float]
    env_states: tuple[float, float, float, float]
    lfo_harmonics: tuple[float, float, float, float, float, float, float, float]
    clamped_controls: list[float]
    overflow_or_saturation_flags: int
    rho_sr: float
    rho_ctrl: float
    alpha_ramp_host: float
    r_ac_host: float
    r_dc_host: float
    tau_ref_samples_host: float
    switch_xfade_samples_host: int

class ValidatedContainerDict(TypedDict):
    magic: str
    version_major: int
    version_minor: int
    flags: int
    num_controls: int
    sample_rate_train_hz: int
    json_header_byte_len: int
    tensor_arena_offset: int
    tensor_arena_byte_len: int
    arena_lite_end_cl: int
    arena_full_end_cl: int
    arena_pro_end_cl: int
    arena_ultra_end_cl: int
    arena_offset_lite_end: int
    arena_offset_full_end: int
    arena_offset_pro_end: int
    arena_offset_ultra_end: int
    arena_crc32c_lite: int
    arena_crc32c_full: int
    arena_crc32c_pro: int
    arena_crc32c_ultra: int
    num_ckpt_stems: int
    tail_time_ms: float
    header_crc32c: int
    highest_tier: int
    is_stereo_out: bool
    has_signatures: bool
    is_encrypted_arena: bool
    has_calib_stats: bool
    verification_status: str
    protection_profile: str
    header_json: str
    arena_bytes: bytes | None

class ZohDict(TypedDict):
    sample_rate_host_hz: int
    rho_sr: float
    control_decimation_host: int
    rho_ctrl: float
    alpha_ramp_host: float
    r_ac_host: float
    r_dc_host: float
    tau_ref_samples_host: float
    switch_xfade_samples_host: int

class TempoInversionDict(TypedDict):
    physical_f32: float
    normalized_f32: float
    realized_value: float
    residual: float
    search_steps: tuple[int, int]

class ProvenanceVerifyDict(TypedDict):
    status: str
    signer_id: str | None
    pubkey_ed25519: str | None
    marketplace_id: str | None
    license_id: str | None
    license_tier: str | None

class Dau1DecodedDict(TypedDict):
    auth_kind: int
    creator_pubkey_hex: str
    creator_max_tier: int
    master_prefixes_hex: list[str]
    creator_signature_hex: str
    marketplace_pubkey_hex: str | None
    policy_bytes: bytes
    buyer_max_tier: int | None
    buyer_prefixes_hex: list[str]
    marketplace_signature_hex: str | None

class PyEngine:
    def __init__(
        self,
        container_bytes: bytes,
        tier: int = 1,
        sample_rate_hz: int = 48_000,
        backend: str = "auto",
        max_supported_tier: int | None = None,
        offline_bounce: bool = False,
        maximize_tier_on_bounce: bool = True,
    ) -> None: ...
    def reload(
        self,
        container_bytes: bytes,
        tier: int | None = None,
        sample_rate_hz: int | None = None,
    ) -> None: ...
    def backend_kind(self) -> str: ...
    def active_tier(self) -> int: ...
    def set_tier(self, tier: int) -> None: ...
    def max_available_tier(self) -> int: ...
    def max_supported_tier(self) -> int: ...
    def set_max_supported_tier(self, tier: int) -> None: ...
    def effective_tier(self) -> int: ...
    def offline_bounce(self) -> bool: ...
    def set_offline_bounce(self, enabled: bool) -> None: ...
    def maximize_tier_on_bounce(self) -> bool: ...
    def set_maximize_tier_on_bounce(self, enabled: bool) -> None: ...
    def latency_samples(self) -> int: ...
    def sample_rate_hz(self) -> int: ...
    def set_sample_rate(self, sample_rate_hz: int) -> None: ...
    def num_controls(self) -> int: ...
    def raw_control_values(self) -> list[float]: ...
    def clamped_control_values(self) -> list[float]: ...
    def clamped_control_value(self, ctrl_idx: int) -> float: ...
    def set_control_value(self, ctrl_idx: int, value: float) -> None: ...
    def set_control_immediate(self, ctrl_idx: int, value: float) -> None: ...
    def set_controls_immediate_batch(self, updates: list[tuple[int, float]]) -> None: ...
    def reset_controls(self) -> None: ...
    def reset(self) -> None: ...
    def supports_freeze(self) -> bool: ...
    def set_freeze(self, freeze: bool) -> None: ...
    def set_freeze_amount(self, alpha: float) -> None: ...
    def is_frozen(self) -> bool: ...
    def set_bypass(self, bypassed: bool, trails: bool) -> None: ...
    def bypass_state(self) -> tuple[bool, bool]: ...
    def set_stereo_width(self, width: float) -> None: ...
    def stereo_width(self) -> float: ...
    def configure_host_mix_and_leveling(
        self,
        blend_wet: float = 1.0,
        output_gain_db: float = 0.0,
        apply_unity_leveling: bool = True,
    ) -> None: ...
    def current_bpm(self) -> float: ...
    def sync_tempo(
        self,
        bpm: float,
        subdivision: str | None = None,
        fold_octave: bool = False,
    ) -> int: ...
    def register_tap_tempo(
        self,
        subdivision: str | None = None,
        fold_octave: bool = False,
    ) -> tuple[float, int]: ...
    def apply_preset(self, preset_id: str) -> None: ...
    def render_offline_bounce(
        self,
        left_in: NDArray[np.float32],
        right_in: NDArray[np.float32] | None,
        stereo_out: NDArray[np.float32],
        maximize_tier: bool | None = None,
    ) -> None: ...
    def process_stereo(
        self,
        left_in: NDArray[np.float32],
        right_in: NDArray[np.float32],
        left_out: NDArray[np.float32],
        right_out: NDArray[np.float32],
        buffer_size: int | None = None,
    ) -> None: ...
    def process_stereo_in_place(
        self,
        left_io: NDArray[np.float32],
        right_io: NDArray[np.float32],
        buffer_size: int | None = None,
    ) -> None: ...
    def process_stereo_2d(
        self,
        stereo_in: NDArray[np.float32],
        stereo_out: NDArray[np.float32],
        buffer_size: int | None = None,
    ) -> None: ...
    def process_stereo_2d_in_place(
        self,
        stereo_io: NDArray[np.float32],
        buffer_size: int | None = None,
    ) -> None: ...
    def process_mono_to_stereo(
        self,
        mono_in: NDArray[np.float32],
        left_out: NDArray[np.float32],
        right_out: NDArray[np.float32],
        buffer_size: int | None = None,
    ) -> None: ...
    def process_mono_to_stereo_2d(
        self,
        mono_in: NDArray[np.float32],
        stereo_out: NDArray[np.float32],
        buffer_size: int | None = None,
    ) -> None: ...
    def process_mono(
        self,
        mono_in: NDArray[np.float32],
        mono_out: NDArray[np.float32],
        buffer_size: int | None = None,
    ) -> None: ...
    def process_mono_in_place(
        self,
        mono_io: NDArray[np.float32],
        buffer_size: int | None = None,
    ) -> None: ...
    def telemetry(self) -> dict[str, object]: ...

class PyControlSpace:
    def __init__(self, header_json: str) -> None: ...
    def num_controls(self) -> int: ...
    def clamp(
        self, raw_numeric_values: list[float]
    ) -> tuple[list[float], list[float], list[bool], int]: ...
    def to_physical(self, idx: int, normalized_gesture: float) -> float: ...
    def to_normalized(self, idx: int, physical_value: float) -> float: ...

# Model & Checkpoint functions
def native_validate_container(
    container_bytes: bytes,
    device_id_hash: str | None = None,
    device_secret_32: bytes | None = None,
    trusted_marketplace_pubkey_hex: str | None = None,
) -> ValidatedContainerDict: ...
def native_build_container(
    header_json: str,
    arena_bytes: bytes,
    plaintext_arena_crc_source: bytes | None = None,
    encrypted_arena: bool = False,
) -> bytes: ...
def native_build_reference_container(
    supports_freeze: bool = True,
    is_checkpoint: bool = False,
) -> bytes: ...
def native_update_header_in_place(
    container_bytes: bytes,
    updated_header_json: str,
) -> bytes: ...
def native_update_header_in_place_fast(
    container_bytes: bytes,
    updated_header_json: str,
) -> tuple[bytes, int, int]: ...
def native_strip_container_to_tier(
    container_bytes: bytes,
    max_tier: int,
) -> bytes: ...
def native_validate_v2_open_layout(container_bytes: bytes) -> dict[str, object]: ...
def native_serialize_jcs(
    json_str: str,
    project_immutable: bool = False,
    reject_floats: bool = False,
) -> str: ...
def native_build_dmu1_frame(mutable_json_bytes: bytes, capacity: int) -> bytes: ...
def native_parse_dmu1_frame(frame_bytes: bytes) -> str: ...
def native_crc32c(data: bytes) -> int: ...
def native_validate_and_normalize_header(header_json: str) -> str: ...
def native_parse_schema_version(version_str: str) -> tuple[int, int, int, bool]: ...
def native_normalize_allpass_delays(delays: list[int]) -> list[int]: ...
def native_build_reference_checkpoint(include_calib_stats: bool = True) -> bytes: ...
def native_verify_stem_arena4_crc32c(
    arena_bytes: bytes,
    stem_byte_offset: int,
    stem_byte_len: int,
    expected_crc32c: int,
) -> None: ...
def native_verify_arena5_calib_stats(
    arena_bytes: bytes,
    arena5_offset: int,
) -> int: ...
def native_validate_v2_dnck_checkpoint(ckpt_bytes: bytes) -> dict[str, object]: ...
def native_export_runtime_dnam_from_ckpt(ckpt_bytes: bytes) -> bytes: ...
def native_uplo_u_packed_len(n_dim: int) -> int: ...
def native_pack_uplo_u_f64(
    dense_flat: NDArray[np.float64],
    n_dim: int,
) -> NDArray[np.float64]: ...
def native_pack_uplo_u_f64_into(
    dense_flat: NDArray[np.float64],
    n_dim: int,
    out_packed: NDArray[np.float64],
) -> None: ...
def native_unpack_uplo_u_f64(
    packed_u: NDArray[np.float64],
    n_tier: int,
) -> NDArray[np.float64]: ...
def native_unpack_uplo_u_f64_into(
    packed_u: NDArray[np.float64],
    n_tier: int,
    out_dense: NDArray[np.float64],
) -> None: ...

# Controls & Tempo functions
def native_ui_skew_exponent(midpoint: float | None, x_min: float, x_max: float) -> float: ...
def native_ui_normalized_to_physical(
    midpoint: float | None,
    t: float,
    x_min: float,
    x_max: float,
) -> float: ...
def native_ui_physical_to_normalized(
    midpoint: float | None,
    x_val: float,
    x_min: float,
    x_max: float,
) -> float: ...
def native_invert_lobatto_anchors(
    anchors: tuple[float, float, float] | list[float],
    target_val: float,
    fold_octave: bool = True,
) -> tuple[float, bool]: ...
def native_evaluate_lobatto_anchors(
    anchors: tuple[float, float, float] | list[float],
    u: float,
) -> float: ...
def native_resolve_tempo_sync(
    calibration_json: str,
    bpm: float,
    subdivision: str | None = None,
    fold_octave: bool = True,
) -> tuple[tuple[str, str] | None, float, bool] | None: ...
def native_subdivision_beat_multiplier(subdivision: str) -> float: ...
def native_subdivision_delay_ms_at_bpm(subdivision: str, bpm: float) -> float: ...
def native_subdivision_lfo_hz_at_bpm(subdivision: str, bpm: float) -> float: ...
def native_compute_zoh(
    sample_rate_train_hz: int,
    sample_rate_host_hz: int,
    r_ac_train: float = 0.9993,
    r_dc_train: float = 0.995,
    tau_ref_train: float = 64.0,
    switch_xfade_train: int = 128,
) -> ZohDict: ...
def native_control_stride_for_rate(sample_rate_hz: int) -> int: ...
def native_control_tick_ratio(sample_rate_hz: int) -> float: ...
def native_eval_subdivision_phase(
    beat_mod12: int,
    within_beat_q32: int,
    subdivision: str,
    offset_q32: int = 0,
) -> tuple[int, float, float]: ...
def native_eval_3node_calibration(
    anchors: tuple[float, float, float],
    u: float,
) -> float: ...
def native_invert_tempo_calibration(
    declared_range: tuple[float, float],
    safe_range: tuple[float, float],
    anchors: tuple[float, float, float],
    target: float,
    flush_subnormals: bool = True,
) -> TempoInversionDict: ...

# Provenance functions
def native_verify_container_provenance(
    header_json: str,
    plaintext_arena: bytes,
    has_signatures_flag: bool = False,
    device_id_hash: str | None = None,
    device_secret_32: bytes | None = None,
    trusted_marketplace_pubkey_hex: str | None = None,
) -> ProvenanceVerifyDict: ...
def native_enforce_distribution_policy(
    header_json: str,
    active_profile: str,
) -> None: ...
def native_evaluate_anticlone_policy(
    fingerprint_matched: bool,
    buyer_watermark_identified: bool,
    independent_device_source_evidence: bool,
) -> str: ...
def native_is_profile_allowed(allowed_profiles: list[str], profile: str) -> bool: ...
def native_is_oem_device_allowed(
    allowed_oem_vendors: list[str],
    oem_vendor: str,
    model_code: str,
) -> bool: ...
def native_immutable_header_sha256(header_json: str) -> str: ...
def native_sha256_hex(data: bytes) -> str: ...
def native_hmac_sha256_hex(key_32: bytes, message: bytes) -> str: ...
def native_ed25519_pubkey_hex(seed_32: bytes) -> str: ...
def native_ed25519_sign_hex(seed_32: bytes, message: bytes) -> str: ...
def native_x25519_pubkey_hex(secret_32: bytes) -> str: ...
def native_build_creator_signing_message(
    immutable_header_sha256_hex: str,
    master_arena_sha256_hex: str,
) -> bytes: ...
def native_build_marketplace_signing_message(
    immutable_header_sha256_hex: str,
    master_arena_sha256_hex: str,
    receipt_json: str,
) -> bytes: ...
def native_sign_creator_header(
    header_json: str,
    master_arena: bytes,
    creator_seed_32: bytes,
    signer_id: str,
    include_tier_prefixes: bool = True,
) -> str: ...
def native_sign_marketplace_receipt(
    header_json: str,
    buyer_arena: bytes,
    marketplace_seed_32: bytes,
    spec_json: str,
) -> str: ...
def native_sign_oem_device_cert(
    oem_seed_32: bytes,
    oem_vendor: str,
    model_code: str,
    device_id_hash: str,
    pubkey_x25519_32: bytes,
) -> str: ...
def native_verify_oem_device_cert(
    cert_json: str,
    trusted_oem_pubkey_hex: str,
) -> str: ...
def native_build_oem_cert_v2_signing_message(
    oem_vendor: str,
    model_code: str,
    device_id_hash_16: bytes,
    pubkey_x25519_32: bytes,
    issuer_pubkey_ed25519_32: bytes,
    assurance: int = 1,
    attestation_policy_sha256_32: bytes | None = None,
    attestation_evidence_sha256_32: bytes | None = None,
) -> bytes: ...
def native_chacha20_xor_arena(
    arena_bytes: bytes,
    key_32: bytes,
    nonce_12: bytes,
    start_block_counter: int = 0,
) -> bytes: ...
def native_wrap_device_cek_hex(
    ephemeral_secret_32: bytes,
    device_pubkey_32: bytes,
    device_id_hash: str,
    cek_32: bytes,
) -> str: ...
def native_seal_container(
    inner_dnam_bytes: bytes,
    recipient_certs_json: str,
    trusted_oem_pubkeys_json: str,
    marketplace_seed_32: bytes,
    ephemeral_secret_32: bytes,
    cek_32: bytes,
    nonce_12: bytes,
) -> bytes: ...
def native_unseal_container(
    sealed_bytes: bytes,
    device_id_hash: str | None = None,
    device_secret_32: bytes | None = None,
    trusted_marketplace_pubkey_hex: str | None = None,
) -> bytes: ...
def native_encode_dau1_envelope(
    auth_kind: int,
    creator_pubkey_hex: str,
    creator_max_tier: int,
    master_prefixes_hex: list[str],
    creator_signature_hex: str,
    marketplace_pubkey_hex: str | None = None,
    policy_bytes: bytes | None = None,
    buyer_max_tier: int | None = None,
    buyer_prefixes_hex: list[str] = ...,
    marketplace_signature_hex: str | None = None,
) -> bytes: ...
def native_decode_dau1_envelope(wire_bytes: bytes) -> Dau1DecodedDict: ...
def native_build_dme1_header(
    profile: int,
    recipient_count: int,
    ciphertext_length: int,
) -> bytes: ...
def native_parse_dme1_plaintext(
    plaintext: bytes,
    recipient_key_digests_hex: list[str],
) -> str: ...
def native_encode_recipient_key_ids(key_digests_hex: list[str]) -> bytes: ...
def native_parse_recipient_key_ids(
    data: bytes,
    max_recipients: int = 16,
) -> list[str]: ...
def native_encode_hpke_slots(slots_hex: list[tuple[str, str]]) -> bytes: ...
def native_parse_hpke_slots(
    data: bytes,
    expected_count: int,
    max_recipients: int = 16,
) -> list[tuple[str, str]]: ...
def native_encode_record_directory(
    maximum_tier: int,
    records: list[tuple[int, int]],
) -> bytes: ...
def native_parse_record_directory(
    data: bytes,
    maximum_tier: int,
) -> list[tuple[int, int]]: ...
def native_compute_preview_window(
    anchor: float,
    delta: float,
    safe_bounds: tuple[float, float],
) -> tuple[float, float]: ...
def native_validate_preview_switch_options(
    preset: str,
    alternative: str,
    observed: list[str],
    active: list[str],
    safe: list[str],
) -> tuple[str, str]: ...
def native_validate_royalty_splits(
    license_tier: str,
    royalty_splits_json: str,
) -> None: ...
def native_apply_isometric_gauge_watermark(
    arena_bytes: bytes,
    kappa_64: NDArray[np.float32],
    q_buyer_flat_16: NDArray[np.float32],
) -> bytes: ...
def native_generate_tardos_code(
    marketplace_secret_32: bytes,
    model_id: str,
    license_id: str,
) -> NDArray[np.float32]: ...
def native_apply_tardos_watermark(
    arena_bytes: bytes,
    code_7680: NDArray[np.float32],
    epsilon: float,
) -> bytes: ...
def native_score_tardos_watermark(
    master_arena: bytes,
    suspect_arena: bytes,
    code_7680: NDArray[np.float32],
) -> float: ...
def native_extract_structural_hash(arena_bytes: bytes) -> NDArray[np.float64]: ...
def native_structural_cosine_similarity(
    a: NDArray[np.float64],
    b: NDArray[np.float64],
) -> float: ...
def native_derive_watermark_seed_hex(
    key_32: bytes,
    marketplace_pubkey_32: bytes,
    watermark_key_id_16: bytes,
    creator_signed_manifest_sha256_32: bytes,
    model_id_16: bytes,
    watermark_profile_sha256_32: bytes,
    owning_tier: int,
    license_id_16: bytes | None = None,
    buyer_id_16: bytes | None = None,
) -> str: ...
def native_extract_stage3_fingerprint(
    early_fir_128b: bytes,
    kautz_modal_bytes: bytes,
    added_modes: int,
) -> NDArray[np.float64]: ...

# Reference DSP functions
def native_catmull_rom_hermite(
    s_m1: float,
    s_0: float,
    s_1: float,
    s_2: float,
    mu: float,
) -> float: ...
def native_catmull_rom_hermite_slice(
    s_m1: NDArray[np.float64],
    s_0: NDArray[np.float64],
    s_1: NDArray[np.float64],
    s_2: NDArray[np.float64],
    mu: NDArray[np.float64],
    out: NDArray[np.float64],
) -> None: ...
def native_op_t_coercive_step(
    u_in: float,
    g_lin: float,
    sigma_knee: float,
    v_ceil: float,
    r_dc: float,
    dc_x_prev: float,
    dc_y_prev: float,
) -> tuple[float, float, float]: ...
def native_op_t_tank_slice(
    input: NDArray[np.float64],
    output: NDArray[np.float64],
    g_lin: float,
    sigma_knee: float,
    v_ceil: float,
    r_dc: float,
) -> None: ...
def native_op_m_modal_step(
    x: float,
    r_low: float,
    dr_span: float,
    w_sel: float,
    b_gamma_norm: float,
    c_real: float,
    c_imag: float,
    u_rot_cos: float,
    u_rot_sin: float,
    z_re: float,
    z_im: float,
) -> tuple[float, float, float]: ...
def native_op_m_modal_slice(
    input: NDArray[np.float64],
    output: NDArray[np.float64],
    r_low: float,
    dr_span: float,
    w_sel: float,
    b_gamma_norm: float,
    c_real: float,
    c_imag: float,
    u_rot_cos: float,
    u_rot_sin: float,
    z_re: float,
    z_im: float,
    reset: bool,
) -> tuple[float, float]: ...
def native_unit_rotor(phi: float) -> tuple[float, float]: ...
def native_unit_rotor_slice(
    phi: NDArray[np.float64],
    out: NDArray[np.float64],
) -> None: ...
def native_poly_adaa_algebraic(
    u_prev: float,
    u_curr: float,
    beta: float,
    delta: float,
) -> float: ...
def native_poly_adaa_step(
    x: float,
    b_conv: float,
    alpha: float,
    beta: float,
    delta: float,
    r_ac: float,
    u_prev: float,
    a_prev: float,
    m_dc: float,
) -> tuple[float, float, float, float]: ...
def native_poly_adaa_slice(
    input: NDArray[np.float64],
    output: NDArray[np.float64],
    b_conv: float,
    alpha: float,
    beta: float,
    delta: float,
    r_ac: float,
    u_prev: float,
    a_prev: float,
    m_dc: float,
    reset: bool,
) -> tuple[float, float, float]: ...
def native_eval_lobatto_basis(
    u: float,
    anchors: tuple[float, float, float],
    degree: int,
) -> tuple[float, float, float]: ...
def native_eval_lobatto_basis_slice(
    u: NDArray[np.float64],
    out: NDArray[np.float64],
    anchors: tuple[float, float, float],
    degree: int,
) -> None: ...
def native_eval_rank4_fm(
    features: NDArray[np.float64],
    v_factors_flat: NDArray[np.float64],
) -> tuple[float, float, float, float]: ...
def native_eval_chebyshev_lfo(cos_phi: float) -> tuple[float, float, float, float]: ...
def native_eval_chebyshev_lfo_slice(
    cos_phi: NDArray[np.float64],
    out: NDArray[np.float64],
) -> None: ...
def native_eval_trust_region_softclip(z: float, mh: float) -> float: ...
def native_eval_trust_region_softclip_slice(
    z: NDArray[np.float64],
    out: NDArray[np.float64],
    mh: float,
) -> None: ...
def native_cubic_lagrange_weights(mu: float) -> tuple[float, float, float, float]: ...
def native_cubic_lagrange_weights_slice(
    mu: NDArray[np.float64],
    out: NDArray[np.float64],
) -> None: ...
def native_givens_householder_8ch(
    x: tuple[float, float, float, float, float, float, float, float],
    cos_psi: float,
    sin_psi: float,
) -> tuple[float, float, float, float, float, float, float, float]: ...
def native_givens_householder_16ch(
    x: tuple[
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
    ],
    cos_psi: float,
    sin_psi: float,
    cross_cos: tuple[float, float, float, float, float, float, float, float],
    cross_sin: tuple[float, float, float, float, float, float, float, float],
) -> tuple[
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
]: ...

class _TestingModule:
    @staticmethod
    def render_reference_stereo(
        container_bytes: bytes,
        tier: int,
        backend: str,
        sample_rate_hz: int,
        left_in: NDArray[np.float32],
        right_in: NDArray[np.float32],
        block_size: int | None = None,
        reset: bool = True,
    ) -> NDArray[np.float32]: ...
    @staticmethod
    def render_reference_mono_to_stereo(
        container_bytes: bytes,
        tier: int,
        backend: str,
        sample_rate_hz: int,
        mono_in: NDArray[np.float32],
        block_size: int | None = None,
        reset: bool = True,
    ) -> NDArray[np.float32]: ...
    @staticmethod
    def render_reference_mono(
        container_bytes: bytes,
        tier: int,
        backend: str,
        sample_rate_hz: int,
        mono_in: NDArray[np.float32],
        block_size: int | None = None,
        reset: bool = True,
    ) -> NDArray[np.float32]: ...
    @staticmethod
    def reference_header_json(kind: str = "default") -> str: ...
    @staticmethod
    def error_variant_names() -> list[str]: ...
    @staticmethod
    def raise_error_variant(name: str) -> None: ...

_testing: _TestingModule
