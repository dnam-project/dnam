"""PEP 561 strict type stubs for the Rust PyO3 extension module `dnam_engine._native`."""

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

class NativeControlSpace:
    def __init__(self, header_json: str) -> None: ...
    def clamp(
        self,
        raw_numeric_values: Sequence[float],
    ) -> tuple[list[float], list[float], list[bool]]: ...

class NativeDnamEngine:
    def __init__(
        self,
        container_bytes: bytes,
        tier: int = 1,
        sample_rate_hz: int = 48_000,
        backend: str = "auto",
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
    def sample_rate_hz(self) -> int: ...
    def set_sample_rate(self, sample_rate_hz: int) -> None: ...
    def set_control_value(self, ctrl_idx: int, value: float) -> None: ...
    def set_control_immediate(self, ctrl_idx: int, value: float) -> None: ...
    def reset(self) -> None: ...
    def process_stereo(
        self,
        left_in: NDArray[np.float32],
        right_in: NDArray[np.float32],
        left_out: NDArray[np.float32],
        right_out: NDArray[np.float32],
        buffer_size: int | None = None,
    ) -> None: ...
    def process_stereo_2d(
        self,
        stereo_in: NDArray[np.float32],
        stereo_out: NDArray[np.float32],
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
    def telemetry(self) -> dict[str, Any]: ...

def native_validate_container(
    container_bytes: bytes,
    device_id_hash: str | None = None,
    device_secret_32: bytes | None = None,
    trusted_marketplace_pubkey_hex: str | None = None,
) -> dict[str, Any]: ...
def native_build_container(
    header_json: str,
    arena_bytes: bytes,
    plaintext_arena_crc_source: bytes | None = None,
    encrypted_arena: bool = False,
) -> bytes: ...
def native_build_reference_container(
    core2_active: bool = False,
    is_checkpoint: bool = False,
) -> bytes: ...
def native_update_header_in_place(
    container_bytes: bytes,
    updated_header_json: str,
) -> bytes: ...
def native_clamp_controls(
    header_json: str,
    raw_numeric_values: Sequence[float],
) -> list[dict[str, Any]]: ...
def native_compute_zoh(
    sample_rate_train_hz: int,
    sample_rate_host_hz: int,
    control_decimation_train: int = 16,
    r_ac_train: float = 0.9993,
    r_dc_train: float = 0.995,
    tau_ref_train: float = 64.0,
    switch_xfade_train: int = 128,
) -> dict[str, Any]: ...
def native_pack_uplo_u_f64(
    dense_flat: NDArray[np.float64],
    n_dim: int,
) -> NDArray[np.float64]: ...
def native_unpack_uplo_u_f64(
    packed_u: NDArray[np.float64],
    n_tier: int,
) -> NDArray[np.float64]: ...
def native_crc32c(data: bytes) -> int: ...
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
def native_poly_adaa_scan(
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
def native_selective_rglru_step(
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
def native_selective_rglru_scan(
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
def native_cubic_lagrange_weights(mu: float) -> list[float]: ...
def native_givens_householder_mix_8ch(
    x: Sequence[float],
    cos_psi: float,
    sin_psi: float,
) -> list[float]: ...
def native_mufdn_coercive_step(
    u_in: float,
    g_lin: float,
    sigma_knee: float,
    v_ceil: float,
    r_dc: float,
    dc_x_prev: float,
    dc_y_prev: float,
) -> tuple[float, float, float]: ...
def native_mufdn_slice(
    input: NDArray[np.float64],
    output: NDArray[np.float64],
    delay_samples: int,
    g_lin: float,
    sigma_knee: float,
    v_ceil: float,
    r_dc: float,
) -> None: ...
def native_eval_lobatto_basis(
    u: float,
    anchors: Sequence[float],
    degree: int,
) -> list[float]: ...
def native_eval_lobatto_basis_slice(
    u: NDArray[np.float64],
    out: NDArray[np.float64],
    anchors: Sequence[float],
    degree: int,
) -> None: ...
def native_eval_rank4_fm(
    features: NDArray[np.float64],
    v_factors_flat: NDArray[np.float64],
) -> list[float]: ...
def native_eval_chebyshev_lfo(cos_phi: float) -> list[float]: ...
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
def native_immutable_header_sha256(header_json: str) -> str: ...
def native_sha256_hex(data: bytes) -> str: ...
def native_build_creator_signing_message(
    immutable_header_sha256_hex: str,
    master_arena_sha256_hex: str,
) -> bytes: ...
def native_build_marketplace_signing_message(
    immutable_header_sha256_hex: str,
    master_arena_sha256_hex: str,
    receipt_json: str,
) -> bytes: ...
def native_chacha20_xor_arena(
    arena_bytes: bytes,
    key_32: bytes,
    nonce_12: bytes,
    start_block_counter: int = 0,
) -> bytes: ...
def native_apply_isometric_gauge_watermark(
    arena_bytes: bytes,
    kappa_48: NDArray[np.float32],
    q_buyer_flat_16: NDArray[np.float32],
) -> bytes: ...
def native_generate_tardos_code(
    marketplace_secret_32: bytes,
    model_id: str,
    license_id: str,
) -> NDArray[np.float32]: ...
def native_apply_tardos_watermark(
    arena_bytes: bytes,
    code_4608: NDArray[np.float32],
    epsilon: float,
) -> bytes: ...
def native_score_tardos_watermark(
    master_arena: bytes,
    suspect_arena: bytes,
    code_4608: NDArray[np.float32],
) -> float: ...
def native_extract_structural_hash(arena_bytes: bytes) -> NDArray[np.float64]: ...
def native_structural_cosine_similarity(
    a: NDArray[np.float64],
    b: NDArray[np.float64],
) -> float: ...
def native_ed25519_pubkey_hex(seed_32: bytes) -> str: ...
def native_ed25519_sign_hex(seed_32: bytes, message: bytes) -> str: ...
def native_x25519_pubkey_hex(secret_32: bytes) -> str: ...
def native_wrap_device_cek_hex(
    ephemeral_secret_32: bytes,
    device_pubkey_32: bytes,
    device_id_hash: str,
    cek_32: bytes,
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
