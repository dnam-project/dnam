"""`CL-18` Dual-Ridge `float64` Kronecker Cholesky Head Solver, Orthonormal SVD Control Projector,
and `.dnam-ckpt` `Arena 3` `uplo='U'` Sufficient-Statistics Accumulator (`Test 7B`, `Test 8G`).

Implements the mathematical specifications from:
- [`docs/clamp_and_audit_registry.md`](../../../../../docs/clamp_and_audit_registry.md) (`CL-18`)
- [`docs/training_and_iterative_resume.md`](../../../../../docs/training_and_iterative_resume.md)
§1.2, §1.4, §2
- [`docs/dnam_container_schema.md`](../../../../../docs/dnam_container_schema.md) §6.2
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from dnam_engine.container import GramMatrix, crc32c
from dnam_engine.schema import ExtractedPhysics, StemProvenance, StemRegistryEntry, TensorSliceEntry
from dnam_trainer.training.metrics import split_three_bands_np
from dnam_trainer.training.numerics import (
    ExactGramAccumulator,
    HostDenseSolver,
    recombine_f32_trilimb_to_f64,
    split_f64_to_f32_trilimb,
)
from dnam_trainer.training.selective_cascade import (
    compute_tier32_modulator_vectors,
    synthesize_4th_tier_base_block,
)

ARENA012_BYTE_LEN: int = 64_000
ARENA3_G_AUDIO_FLOATS: int = 32_896  # 256 * 257 // 2
ARENA3_G_AUDIO_BYTES: int = 263_168  # 32,896 * 8 B
ARENA3_C_AUDIO_FLOATS: int = 512  # 256 * 2
ARENA3_C_AUDIO_BYTES: int = 4_096  # 512 * 8 B
ARENA3_STEM_BYTE_LEN: int = 267_264  # 263,168 + 4,096 B (4,176 cache lines of 64 B)
ARENA4_STEM_BYTE_LEN: int = ARENA3_STEM_BYTE_LEN
ARENA3_STEM_BYTE_LEN_V10_192: int = 151_296  # legacy 192-D stem byte length

CL18_SVD_MIN_SIGMA: float = 1e-4
CL18_MAX_HEAD_SVD_RANK: int = 4
CL18_RIDGE_INIT_SCALE: float = 1e-3
CL18_RIDGE_POLISH_SCALE: float = 1e-6


@dataclass(frozen=True, slots=True)
class StemSufficientStatistics:
    """Un-entangled per-stem `float64` audio sufficient statistics stored in `.dnam-ckpt` `Arena
    4` (`267,264 B/stem` for $N=256$)."""

    stem_id: str
    controls_meta: dict[str, float | str]
    input_pcm_sha256: str
    target_pcm_sha256: str
    num_frames: int
    g_audio_uplo_u_f64: NDArray[np.float64]  # shape (32896,) for N=256 (or (18528,) for N=192)
    c_audio_tier_f64: NDArray[np.float64]  # shape (256, 2) (or (192, 2))
    extracted_physics: ExtractedPhysics | None = None

    @property
    def active_feature_dim(self) -> int:
        """Returns `256` if the 4th tier group (`192..256`) has non-zero energy, else `192`."""
        if (
            self.c_audio_tier_f64.shape[0] >= 256
            and float(np.max(np.abs(self.c_audio_tier_f64[192:256]))) > 1e-18
        ):
            return 256
        if self.g_audio_uplo_u_f64.size >= ARENA3_G_AUDIO_FLOATS:
            # Check diagonal entries of features 192..256 in uplo='U' packed array
            col192_diag_idx = (192 * 193) // 2 + 192
            if float(np.abs(self.g_audio_uplo_u_f64[col192_diag_idx])) > 1e-18:
                return 256
        return 192

    def _normalized_256_arrays(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Returns `(g_uplo_32896, c_tier_256x2)` padded to $N=256$ if stored as $N=192$."""
        if (
            self.g_audio_uplo_u_f64.size == ARENA3_G_AUDIO_FLOATS
            and self.c_audio_tier_f64.shape == (256, 2)
        ):
            return self.g_audio_uplo_u_f64, self.c_audio_tier_f64
        g_pad = np.zeros(ARENA3_G_AUDIO_FLOATS, dtype=np.float64)
        g_pad[: min(ARENA3_G_AUDIO_FLOATS, self.g_audio_uplo_u_f64.size)] = (
            self.g_audio_uplo_u_f64.ravel()[:ARENA3_G_AUDIO_FLOATS]
        )
        c_pad = np.zeros((256, 2), dtype=np.float64)
        n_rows = min(256, int(self.c_audio_tier_f64.shape[0]))
        c_pad[:n_rows, :] = self.c_audio_tier_f64[:n_rows, :2]
        return g_pad, c_pad

    def to_arena3_bytes(self) -> bytes:
        """Serializes `(g_audio_uplo_u_f64, c_audio_tier_f64)` into a `267,264 B` little-endian
        slice."""
        g_256, c_256 = self._normalized_256_arrays()
        g_bytes = np.ascontiguousarray(g_256, dtype="<f8").tobytes()
        c_bytes = np.ascontiguousarray(c_256, dtype="<f8").tobytes()
        if len(g_bytes) != ARENA3_G_AUDIO_BYTES or len(c_bytes) != ARENA3_C_AUDIO_BYTES:
            raise ValueError(
                f"Invalid Arena 4 stem byte lengths: g={len(g_bytes)}, c={len(c_bytes)}"
            )
        return g_bytes + c_bytes

    def to_registry_entry(self, stem_index: int) -> tuple[StemRegistryEntry, bytes]:
        """
        Builds the [`StemRegistryEntry`] and `267,264 B` payload for `stem_index` in `Arena 4`.
        """
        payload = self.to_arena3_bytes()
        stem_crc = crc32c(payload)
        base_offset = ARENA012_BYTE_LEN + stem_index * ARENA3_STEM_BYTE_LEN
        g_slice = TensorSliceEntry(
            arena=4,
            dtype="f64",
            packing="uplo_upper_col_major_tier_ordered",
            feature_dim=256,
            packed_len=ARENA3_G_AUDIO_FLOATS,
            shape=None,
            byte_offset=base_offset,
            byte_length=ARENA3_G_AUDIO_BYTES,
        )
        c_slice = TensorSliceEntry(
            arena=4,
            dtype="f64",
            packing="row_major_tier_ordered",
            feature_dim=256,
            packed_len=None,
            shape=[256, 2],
            byte_offset=base_offset + ARENA3_G_AUDIO_BYTES,
            byte_length=ARENA3_C_AUDIO_BYTES,
        )
        entry = StemRegistryEntry(
            stem_id=self.stem_id,
            controls_meta=dict(self.controls_meta),
            provenance=StemProvenance(
                input_pcm_sha256=self.input_pcm_sha256,
                target_pcm_sha256=self.target_pcm_sha256,
                num_frames=self.num_frames,
            ),
            stem_arena3_crc32c=stem_crc,
            user_metadata={},
            extracted_physics=self.extracted_physics,
            g_audio_uplo_u_f64=g_slice,
            c_audio_tier_f64=c_slice,
        )
        return entry, payload

    @classmethod
    def from_checkpoint_entry(
        cls,
        entry: StemRegistryEntry,
        arena_bytes: bytes | memoryview,
    ) -> StemSufficientStatistics:
        """Deserializes and CRC32C-verifies a stem's `Arena 4` (or legacy `Arena 3`) statistics
        from a `.dnam-ckpt` buffer."""
        raw = memoryview(arena_bytes)
        g_off = entry.g_audio_uplo_u_f64.byte_offset
        g_len = entry.g_audio_uplo_u_f64.byte_length
        c_off = entry.c_audio_tier_f64.byte_offset
        c_len = entry.c_audio_tier_f64.byte_length

        stem_slice = bytes(raw[g_off : c_off + c_len])
        computed_crc = crc32c(stem_slice)
        if computed_crc != entry.stem_arena3_crc32c:
            raise ValueError(
                f"Arena 3 CRC32C mismatch for stem '{entry.stem_id}': "
                f"expected {entry.stem_arena3_crc32c:#010x}, got {computed_crc:#010x}"
            )

        g_arr_raw = np.frombuffer(raw[g_off : g_off + g_len], dtype="<f8").copy()
        c_rows = c_len // 16
        c_arr_raw = (
            np.frombuffer(raw[c_off : c_off + c_len], dtype="<f8").reshape((c_rows, 2)).copy()
        )
        if g_arr_raw.size < ARENA3_G_AUDIO_FLOATS:
            g_arr = np.zeros(ARENA3_G_AUDIO_FLOATS, dtype=np.float64)
            g_arr[: g_arr_raw.size] = g_arr_raw
        else:
            g_arr = g_arr_raw
        if c_rows < 256:
            c_arr = np.zeros((256, 2), dtype=np.float64)
            c_arr[:c_rows, :] = c_arr_raw
        else:
            c_arr = c_arr_raw
        controls_clean: dict[str, float | str] = {
            k: (float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else str(v))
            for k, v in entry.controls_meta.items()
        }
        return cls(
            stem_id=entry.stem_id,
            controls_meta=controls_clean,
            input_pcm_sha256=entry.provenance.input_pcm_sha256,
            target_pcm_sha256=entry.provenance.target_pcm_sha256,
            num_frames=entry.provenance.num_frames,
            g_audio_uplo_u_f64=g_arr,
            c_audio_tier_f64=c_arr,
            extracted_physics=entry.extracted_physics,
        )

    def to_trilimb_f32_tensors(
        self,
        device: torch.device | str = "cpu",
    ) -> tuple[tuple[Tensor, Tensor, Tensor], tuple[Tensor, Tensor, Tensor]]:
        """Splits `g_audio_uplo_u_f64` and `c_audio_tier_f64` into lossless 3-limb `float32`
        expansions `(hi, mid, lo)` on `device` without allocating 64-bit device tensors.
        """
        g_t = split_f64_to_f32_trilimb(self.g_audio_uplo_u_f64, device=device)
        c_t = split_f64_to_f32_trilimb(self.c_audio_tier_f64, device=device)
        return g_t, c_t

    @classmethod
    def from_trilimb_f32_tensors(
        cls,
        *,
        stem_id: str,
        controls_meta: dict[str, float | str],
        input_pcm_sha256: str,
        target_pcm_sha256: str,
        num_frames: int,
        g_trilimb: tuple[Tensor, Tensor, Tensor],
        c_trilimb: tuple[Tensor, Tensor, Tensor],
        extracted_physics: ExtractedPhysics | None = None,
    ) -> StemSufficientStatistics:
        """Recombines 3-limb `float32` tensors `(hi, mid, lo)` on the host into `float64`
        `StemSufficientStatistics`.
        """
        g_f64 = recombine_f32_trilimb_to_f64(*g_trilimb)
        c_f64 = recombine_f32_trilimb_to_f64(*c_trilimb)
        return cls(
            stem_id=stem_id,
            controls_meta=dict(controls_meta),
            input_pcm_sha256=input_pcm_sha256,
            target_pcm_sha256=target_pcm_sha256,
            num_frames=num_frames,
            g_audio_uplo_u_f64=g_f64,
            c_audio_tier_f64=c_f64,
            extracted_physics=extracted_physics,
        )


def _compute_band_weights_f64(
    y_lf: NDArray[np.float64],
    y_mf: NDArray[np.float64],
    y_hf: NDArray[np.float64],
    *,
    num_frames: int,
    floor_dbfs: float = -60.0,
    band_floors_dbfs: tuple[float, float, float] | None = None,
    is_silent: bool = False,
) -> tuple[float, float, float]:
    """Computes target-adaptive 3-band weights `(w_lf, w_mf, w_hf)` using per-band measured noise
    floors when available, preserving full tonal weight on quiet active stems while regularizing
    genuinely silent stems.
    """
    n_elems = float(max(1, y_lf.size))
    e_lf = float(np.sum(y_lf * y_lf))
    e_mf = float(np.sum(y_mf * y_mf))
    e_hf = float(np.sum(y_hf * y_hf))
    e_bb = e_lf + e_mf + e_hf

    if band_floors_dbfs is not None:
        fl_lf = n_elems * (10.0 ** (float(band_floors_dbfs[0]) / 10.0))
        fl_mf = n_elems * (10.0 ** (float(band_floors_dbfs[1]) / 10.0))
        fl_hf = n_elems * (10.0 ** (float(band_floors_dbfs[2]) / 10.0))
        if is_silent:
            mute_reg = n_elems * 1.0e-3
            fl_lf = max(fl_lf, mute_reg)
            fl_mf = max(fl_mf, mute_reg)
            fl_hf = max(fl_hf, mute_reg)
    else:
        fl_base = n_elems * (10.0 ** (float(floor_dbfs) / 10.0))
        if is_silent or e_bb <= fl_base:
            fl_base = max(fl_base, n_elems * 1.0e-3)
        fl_lf = fl_base
        fl_mf = fl_base
        fl_hf = fl_base

    scale = float(num_frames) / 3.0
    return (
        scale / (e_lf + fl_lf),
        scale / (e_mf + fl_mf),
        scale / (e_hf + fl_hf),
    )


def compute_svd_control_projector(
    psi_stems: NDArray[np.float64],
    *,
    svd_tol: float = CL18_SVD_MIN_SIGMA,
    max_rank: int = CL18_MAX_HEAD_SVD_RANK,
    trust_region_radius: float = 2.5,
) -> tuple[NDArray[np.float64], NDArray[np.float32], int, NDArray[np.float64]]:
    """Computes the `CL-18` Orthonormal SVD Principal Control Subspace Projector
    $\\mathbf{P}_{\\text{svd}}$.

    Args:
        psi_stems: `(num_stems, D_basis)` matrix of raw control basis vectors
        $\\boldsymbol{\\psi}_{\\text{raw}}(\\mathbf{u}_s, \\mathbf{s}_s)$
            (excluding or stripping the constant intercept column 0 if present).
        svd_tol: Minimum singular value threshold $\\sigma_k > 10^{-4}$ (`CL-18`).
        max_rank: Maximum head modulation rank $D_h \\le 4$.
        trust_region_radius: Smooth $\\tanh$ trust-region bound $R_{\\text{trust}} = 2.5$ (`CL-17`).

    Returns:
        `(p_svd_dbasis_x_4, p_svd_16x4_f32, active_rank, z_head_stems)` where:
        - `p_svd_dbasis_x_4` has shape `(D_basis, 4)` in `float64`,
        - `p_svd_16x4_f32` has shape `(16, 4)` in `float32` for `Arena 0` offset `8960..9216`,
        - `active_rank` is $\\max(1, |\\{k : \\sigma_k > 10^{-4}\\}|) \\le 4$,
        - `z_head_stems` has shape `(num_stems, 4)` in `float64`.
    """
    arr = np.asarray(psi_stems, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"psi_stems must be 2D, got shape {arr.shape}")
    num_stems, d_basis = arr.shape

    p_svd = np.zeros((d_basis, 4), dtype=np.float64)
    p_svd_16x4 = np.zeros((16, 4), dtype=np.float32)

    if d_basis == 0 or num_stems == 0:
        return p_svd, p_svd_16x4, 1, np.zeros((num_stems, 4), dtype=np.float64)

    max_cols = min(max_rank, 4)
    primary_cols: list[int] = []
    secondary_cols: list[int] = []
    b = 0
    while b < d_basis:
        if (
            b + 2 < d_basis
            and float(np.min(arr[:, b])) < -1e-4
            and float(np.max(arr[:, b])) > 1e-4
            and float(np.min(arr[:, b + 1])) >= -1e-6
            and float(np.max(np.abs(arr[:, b + 2]))) <= float(np.max(np.abs(arr[:, b + 1]))) + 1e-6
        ):
            primary_cols.append(b)
            secondary_cols.extend([b + 1, b + 2])
            b += 3
        else:
            primary_cols.append(b)
            b += 1

    col_idx = 0
    if secondary_cols and primary_cols:
        arr_p = arr[:, primary_cols]
        u_p, sigmas_p, vt_p = np.linalg.svd(arr_p, full_matrices=False)
        valid_p = [k for k in range(min(len(sigmas_p), max_cols)) if float(sigmas_p[k]) > svd_tol]
        for k in valid_p:
            v_sub = vt_p[k, :]
            v_full = np.zeros(d_basis, dtype=np.float64)
            v_full[primary_cols] = v_sub
            max_i = int(np.argmax(np.abs(v_full)))
            if v_full[max_i] < 0.0:
                v_full = -v_full
            p_svd[:, col_idx] = v_full
            col_idx += 1

        rem_slots = max_cols - col_idx
        if rem_slots > 0:
            arr_s = arr[:, secondary_cols]
            if valid_p:
                u_act = u_p[:, valid_p]
                arr_s = arr_s - u_act @ (u_act.T @ arr_s)
            col_norms = [float(np.linalg.norm(arr_s[:, j])) for j in range(len(secondary_cols))]
            ranked_sec = [
                j
                for j in sorted(
                    range(len(secondary_cols)),
                    key=lambda j: (-round(col_norms[j], 4), j),
                )
                if col_norms[j] > svd_tol
            ]
            for j in ranked_sec[:rem_slots]:
                v_full = np.zeros(d_basis, dtype=np.float64)
                v_full[secondary_cols[j]] = 1.0
                p_svd[:, col_idx] = v_full
                col_idx += 1
    else:
        _, sigmas, vt = np.linalg.svd(arr, full_matrices=False)
        valid_indices = [k for k in range(min(len(sigmas), max_cols)) if float(sigmas[k]) > svd_tol]
        for k in valid_indices:
            v_col = vt[k, :]
            max_idx = int(np.argmax(np.abs(v_col)))
            if v_col[max_idx] < 0.0:
                v_col = -v_col
            p_svd[:, col_idx] = v_col
            col_idx += 1

    active_rank = max(1, col_idx)
    rows_16 = min(d_basis, 16)
    p_svd_16x4[:rows_16, :] = p_svd[:rows_16, :].astype(np.float32)

    raw_z = arr @ p_svd  # (num_stems, 4)
    # Apply CL-17 smooth tanh trust-region bound R_trust * tanh(z / R_trust)
    z_head = trust_region_radius * np.tanh(raw_z / trust_region_radius)
    return p_svd, p_svd_16x4, active_rank, z_head


def compute_svd_control_projector_ultra(
    psi_stems: NDArray[np.float64],
    p_svd_base: NDArray[np.float64],
    *,
    base_active_rank: int = 1,
    svd_tol: float = CL18_SVD_MIN_SIGMA,
    trust_region_radius: float = 2.5,
) -> tuple[NDArray[np.float64], NDArray[np.float32], int, NDArray[np.float64]]:
    """Computes the `Tier::Ultra` Nested SVD Prefix modes `5..8` (`z_head_ultra`, $D_h = 8$)
    orthogonal to the base 4-mode projector `p_svd_base`.

    Returns:
        `(p_svd_ultra_dbasis_x_4, p_svd_ultra_16x4_f32, active_rank_ultra, z_head_ultra_stems)`
    """
    arr = np.asarray(psi_stems, dtype=np.float64)
    num_stems, d_basis = arr.shape
    p_svd_u = np.zeros((d_basis, 4), dtype=np.float64)
    p_svd_u_16x4 = np.zeros((16, 4), dtype=np.float32)
    if d_basis <= 4 or num_stems == 0:
        return (
            p_svd_u,
            p_svd_u_16x4,
            max(1, base_active_rank),
            np.zeros((num_stems, 4), dtype=np.float64),
        )

    # Deflate arr by the orthonormal columns of p_svd_base
    q_base, _ = np.linalg.qr(p_svd_base, mode="reduced")
    arr_rem = arr - (arr @ q_base) @ q_base.T
    _, sigmas_u, vt_u = np.linalg.svd(arr_rem, full_matrices=False)
    valid_u = [k for k in range(min(len(sigmas_u), 4)) if float(sigmas_u[k]) > svd_tol]
    col_idx = 0
    for k in valid_u:
        v_col = vt_u[k, :]
        v_col = v_col - q_base @ (q_base.T @ v_col)
        v_norm = float(np.linalg.norm(v_col))
        if v_norm <= 1e-10:
            continue
        v_col = v_col / v_norm
        max_idx = int(np.argmax(np.abs(v_col)))
        if v_col[max_idx] < 0.0:
            v_col = -v_col
        p_svd_u[:, col_idx] = v_col
        col_idx += 1

    rows_16 = min(d_basis, 16)
    p_svd_u_16x4[:rows_16, :] = p_svd_u[:rows_16, :].astype(np.float32)
    raw_z_u = arr @ p_svd_u
    z_head_u = trust_region_radius * np.tanh(raw_z_u / trust_region_radius)
    active_rank_ultra = max(base_active_rank, 4 + col_idx if col_idx > 0 else base_active_rank)
    return p_svd_u, p_svd_u_16x4, active_rank_ultra, z_head_u


def compute_stem_sufficient_statistics(
    features_t_x_192: NDArray[np.floating[Any]],
    target_t_x_2: NDArray[np.floating[Any]],
    *,
    stem_id: str,
    controls_meta: dict[str, float | str],
    input_pcm_sha256: str,
    target_pcm_sha256: str,
    extracted_physics: ExtractedPhysics | None = None,
    window_size: int = 32_768,
    band_balance_weight: float = 0.25,
    floor_dbfs: float = -60.0,
    band_floors_dbfs: tuple[float, float, float] | None = None,
    is_silent: bool = False,
    device: torch.device | str | None = None,
) -> StemSufficientStatistics:
    """Computes a stem's tier-ordered `float64` audio Gram matrix
    $\\mathbf{G}_{\\text{audio}}^{(s)} \\in \\mathbb{R}^{192 \\times 192}$
    and cross-correlation $\\mathbf{C}_{\\text{audio}}^{(s)} \\in \\mathbb{R}^{192 \\times 2}$ via
    window-streamed accumulation with **Floor-Regularized Target-Adaptive Band Balancing**
    across Low (`< 250 Hz`), Mid (`250-2500 Hz`), and High (`> 2500 Hz`) bands, where
    $H_{\\text{LF}}(z) + H_{\\text{MF}}(z) + H_{\\text{HF}}(z) \\equiv 1$ is a complementary
    3-band partition of unity and $\\tilde{\\gamma}_b^{(s)}$ normalizes each band by its
    floor-clamped target energy, then packs $\\mathbf{G}_{\\text{audio}}^{(s)}$ into
    `LAPACK uplo='U'` format (`18,528` `f64`).
    """
    phi = np.asarray(features_t_x_192)
    y = np.asarray(target_t_x_2)
    if phi.ndim != 2 or phi.shape[1] not in (192, 256):
        raise ValueError(f"features_t_x_192 must have shape (T, 256) or (T, 192), got {phi.shape}")
    if y.ndim == 1:
        y = np.column_stack([y, y])
    if y.ndim != 2 or y.shape[0] != phi.shape[0] or y.shape[1] not in (1, 2):
        raise ValueError(f"target_t_x_2 must have shape ({phi.shape[0]}, 2), got {y.shape}")
    if y.shape[1] == 1:
        y = np.repeat(y, 2, axis=1)

    num_frames = int(phi.shape[0])
    feat_dim = int(phi.shape[1])
    g_dense = np.zeros((256, 256), dtype=np.float64)
    c_dense = np.zeros((256, 2), dtype=np.float64)

    step = max(1, window_size)
    use_bands = band_balance_weight > 0.0 and num_frames >= 8
    y_f64_all = np.asarray(y, dtype=np.float64)

    if use_bands:
        y_lf_all, y_mf_all, y_hf_all = split_three_bands_np(y_f64_all)
        w_lf, w_mf, w_hf = _compute_band_weights_f64(
            y_lf_all,
            y_mf_all,
            y_hf_all,
            num_frames=num_frames,
            floor_dbfs=floor_dbfs,
            band_floors_dbfs=band_floors_dbfs,
            is_silent=is_silent,
        )

        for start in range(0, num_frames, step):
            end = min(start + step, num_frames)
            pad_h = min(start, 4096)
            phi_ext = np.ascontiguousarray(phi[start - pad_h : end], dtype=np.float64)
            p_lf_ext, p_mf_ext, p_hf_ext = split_three_bands_np(phi_ext)
            p_lf = p_lf_ext[pad_h:]
            p_mf = p_mf_ext[pad_h:]
            p_hf = p_hf_ext[pad_h:]
            ylf_win = y_lf_all[start:end]
            ymf_win = y_mf_all[start:end]
            yhf_win = y_hf_all[start:end]
            g_dense[:feat_dim, :feat_dim] += (
                w_lf * (p_lf.T @ p_lf) + w_mf * (p_mf.T @ p_mf) + w_hf * (p_hf.T @ p_hf)
            )
            c_dense[:feat_dim, :] += (
                w_lf * (p_lf.T @ ylf_win) + w_mf * (p_mf.T @ ymf_win) + w_hf * (p_hf.T @ yhf_win)
            )
    else:
        for start in range(0, num_frames, step):
            end = min(start + step, num_frames)
            phi_win = np.ascontiguousarray(phi[start:end], dtype=np.float64)
            y_win = np.ascontiguousarray(y_f64_all[start:end], dtype=np.float64)
            g_dense[:feat_dim, :feat_dim] += phi_win.T @ phi_win
            c_dense[:feat_dim, :] += phi_win.T @ y_win

    # Symmetrize numerical rounding before packing upper triangle
    g_dense = 0.5 * (g_dense + g_dense.T)
    g_packed = GramMatrix.pack(g_dense)

    return StemSufficientStatistics(
        stem_id=stem_id,
        controls_meta=dict(controls_meta),
        input_pcm_sha256=input_pcm_sha256,
        target_pcm_sha256=target_pcm_sha256,
        num_frames=num_frames,
        g_audio_uplo_u_f64=g_packed,
        c_audio_tier_f64=c_dense,
        extracted_physics=extracted_physics,
    )


@dataclass(frozen=True, slots=True)
class Base48BandGramCache:
    """Pre-filtered base cascade blocks `(b_lite, b_full, b_pro, b_ultra)` (`48`..`128` channels)
    and their exact full-length 3-band Gram matrices for $O(1)$ per-stem `Arena 4`
    sufficient-statistics assembly (`60x` faster Stage 4 with zero window-boundary truncation).
    """

    b48_lf: NDArray[np.floating[Any]]  # (T, num_tiers * B_cols)
    b48_mf: NDArray[np.floating[Any]]  # (T, num_tiers * B_cols)
    b48_hf: NDArray[np.floating[Any]]  # (T, num_tiers * B_cols)
    g48_lf: NDArray[np.float64]  # (num_tiers * B_cols, num_tiers * B_cols)
    g48_mf: NDArray[np.float64]  # (num_tiers * B_cols, num_tiers * B_cols)
    g48_hf: NDArray[np.float64]  # (num_tiers * B_cols, num_tiers * B_cols)
    c_knots: NDArray[np.float64] | None = None


@torch.no_grad()
def precompute_base48_band_cache(
    b_lite_16: NDArray[np.float32],
    b_full_16: NDArray[np.float32],
    b_ultra_16: NDArray[np.float32],
    b_ultra_extra_16: NDArray[np.float32] | None = None,
    *,
    c_knots: NDArray[np.float64] | None = None,
    device: torch.device | str | None = None,
    sample_rate_hz: float = 48_000.0,
) -> Base48BandGramCache:
    """Filters the base cascade channels `[b_lite, b_full, b_pro, b_ultra]` once over the full
    signal length in 32-bit `float32` on `device` and accumulates their exact band Gram matrices
    via [`ExactGramAccumulator`] before recombining to `float64` on the host.
    """
    b_u4 = (
        b_ultra_extra_16
        if b_ultra_extra_16 is not None
        else synthesize_4th_tier_base_block(b_lite_16, b_full_16, b_ultra_16)
    )
    b_all = np.ascontiguousarray(
        np.concatenate([b_lite_16, b_full_16, b_ultra_16, b_u4], axis=-1), dtype=np.float32
    )
    t_len, n_ch = b_all.shape
    sr = float(sample_rate_hz)
    store_dtype = np.float32 if t_len > 65_536 else np.float64

    b_lf = np.empty((t_len, n_ch), dtype=store_dtype)
    b_mf = np.empty((t_len, n_ch), dtype=store_dtype)
    b_hf = np.empty((t_len, n_ch), dtype=store_dtype)

    for c0 in range(0, n_ch, 16):
        c1 = min(n_ch, c0 + 16)
        lf_c, mf_c, hf_c = split_three_bands_np(
            np.ascontiguousarray(b_all[:, c0:c1], dtype=np.float64),
            sample_rate_hz=sr,
        )
        b_lf[:, c0:c1] = lf_c.astype(store_dtype, copy=False)
        b_mf[:, c0:c1] = mf_c.astype(store_dtype, copy=False)
        b_hf[:, c0:c1] = hf_c.astype(store_dtype, copy=False)

    if device is not None and torch.device(device).type != "cpu":
        dev = torch.device(device)
        acc_lf = ExactGramAccumulator(n_ch, 0, device=dev)
        acc_mf = ExactGramAccumulator(n_ch, 0, device=dev)
        acc_hf = ExactGramAccumulator(n_ch, 0, device=dev)
        for r0 in range(0, t_len, 16_384):
            r1 = min(t_len, r0 + 16_384)
            sl_lf_t = torch.from_numpy(np.ascontiguousarray(b_lf[r0:r1], dtype=np.float32)).to(
                device=dev
            )
            sl_mf_t = torch.from_numpy(np.ascontiguousarray(b_mf[r0:r1], dtype=np.float32)).to(
                device=dev
            )
            sl_hf_t = torch.from_numpy(np.ascontiguousarray(b_hf[r0:r1], dtype=np.float32)).to(
                device=dev
            )
            acc_lf.add(sl_lf_t, None)
            acc_mf.add(sl_mf_t, None)
            acc_hf.add(sl_hf_t, None)
            del sl_lf_t, sl_mf_t, sl_hf_t
        g_lf = recombine_f32_trilimb_to_f64(acc_lf.g_hi, acc_lf.g_mid, acc_lf.g_lo)
        g_mf = recombine_f32_trilimb_to_f64(acc_mf.g_hi, acc_mf.g_mid, acc_mf.g_lo)
        g_hf = recombine_f32_trilimb_to_f64(acc_hf.g_hi, acc_hf.g_mid, acc_hf.g_lo)
    else:
        g_lf = np.zeros((n_ch, n_ch), dtype=np.float64)
        g_mf = np.zeros((n_ch, n_ch), dtype=np.float64)
        g_hf = np.zeros((n_ch, n_ch), dtype=np.float64)
        for r0 in range(0, t_len, 32_768):
            r1 = min(t_len, r0 + 32_768)
            sl_lf = b_lf[r0:r1].astype(np.float64, copy=False)
            sl_mf = b_mf[r0:r1].astype(np.float64, copy=False)
            sl_hf = b_hf[r0:r1].astype(np.float64, copy=False)
            g_lf += sl_lf.T @ sl_lf
            g_mf += sl_mf.T @ sl_mf
            g_hf += sl_hf.T @ sl_hf

    g_lf = 0.5 * (g_lf + g_lf.T)
    g_mf = 0.5 * (g_mf + g_mf.T)
    g_hf = 0.5 * (g_hf + g_hf.T)

    return Base48BandGramCache(
        b48_lf=b_lf,
        b48_mf=b_mf,
        b48_hf=b_hf,
        g48_lf=g_lf,
        g48_mf=g_mf,
        g48_hf=g_hf,
        c_knots=c_knots.copy() if c_knots is not None else None,
    )


def compute_modulated_stem_sufficient_statistics(
    base_cache: Base48BandGramCache,
    u_ctrl_3: Sequence[float],
    target_t_x_2: NDArray[np.floating[Any]],
    *,
    stem_id: str,
    controls_meta: dict[str, float | str],
    input_pcm_sha256: str,
    target_pcm_sha256: str,
    extracted_physics: ExtractedPhysics | None = None,
    floor_dbfs: float = -60.0,
    band_floors_dbfs: tuple[float, float, float] | None = None,
    is_silent: bool = False,
    device: torch.device | str | None = None,
) -> StemSufficientStatistics:
    """Computes exact full-length 3-band `float64` [`StemSufficientStatistics`] directly from
    [`Base48BandGramCache`] and `u_ctrl_3` in `< 5 ms` per stem without re-filtering the 192
    modulated feature channels.
    """
    y = np.asarray(target_t_x_2, dtype=np.float64)
    if y.ndim == 1:
        y = np.column_stack([y, y])
    elif y.shape[1] == 1:
        y = np.repeat(y, 2, axis=1)

    num_frames = int(base_cache.b48_lf.shape[0])
    y = y[:num_frames]
    y_lf, y_mf, y_hf = split_three_bands_np(y)
    w_lf, w_mf, w_hf = _compute_band_weights_f64(
        y_lf,
        y_mf,
        y_hf,
        num_frames=num_frames,
        floor_dbfs=floor_dbfs,
        band_floors_dbfs=band_floors_dbfs,
        is_silent=is_silent,
    )

    g_base = w_lf * base_cache.g48_lf + w_mf * base_cache.g48_mf + w_hf * base_cache.g48_hf
    c_base = (
        w_lf * np.asarray(base_cache.b48_lf.T @ y_lf, dtype=np.float64)
        + w_mf * np.asarray(base_cache.b48_mf.T @ y_mf, dtype=np.float64)
        + w_hf * np.asarray(base_cache.b48_hf.T @ y_hf, dtype=np.float64)
    )

    g_dense = np.zeros((256, 256), dtype=np.float64)
    c_dense = np.zeros((256, 2), dtype=np.float64)

    if g_base.shape[0] in (96, 128):
        n_tiers = g_base.shape[0] // 32
        f0, f1 = compute_tier32_modulator_vectors(u_ctrl_3, base_cache.c_knots)
        f00 = np.outer(f0, f0)
        f01 = np.outer(f0, f1)
        f10 = np.outer(f1, f0)
        f11 = np.outer(f1, f1)
        for t1 in range(n_tiers):
            b1_s, b1_e = 32 * t1, 32 * (t1 + 1)
            c_t1 = c_base[b1_s:b1_e, :]
            r0 = 64 * t1
            r1 = r0 + 32
            c_dense[r0 : r0 + 32, :] = f0[:, None] * c_t1
            c_dense[r1 : r1 + 32, :] = f1[:, None] * c_t1
            for t2 in range(n_tiers):
                b2_s, b2_e = 32 * t2, 32 * (t2 + 1)
                g_sub = g_base[b1_s:b1_e, b2_s:b2_e]
                c0 = 64 * t2
                c1 = c0 + 32
                g_dense[r0 : r0 + 32, c0 : c0 + 32] = g_sub * f00
                g_dense[r0 : r0 + 32, c1 : c1 + 32] = g_sub * f01
                g_dense[r1 : r1 + 32, c0 : c0 + 32] = g_sub * f10
                g_dense[r1 : r1 + 32, c1 : c1 + 32] = g_sub * f11
    else:
        n_tiers = min(4, g_base.shape[0] // 16)
        u0 = float(u_ctrl_3[0]) if len(u_ctrl_3) > 0 else 0.0
        u1 = float(u_ctrl_3[1]) if len(u_ctrl_3) > 1 else 0.0
        u2 = float(u_ctrl_3[2]) if len(u_ctrl_3) > 2 else 0.0
        m_vec = np.array([1.0, u0, u1, u2], dtype=np.float64)
        m_outer = np.outer(m_vec, m_vec)  # (4, 4)
        for t1 in range(n_tiers):
            b1_s, b1_e = 16 * t1, 16 * (t1 + 1)
            c48_t1 = c_base[b1_s:b1_e, :]
            for k1 in range(4):
                r_s = 64 * t1 + 16 * k1
                r_e = r_s + 16
                c_dense[r_s:r_e, :] = m_vec[k1] * c48_t1
                for t2 in range(n_tiers):
                    b2_s, b2_e = 16 * t2, 16 * (t2 + 1)
                    g48_sub = g_base[b1_s:b1_e, b2_s:b2_e]
                    for k2 in range(4):
                        c_s = 64 * t2 + 16 * k2
                        c_e = c_s + 16
                        g_dense[r_s:r_e, c_s:c_e] = m_outer[k1, k2] * g48_sub

    g_dense = 0.5 * (g_dense + g_dense.T)
    g_packed = GramMatrix.pack(g_dense)
    return StemSufficientStatistics(
        stem_id=stem_id,
        controls_meta=dict(controls_meta),
        input_pcm_sha256=input_pcm_sha256,
        target_pcm_sha256=target_pcm_sha256,
        num_frames=num_frames,
        g_audio_uplo_u_f64=g_packed,
        c_audio_tier_f64=c_dense,
        extracted_physics=extracted_physics,
    )


def solve_kronecker_head_float64(
    stem_stats: Sequence[StemSufficientStatistics],
    z_head_stems: NDArray[np.float64],
    *,
    mode: Literal["init", "polish"] = "polish",
    tier_dim: Literal[64, 128, 192, 256] = 192,
    ridge_scale_mult: float = 1.0,
    solver: HostDenseSolver | None = None,
) -> tuple[NDArray[np.float64], float]:
    """Solves the `CL-18` `float64` Kronecker normal equations for a single slimmable tier (`64`,
    `128`, `192`, or `256`).

    Uses the **Nested Principal Submatrix Prefix Theorem**:
    `GramMatrix.unpack(s.g_audio_uplo_u_f64, dim=tier_dim)` extracts the exact leading
    principal submatrix for `Lite` ($N=64$), `Full` ($N=128$), `Pro` ($N=192$), or
    `Ultra` ($N=256$).

    Args:
        stem_stats: Sequence of per-stem `StemSufficientStatistics`.
        z_head_stems: `(num_stems, 4)` projected control coordinates
            $\\mathbf{z}_{\\text{head}}(\\mathbf{u}_s, \\mathbf{s}_s)$.
        mode: `'init'` ($\\lambda_{\\text{init}} = 10^{-3}\\bar{\\tau}$) or `'polish'`
            ($\\lambda_{\\text{polish}} = 10^{-6}\\bar{\\tau}$).
        tier_dim: `64` (`Lite`), `128` (`Full`), `192` (`Pro`), or `256` (`Ultra`).
        ridge_scale_mult: Multiplier on the base ridge regularization scale (`1.0` default).

    Returns:
        `(w_head_5_2_tier, ridge_lambda)` where `w_head_5_2_tier` has shape `(5, 2, tier_dim)`.
    """
    _ = solver
    num_stems = len(stem_stats)
    if num_stems == 0:
        raise ValueError("solve_kronecker_head_float64 requires at least 1 stem")
    z_arr = np.asarray(z_head_stems, dtype=np.float64)
    if z_arr.shape != (num_stems, 4):
        raise ValueError(f"z_head_stems must have shape ({num_stems}, 4), got {z_arr.shape}")

    num_modes = 5  # [1, z_1, z_2, z_3, z_4]
    total_dim = num_modes * tier_dim
    g_kron = np.zeros((total_dim, total_dim), dtype=np.float64)
    c_kron = np.zeros((total_dim, 2), dtype=np.float64)

    for s_idx, stat in enumerate(stem_stats):
        g_sub = GramMatrix.unpack(stat.g_audio_uplo_u_f64, dim=tier_dim)
        c_sub = stat.c_audio_tier_f64[:tier_dim, :]  # (tier_dim, 2)
        psi_tilde = np.empty(num_modes, dtype=np.float64)
        psi_tilde[0] = 1.0
        psi_tilde[1:] = z_arr[s_idx]
        psi_outer = np.outer(psi_tilde, psi_tilde)  # (5, 5)

        # Block Kronecker accumulation: mode_i * tier_dim .. (mode_i + 1) * tier_dim
        g_kron += np.kron(psi_outer, g_sub)
        c_kron += np.kron(psi_tilde[:, None], c_sub)

    trace_avg = float(np.trace(g_kron)) / float(max(total_dim, 1))
    scale = (CL18_RIDGE_INIT_SCALE if mode == "init" else CL18_RIDGE_POLISH_SCALE) * float(
        ridge_scale_mult
    )
    ridge_lambda = scale * max(trace_avg, 1e-12)

    g_reg = g_kron + ridge_lambda * np.eye(total_dim, dtype=np.float64)
    try:
        w_flat = np.linalg.solve(g_reg, c_kron)  # (5 * tier_dim, 2)
    except np.linalg.LinAlgError:
        w_flat, _, _, _ = np.linalg.lstsq(g_reg, c_kron, rcond=None)

    # Reshape (5, tier_dim, 2) -> transpose to (5, 2, tier_dim)
    w_modes = w_flat.reshape((num_modes, tier_dim, 2)).transpose(0, 2, 1).copy()
    return w_modes, ridge_lambda


def _solve_ultra_modes5to8_residual_float64(
    stem_stats: Sequence[StemSufficientStatistics],
    z_head_stems: NDArray[np.float64],
    z_head_ultra_stems: NDArray[np.float64],
    w_base_5x2x_tier: NDArray[np.float64],
    *,
    mode: Literal["init", "polish"] = "polish",
) -> NDArray[np.float64]:
    """Solves the 4 `Tier::Ultra` SVD residual modes `5..8` (`(4, 2, tier_dim)`) on the residual
    cross-correlation $\\mathbf{C}_{\\text{res}}^{(s)} = \\mathbf{C}_{\\text{audio}}^{(s)} -
    \\mathbf{G}_{\\text{audio}}^{(s)} \\mathbf{W}_{\\text{base}}(\\mathbf{z}_s)^\\top$.
    """
    tier_dim = int(w_base_5x2x_tier.shape[2])
    num_u_modes = 4
    total_dim = num_u_modes * tier_dim
    g_kron = np.zeros((total_dim, total_dim), dtype=np.float64)
    c_kron = np.zeros((total_dim, 2), dtype=np.float64)

    z_base = np.asarray(z_head_stems, dtype=np.float64)
    z_u = np.asarray(z_head_ultra_stems, dtype=np.float64)
    if float(np.max(np.abs(z_u))) <= 1e-6:
        return np.zeros((4, 2, tier_dim), dtype=np.float64)

    for s_idx, stat in enumerate(stem_stats):
        g_sub = GramMatrix.unpack(stat.g_audio_uplo_u_f64, dim=tier_dim)
        c_sub = stat.c_audio_tier_f64[:tier_dim, :]
        psi_base = np.array(
            [1.0, z_base[s_idx, 0], z_base[s_idx, 1], z_base[s_idx, 2], z_base[s_idx, 3]],
            dtype=np.float64,
        )
        w_eff_base = np.tensordot(psi_base, w_base_5x2x_tier, axes=(0, 0))  # (2, tier_dim)
        c_res = c_sub - g_sub @ w_eff_base.T  # (tier_dim, 2)

        zu_s = z_u[s_idx]  # (4,)
        zu_outer = np.outer(zu_s, zu_s)  # (4, 4)
        g_kron += np.kron(zu_outer, g_sub)
        c_kron += np.kron(zu_s[:, None], c_res)

    trace_avg = float(np.trace(g_kron)) / float(max(total_dim, 1))
    if trace_avg <= 1e-18:
        return np.zeros((4, 2, tier_dim), dtype=np.float64)
    scale = CL18_RIDGE_INIT_SCALE if mode == "init" else CL18_RIDGE_POLISH_SCALE
    ridge_lambda = scale * max(trace_avg, 1e-12)
    g_reg = g_kron + ridge_lambda * np.eye(total_dim, dtype=np.float64)
    try:
        w_flat = np.linalg.solve(g_reg, c_kron)
    except np.linalg.LinAlgError:
        w_flat, _, _, _ = np.linalg.lstsq(g_reg, c_kron, rcond=None)
    return w_flat.reshape((num_u_modes, tier_dim, 2)).transpose(0, 2, 1).copy()


@dataclass(frozen=True, slots=True)
class MultiTierKroneckerHeadSolution:
    """Simultaneous `Lite` ($N=64$), `Full` ($N=128$), `Pro` ($N=192, D_h=4$), and
    `Ultra` ($N=256, D_h=8$) Kronecker head solution."""

    w_lite_5x2x64: NDArray[np.float64]
    w_full_5x2x128: NDArray[np.float64]
    w_ultra_5x2x192: NDArray[np.float64]
    ridge_lite: float
    ridge_full: float
    ridge_ultra: float
    w_pro_5x2x192: NDArray[np.float64] | None = None
    ridge_pro: float = 0.0
    w_ultra_5x2x256: NDArray[np.float64] | None = None
    w_ultra_modes5to8_4x2x192: NDArray[np.float64] | None = None

    @property
    def pro_head_5x2x192(self) -> NDArray[np.float64]:
        """Returns the `Pro` ($N=192, D_h=4$) head tensor `(5, 2, 192)`."""
        if self.w_pro_5x2x192 is not None:
            return self.w_pro_5x2x192[:, :, :192]
        return self.w_ultra_5x2x192[:, :, :192]

    @property
    def ultra_head_5x2x256(self) -> NDArray[np.float64]:
        """Returns the `Ultra` ($N=256, D_h=4$) base head tensor `(5, 2, 256)`."""
        if self.w_ultra_5x2x256 is not None:
            return self.w_ultra_5x2x256
        if self.w_ultra_5x2x192.shape[2] >= 256:
            return self.w_ultra_5x2x192[:, :, :256]
        out = np.zeros((5, 2, 256), dtype=np.float64)
        n = min(256, int(self.w_ultra_5x2x192.shape[2]))
        out[:, :, :n] = self.w_ultra_5x2x192[:, :, :n]
        return out

    @property
    def arena0_head_5x2x64_f32(self) -> NDArray[np.float32]:
        """
        `Arena 0 (Lite)` head slice `(5, 2, 64)` (`5 * 128 floats = 2,560 B` at offset `4224`).
        """
        return self.w_lite_5x2x64.astype(np.float32)

    @property
    def arena1_head_delta_5x2x64_f32(self) -> NDArray[np.float32]:
        """`Arena 1 (Full Delta)` head slice `(5, 2, 64)` for features `64..128` (`2,560 B` at
        offset `17984`)."""
        return self.w_full_5x2x128[:, :, 64:128].astype(np.float32)

    @property
    def arena2_head_delta_5x2x64_f32(self) -> NDArray[np.float32]:
        """`Arena 2 (Pro Delta)` head slice `(5, 2, 64)` for features `128..192` (`2,560 B` at
        offset `32640`)."""
        return self.pro_head_5x2x192[:, :, 128:192].astype(np.float32)

    @property
    def arena3_head_delta_5x2x64_f32(self) -> NDArray[np.float32]:
        """`Arena 3 (Ultra Delta)` head slice `(5, 2, 64)` for features `192..256` (`2,560 B` at
        offset `49600`)."""
        return self.ultra_head_5x2x256[:, :, 192:256].astype(np.float32)

    @property
    def arena3_ultra_modes5to8_4x4x2x64_f32(self) -> NDArray[np.float32]:
        """`Arena 3 (Ultra SVD modes 5..8)` head slice `(4 tier groups, 4 modes, 2 ch, 64 taps)`
        (`8,192 B` at offset `52160`)."""
        out = np.zeros((4, 4, 2, 64), dtype=np.float32)
        if self.w_ultra_modes5to8_4x2x192 is not None:
            n_grp = min(4, int(self.w_ultra_modes5to8_4x2x192.shape[2]) // 64)
            for t_grp in range(n_grp):
                out[t_grp] = self.w_ultra_modes5to8_4x2x192[
                    :, :, t_grp * 64 : (t_grp + 1) * 64
                ].astype(np.float32)
        return out


def solve_all_tiers_kronecker_head_float64(
    stem_stats: Sequence[StemSufficientStatistics],
    z_head_stems: NDArray[np.float64],
    *,
    z_head_ultra_stems: NDArray[np.float64] | None = None,
    mode: Literal["init", "polish"] = "polish",
    solver: HostDenseSolver | None = None,
) -> MultiTierKroneckerHeadSolution:
    """
    Solves all 4 slimmable tiers (`Lite` $N=64$, `Full` $N=128$, `Pro` $N=192, D_h=4$,
    and `Ultra` $N=256, D_h=8$) in `float64`.
    """
    has_active_256 = bool(stem_stats) and all(stat.active_feature_dim == 256 for stat in stem_stats)
    w_lite, r_lite = solve_kronecker_head_float64(
        stem_stats, z_head_stems, mode=mode, tier_dim=64, solver=solver
    )
    w_full, r_full = solve_kronecker_head_float64(
        stem_stats, z_head_stems, mode=mode, tier_dim=128, solver=solver
    )
    pro_ridge_mult = 1.0 if has_active_256 else 5.0
    w_pro, r_pro = solve_kronecker_head_float64(
        stem_stats,
        z_head_stems,
        mode=mode,
        tier_dim=192,
        ridge_scale_mult=pro_ridge_mult,
        solver=solver,
    )
    ultra_tier_dim: Literal[192, 256] = 256 if has_active_256 else 192
    w_ultra, r_ultra = solve_kronecker_head_float64(
        stem_stats,
        z_head_stems,
        mode=mode,
        tier_dim=ultra_tier_dim,
        ridge_scale_mult=1.0,
        solver=solver,
    )
    w_u_modes = None
    if z_head_ultra_stems is not None:
        w_u_modes = _solve_ultra_modes5to8_residual_float64(
            stem_stats, z_head_stems, z_head_ultra_stems, w_ultra, mode=mode
        )
    if solver is not None:
        solver.total_solves += 4
    w_ultra_256 = np.zeros((5, 2, 256), dtype=np.float64)
    w_ultra_256[:, :, : w_ultra.shape[2]] = w_ultra
    return MultiTierKroneckerHeadSolution(
        w_lite_5x2x64=w_lite,
        w_full_5x2x128=w_full,
        w_ultra_5x2x192=w_ultra,
        ridge_lite=r_lite,
        ridge_full=r_full,
        ridge_ultra=r_ultra,
        w_pro_5x2x192=w_pro,
        ridge_pro=r_pro,
        w_ultra_5x2x256=w_ultra_256,
        w_ultra_modes5to8_4x2x192=w_u_modes,
    )


def evaluate_kronecker_head_features(
    features_t_x_192: NDArray[np.floating[Any]],
    z_head_4: NDArray[np.floating[Any]],
    w_head_5x2x_tier: NDArray[np.floating[Any]],
    *,
    z_head_ultra_4: NDArray[np.floating[Any]] | None = None,
    w_ultra_modes5to8_4x2x_tier: NDArray[np.floating[Any]] | None = None,
) -> NDArray[np.float64]:
    """Evaluates the stereo Kronecker head $\\hat{\\mathbf{Y}} = \\sum_{m=0}^4 \\tilde{\\psi}_m
    \\mathbf{\\Phi}_{:, :N_{\\text{tier}}} \\mathbf{W}_{\\text{head},m}^\\top$ (plus `Tier::Ultra`
    SVD modes `5..8` when present)."""
    tier_dim = min(int(w_head_5x2x_tier.shape[2]), int(features_t_x_192.shape[1]))
    phi = np.asarray(features_t_x_192[:, :tier_dim], dtype=np.float64)
    z = np.asarray(z_head_4, dtype=np.float64).ravel()
    psi_tilde = np.array([1.0, z[0], z[1], z[2], z[3]], dtype=np.float64)
    w_eff = np.tensordot(
        psi_tilde, np.asarray(w_head_5x2x_tier[:, :, :tier_dim], dtype=np.float64), axes=(0, 0)
    )
    if z_head_ultra_4 is not None and w_ultra_modes5to8_4x2x_tier is not None:
        zu = np.asarray(z_head_ultra_4, dtype=np.float64).ravel()
        if float(np.max(np.abs(zu))) > 1e-12:
            wu = np.asarray(w_ultra_modes5to8_4x2x_tier[:, :, :tier_dim], dtype=np.float64)
            w_eff = w_eff + np.tensordot(zu, wu, axes=(0, 0))
    # w_eff has shape (2, tier_dim) -> phi @ w_eff.T has shape (T, 2)
    return phi @ w_eff.T


def evaluate_stem_besr_4tier_from_sufficient_stats(
    stat: StemSufficientStatistics,
    z_head_4: NDArray[np.floating[Any]],
    head_sol: MultiTierKroneckerHeadSolution,
    target_t_x_2: NDArray[np.floating[Any]],
    *,
    z_head_ultra_4: NDArray[np.floating[Any]] | None = None,
    floor_dbfs: float = -60.0,
    band_floors_dbfs: tuple[float, float, float] | None = None,
    is_silent: bool = False,
) -> tuple[float, float, float, float]:
    """Evaluates exact `Lite` ($N=64$), `Full` ($N=128$), `Pro` ($N=192, D_h=4$), and
    `Ultra` ($N=256, D_h=8$) `bESR` in dB directly from a stem's cached `Arena 4`
    [`StemSufficientStatistics`] via the quadratic form identity.
    """
    y = np.asarray(target_t_x_2, dtype=np.float64)
    if y.ndim == 1:
        y = np.column_stack([y, y])
    elif y.shape[1] == 1:
        y = np.repeat(y, 2, axis=1)
    num_frames = stat.num_frames
    y = y[:num_frames]

    y_lf, y_mf, y_hf = split_three_bands_np(y)
    w_lf, w_mf, w_hf = _compute_band_weights_f64(
        y_lf,
        y_mf,
        y_hf,
        num_frames=num_frames,
        floor_dbfs=floor_dbfs,
        band_floors_dbfs=band_floors_dbfs,
        is_silent=is_silent,
    )
    target_quad = (
        w_lf * float(np.sum(y_lf * y_lf))
        + w_mf * float(np.sum(y_mf * y_mf))
        + w_hf * float(np.sum(y_hf * y_hf))
    )

    z = np.asarray(z_head_4, dtype=np.float64).ravel()
    psi_tilde = np.array([1.0, z[0], z[1], z[2], z[3]], dtype=np.float64)

    w_u = head_sol.w_ultra_5x2x192
    u_dim = int(w_u.shape[2])
    tier_besrs: list[float] = []
    for tier_idx, (tier_dim, w_head) in enumerate(
        (
            (64, head_sol.w_lite_5x2x64),
            (128, head_sol.w_full_5x2x128),
            (192, head_sol.pro_head_5x2x192),
            (u_dim, w_u),
        )
    ):
        w_eff = np.tensordot(psi_tilde, np.asarray(w_head, dtype=np.float64), axes=(0, 0))
        if (
            tier_idx == 3
            and z_head_ultra_4 is not None
            and head_sol.w_ultra_modes5to8_4x2x192 is not None
        ):
            zu = np.asarray(z_head_ultra_4, dtype=np.float64).ravel()
            if float(np.max(np.abs(zu))) > 1e-12:
                w_eff = w_eff + np.tensordot(
                    zu,
                    np.asarray(
                        head_sol.w_ultra_modes5to8_4x2x192[:, :, :tier_dim], dtype=np.float64
                    ),
                    axes=(0, 0),
                )
        g_sub = GramMatrix.unpack(stat.g_audio_uplo_u_f64, dim=tier_dim)
        c_sub = stat.c_audio_tier_f64[:tier_dim, :]
        quad_err = float(
            np.sum(w_eff * (w_eff @ g_sub)) - 2.0 * np.sum(w_eff * c_sub.T) + target_quad
        )
        besr_lin = max(quad_err / float(max(1, num_frames)), 1e-18)
        tier_besrs.append(float(10.0 * np.log10(besr_lin)))

    return tier_besrs[0], tier_besrs[1], tier_besrs[2], tier_besrs[3]


def evaluate_stem_besr_from_sufficient_stats(
    stat: StemSufficientStatistics,
    z_head_4: NDArray[np.floating[Any]],
    head_sol: MultiTierKroneckerHeadSolution,
    target_t_x_2: NDArray[np.floating[Any]],
    *,
    z_head_ultra_4: NDArray[np.floating[Any]] | None = None,
    floor_dbfs: float = -60.0,
    band_floors_dbfs: tuple[float, float, float] | None = None,
    is_silent: bool = False,
) -> tuple[float, float, float]:
    """Evaluates exact `Lite` ($N=64$), `Full` ($N=128$), and `Ultra` ($N=192..256$) `bESR` in dB
    directly from a stem's cached `Arena 4` [`StemSufficientStatistics`] via the quadratic form
    identity without re-evaluating the 4-stage cascade.
    """
    bl, bf, _bp, bu = evaluate_stem_besr_4tier_from_sufficient_stats(
        stat,
        z_head_4,
        head_sol,
        target_t_x_2,
        z_head_ultra_4=z_head_ultra_4,
        floor_dbfs=floor_dbfs,
        band_floors_dbfs=band_floors_dbfs,
        is_silent=is_silent,
    )
    return bl, bf, bu


def merge_stem_statistics_for_resume(
    existing_stats: Sequence[StemSufficientStatistics],
    incoming_candidates: Sequence[tuple[str, dict[str, float | str], str, str]],
) -> tuple[list[StemSufficientStatistics | None], list[int]]:
    """Deduplicates incoming stems against `.dnam-ckpt` `existing_stats` by `(input_pcm_sha256,
    target_pcm_sha256, controls_meta)`.

    Args:
        existing_stats: Cached `StemSufficientStatistics` loaded from `.dnam-ckpt` `Arena 3`.
        incoming_candidates: Sequence of `(stem_id, controls_meta, input_pcm_sha256,
        target_pcm_sha256)`.

    Returns:
        `(resolved_slots, indices_requiring_forward_pass)` where:
        - `resolved_slots[i]` is the reused `StemSufficientStatistics` if unchanged (`0 ms` fast
        path),
          or `None` if `incoming_candidates[i]` is new or modified and must be forward-passed.
        - `indices_requiring_forward_pass` lists the indices `i` where `resolved_slots[i] is None`.
    """
    exact_lookup: dict[
        tuple[str, str, tuple[tuple[str, float | str], ...]], StemSufficientStatistics
    ] = {}
    for stat in existing_stats:
        key_ctrl = tuple(sorted(stat.controls_meta.items(), key=lambda kv: kv[0]))
        exact_lookup[(stat.input_pcm_sha256, stat.target_pcm_sha256, key_ctrl)] = stat

    resolved_slots: list[StemSufficientStatistics | None] = []
    needs_eval: list[int] = []

    for idx, (_stem_id, ctrl_meta, in_sha, tgt_sha) in enumerate(incoming_candidates):
        key_ctrl = tuple(sorted(ctrl_meta.items(), key=lambda kv: kv[0]))
        cached = exact_lookup.get((in_sha, tgt_sha, key_ctrl))
        if cached is not None:
            resolved_slots.append(cached)
        else:
            resolved_slots.append(None)
            needs_eval.append(idx)

    return resolved_slots, needs_eval
