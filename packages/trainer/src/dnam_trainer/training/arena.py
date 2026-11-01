"""64-Byte Cache-Line Aligned `Arena 0..3` Tensor Packer, Unpacker & Signed Container Builder.

Matches the exact byte offsets required by:
- [`docs/dnam_container_schema.md`](../../../../../docs/dnam_container_schema.md)
- `packages/engine/src/container/view.rs` (`10,752 B` Lite, `22,336 B` Full, `36,224 B` Ultra)
- `packages/engine/src/provenance/watermark.rs` (`rglru_soa_byte_offset`, `W_head`, `b_head`,
`P_svd`, `M-UFDN` delays, and 4,608 dilated conv weights)
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from dnam_engine import DnamContainer, provenance
from dnam_engine.schema import (
    CreatorSignature,
    DnamJsonHeader,
    ModelSignatures,
    StemRegistryEntry,
    TensorSliceEntry,
)
from dnam_trainer.training.calib_cache import Arena4CalibrationData
from dnam_trainer.training.kronecker_head import (
    ARENA3_STEM_BYTE_LEN,
    MultiTierKroneckerHeadSolution,
    StemSufficientStatistics,
)

ARENA_OFFSET_LITE_END: int = 10_752
ARENA_OFFSET_FULL_END: int = 22_336
ARENA_OFFSET_PRO_END: int = 36_224
ARENA_OFFSET_ULTRA_END: int = 64_000


def rglru_soa_byte_offset(tier: int, block_idx: int) -> int:
    """
    Returns the byte offset of the 128-byte `SelectiveRgLru` SoA struct for `(tier, block_idx)`.
    """
    if tier == 0:
        return 128 + block_idx * 1024 + 896
    if tier == 1:
        return 10816 + block_idx * 1792 + 1664
    if tier == 2:
        return 22400 + block_idx * 2560 + 2432
    return 36288 + block_idx * 3328 + 3200


def block_base_byte_offset(tier: int, block_idx: int) -> int:
    """Returns the byte offset of `Core1BlockSlice` for `(tier, block_idx)`."""
    if tier == 0:
        return 128 + block_idx * 1024
    if tier == 1:
        return 10816 + block_idx * 1792
    if tier == 2:
        return 22400 + block_idx * 2560
    return 36288 + block_idx * 3328


@dataclass
class ArenaWeightLayout:
    """Structured view of the `36,224 B` `Arena 0..2` execution-order SoA tensor arena."""

    # [tier 0..2, block 0..3, lane 0..3] -> (3, 4, 4)
    rglru_r_low: NDArray[np.float32] = field(
        default_factory=lambda: np.full((4, 4, 4), 0.85, dtype=np.float32)
    )
    rglru_dr_span: NDArray[np.float32] = field(
        default_factory=lambda: np.full((4, 4, 4), 0.10, dtype=np.float32)
    )
    rglru_theta0_rad: NDArray[np.float32] = field(
        default_factory=lambda: np.full((4, 4, 4), 0.25, dtype=np.float32)
    )
    rglru_b_gamma_norm: NDArray[np.float32] = field(
        default_factory=lambda: np.full((4, 4, 4), 0.35, dtype=np.float32)
    )
    rglru_w_sel: NDArray[np.float32] = field(
        default_factory=lambda: np.full((4, 4, 4), 0.50, dtype=np.float32)
    )
    rglru_c_real: NDArray[np.float32] = field(
        default_factory=lambda: np.full((4, 4, 4), 0.40, dtype=np.float32)
    )
    rglru_c_imag: NDArray[np.float32] = field(
        default_factory=lambda: np.full((4, 4, 4), -0.15, dtype=np.float32)
    )

    # PolyADAA per [tier 0..3, block 0..3, sublayer 0..1, param 0..3 (b_conv, alpha, beta, delta),
    # lane 0..3]
    poly_adaa: NDArray[np.float32] = field(default_factory=lambda: _default_poly_adaa_weights())

    # 4,608 nonlinear dilated conv weights across Arena 0 (768), Arena 1 (1536), Arena 2 (2304)
    conv_weights_4608: NDArray[np.float32] = field(
        default_factory=lambda: np.zeros(4608, dtype=np.float32)
    )

    # Stereo Kronecker head weights per tier slice: (4, 5, 2, 64)
    w_head_tiers: NDArray[np.float32] = field(
        default_factory=lambda: np.zeros((4, 5, 2, 64), dtype=np.float32)
    )
    # Head stereo bias b_head: (5, 2)
    b_head: NDArray[np.float32] = field(default_factory=lambda: np.zeros((5, 2), dtype=np.float32))
    # Per-tier output loudness leveling linear gain trims [Lite, Full, Pro, Ultra]: (4,)
    tier_gain_trim_lin: NDArray[np.float32] = field(
        default_factory=lambda: np.ones(4, dtype=np.float32)
    )
    # SVD control subspace projector P_svd: (16, 4)
    p_svd_16x4: NDArray[np.float32] = field(
        default_factory=lambda: np.zeros((16, 4), dtype=np.float32)
    )
    # 8 M-UFDN base delays (samples): (8,)
    mufdn_delays_8: NDArray[np.float32] = field(
        default_factory=lambda: np.array(
            [149.0, 211.0, 307.0, 443.0, 619.0, 887.0, 1277.0, 1811.0],
            dtype=np.float32,
        )
    )
    # 8 WT-UFDN grain lengths (samples): (8,)
    mufdn_grain_len_8: NDArray[np.float32] = field(
        default_factory=lambda: np.full(8, 2048.0, dtype=np.float32)
    )
    # 8 WT-UFDN grain warp rates (nu = 1 - 2^(st/12)): (8,)
    mufdn_grain_warp_8: NDArray[np.float32] = field(
        default_factory=lambda: np.zeros(8, dtype=np.float32)
    )
    # Group 0 WT-UFDN grain meta: [window_mix, spray_samples, feedback_mix, head_tap_inject_gain]
    mufdn_grain_meta_0: NDArray[np.float32] = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0, 0.1], dtype=np.float32)
    )
    # Group 1 WT-UFDN grain metadata: [freeze_alpha_default, 0.0, onset_sync_thresh, 0.0]
    mufdn_grain_meta_1: NDArray[np.float32] = field(
        default_factory=lambda: np.zeros(4, dtype=np.float32)
    )
    # 3 kHz Core 2 state-space modulation matrix W_ctrl_core2: (8, 32) at 9216..10240
    w_ctrl_core2_8x32: NDArray[np.float32] = field(
        default_factory=lambda: np.zeros((8, 32), dtype=np.float32)
    )
    # Arena 3 (Ultra Delta, 36224..64000 B) 3,072 nonlinear dilated conv weights
    conv_weights_ultra_3072: NDArray[np.float32] = field(
        default_factory=lambda: np.zeros(3072, dtype=np.float32)
    )
    # Arena 3 (Ultra Delta) M-UFDN lines 8..15 base delays (samples): (8,)
    mufdn_delays_ultra_8: NDArray[np.float32] = field(
        default_factory=lambda: np.array(
            [179.0, 257.0, 373.0, 521.0, 743.0, 1061.0, 1523.0, 2179.0],
            dtype=np.float32,
        )
    )
    # Arena 3 (Ultra Delta) SVD modes 5..8 head weights: (4 tier groups, 4 modes, 2 ch, 64 taps)
    w_head_ultra_modes5to8: NDArray[np.float32] = field(
        default_factory=lambda: np.zeros((4, 4, 2, 64), dtype=np.float32)
    )
    # Arena 3 (Ultra Delta) SVD modes 5..8 control projector: (16, 4) at 63168..63424
    p_svd_ultra_16x4: NDArray[np.float32] = field(
        default_factory=lambda: np.zeros((16, 4), dtype=np.float32)
    )
    # Arena 3 (Ultra Delta) SVD modes 5..8 stereo bias: (4, 2) at 63424..63456
    b_head_ultra_modes5to8: NDArray[np.float32] = field(
        default_factory=lambda: np.zeros((4, 2), dtype=np.float32)
    )

    def to_arena012_bytes(self) -> bytes:
        """Packs all fields into a 64-byte aligned `64,000 B` `Arena 0..3` byte buffer."""
        buf = bytearray(ARENA_OFFSET_ULTRA_END)

        def write_f32_slice(byte_offset: int, arr: NDArray[np.float32]) -> None:
            raw = np.ascontiguousarray(arr, dtype="<f4").tobytes()
            buf[byte_offset : byte_offset + len(raw)] = raw

        # 1. b_head in Arena 0 CL1 (64..104: 5 modes * 2 stereo channels = 40 B)
        # + Lite/Full/Pro linear gain trims (104..116 = 12 B)
        write_f32_slice(64, self.b_head.ravel())
        trims_flat = np.ascontiguousarray(self.tier_gain_trim_lin, dtype=np.float32).ravel()
        if len(trims_flat) >= 3:
            write_f32_slice(104, trims_flat[:3])

        # 2. Dilated conv weights (4,608 floats in Arena 0..2 + 3,072 in Arena 3)
        # + PolyADAA + RG-LRU SoA
        conv_flat = np.ascontiguousarray(self.conv_weights_4608, dtype=np.float32).ravel()
        conv_cursor = 0
        conv_counts_per_sublayer = (96, 192, 288)
        poly0_offsets = (384, 768, 1152)
        conv1_offsets = (448, 832, 1216)
        poly1_offsets = (832, 1600, 2368)

        for tier in range(3):
            n_conv = conv_counts_per_sublayer[tier]
            for blk in range(4):
                base = block_base_byte_offset(tier, blk)
                # Conv0
                write_f32_slice(base, conv_flat[conv_cursor : conv_cursor + n_conv])
                conv_cursor += n_conv
                # PolyADAA0 (16 floats = 64 B)
                write_f32_slice(base + poly0_offsets[tier], self.poly_adaa[tier, blk, 0].ravel())
                # Conv1
                write_f32_slice(
                    base + conv1_offsets[tier],
                    conv_flat[conv_cursor : conv_cursor + n_conv],
                )
                conv_cursor += n_conv
                # PolyADAA1 (16 floats = 64 B)
                write_f32_slice(base + poly1_offsets[tier], self.poly_adaa[tier, blk, 1].ravel())

                # SelectiveRgLru SoA (32 floats = 128 B)
                rglru_base = rglru_soa_byte_offset(tier, blk)
                theta = self.rglru_theta0_rad[tier, blk]
                u_rot_init = np.cos(theta).astype(np.float32)
                rglru_soa = np.concatenate(
                    [
                        self.rglru_r_low[tier, blk],
                        self.rglru_dr_span[tier, blk],
                        theta,
                        self.rglru_b_gamma_norm[tier, blk],
                        self.rglru_w_sel[tier, blk],
                        self.rglru_c_real[tier, blk],
                        self.rglru_c_imag[tier, blk],
                        u_rot_init,
                    ]
                )
                write_f32_slice(rglru_base, rglru_soa)

        # Arena 3 (Tier 3 = Ultra Delta, 36224..64000 B)
        conv_u_flat = np.ascontiguousarray(self.conv_weights_ultra_3072, dtype=np.float32).ravel()
        conv_u_cursor = 0
        t3_src = 3 if self.rglru_r_low.shape[0] > 3 else 2
        p3_src = 3 if self.poly_adaa.shape[0] > 3 else 2
        for blk in range(4):
            base = block_base_byte_offset(3, blk)
            write_f32_slice(base, conv_u_flat[conv_u_cursor : conv_u_cursor + 384])
            conv_u_cursor += 384
            write_f32_slice(base + 1536, self.poly_adaa[p3_src, blk, 0].ravel())
            write_f32_slice(base + 1600, conv_u_flat[conv_u_cursor : conv_u_cursor + 384])
            conv_u_cursor += 384
            write_f32_slice(base + 3136, self.poly_adaa[p3_src, blk, 1].ravel())

            rglru_base = rglru_soa_byte_offset(3, blk)
            theta3 = self.rglru_theta0_rad[t3_src, blk]
            u_rot_init3 = np.cos(theta3).astype(np.float32)
            rglru_soa3 = np.concatenate(
                [
                    self.rglru_r_low[t3_src, blk],
                    self.rglru_dr_span[t3_src, blk],
                    theta3,
                    self.rglru_b_gamma_norm[t3_src, blk],
                    self.rglru_w_sel[t3_src, blk],
                    self.rglru_c_real[t3_src, blk],
                    self.rglru_c_imag[t3_src, blk],
                    u_rot_init3,
                ]
            )
            write_f32_slice(rglru_base, rglru_soa3)

        # 3. Stereo Kronecker output head W_head across Arena 0 (4224), Arena 1 (17984), Arena 2
        # (32640), Arena 3 (49600)
        head_offsets = (4224, 17984, 32640, 49600)
        num_head_tiers = min(len(head_offsets), int(self.w_head_tiers.shape[0]))
        for tier in range(num_head_tiers):
            write_f32_slice(head_offsets[tier], self.w_head_tiers[tier].ravel())

        # 4. M-UFDN base delays & WT-UFDN grain weights (Arena 0: 6784..7040, Arena 1: 20544..20800,
        # Arena 2: 35200, Arena 3: 62400)
        write_f32_slice(6784, self.mufdn_delays_8[:4])
        write_f32_slice(6784 + 208, self.mufdn_grain_len_8[:4])
        write_f32_slice(6784 + 224, self.mufdn_grain_warp_8[:4])
        write_f32_slice(6784 + 240, self.mufdn_grain_meta_0[:4])
        write_f32_slice(20544, self.mufdn_delays_8[4:8])
        write_f32_slice(20544 + 208, self.mufdn_grain_len_8[4:8])
        write_f32_slice(20544 + 224, self.mufdn_grain_warp_8[4:8])
        write_f32_slice(20544 + 240, self.mufdn_grain_meta_1[:4])
        write_f32_slice(35200, self.mufdn_delays_8[4:8])
        write_f32_slice(62400, self.mufdn_delays_ultra_8[:4])
        write_f32_slice(62656, self.mufdn_delays_ultra_8[4:8])

        # 5. P_svd (16x4 = 256 B) at 8960, W_ctrl_core2 (8x32 = 1024 B) at 9216,
        # Ultra SVD modes 5..8 head (8192 B) at 52160, P_svd_ultra (16x4 = 256 B) at 63168,
        # b_head_ultra (4x2 = 32 B) at 63424, and Ultra linear gain trim (4 B) at 63456
        write_f32_slice(8960, self.p_svd_16x4.ravel())
        write_f32_slice(9216, self.w_ctrl_core2_8x32.ravel())
        write_f32_slice(52160, self.w_head_ultra_modes5to8.ravel())
        write_f32_slice(63168, self.p_svd_ultra_16x4.ravel())
        write_f32_slice(63424, self.b_head_ultra_modes5to8.ravel())
        if len(trims_flat) >= 4:
            write_f32_slice(63456, trims_flat[3:4])

        return bytes(buf)

    @classmethod
    def from_arena012_bytes(cls, arena_bytes: bytes | memoryview) -> ArenaWeightLayout:
        """Unpacks an [`ArenaWeightLayout`] from a `64,000 B` (or `>= 36,224 B`) arena buffer."""
        raw = memoryview(arena_bytes)
        if len(raw) < ARENA_OFFSET_PRO_END:
            raise ValueError(
                f"Arena buffer too small: expected >= {ARENA_OFFSET_PRO_END} B, got {len(raw)} B"
            )

        def read_f32_slice(byte_offset: int, count: int) -> NDArray[np.float32]:
            return np.frombuffer(raw[byte_offset : byte_offset + count * 4], dtype="<f4").copy()

        layout = cls()
        layout.b_head = read_f32_slice(64, 10).reshape((5, 2))
        trims_012 = read_f32_slice(104, 3)
        for t_idx in range(3):
            if float(trims_012[t_idx]) > 1e-4:
                layout.tier_gain_trim_lin[t_idx] = trims_012[t_idx]

        conv_list: list[NDArray[np.float32]] = []
        conv_counts_per_sublayer = (96, 192, 288)
        poly0_offsets = (384, 768, 1152)
        conv1_offsets = (448, 832, 1216)
        poly1_offsets = (832, 1600, 2368)

        for tier in range(3):
            n_conv = conv_counts_per_sublayer[tier]
            for blk in range(4):
                base = block_base_byte_offset(tier, blk)
                conv_list.append(read_f32_slice(base, n_conv))
                layout.poly_adaa[tier, blk, 0] = read_f32_slice(
                    base + poly0_offsets[tier], 16
                ).reshape((4, 4))
                conv_list.append(read_f32_slice(base + conv1_offsets[tier], n_conv))
                layout.poly_adaa[tier, blk, 1] = read_f32_slice(
                    base + poly1_offsets[tier], 16
                ).reshape((4, 4))

                rglru_base = rglru_soa_byte_offset(tier, blk)
                soa = read_f32_slice(rglru_base, 32).reshape((8, 4))
                layout.rglru_r_low[tier, blk] = soa[0]
                layout.rglru_dr_span[tier, blk] = soa[1]
                layout.rglru_theta0_rad[tier, blk] = soa[2]
                layout.rglru_b_gamma_norm[tier, blk] = soa[3]
                layout.rglru_w_sel[tier, blk] = soa[4]
                layout.rglru_c_real[tier, blk] = soa[5]
                layout.rglru_c_imag[tier, blk] = soa[6]

        layout.conv_weights_4608 = np.concatenate(conv_list)

        if len(raw) >= ARENA_OFFSET_ULTRA_END:
            conv_u_list: list[NDArray[np.float32]] = []
            for blk in range(4):
                base = block_base_byte_offset(3, blk)
                conv_u_list.append(read_f32_slice(base, 384))
                layout.poly_adaa[3, blk, 0] = read_f32_slice(base + 1536, 16).reshape((4, 4))
                conv_u_list.append(read_f32_slice(base + 1600, 384))
                layout.poly_adaa[3, blk, 1] = read_f32_slice(base + 3136, 16).reshape((4, 4))

                rglru_base = rglru_soa_byte_offset(3, blk)
                soa3 = read_f32_slice(rglru_base, 32).reshape((8, 4))
                layout.rglru_r_low[3, blk] = soa3[0]
                layout.rglru_dr_span[3, blk] = soa3[1]
                layout.rglru_theta0_rad[3, blk] = soa3[2]
                layout.rglru_b_gamma_norm[3, blk] = soa3[3]
                layout.rglru_w_sel[3, blk] = soa3[4]
                layout.rglru_c_real[3, blk] = soa3[5]
                layout.rglru_c_imag[3, blk] = soa3[6]
            layout.conv_weights_ultra_3072 = np.concatenate(conv_u_list)
            layout.w_head_tiers[3] = read_f32_slice(49600, 640).reshape((5, 2, 64))
            layout.w_head_ultra_modes5to8 = read_f32_slice(52160, 2048).reshape((4, 4, 2, 64))
            layout.mufdn_delays_ultra_8[:4] = read_f32_slice(62400, 4)
            layout.mufdn_delays_ultra_8[4:8] = read_f32_slice(62656, 4)
            layout.p_svd_ultra_16x4 = read_f32_slice(63168, 64).reshape((16, 4))
            layout.b_head_ultra_modes5to8 = read_f32_slice(63424, 8).reshape((4, 2))
            trim_u = float(read_f32_slice(63456, 1)[0])
            if trim_u > 1e-4:
                layout.tier_gain_trim_lin[3] = np.float32(trim_u)

        head_offsets = (4224, 17984, 32640)
        for tier, off in enumerate(head_offsets):
            layout.w_head_tiers[tier] = read_f32_slice(off, 640).reshape((5, 2, 64))

        layout.mufdn_delays_8[:4] = read_f32_slice(6784, 4)
        layout.mufdn_grain_len_8[:4] = read_f32_slice(6784 + 208, 4)
        layout.mufdn_grain_warp_8[:4] = read_f32_slice(6784 + 224, 4)
        layout.mufdn_grain_meta_0[:4] = read_f32_slice(6784 + 240, 4)
        layout.mufdn_delays_8[4:8] = read_f32_slice(20544, 4)
        layout.mufdn_grain_len_8[4:8] = read_f32_slice(20544 + 208, 4)
        layout.mufdn_grain_warp_8[4:8] = read_f32_slice(20544 + 224, 4)
        layout.mufdn_grain_meta_1[:4] = read_f32_slice(20544 + 240, 4)
        layout.p_svd_16x4 = read_f32_slice(8960, 64).reshape((16, 4))
        layout.w_ctrl_core2_8x32 = read_f32_slice(9216, 256).reshape((8, 32))
        return layout

    def with_kronecker_solution(
        self,
        head_sol: MultiTierKroneckerHeadSolution,
        p_svd_16x4: NDArray[np.float32],
        p_svd_ultra_16x4: NDArray[np.float32] | None = None,
    ) -> ArenaWeightLayout:
        """Updates `w_head_tiers`, `w_head_ultra_modes5to8`, `p_svd_16x4`, and `p_svd_ultra_16x4`
        in-place from a [`MultiTierKroneckerHeadSolution`]."""
        self.w_head_tiers[0] = head_sol.arena0_head_5x2x64_f32
        self.w_head_tiers[1] = head_sol.arena1_head_delta_5x2x64_f32
        self.w_head_tiers[2] = head_sol.arena2_head_delta_5x2x64_f32
        if self.w_head_tiers.shape[0] >= 4:
            self.w_head_tiers[3] = head_sol.arena3_head_delta_5x2x64_f32
        self.w_head_ultra_modes5to8 = head_sol.arena3_ultra_modes5to8_4x4x2x64_f32
        self.p_svd_16x4 = np.ascontiguousarray(p_svd_16x4, dtype=np.float32).reshape((16, 4))
        if p_svd_ultra_16x4 is not None:
            self.p_svd_ultra_16x4 = np.ascontiguousarray(
                p_svd_ultra_16x4, dtype=np.float32
            ).reshape((16, 4))
        return self


def _default_poly_adaa_weights() -> NDArray[np.float32]:
    arr = np.zeros((4, 4, 2, 4, 4), dtype=np.float32)
    arr[:, :, :, 1, :] = 1.0  # alpha = 1.0 linear bypass
    return arr


def default_tensor_directory() -> dict[str, TensorSliceEntry]:
    """Returns the canonical 64-byte aligned `tensor_directory` for `Arena 0..3`."""
    return {
        "arena0_lite": TensorSliceEntry(
            arena=0,
            dtype="f32",
            byte_offset=0,
            byte_length=ARENA_OFFSET_LITE_END,
        ),
        "arena1_full_delta": TensorSliceEntry(
            arena=1,
            dtype="f32",
            byte_offset=ARENA_OFFSET_LITE_END,
            byte_length=ARENA_OFFSET_FULL_END - ARENA_OFFSET_LITE_END,
        ),
        "arena2_pro_delta": TensorSliceEntry(
            arena=2,
            dtype="f32",
            byte_offset=ARENA_OFFSET_FULL_END,
            byte_length=ARENA_OFFSET_PRO_END - ARENA_OFFSET_FULL_END,
        ),
        "arena3_ultra_delta": TensorSliceEntry(
            arena=3,
            dtype="f32",
            byte_offset=ARENA_OFFSET_PRO_END,
            byte_length=ARENA_OFFSET_ULTRA_END - ARENA_OFFSET_PRO_END,
        ),
    }


def sign_header_with_creator_key(
    header: DnamJsonHeader,
    arena012_bytes: bytes,
    *,
    creator_seed_hex: str,
    signer_id: str,
) -> DnamJsonHeader:
    """
    Computes `immutable_header_sha256`, `master_arena_sha256`, and `tier_arena_sha256`
    and attaches `ModelSignatures`.
    """
    if len(arena012_bytes) != ARENA_OFFSET_ULTRA_END:
        raise ValueError(
            f"arena012_bytes must be exactly {ARENA_OFFSET_ULTRA_END} B, got {len(arena012_bytes)}"
        )
    imm_sha_hex = provenance.immutable_header_sha256(header)
    arena_sha_hex = hashlib.sha256(arena012_bytes).hexdigest()
    tier_shas: tuple[str, str, str, str] = (
        hashlib.sha256(arena012_bytes[:ARENA_OFFSET_LITE_END]).hexdigest(),
        hashlib.sha256(arena012_bytes[:ARENA_OFFSET_FULL_END]).hexdigest(),
        hashlib.sha256(arena012_bytes[:ARENA_OFFSET_PRO_END]).hexdigest(),
        arena_sha_hex,
    )
    msg_64 = bytes.fromhex(imm_sha_hex) + bytes.fromhex(arena_sha_hex)

    seed_bytes = bytes.fromhex(creator_seed_hex)
    pubkey_hex = provenance.ed25519_pubkey_hex(seed_bytes)
    sig_hex = provenance.ed25519_sign_hex(seed_bytes, msg_64)

    sigs = ModelSignatures(
        immutable_header_sha256=imm_sha_hex,
        master_arena_sha256=arena_sha_hex,
        tier_arena_sha256=tier_shas,
        creator_signature=CreatorSignature(
            signer_id=signer_id,
            pubkey_ed25519=pubkey_hex,
            sig_ed25519=sig_hex,
        ),
        marketplace_receipt=None,
    )
    updated_meta = header.metadata.model_copy(update={"signatures": sigs})
    return header.model_copy(update={"metadata": updated_meta})


def build_signed_dnam_containers(
    base_header: DnamJsonHeader,
    arena_layout: ArenaWeightLayout,
    stem_stats: Sequence[StemSufficientStatistics],
    *,
    creator_seed_hex: str | None = None,
    signer_id: str = "dnam-creator",
    arena4_calib: Arena4CalibrationData | None = None,
) -> tuple[DnamContainer, DnamContainer]:
    """Builds both the deployment `.dnam` (`DNAM`) and single-checkpoint `.dnam-ckpt` (`DNCK`)
    containers.

    `Arena 0..3` (`64,000 B`) are bit-identical across `.dnam` and `.dnam-ckpt`.
    `.dnam-ckpt` appends `Arena 4` (`267,264 B/stem`) containing `uplo='U'` `float64`
    audio statistics, and optional `Arena 5` (`DNCA` 64-byte aligned calibration stats).
    """
    arena_buf = bytearray(arena_layout.to_arena012_bytes())
    if np.allclose(arena_layout.tier_gain_trim_lin, 1.0):
        st = base_header.slimmable_tiers
        trims_lin = np.array(
            [
                10.0 ** (float(st.lite.output_trim_db) / 20.0),
                10.0 ** (float(st.full.output_trim_db) / 20.0),
                10.0 ** (float(st.pro.output_trim_db) / 20.0),
                10.0 ** (float(st.ultra.output_trim_db) / 20.0),
            ],
            dtype="<f4",
        )
        arena_buf[104:116] = trims_lin[:3].tobytes()
        arena_buf[63456:63460] = trims_lin[3:4].tobytes()
    arena012 = bytes(arena_buf)
    t_dir = default_tensor_directory()
    unsigned_meta = base_header.metadata.model_copy(update={"signatures": None})

    # 1. Build deployment .dnam header & container (first pass populates binary clamp descriptors &
    # prefix CRCs)
    deploy_hdr = base_header.model_copy(
        update={
            "version": "1.0.0",
            "container_kind": "dnam",
            "tensor_directory": t_dir,
            "stems_registry": None,
            "metadata": unsigned_meta,
        }
    )
    dnam_unsigned = DnamContainer.build(deploy_hdr, arena012)
    final_arena012 = bytes(dnam_unsigned.arena)

    if creator_seed_hex is not None:
        signed_deploy_hdr = sign_header_with_creator_key(
            dnam_unsigned.header,
            final_arena012,
            creator_seed_hex=creator_seed_hex,
            signer_id=signer_id,
        )
        dnam_container = DnamContainer.build(signed_deploy_hdr, final_arena012)
    else:
        dnam_container = dnam_unsigned

    # 2. Build .dnam-ckpt header & Arena 0..3(+4) payload using bit-identical final_arena012
    registry_entries: list[StemRegistryEntry] = []
    arena3_chunks: list[bytes] = [final_arena012]
    for idx, stat in enumerate(stem_stats):
        entry, payload = stat.to_registry_entry(idx)
        registry_entries.append(entry)
        arena3_chunks.append(payload)

    arena0123 = b"".join(arena3_chunks)
    assert len(arena0123) == ARENA_OFFSET_ULTRA_END + len(stem_stats) * ARENA3_STEM_BYTE_LEN
    arena01234 = arena0123 + arena4_calib.to_bytes() if arena4_calib is not None else arena0123

    ckpt_hdr = dnam_unsigned.header.model_copy(
        update={
            "version": "1.1.0" if arena4_calib is not None else dnam_unsigned.header.version,
            "container_kind": "dnam-ckpt",
            "tensor_directory": t_dir,
            "stems_registry": registry_entries,
            "metadata": unsigned_meta,
        }
    )
    ckpt_unsigned = DnamContainer.build(
        ckpt_hdr,
        arena01234,
        plaintext_arena_crc_source=final_arena012,
    )
    if creator_seed_hex is not None:
        signed_ckpt_hdr = sign_header_with_creator_key(
            ckpt_unsigned.header,
            final_arena012,
            creator_seed_hex=creator_seed_hex,
            signer_id=signer_id,
        )
        ckpt_container = DnamContainer.build(
            signed_ckpt_hdr,
            arena01234,
            plaintext_arena_crc_source=final_arena012,
        )
    else:
        ckpt_container = ckpt_unsigned
    return dnam_container, ckpt_container
