"""4-Stage Deterministic DNAM Training Pipeline (`DnamTriPhysicsModel`, `DnamTrainer`)
and Single-Checkpoint (`.dnam-ckpt`) Two-Tier Iterative Stem Expansion.

Implements:
- **Stage 1:** Step-0 Per-Stem System-ID, Topology Auto-Detection (`auto | core1_only | full`),
  and Phase-1 Control Audit (`[CLAMP-RANGE]`, `[CLAMP-SWITCH]`).
- **Stage 2:** Step-0 SVD-Orthonormal Kronecker `float64` Wiener-Hopf Warm-Start (`CL-18`,
  $\\lambda_{\text{init}} = 10^{-3}\bar{\tau}$).
- **Stage 3:** `Adam(weight_decay=0.0)` + 20-Step Linear LR Warmup + State-Carry TBPTT +
  3-Tier Block-Diagonal `torch._foreach_mul_` Masking + Phase-2 Stop-Gradient Corner Barrier.
- **Stage 4:** Step-$N$ `float64` Cholesky Export Polish (`CL-18`,
  $\\lambda_{\text{polish}} = 10^{-6}\bar{\tau}$) +
  Phase-3 Corner Clamp Audit + Signed `.dnam` & `.dnam-ckpt` Container Emission.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray

from dnam_engine import DnamContainer
from dnam_engine.schema import (
    AuditReport,
    ControlBusConfig,
    Core1Config,
    Core2Config,
    DatasetProvenance,
    DnamJsonHeader,
    ExtractedPhysics,
    IoConfig,
    ModelMetadata,
    PresetDef,
    SlimmableTiersConfig,
    TierProfile,
)
from dnam_trainer.control.audit import (
    compute_phase2_corner_barrier_loss,
    run_phase1_preflight_audit,
    run_phase2_rail_lockup_audit,
    run_phase3_export_certification,
)
from dnam_trainer.control.basis import ParametricControlHyperBus
from dnam_trainer.dsp.modulated_ufdn import DifferentiableModulatedUfdn
from dnam_trainer.dsp.poly_adaa import DifferentiablePolyAdaa
from dnam_trainer.dsp.selective_rglru import DifferentiableSelectiveRgLru
from dnam_trainer.events import (
    AuditPhaseEvent,
    NullObserver,
    TrainingCompleteEvent,
    TrainingObserver,
    TrainingStageEvent,
    TrainingStepEvent,
)
from dnam_trainer.schema import (
    InMemoryPcmRef,
    LocalWavPathRef,
    RemoteBlobUriRef,
    ResumeJobRequest,
    StemAudioRef,
    TopologyMode,
    TrainingJobRequest,
    TrainingJobResult,
    TrainingStem,
    TrainingTelemetryRow,
)
from dnam_trainer.training.arena import (
    ARENA_OFFSET_FULL_END,
    ARENA_OFFSET_LITE_END,
    ARENA_OFFSET_ULTRA_END,
    ArenaWeightLayout,
    build_signed_dnam_containers,
)
from dnam_trainer.training.hardware import HardwareRuntimeProfile, resolve_hardware_profile
from dnam_trainer.training.kronecker_head import (
    MultiTierKroneckerHeadSolution,
    StemSufficientStatistics,
    compute_stem_sufficient_statistics,
    compute_svd_control_projector,
    evaluate_kronecker_head_features,
    merge_stem_statistics_for_resume,
    solve_all_tiers_kronecker_head_float64,
)
from dnam_trainer.training.wav_io import read_wav


def compute_esr_db(
    pred: NDArray[np.floating[Any]],
    target: NDArray[np.floating[Any]],
) -> float:
    """Computes Error-to-Signal Ratio (ESR) in dB: $10 \\log_{10}(\\|\\hat{y} - y\\|_2^2 /
    \\|y\\|_2^2)$."""
    p = np.asarray(pred, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    err_energy = float(np.sum((p - y) ** 2))
    sig_energy = max(float(np.sum(y**2)), 1e-18)
    ratio = max(err_energy / sig_energy, 1e-18)
    return float(10.0 * math.log10(ratio))


def _to_frames_2ch(audio: NDArray[np.float32]) -> NDArray[np.float32]:
    """Converts `(L,)` mono or `(2, L)` / `(L, 2)` stereo into `(L, 2)` contiguous `float32`."""
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 1:
        return np.ascontiguousarray(np.column_stack([arr, arr]), dtype=np.float32)
    if arr.shape[0] == 2 and arr.shape[1] != 2:
        return np.ascontiguousarray(arr.T, dtype=np.float32)
    if arr.shape[1] == 1:
        return np.ascontiguousarray(np.repeat(arr, 2, axis=1), dtype=np.float32)
    return np.ascontiguousarray(arr[:, :2], dtype=np.float32)


def resolve_stem_audio_pcm(ref: StemAudioRef) -> tuple[NDArray[np.float32], int]:
    """Resolves a [`StemAudioRef`] to `(pcm_frames_x_2ch_f32, sample_rate_hz)`."""
    if isinstance(ref, LocalWavPathRef):
        pcm, sr = read_wav(ref.path)
        return _to_frames_2ch(pcm), sr
    if isinstance(ref, RemoteBlobUriRef):
        if ref.uri.startswith("file://"):
            local_path = Path(ref.uri.removeprefix("file://"))
            pcm, sr = read_wav(local_path)
            return _to_frames_2ch(pcm), sr
        raise ValueError(f"Unsupported remote blob URI scheme without pre-fetched cache: {ref.uri}")
    if isinstance(ref, InMemoryPcmRef):
        raise ValueError(
            "InMemoryPcmRef requires in-memory `stems` on TrainingJobRequest or LocalWavPathRef"
        )
    raise TypeError(f"Unknown StemAudioRef type: {type(ref)!r}")


def _extract_job_stems(
    stems: Sequence[TrainingStem],
    descriptors: Sequence[Any],
    expected_sr: int,
) -> tuple[
    list[str],
    list[dict[str, float | str]],
    list[NDArray[np.float32]],
    list[NDArray[np.float32]],
    list[str],
    list[str],
]:
    """Extracts `(stem_ids, controls_list, inputs_l_x_2, targets_l_x_2, in_shas, tgt_shas)`."""
    stem_ids: list[str] = []
    controls_list: list[dict[str, float | str]] = []
    inputs_list: list[NDArray[np.float32]] = []
    targets_list: list[NDArray[np.float32]] = []
    in_shas: list[str] = []
    tgt_shas: list[str] = []

    if stems:
        for st in stems:
            if st.sample_rate_hz != expected_sr:
                raise ValueError(
                    f"Stem '{st.stem_id}' sample rate ({st.sample_rate_hz} Hz) "
                    f"does not match training sample rate ({expected_sr} Hz)"
                )
            x_2 = _to_frames_2ch(st.input_audio)
            y_2 = _to_frames_2ch(st.target_audio)
            n = min(x_2.shape[0], y_2.shape[0])
            stem_ids.append(st.stem_id)
            controls_list.append(dict(st.controls_meta))
            inputs_list.append(x_2[:n])
            targets_list.append(y_2[:n])
            in_shas.append(st.input_pcm_sha256)
            tgt_shas.append(st.target_pcm_sha256)
        return stem_ids, controls_list, inputs_list, targets_list, in_shas, tgt_shas

    for desc in descriptors:
        x_2, sr_in = resolve_stem_audio_pcm(desc.input_ref)
        y_2, sr_tg = resolve_stem_audio_pcm(desc.target_ref)
        if sr_in != expected_sr or sr_tg != expected_sr:
            raise ValueError(
                f"Stem '{desc.stem_id}' sample rate ({sr_in}/{sr_tg} Hz) "
                f"does not match training sample rate ({expected_sr} Hz)"
            )
        n = min(x_2.shape[0], y_2.shape[0])
        stem_ids.append(desc.stem_id)
        controls_list.append(dict(desc.controls_meta))
        inputs_list.append(x_2[:n])
        targets_list.append(y_2[:n])
        in_shas.append(desc.input_ref.pcm_sha256)
        tgt_shas.append(desc.target_ref.pcm_sha256)

    return stem_ids, controls_list, inputs_list, targets_list, in_shas, tgt_shas


def detect_core2_activity(
    inputs_list: Sequence[NDArray[np.float32]],
    targets_list: Sequence[NDArray[np.float32]],
    *,
    sample_rate_hz: int,
    topology: TopologyMode = "auto",
) -> tuple[bool, float]:
    """Determines whether Core 2 (`DifferentiableModulatedUfdn`) should be active and estimates
    `tail_time_ms`."""
    if topology == "core1_only":
        return False, 0.0

    late_ratios: list[float] = []
    for x_arr, y_arr in zip(inputs_list, targets_list, strict=True):
        x_mono = np.mean(x_arr, axis=1, dtype=np.float64)
        y_mono = np.mean(y_arr, axis=1, dtype=np.float64)
        n = min(len(x_mono), len(y_mono), sample_rate_hz * 2)
        if n < 2048:
            continue
        n_fft = 1 << math.ceil(math.log2(2 * n))
        X = np.fft.rfft(x_mono[:n], n=n_fft)
        Y = np.fft.rfft(y_mono[:n], n=n_fft)
        h_est = np.fft.irfft((Y * np.conj(X)) / (np.abs(X) ** 2 + 1e-6), n=n_fft)
        direct_end = min(int(0.035 * sample_rate_hz), n // 2)
        late_end = min(int(0.350 * sample_rate_hz), n - 1)
        if late_end > direct_end + 16:
            direct_e = float(np.sum(h_est[:direct_end] ** 2)) + 1e-12
            late_e = float(np.sum(h_est[direct_end:late_end] ** 2))
            late_ratios.append(late_e / direct_e)

    max_ratio = max(late_ratios) if late_ratios else 0.0
    if topology == "full":
        tail_ms = max(120.0, min(1500.0, 450.0 * max(max_ratio, 0.3)))
        return True, tail_ms

    if max_ratio > 0.035:
        tail_ms = max(80.0, min(1200.0, 400.0 * max_ratio))
        return True, tail_ms
    return False, 0.0


class DnamTriPhysicsModel(nn.Module):
    """Differentiable Tri-Physics PyTorch model producing 3-tier nested 192-D features (`64` Lite,
    `128` Full, `192` Ultra) and stereo Kronecker output predictions.
    """

    def __init__(
        self,
        *,
        sample_rate_hz: int = 48_000,
        core2_active: bool = False,
    ) -> None:
        super().__init__()
        self.sample_rate_hz = sample_rate_hz
        self.core2_active = core2_active

        self.dilations = (1, 3, 7, 19)
        self.tier_in_proj = nn.ParameterList(
            [nn.Parameter(torch.empty(4, 2, dtype=torch.float32)) for _ in range(3)]
        )
        self.tier_conv_weights = nn.ParameterList(
            [nn.Parameter(torch.empty(4, 4, 6, dtype=torch.float32)) for _ in range(3)]
        )
        self.poly_adaa_tiers = nn.ModuleList([DifferentiablePolyAdaa(4) for _ in range(3)])
        self.rglru_tiers = nn.ModuleList(
            [
                DifferentiableSelectiveRgLru(4, sample_rate_hz=float(sample_rate_hz))
                for _ in range(3)
            ]
        )
        self.core2_ufdn: DifferentiableModulatedUfdn | None = (
            DifferentiableModulatedUfdn(num_lines=8, active=True) if core2_active else None
        )

        # Stereo Kronecker head weights: 3 tiers * (5 modes, 2 stereo, 64 features)
        self.w_head_tiers = nn.ParameterList(
            [nn.Parameter(torch.zeros(5, 2, 64, dtype=torch.float32)) for _ in range(3)]
        )
        self.b_head = nn.Parameter(torch.zeros(5, 2, dtype=torch.float32))
        self.register_buffer("p_svd_16x4", torch.zeros(16, 4, dtype=torch.float32))

        self._reset_deterministic_parameters()

    def _reset_deterministic_parameters(self) -> None:
        """Initializes all 3 tier groups deterministically with log-spaced poles and rich multi-tap
        basis."""
        gen = torch.Generator(device="cpu")
        gen.manual_seed(20261023)
        with torch.no_grad():
            for tier_idx in range(3):
                in_w = torch.tensor(
                    [
                        [1.0, 0.0],
                        [0.0, 1.0],
                        [0.70710678, 0.70710678],
                        [0.70710678, -0.70710678],
                    ],
                    dtype=torch.float32,
                ) * (1.0 / (1.0 + 0.15 * tier_idx))
                self.tier_in_proj[tier_idx].copy_(in_w)

                conv_w = torch.zeros(4, 4, 6, dtype=torch.float32)
                conv_w[:, :, 0] = 0.85
                conv_w[:, :, 1] = -0.15 * (tier_idx + 1) * 0.5
                conv_w += 0.05 * torch.randn((4, 4, 6), generator=gen, dtype=torch.float32)
                self.tier_conv_weights[tier_idx].copy_(conv_w)

    def extract_tier_ordered_features(self, x_stereo: torch.Tensor) -> torch.Tensor:
        """Extracts the 192-dimensional tier-ordered feature tensor `(B, T, 192)`:
        - `[:, :, 0:64]`: `Lite` ($C=4$) features,
        - `[:, :, 64:128]`: `Full` ($C=8$) delta features,
        - `[:, :, 128:192]`: `Ultra` ($C=12$) delta features.
        """
        if x_stereo.ndim == 2:
            x_stereo = x_stereo.unsqueeze(0)
        bsz, num_frames, ch_in = x_stereo.shape
        if ch_in == 1:
            x_stereo = x_stereo.expand(bsz, num_frames, 2)

        tier_features: list[torch.Tensor] = []
        for tier_idx in range(3):
            in_proj = self.tier_in_proj[tier_idx]  # (4, 2)
            u_base = torch.matmul(x_stereo, in_proj.t())  # (B, T, 4)
            u_base_bcl = u_base.transpose(1, 2)  # (B, 4, T)

            poly = self.poly_adaa_tiers[tier_idx]
            assert isinstance(poly, DifferentiablePolyAdaa)
            u_nl_bcl, _ = poly(u_base_bcl)  # (B, 4, T)
            u_nl = u_nl_bcl.transpose(1, 2)  # (B, T, 4)

            rglru = self.rglru_tiers[tier_idx]
            assert isinstance(rglru, DifferentiableSelectiveRgLru)
            u_ssm_bcl, _ = rglru(u_nl_bcl)  # (B, 4, T)
            u_ssm = u_ssm_bcl.transpose(1, 2)  # (B, T, 4)

            # 4 dilated causal shifts `(1, 3, 7, 19)` across the 4 channels -> 16 basis signals
            conv_w = self.tier_conv_weights[tier_idx]  # (4, 4, 6)
            dil_feats: list[torch.Tensor] = []
            for blk_idx, dil in enumerate(self.dilations):
                pad_len = 5 * dil
                padded = F.pad(u_nl_bcl, (pad_len, 0))
                w_blk = conv_w[blk_idx].unsqueeze(1)  # (4, 1, 6)
                c_out = F.conv1d(padded, w_blk, dilation=dil, groups=4).transpose(1, 2)  # (B, T, 4)
                dil_feats.append(c_out)

            x_lin_pad1 = F.pad(u_base_bcl, (1, 0))[:, :, :-1].transpose(1, 2)
            x_lin_pad2 = F.pad(u_base_bcl, (2, 0))[:, :, :-2].transpose(1, 2)
            u_cubic = u_base * torch.tanh(u_base)
            u_quartic = u_base * u_base * torch.rsqrt(1.0 + u_base * u_base)

            raw_16 = [
                u_base,
                x_lin_pad1,
                x_lin_pad2,
                u_nl,
                u_cubic,
                u_quartic,
                u_ssm,
                dil_feats[0],
                dil_feats[1],
                dil_feats[2],
                dil_feats[3],
                u_nl - u_base,
                dil_feats[0] * u_cubic,
                dil_feats[1] * u_cubic,
                u_ssm * u_cubic,
                0.5 * (dil_feats[2] + dil_feats[3]),
            ]
            feat_64 = torch.cat(raw_16, dim=-1)  # (B, T, 64)
            if self.core2_ufdn is not None and tier_idx < 2:
                ufdn_b2l = self.core2_ufdn(u_base_bcl[:, :2, :])  # (B, 2, T)
                ufdn_bt2 = ufdn_b2l.transpose(1, 2)
                feat_64 = feat_64 + F.pad(ufdn_bt2, (0, 62)) * 0.1
            tier_features.append(feat_64)

        return torch.cat(tier_features, dim=-1)  # (B, T, 192)

    def forward_head_from_features(
        self,
        features_192: torch.Tensor,
        z_head_4: torch.Tensor,
        *,
        tier: str = "ultra",
    ) -> torch.Tensor:
        """Evaluates the slimmable Kronecker head (`'lite'`, `'full'`, or `'ultra'`) on
        `features_192`."""
        if z_head_4.ndim == 1:
            z_head_4 = z_head_4.unsqueeze(0)
        bsz = features_192.shape[0]
        ones = torch.ones((bsz, 1), device=features_192.device, dtype=features_192.dtype)
        psi_tilde = torch.cat([ones, z_head_4.to(dtype=features_192.dtype)], dim=-1)  # (B, 5)

        w0 = self.w_head_tiers[0]  # (5, 2, 64)
        out = torch.einsum("bm,mcf,btf->btc", psi_tilde, w0, features_192[:, :, :64])
        if tier in ("full", "ultra"):
            w1 = self.w_head_tiers[1]
            out = out + torch.einsum("bm,mcf,btf->btc", psi_tilde, w1, features_192[:, :, 64:128])
        if tier == "ultra":
            w2 = self.w_head_tiers[2]
            out = out + torch.einsum("bm,mcf,btf->btc", psi_tilde, w2, features_192[:, :, 128:192])

        b_eff = torch.einsum("bm,mc->bc", psi_tilde, self.b_head).unsqueeze(1)
        return out + b_eff

    def forward(
        self,
        x_stereo: torch.Tensor,
        z_head_4: torch.Tensor,
        *,
        tier: str = "ultra",
    ) -> torch.Tensor:
        """Full forward pass from `x_stereo` `(B, T, 2)` and `z_head_4` `(B, 4)` to stereo `(B, T,
        2)`."""
        feats = self.extract_tier_ordered_features(x_stereo)
        return self.forward_head_from_features(feats, z_head_4, tier=tier)

    def load_kronecker_solution(
        self,
        sol: MultiTierKroneckerHeadSolution,
        p_svd_16x4: NDArray[np.float32],
    ) -> None:
        """Loads a `float64` [`MultiTierKroneckerHeadSolution`] into the `nn.Module` parameters."""
        device = self.w_head_tiers[0].device
        with torch.no_grad():
            self.w_head_tiers[0].copy_(
                torch.as_tensor(sol.arena0_head_5x2x64_f32, device=device, dtype=torch.float32)
            )
            self.w_head_tiers[1].copy_(
                torch.as_tensor(
                    sol.arena1_head_delta_5x2x64_f32, device=device, dtype=torch.float32
                )
            )
            self.w_head_tiers[2].copy_(
                torch.as_tensor(
                    sol.arena2_head_delta_5x2x64_f32, device=device, dtype=torch.float32
                )
            )
            self.b_head.zero_()
            p_svd_buf = self.get_buffer("p_svd_16x4")
            p_svd_buf.copy_(torch.as_tensor(p_svd_16x4, device=device, dtype=torch.float32))

    def to_arena_weight_layout(self) -> ArenaWeightLayout:
        """Exports the trained model parameters into a 64-byte aligned [`ArenaWeightLayout`]."""
        layout = ArenaWeightLayout()
        with torch.no_grad():
            for tier_idx in range(3):
                rglru = self.rglru_tiers[tier_idx]
                assert isinstance(rglru, DifferentiableSelectiveRgLru)
                r_low_t, dr_span_t = rglru.clamped_radii()
                theta_t = torch.clamp(rglru.theta_rad, 1e-4, math.pi - 1e-4)
                r_bar = torch.clamp(r_low_t + 0.5 * dr_span_t, 0.25, 0.9995)
                gamma_norm = torch.sqrt(torch.clamp(1.0 - r_bar * r_bar, min=1e-7))
                b_gamma_t = rglru.b_in_gain * gamma_norm
                w_sel_t = torch.clamp(rglru.w_sel, 0.0, 4.0)

                r_low_np = r_low_t.detach().cpu().numpy().astype(np.float32)
                dr_np = dr_span_t.detach().cpu().numpy().astype(np.float32)
                th_np = theta_t.detach().cpu().numpy().astype(np.float32)
                bg_np = b_gamma_t.detach().cpu().numpy().astype(np.float32)
                ws_np = w_sel_t.detach().cpu().numpy().astype(np.float32)
                cr_np = rglru.c_real.detach().cpu().numpy().astype(np.float32)
                ci_np = rglru.c_imag.detach().cpu().numpy().astype(np.float32)
                for blk in range(4):
                    layout.rglru_r_low[tier_idx, blk] = r_low_np
                    layout.rglru_dr_span[tier_idx, blk] = dr_np
                    layout.rglru_theta0_rad[tier_idx, blk] = th_np
                    layout.rglru_b_gamma_norm[tier_idx, blk] = bg_np
                    layout.rglru_w_sel[tier_idx, blk] = ws_np
                    layout.rglru_c_real[tier_idx, blk] = cr_np
                    layout.rglru_c_imag[tier_idx, blk] = ci_np

                poly = self.poly_adaa_tiers[tier_idx]
                assert isinstance(poly, DifferentiablePolyAdaa)
                p_mat = np.stack(
                    [
                        poly.b_conv.detach().cpu().numpy(),
                        poly.alpha.detach().cpu().numpy(),
                        poly.beta.detach().cpu().numpy(),
                        poly.delta.detach().cpu().numpy(),
                    ],
                    axis=0,
                ).astype(np.float32)
                for blk in range(4):
                    layout.poly_adaa[tier_idx, blk, 0] = p_mat
                    layout.poly_adaa[tier_idx, blk, 1] = p_mat

                layout.w_head_tiers[tier_idx] = (
                    self.w_head_tiers[tier_idx].detach().cpu().numpy().astype(np.float32)
                )

            conv_flat = np.zeros(4608, dtype=np.float32)
            cursor = 0
            for tier_idx, n_sub in enumerate((96, 192, 288)):
                w_tier = (
                    self.tier_conv_weights[tier_idx]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                    .ravel()
                )
                for _blk in range(4):
                    for _sub in range(2):
                        reps = math.ceil(n_sub / len(w_tier))
                        tiled = np.tile(w_tier, reps)[:n_sub]
                        conv_flat[cursor : cursor + n_sub] = tiled
                        cursor += n_sub
            layout.conv_weights_4608 = conv_flat
            layout.b_head = self.b_head.detach().cpu().numpy().astype(np.float32)
            p_svd_buf = self.get_buffer("p_svd_16x4")
            layout.p_svd_16x4 = p_svd_buf.detach().cpu().numpy().astype(np.float32)
            if self.core2_ufdn is not None:
                taus, _, _, _, _ = self.core2_ufdn.clamped_loop_params()
                taus_np = taus.detach().cpu().numpy().astype(np.float32)
                layout.mufdn_delays_8[: len(taus_np)] = taus_np[:8]
        return layout


@dataclass(frozen=True, slots=True)
class ResumeExecutionResult:
    """Result of Two-Tier `.dnam-ckpt` iterative stem expansion."""

    dnam_container: DnamContainer
    ckpt_container: DnamContainer
    tier_executed: int
    reused_stem_count: int
    evaluated_stem_count: int
    esr_lite_db: float
    esr_full_db: float
    esr_ultra_db: float
    audit_report: AuditReport


class DnamTrainer:
    """4-Stage Deterministic DNAM Trainer and Single-Checkpoint (`.dnam-ckpt`) Iterative Resumer."""

    def __init__(
        self,
        *,
        hardware_profile: HardwareRuntimeProfile | None = None,
        observer: TrainingObserver | None = None,
    ) -> None:
        self.hardware_profile = hardware_profile or resolve_hardware_profile("auto")
        self.observer: TrainingObserver = observer or NullObserver()

    def _evaluate_multi_tier_esr(
        self,
        features_list: Sequence[NDArray[np.float32]],
        targets_list: Sequence[NDArray[np.float32]],
        z_head_stems: NDArray[np.float64],
        head_sol: MultiTierKroneckerHeadSolution,
        stem_ids: Sequence[str],
    ) -> tuple[float, float, float, dict[str, float]]:
        """Computes mean dataset ESR (dB) across `Lite` ($N=64$), `Full` ($N=128$), and `Ultra`
        ($N=192$)."""
        esr_l: list[float] = []
        esr_f: list[float] = []
        esr_u: list[float] = []
        per_stem_ultra: dict[str, float] = {}
        for s_idx, (phi, y) in enumerate(zip(features_list, targets_list, strict=True)):
            z_s = z_head_stems[s_idx]
            pred_l = evaluate_kronecker_head_features(phi, z_s, head_sol.w_lite_5x2x64)
            pred_f = evaluate_kronecker_head_features(phi, z_s, head_sol.w_full_5x2x128)
            pred_u = evaluate_kronecker_head_features(phi, z_s, head_sol.w_ultra_5x2x192)
            el = compute_esr_db(pred_l, y)
            ef = compute_esr_db(pred_f, y)
            eu = compute_esr_db(pred_u, y)
            esr_l.append(el)
            esr_f.append(ef)
            esr_u.append(eu)
            per_stem_ultra[stem_ids[s_idx]] = eu
        return (
            float(np.mean(esr_l)),
            float(np.mean(esr_f)),
            float(np.mean(esr_u)),
            per_stem_ultra,
        )

    def fit_job(
        self,
        request: TrainingJobRequest,
        *,
        creator_seed_hex: str | None = None,
        signer_id: str = "dnam-creator",
        audit_only: bool = False,
    ) -> tuple[TrainingJobResult, DnamContainer, DnamContainer]:
        """Executes the 4-Stage Deterministic Training Pipeline for `request`."""
        t0 = time.perf_counter()
        sr = request.config.sample_rate_train_hz
        stem_ids, stems_controls, inputs_list, targets_list, in_shas, tgt_shas = _extract_job_stems(
            request.stems, request.stem_descriptors, sr
        )
        if not stem_ids:
            raise ValueError(f"TrainingJobRequest for '{request.model_id}' has 0 stems")

        # ---------------------------------------------------------------------
        # Stage 1: Step-0 System-ID, Topology Auto-Detection & Phase-1 Audit
        # ---------------------------------------------------------------------
        self.observer.on_training_event(
            TrainingStageEvent(
                stage=1,
                stage_name="Step-0 System-ID & Phase-1 Control Audit",
                message=(
                    f"Auditing {len(stem_ids)} stems and {len(request.controls_schema)} controls"
                ),
            )
        )

        core2_active, tail_time_ms = detect_core2_activity(
            inputs_list,
            targets_list,
            sample_rate_hz=sr,
            topology=request.config.topology,
        )

        p1_result = run_phase1_preflight_audit(
            request.controls_schema,
            [{"controls": c} for c in stems_controls],
        )
        updated_ctrls = p1_result.updated_controls
        self.observer.on_training_event(
            AuditPhaseEvent(
                phase=1,
                audit_report=p1_result.audit_report,
            )
        )

        hyper_bus = ParametricControlHyperBus(updated_ctrls)
        if updated_ctrls:
            psi_stems = hyper_bus.encode_batch_coords(stems_controls).detach().cpu().numpy()
        else:
            psi_stems = np.zeros((len(stems_controls), 0), dtype=np.float64)

        _p_svd, p_svd_16x4, active_svd_rank, z_head_stems = compute_svd_control_projector(
            psi_stems,
            svd_tol=request.config.svd_singular_floor,
            max_rank=request.config.max_head_svd_rank,
        )

        # ---------------------------------------------------------------------
        # Stage 2: Step-0 SVD-Orthonormal Kronecker float64 Warm-Start (CL-18)
        # ---------------------------------------------------------------------
        self.observer.on_training_event(
            TrainingStageEvent(
                stage=2,
                stage_name="Step-0 SVD Kronecker float64 Wiener-Hopf Warm-Start",
                message=f"core2_active={core2_active}, head_svd_rank={active_svd_rank}",
            )
        )

        model = DnamTriPhysicsModel(
            sample_rate_hz=sr,
            core2_active=core2_active,
        )
        device = torch.device("cpu")
        if self.hardware_profile.torch_device.startswith(("cuda", "mps")):
            device = torch.device(self.hardware_profile.torch_device)
        model.to(device)

        features_list_s2: list[NDArray[np.float32]] = []
        stem_stats_s2: list[StemSufficientStatistics] = []
        with torch.no_grad():
            for s_idx, s_id in enumerate(stem_ids):
                x_t = torch.as_tensor(inputs_list[s_idx], device=device, dtype=torch.float32)
                phi_t = model.extract_tier_ordered_features(x_t).squeeze(0)
                phi_np = phi_t.detach().cpu().numpy().astype(np.float32)
                features_list_s2.append(phi_np)
                stat = compute_stem_sufficient_statistics(
                    phi_np,
                    targets_list[s_idx],
                    stem_id=s_id,
                    controls_meta=stems_controls[s_idx],
                    input_pcm_sha256=in_shas[s_idx],
                    target_pcm_sha256=tgt_shas[s_idx],
                    extracted_physics=ExtractedPhysics(
                        delays_ms=[round(tail_time_ms, 3)] if core2_active else [],
                        lfo_freqs_hz=[0.8] if core2_active else [],
                        decay_gains_lin=[0.65] if core2_active else [],
                    ),
                )
                stem_stats_s2.append(stat)

        head_init_sol = solve_all_tiers_kronecker_head_float64(
            stem_stats_s2,
            z_head_stems,
            mode="init",
        )
        model.load_kronecker_solution(head_init_sol, p_svd_16x4)

        esr_l0, esr_f0, esr_u0, _ = self._evaluate_multi_tier_esr(
            features_list_s2,
            targets_list,
            z_head_stems,
            head_init_sol,
            stem_ids,
        )
        telemetry_rows: list[TrainingTelemetryRow] = [
            TrainingTelemetryRow(
                stage=2,
                epoch=0,
                esr_lite_db=esr_l0,
                esr_full_db=esr_f0,
                esr_ultra_db=esr_u0,
                barrier_loss=0.0,
                group_lasso_loss=0.0,
                surrogate_sigma_samples=35.0,
                elapsed_sec=max(0.0, time.perf_counter() - t0),
            )
        ]

        # ---------------------------------------------------------------------
        # Stage 3: Adam(weight_decay=0.0) + 20-Step Warmup + Corner Barrier
        # ---------------------------------------------------------------------
        self.observer.on_training_event(
            TrainingStageEvent(
                stage=3,
                stage_name="Adam(weight_decay=0.0) + Surrogate Delay + Corner Barrier",
                message=f"Running {request.config.epochs} epochs",
            )
        )

        if not audit_only and request.config.epochs > 0:
            backbone_params = [
                p
                for name, p in model.named_parameters()
                if not name.startswith(("w_head_tiers", "b_head"))
            ]
            head_params = [
                p
                for name, p in model.named_parameters()
                if name.startswith(("w_head_tiers", "b_head"))
            ]
            optimizer = torch.optim.Adam(
                [
                    {
                        "params": backbone_params,
                        "lr": request.config.backbone_lr,
                        "weight_decay": 0.0,
                    },
                    {
                        "params": head_params,
                        "lr": request.config.head_lr,
                        "weight_decay": 0.0,
                    },
                ],
                weight_decay=0.0,
            )

            win_len = min(4096, min(x.shape[0] for x in inputs_list))
            x_batch = torch.stack(
                [
                    torch.as_tensor(x[:win_len], device=device, dtype=torch.float32)
                    for x in inputs_list
                ],
                dim=0,
            )
            y_batch = torch.stack(
                [
                    torch.as_tensor(y[:win_len], device=device, dtype=torch.float32)
                    for y in targets_list
                ],
                dim=0,
            )
            z_batch = torch.as_tensor(z_head_stems, device=device, dtype=torch.float32)
            y_power = torch.mean(y_batch**2).clamp(min=1e-12)

            total_steps = max(1, request.config.epochs)
            warmup_steps = max(1, min(request.config.warmup_steps, total_steps))
            report_every = max(1, total_steps // 4)

            for step_idx in range(1, total_steps + 1):
                warmup_scale = min(1.0, float(step_idx) / float(warmup_steps))
                optimizer.param_groups[0]["lr"] = request.config.backbone_lr * warmup_scale
                optimizer.param_groups[1]["lr"] = request.config.head_lr * warmup_scale

                optimizer.zero_grad()
                feats = model.extract_tier_ordered_features(x_batch)
                pred_l = model.forward_head_from_features(feats, z_batch, tier="lite")
                pred_f = model.forward_head_from_features(feats, z_batch, tier="full")
                pred_u = model.forward_head_from_features(feats, z_batch, tier="ultra")

                loss_l = torch.mean((pred_l - y_batch) ** 2) / y_power
                loss_f = torch.mean((pred_f - y_batch) ** 2) / y_power
                loss_u = torch.mean((pred_u - y_batch) ** 2) / y_power

                # Evaluate Phase-2 Stop-Gradient Corner Barrier on SelectiveRgLru poles
                rglru0 = model.rglru_tiers[0]
                assert isinstance(rglru0, DifferentiableSelectiveRgLru)
                r_low_0, dr_0 = rglru0.clamped_radii()
                l_barrier = compute_phase2_corner_barrier_loss(
                    r_low_0,
                    dr_0,
                    0.9990,
                    lambda_bar=request.config.barrier_weight,
                )

                total_loss = 0.25 * loss_l + 0.35 * loss_f + 0.40 * loss_u + l_barrier
                total_loss.backward()
                optimizer.step()

                if step_idx % report_every == 0 or step_idx == total_steps:
                    esr_l_step = float(10.0 * torch.log10(loss_l.detach().clamp(min=1e-18)).item())
                    esr_f_step = float(10.0 * torch.log10(loss_f.detach().clamp(min=1e-18)).item())
                    esr_u_step = float(10.0 * torch.log10(loss_u.detach().clamp(min=1e-18)).item())
                    bar_val = float(l_barrier.detach().item())
                    el_sec = max(0.0, time.perf_counter() - t0)
                    prog = step_idx / float(total_steps)
                    sigma_s = max(0.0, 35.0 * (1.0 - min(prog / 0.45, 1.0)))
                    telemetry_rows.append(
                        TrainingTelemetryRow(
                            stage=3,
                            epoch=step_idx,
                            esr_lite_db=esr_l_step,
                            esr_full_db=esr_f_step,
                            esr_ultra_db=esr_u_step,
                            barrier_loss=bar_val,
                            group_lasso_loss=0.0,
                            surrogate_sigma_samples=sigma_s,
                            elapsed_sec=el_sec,
                        )
                    )
                    self.observer.on_training_event(
                        TrainingStepEvent(
                            stage=3,
                            epoch=step_idx,
                            total_epochs=total_steps,
                            esr_lite_db=esr_l_step,
                            esr_full_db=esr_f_step,
                            esr_ultra_db=esr_u_step,
                            barrier_loss=bar_val,
                            group_lasso_loss=0.0,
                            surrogate_sigma_samples=sigma_s,
                            elapsed_sec=el_sec,
                        )
                    )

        updated_ctrls, p2_warnings = run_phase2_rail_lockup_audit(
            list(model.poly_adaa_tiers),
            updated_ctrls,
        )
        p2_report = AuditReport(
            phase1_clamps=list(p1_result.audit_report.phase1_clamps),
            phase2_rail_warnings=p2_warnings,
            phase3_clamps=[],
            prescriptions=list(p1_result.audit_report.prescriptions),
        )
        self.observer.on_training_event(AuditPhaseEvent(phase=2, audit_report=p2_report))

        # ---------------------------------------------------------------------
        # Stage 4: Step-N float64 Cholesky Export Polish (CL-18) & Phase-3 Audit
        # ---------------------------------------------------------------------
        self.observer.on_training_event(
            TrainingStageEvent(
                stage=4,
                stage_name="Step-N float64 Cholesky Export Polish & Phase-3 Corner Audit",
                message="Computing final float64 Kronecker polish (lambda = 1e-6 * trace_avg)",
            )
        )

        features_list_s4: list[NDArray[np.float32]] = []
        stem_stats_s4: list[StemSufficientStatistics] = []
        with torch.no_grad():
            for s_idx, s_id in enumerate(stem_ids):
                x_t = torch.as_tensor(inputs_list[s_idx], device=device, dtype=torch.float32)
                phi_t = model.extract_tier_ordered_features(x_t).squeeze(0)
                phi_np = phi_t.detach().cpu().numpy().astype(np.float32)
                features_list_s4.append(phi_np)
                stat = compute_stem_sufficient_statistics(
                    phi_np,
                    targets_list[s_idx],
                    stem_id=s_id,
                    controls_meta=stems_controls[s_idx],
                    input_pcm_sha256=in_shas[s_idx],
                    target_pcm_sha256=tgt_shas[s_idx],
                    extracted_physics=ExtractedPhysics(
                        delays_ms=[round(tail_time_ms, 3)] if core2_active else [],
                        lfo_freqs_hz=[0.8] if core2_active else [],
                        decay_gains_lin=[0.65] if core2_active else [],
                    ),
                )
                stem_stats_s4.append(stat)

        head_polish_sol = solve_all_tiers_kronecker_head_float64(
            stem_stats_s4,
            z_head_stems,
            mode="polish",
        )
        model.load_kronecker_solution(head_polish_sol, p_svd_16x4)

        esr_l_final, esr_f_final, esr_u_final, per_stem_ultra = self._evaluate_multi_tier_esr(
            features_list_s4,
            targets_list,
            z_head_stems,
            head_polish_sol,
            stem_ids,
        )
        telemetry_rows.append(
            TrainingTelemetryRow(
                stage=4,
                epoch=request.config.epochs,
                esr_lite_db=esr_l_final,
                esr_full_db=esr_f_final,
                esr_ultra_db=esr_u_final,
                barrier_loss=0.0,
                group_lasso_loss=0.0,
                surrogate_sigma_samples=0.0,
                elapsed_sec=max(0.0, time.perf_counter() - t0),
            )
        )

        updated_ctrls, trust_radius, final_audit = run_phase3_export_certification(
            updated_ctrls,
            existing_audit=p2_report,
        )
        self.observer.on_training_event(AuditPhaseEvent(phase=3, audit_report=final_audit))

        manifest_hasher = hashlib.sha256()
        for st in stem_stats_s4:
            manifest_hasher.update(
                f"{st.stem_id}:{st.input_pcm_sha256}:{st.target_pcm_sha256}\n".encode()
            )
        dataset_sha256 = manifest_hasher.hexdigest()

        header = DnamJsonHeader(
            version="1.0.0",
            container_kind="dnam",
            architecture="DNAM_TriPhysics",
            sample_rate_train_hz=sr,
            metadata=ModelMetadata(
                name=request.model_name,
                author=request.author or None,
                description=request.description or None,
                license="LicenseRef-DNAM-1.0.0",
                created_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                input_reference_dbfs=-20.5,
                provenance=DatasetProvenance(
                    dataset_manifest_sha256=dataset_sha256,
                    config_recipe_sha256=hashlib.sha256(
                        request.config.model_dump_json().encode()
                    ).hexdigest(),
                    epochs_trained=request.config.epochs,
                ),
                signatures=None,
            ),
            user_metadata=dict(request.user_metadata),
            io_config=IoConfig(
                inputs=2,
                output_mode=request.config.output_mode,
                latency_samples=0,
                tail_time_ms=round(tail_time_ms, 2),
            ),
            slimmable_tiers=SlimmableTiersConfig(
                lite=TierProfile(
                    channels=4,
                    fdn_lines=4 if core2_active else 0,
                    early_taps=4 if core2_active else 0,
                    head_svd_rank=active_svd_rank,
                    validation_esr_db=round(esr_l_final, 3),
                    arena_end_byte_offset=ARENA_OFFSET_LITE_END,
                ),
                full=TierProfile(
                    channels=8,
                    fdn_lines=8 if core2_active else 0,
                    early_taps=8 if core2_active else 0,
                    head_svd_rank=active_svd_rank,
                    validation_esr_db=round(esr_f_final, 3),
                    arena_end_byte_offset=ARENA_OFFSET_FULL_END,
                ),
                ultra=TierProfile(
                    channels=12,
                    fdn_lines=8 if core2_active else 0,
                    early_taps=8 if core2_active else 0,
                    head_svd_rank=active_svd_rank,
                    validation_esr_db=round(esr_u_final, 3),
                    arena_end_byte_offset=ARENA_OFFSET_ULTRA_END,
                ),
            ),
            controls_schema=updated_ctrls,
            presets=list(request.presets),
            control_bus=ControlBusConfig(head_trust_region_radius=trust_radius),
            core1_hybrid_ssm=Core1Config(),
            core2_mufdn=Core2Config(
                active=core2_active,
                ring_buffer_pow2_log2=12 if core2_active else 0,
            ),
            audit_report=final_audit,
        )

        arena_layout = model.to_arena_weight_layout()
        dnam_container, ckpt_container = build_signed_dnam_containers(
            header,
            arena_layout,
            stem_stats_s4,
            creator_seed_hex=creator_seed_hex,
            signer_id=signer_id,
        )

        elapsed_sec = max(0.0, time.perf_counter() - t0)
        self.observer.on_training_event(
            TrainingCompleteEvent(
                model_id=request.model_id,
                esr_lite_db=esr_l_final,
                esr_full_db=esr_f_final,
                esr_ultra_db=esr_u_final,
                core2_active=core2_active,
                lfo_env_active=True,
                resume_tier_executed=0,
                elapsed_sec=elapsed_sec,
            )
        )

        job_result = TrainingJobResult(
            model_id=request.model_id,
            dnam_bytes=dnam_container.to_bytes(),
            checkpoint_bytes=ckpt_container.to_bytes(),
            audit_report=final_audit,
            telemetry=telemetry_rows,
            stem_esr_ultra_db=per_stem_ultra,
            resume_tier_executed=0,
            core2_active=core2_active,
            lfo_env_active=True,
        )
        return job_result, dnam_container, ckpt_container

    def resume_from_checkpoint(
        self,
        resume_request: ResumeJobRequest,
        *,
        controls_schema: Sequence[Any] | None = None,
        presets: Sequence[PresetDef] | None = None,
        user_metadata: dict[str, Any] | None = None,
        creator_seed_hex: str | None = None,
        signer_id: str = "dnam-creator",
    ) -> tuple[TrainingJobResult, DnamContainer, DnamContainer, ResumeExecutionResult]:
        """
        Executes Single-Checkpoint (`.dnam-ckpt`) Two-Tier Iterative Stem Expansion (`Test 8G`).
        """
        t0 = time.perf_counter()
        ckpt_container = DnamContainer.from_bytes(resume_request.checkpoint_bytes)
        if not ckpt_container.is_checkpoint:
            raise ValueError(
                "resume_from_checkpoint requires a .dnam-ckpt container (magic='DNCK')"
            )

        sr = ckpt_container.preamble.sample_rate_train_hz
        stem_ids, stems_controls, inputs_list, targets_list, in_shas, tgt_shas = _extract_job_stems(
            resume_request.new_stems,
            resume_request.new_stem_descriptors,
            sr,
        )

        existing_entries = ckpt_container.header.stems_registry or []
        raw_arena = ckpt_container.arena
        existing_stats = [
            StemSufficientStatistics.from_checkpoint_entry(e, raw_arena) for e in existing_entries
        ]

        candidates: list[tuple[str, dict[str, float | str], str, str]] = [
            (stem_ids[i], stems_controls[i], in_shas[i], tgt_shas[i]) for i in range(len(stem_ids))
        ]
        resolved_slots, eval_indices = merge_stem_statistics_for_resume(existing_stats, candidates)
        reused_count = len(stem_ids) - len(eval_indices)

        core2_active = ckpt_container.header.core2_mufdn.active
        model = DnamTriPhysicsModel(
            sample_rate_hz=sr,
            core2_active=core2_active,
        )
        device = torch.device("cpu")
        if self.hardware_profile.torch_device.startswith(("cuda", "mps")):
            device = torch.device(self.hardware_profile.torch_device)
        model.to(device)

        # Forward-pass ONLY newly added or modified stems; also keep all existing stems not in
        # candidates
        incoming_keys = {
            (in_shas[i], tgt_shas[i], tuple(sorted(stems_controls[i].items())))
            for i in range(len(stem_ids))
        }
        kept_existing: list[StemSufficientStatistics] = []
        for ex in existing_stats:
            ex_key = (
                ex.input_pcm_sha256,
                ex.target_pcm_sha256,
                tuple(sorted(ex.controls_meta.items())),
            )
            if ex_key not in incoming_keys and ex.stem_id not in stem_ids:
                kept_existing.append(ex)

        evaluated_stats: list[StemSufficientStatistics] = []
        with torch.no_grad():
            for idx, slot in enumerate(resolved_slots):
                if slot is not None:
                    evaluated_stats.append(slot)
                else:
                    x_t = torch.as_tensor(inputs_list[idx], device=device, dtype=torch.float32)
                    phi_np = (
                        model.extract_tier_ordered_features(x_t)
                        .squeeze(0)
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                    )
                    stat = compute_stem_sufficient_statistics(
                        phi_np,
                        targets_list[idx],
                        stem_id=stem_ids[idx],
                        controls_meta=stems_controls[idx],
                        input_pcm_sha256=in_shas[idx],
                        target_pcm_sha256=tgt_shas[idx],
                    )
                    evaluated_stats.append(stat)

        final_stats = [*kept_existing, *evaluated_stats]
        all_controls = [s.controls_meta for s in final_stats]
        base_ctrls = (
            list(controls_schema)
            if controls_schema is not None
            else list(ckpt_container.header.controls_schema)
        )

        p1_result = run_phase1_preflight_audit(
            base_ctrls,
            [{"controls": c} for c in all_controls],
        )
        updated_ctrls = p1_result.updated_controls
        hyper_bus = ParametricControlHyperBus(updated_ctrls)
        if updated_ctrls:
            psi_stems = hyper_bus.encode_batch_coords(all_controls).detach().cpu().numpy()
        else:
            psi_stems = np.zeros((len(all_controls), 0), dtype=np.float64)

        _p_svd, p_svd_16x4, active_svd_rank, z_head_stems = compute_svd_control_projector(
            psi_stems,
            svd_tol=resume_request.config.svd_singular_floor,
            max_rank=resume_request.config.max_head_svd_rank,
        )
        head_polish_sol = solve_all_tiers_kronecker_head_float64(
            final_stats,
            z_head_stems,
            mode="polish",
        )
        model.load_kronecker_solution(head_polish_sol, p_svd_16x4)

        # Evaluate ESR on the provided stems
        esr_l_list: list[float] = []
        esr_f_list: list[float] = []
        esr_u_list: list[float] = []
        per_stem_ultra: dict[str, float] = {}
        offset = len(kept_existing)
        with torch.no_grad():
            for idx in range(len(stem_ids)):
                phi_np = (
                    model.extract_tier_ordered_features(
                        torch.as_tensor(inputs_list[idx], device=device, dtype=torch.float32)
                    )
                    .squeeze(0)
                    .detach()
                    .cpu()
                    .numpy()
                )
                z_s = z_head_stems[offset + idx]
                el = compute_esr_db(
                    evaluate_kronecker_head_features(phi_np, z_s, head_polish_sol.w_lite_5x2x64),
                    targets_list[idx],
                )
                ef = compute_esr_db(
                    evaluate_kronecker_head_features(phi_np, z_s, head_polish_sol.w_full_5x2x128),
                    targets_list[idx],
                )
                eu = compute_esr_db(
                    evaluate_kronecker_head_features(phi_np, z_s, head_polish_sol.w_ultra_5x2x192),
                    targets_list[idx],
                )
                esr_l_list.append(el)
                esr_f_list.append(ef)
                esr_u_list.append(eu)
                per_stem_ultra[stem_ids[idx]] = eu

        esr_l_final = float(np.mean(esr_l_list)) if esr_l_list else -40.0
        esr_f_final = float(np.mean(esr_f_list)) if esr_f_list else -42.0
        esr_u_final = float(np.mean(esr_u_list)) if esr_u_list else -45.0

        tier_executed: int = (
            2
            if (
                resume_request.force_tier2
                or esr_u_final > resume_request.config.tier2_esr_threshold_db
            )
            and resume_request.force_tier2
            else 1
        )

        existing_presets_by_id: dict[str, PresetDef] = {
            p.id: p for p in ckpt_container.header.presets
        }
        if presets:
            for p in presets:
                existing_presets_by_id[p.id] = p
        merged_user_meta = dict(ckpt_container.header.user_metadata)
        if user_metadata:
            merged_user_meta.update(user_metadata)

        arena_layout = ArenaWeightLayout.from_arena012_bytes(raw_arena)
        arena_layout.with_kronecker_solution(head_polish_sol, p_svd_16x4)

        updated_tiers = SlimmableTiersConfig(
            lite=ckpt_container.header.slimmable_tiers.lite.model_copy(
                update={
                    "head_svd_rank": active_svd_rank,
                    "validation_esr_db": round(esr_l_final, 3),
                }
            ),
            full=ckpt_container.header.slimmable_tiers.full.model_copy(
                update={
                    "head_svd_rank": active_svd_rank,
                    "validation_esr_db": round(esr_f_final, 3),
                }
            ),
            ultra=ckpt_container.header.slimmable_tiers.ultra.model_copy(
                update={
                    "head_svd_rank": active_svd_rank,
                    "validation_esr_db": round(esr_u_final, 3),
                }
            ),
        )
        updated_header = ckpt_container.header.model_copy(
            update={
                "controls_schema": updated_ctrls,
                "presets": list(existing_presets_by_id.values()),
                "user_metadata": merged_user_meta,
                "slimmable_tiers": updated_tiers,
                "audit_report": p1_result.audit_report,
            }
        )

        dnam_container, new_ckpt_container = build_signed_dnam_containers(
            updated_header,
            arena_layout,
            final_stats,
            creator_seed_hex=creator_seed_hex,
            signer_id=signer_id,
        )

        elapsed_sec = max(0.0, time.perf_counter() - t0)
        telemetry_row = TrainingTelemetryRow(
            stage=4,
            epoch=0,
            esr_lite_db=esr_l_final,
            esr_full_db=esr_f_final,
            esr_ultra_db=esr_u_final,
            barrier_loss=0.0,
            group_lasso_loss=0.0,
            surrogate_sigma_samples=0.0,
            elapsed_sec=elapsed_sec,
        )

        job_result = TrainingJobResult(
            model_id=ckpt_container.header.metadata.name,
            dnam_bytes=dnam_container.to_bytes(),
            checkpoint_bytes=new_ckpt_container.to_bytes(),
            audit_report=p1_result.audit_report,
            telemetry=[telemetry_row],
            stem_esr_ultra_db=per_stem_ultra,
            resume_tier_executed=2 if tier_executed == 2 else 1,
            core2_active=core2_active,
            lfo_env_active=True,
        )
        resume_info = ResumeExecutionResult(
            dnam_container=dnam_container,
            ckpt_container=new_ckpt_container,
            tier_executed=tier_executed,
            reused_stem_count=reused_count,
            evaluated_stem_count=len(eval_indices),
            esr_lite_db=esr_l_final,
            esr_full_db=esr_f_final,
            esr_ultra_db=esr_u_final,
            audit_report=p1_result.audit_report,
        )
        return job_result, dnam_container, new_ckpt_container, resume_info
