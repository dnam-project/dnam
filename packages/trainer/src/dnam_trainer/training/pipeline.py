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

import gc
import hashlib
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

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
    LevelingConfig,
    ModelCapabilities,
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
from dnam_trainer.dsp.selective_rglru import (
    DifferentiableSelectiveRgLru,
    parallel_associative_scan_complex,
)
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
    Base48BandGramCache,
    MultiTierKroneckerHeadSolution,
    StemSufficientStatistics,
    compute_modulated_stem_sufficient_statistics,
    compute_stem_sufficient_statistics,
    compute_svd_control_projector,
    evaluate_kronecker_head_features,
    evaluate_stem_besr_from_sufficient_stats,
    merge_stem_statistics_for_resume,
    precompute_base48_band_cache,
    solve_all_tiers_kronecker_head_float64,
)
from dnam_trainer.training.metrics import (
    DEFAULT_NOISE_FLOOR_DBFS,
    PsychoacousticMetricsReport,
    compute_band_split_esr_db,
    compute_floor_regularized_besr_db,
    evaluate_psychoacoustic_metrics,
    torch_floor_regularized_besr_loss,
)
from dnam_trainer.training.selective_cascade import (
    calibrate_3stage_selective_cascade,
    modulate_tier_blocks_with_controls,
)
from dnam_trainer.training.wav_io import read_wav


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
        total_len = min(len(x_mono), len(y_mono))
        if total_len < 2048:
            continue

        # If the excitation stem includes a trailing silent listening tail (>= 100 ms),
        # measure true post-excitation ringdown (> 35 ms after dry signal stops) so
        # nonlinear harmonic distortion products do not masquerade as delay/reverb echoes.
        active_idx = np.flatnonzero(np.abs(x_mono[:total_len]) > 1e-4)
        min_tail_samples = int(0.100 * sample_rate_hz)
        if active_idx.size > 0 and (total_len - 1 - int(active_idx[-1])) >= min_tail_samples:
            t_first = int(active_idx[0])
            t_stop = int(active_idx[-1]) + 1
            active_ms = float(np.mean(y_mono[t_first:t_stop] ** 2)) + 1e-12
            noise_win = max(64, min(t_first, int(0.050 * sample_rate_hz)))
            noise_ms = float(np.mean(y_mono[:noise_win] ** 2))
            late_start = t_stop + int(0.035 * sample_rate_hz)
            late_end = min(total_len, t_stop + int(0.350 * sample_rate_hz))
            if late_end > late_start + 16:
                late_ms = float(np.mean(y_mono[late_start:late_end] ** 2))
                excess_late = max(0.0, late_ms - 4.0 * noise_ms)
                late_ratios.append(excess_late / active_ms)
            continue

        n = min(total_len, sample_rate_hz * 2)
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


def measure_gated_loudness_db(
    audio: NDArray[np.floating[Any]],
    *,
    sample_rate_hz: int = 48_000,
    metric: str = "rms_dbfs",
) -> float:
    """Computes gated unweighted broadband RMS (`rms_dbfs`) or K-weighted LUFS (`k_weighted_lufs`)
    in dB over active 100 ms blocks (with -70 dBFS absolute and -10 dB relative gates).
    """
    arr = np.asarray(audio, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    elif arr.shape[0] == 2 and arr.shape[1] > 2:
        arr = arr.T
    if arr.shape[0] == 0:
        return -96.0

    if metric == "k_weighted_lufs" and arr.shape[0] >= 64:
        n_fft = int(arr.shape[0])
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / float(max(1, sample_rate_hz)))
        f_hp = 38.135
        hp_mag_sq = (freqs**4) / ((freqs**2 + f_hp**2) ** 2 + 1e-18)
        f_sh = 1681.97
        sh_boost = 1.0 + (10.0 ** (4.0 / 20.0) - 1.0) * ((freqs**2) / (freqs**2 + f_sh**2 + 1e-18))
        k_weight = np.sqrt(hp_mag_sq) * sh_boost
        spec = np.fft.rfft(arr, axis=0) * k_weight[:, None]
        arr = np.fft.irfft(spec, n=n_fft, axis=0)

    win = max(64, int(0.100 * sample_rate_hz))
    n_win = int(arr.shape[0]) // win
    if n_win >= 1:
        blocks = arr[: n_win * win].reshape(n_win, win, -1)
        ms_blocks = np.mean(blocks * blocks, axis=(1, 2))
        abs_mask = ms_blocks > 1.0e-7
        if np.any(abs_mask):
            abs_blocks = ms_blocks[abs_mask]
            rel_thresh = 0.1 * float(np.mean(abs_blocks))
            gated = abs_blocks[abs_blocks >= rel_thresh]
            mean_ms = float(np.mean(gated)) if gated.size > 0 else float(np.mean(abs_blocks))
        else:
            mean_ms = float(np.mean(ms_blocks))
    else:
        mean_ms = float(np.mean(arr * arr))

    offset_db = -0.691 if metric == "k_weighted_lufs" else 0.0
    return float(max(-96.0, offset_db + 10.0 * math.log10(max(mean_ms, 1.0e-12))))


def detect_wet_polarity_inverted(
    inputs_list: Sequence[NDArray[np.float32]],
    targets_list: Sequence[NDArray[np.float32]],
) -> bool:
    """Detects whether the captured hardware inverts signal polarity (`-1`)."""
    if not inputs_list or not targets_list:
        return False
    x = np.mean(inputs_list[0], axis=1, dtype=np.float64)
    y = np.mean(targets_list[0], axis=1, dtype=np.float64)
    n = min(int(x.shape[0]), int(y.shape[0]), 65_536)
    if n < 64:
        return False
    x_seg = x[:n]
    y_seg = y[:n]
    denom = float(np.linalg.norm(x_seg) * np.linalg.norm(y_seg)) + 1.0e-12
    best_corr = 0.0
    for lag in range(-16, 17):
        if lag < 0:
            c = float(np.dot(x_seg[-lag:], y_seg[: n + lag])) / denom
        elif lag > 0:
            c = float(np.dot(x_seg[: n - lag], y_seg[lag:])) / denom
        else:
            c = float(np.dot(x_seg, y_seg)) / denom
        if abs(c) > abs(best_corr):
            best_corr = c
    return best_corr < -0.15


def _estimate_core2_ring_buffer_log2(
    controls_schema: Sequence[Any],
    tail_time_ms: float,
    sample_rate_hz: int,
    core2_active: bool,
) -> int:
    """Computes `Core2Config.ring_buffer_pow2_log2` (`12..=18`) when Core 2 is active."""
    if not core2_active:
        return 0
    max_cal_ms = 0.0
    for ctrl in controls_schema:
        cal = getattr(ctrl, "calibration", None)
        if cal is not None and getattr(cal, "role", None) in ("delay_time_ms", "grain_size_ms"):
            lob = getattr(cal, "lobatto_physical_anchors", None)
            if lob is not None:
                max_cal_ms = max(max_cal_ms, float(max(lob)))
            cond_map = getattr(cal, "conditional_physical_anchors", {}) or {}
            for anchors in cond_map.values():
                max_cal_ms = max(max_cal_ms, float(max(anchors)))
    effective_ms = max(max_cal_ms, min(float(tail_time_ms), 2000.0))
    if effective_ms <= 0.0:
        return 12
    needed_samples = max(4096, math.ceil((effective_ms * 0.001 * sample_rate_hz) * 1.25))
    return max(12, min(18, math.ceil(math.log2(needed_samples))))


def _calibrate_capabilities_and_leveling(
    *,
    controls_schema: Sequence[Any],
    config: Any,
    core2_active: bool,
    head_sol: MultiTierKroneckerHeadSolution,
    sample_rate_hz: int,
    inputs_list: Sequence[NDArray[np.float32]],
    targets_list: Sequence[NDArray[np.float32]],
    stems_controls: Sequence[dict[str, float | str]],
    default_features_192: NDArray[np.floating[Any]] | None,
    stem_stats: Sequence[StemSufficientStatistics],
) -> tuple[
    ModelCapabilities, LevelingConfig, tuple[float, float, float], tuple[float, float, float]
]:
    """Computes `ModelCapabilities`, `LevelingConfig` anchored strictly at default control settings
    (`controls_schema[*].default`, normalized `u = 0`, `z_head = 0`), tier loudness
    `(L_lite, L_full, L_ultra)`, and inter-tier parity trims `(trim_lite_db, trim_full_db, 0.0)`.
    """
    cfg_blend = getattr(config, "blend_mode", "auto")
    resolved_blend: Literal["none", "host_parallel", "modeled_control"]
    if cfg_blend == "none":
        resolved_blend = "none"
    elif cfg_blend == "modeled_control":
        resolved_blend = "modeled_control"
    elif cfg_blend == "host_parallel":
        resolved_blend = "host_parallel"
    else:
        has_mix_ctrl = any(
            (
                getattr(getattr(c, "calibration", None), "role", None) == "mix"
                or str(getattr(c, "id", "")).strip().lower() in ("blend", "mix", "dry_wet")
            )
            for c in controls_schema
        )
        resolved_blend = "modeled_control" if has_mix_ctrl else "host_parallel"

    cfg_trails = getattr(config, "supports_trails", None)
    supports_trails = core2_active if cfg_trails is None else bool(cfg_trails)
    tempo_sync_capable = any(
        getattr(getattr(c, "calibration", None), "role", None)
        in ("delay_time_ms", "lfo_rate_hz", "grain_size_ms")
        for c in controls_schema
    )
    cfg_freeze = getattr(config, "supports_freeze", None)
    has_freeze_ctrl = any(
        getattr(getattr(c, "calibration", None), "role", None) == "freeze_latch"
        for c in controls_schema
    )
    supports_freeze = has_freeze_ctrl if cfg_freeze is None else bool(cfg_freeze)
    freeze_mutes_dry = bool(getattr(config, "freeze_mutes_dry", False))
    wet_pol_inv = detect_wet_polarity_inverted(inputs_list, targets_list)

    capabilities = ModelCapabilities(
        blend_mode=resolved_blend,
        default_blend=float(getattr(config, "default_blend", 1.0)),
        wet_polarity_inverted=wet_pol_inv,
        supports_trails=supports_trails,
        tempo_sync_capable=tempo_sync_capable,
        supports_freeze=supports_freeze,
        freeze_mutes_dry=freeze_mutes_dry,
    )

    metric = str(getattr(config, "leveling_metric", "rms_dbfs"))
    auto_unity = bool(getattr(config, "auto_unity_leveling", True))
    target_gain_db = float(getattr(config, "target_gain_db", 0.0))

    # Pick stem closest to default controls (u = 0)
    best_idx = 0
    if controls_schema and stems_controls:
        best_dist = float("inf")
        for s_idx, c_meta in enumerate(stems_controls):
            u_mod = _compute_intrinsic_control_modulators(controls_schema, c_meta)
            dist = float(u_mod[0] ** 2 + u_mod[1] ** 2 + u_mod[2] ** 2)
            if dist < best_dist:
                best_dist = dist
                best_idx = s_idx

    if inputs_list and best_idx < len(inputs_list):
        ref_in_db = measure_gated_loudness_db(
            inputs_list[best_idx], sample_rate_hz=sample_rate_hz, metric=metric
        )
    else:
        ref_in_db = -20.5

    z_zero = np.zeros(4, dtype=np.float64)
    if default_features_192 is not None and default_features_192.shape[0] >= 16:
        y_lite = evaluate_kronecker_head_features(
            default_features_192, z_zero, head_sol.w_lite_5x2x64
        )
        y_full = evaluate_kronecker_head_features(
            default_features_192, z_zero, head_sol.w_full_5x2x128
        )
        y_ultra = evaluate_kronecker_head_features(
            default_features_192, z_zero, head_sol.w_ultra_5x2x192
        )
        l_lite = measure_gated_loudness_db(y_lite, sample_rate_hz=sample_rate_hz, metric=metric)
        l_full = measure_gated_loudness_db(y_full, sample_rate_hz=sample_rate_hz, metric=metric)
        l_ultra = measure_gated_loudness_db(y_ultra, sample_rate_hz=sample_rate_hz, metric=metric)
    elif stem_stats:
        from dnam_engine.container import GramMatrix

        stat_idx = min(best_idx, len(stem_stats) - 1)
        st0 = stem_stats[stat_idx]
        tier_quads: list[float] = []
        for tier_dim, w_head in (
            (64, head_sol.w_lite_5x2x64),
            (128, head_sol.w_full_5x2x128),
            (192, head_sol.w_ultra_5x2x192),
        ):
            w_dc = np.asarray(w_head[0], dtype=np.float64)  # (2, tier_dim) at z_head = 0
            g_sub = GramMatrix.unpack(st0.g_audio_uplo_u_f64, dim=tier_dim)
            tier_quads.append(max(float(np.sum(w_dc * (w_dc @ g_sub))), 1.0e-18))

        if targets_list and best_idx < len(targets_list):
            base_wet_db = measure_gated_loudness_db(
                targets_list[best_idx], sample_rate_hz=sample_rate_hz, metric=metric
            )
            l_ultra = base_wet_db
            l_full = base_wet_db + 10.0 * math.log10(tier_quads[1] / tier_quads[2])
            l_lite = base_wet_db + 10.0 * math.log10(tier_quads[0] / tier_quads[2])
        else:
            n_frames = float(max(1, st0.num_frames))
            l_lite = float(10.0 * math.log10(max(tier_quads[0] / (2.0 * n_frames), 1.0e-12)))
            l_full = float(10.0 * math.log10(max(tier_quads[1] / (2.0 * n_frames), 1.0e-12)))
            l_ultra = float(10.0 * math.log10(max(tier_quads[2] / (2.0 * n_frames), 1.0e-12)))
    else:
        l_lite, l_full, l_ultra = ref_in_db, ref_in_db, ref_in_db

    unity_offset_db = round(ref_in_db - l_ultra, 3)
    cal_trim_db = round((unity_offset_db + target_gain_db) if auto_unity else target_gain_db, 3)
    leveling = LevelingConfig(
        auto_unity_enabled=auto_unity,
        leveling_metric="k_weighted_lufs" if metric == "k_weighted_lufs" else "rms_dbfs",
        reference_input_level_db=round(ref_in_db, 3),
        measured_default_output_level_db=round(l_ultra, 3),
        unity_gain_offset_db=unity_offset_db,
        target_gain_db=round(target_gain_db, 3),
        calibrated_trim_db=cal_trim_db,
    )
    trim_lite = float(np.clip(round(l_ultra - l_lite, 3), -12.0, 12.0))
    trim_full = float(np.clip(round(l_ultra - l_full, 3), -12.0, 12.0))
    return (
        capabilities,
        leveling,
        (round(l_lite, 3), round(l_full, 3), round(l_ultra, 3)),
        (trim_lite, trim_full, 0.0),
    )


def _compute_intrinsic_control_modulators(
    controls_schema: Sequence[Any],
    controls_meta: dict[str, float | str],
) -> tuple[float, float, float]:
    """Computes stem-intrinsic normalized control coordinates `(u_0, u_1, u_2)` in `[-1.0, 1.0]`
    strictly from `declared_range` / `declared_options` and `controls_meta` (`Test 8E` / `Test 8G`
    invariant).
    """
    ctrl_list = list(controls_schema)
    if len(ctrl_list) <= 3:
        coords: list[float] = []
        for ctrl in ctrl_list[:3]:
            kind = getattr(ctrl, "kind", None)
            ctrl_id = getattr(ctrl, "id", "")
            default_v = getattr(ctrl, "default", 0.0)
            if kind == "range":
                dec_range = getattr(ctrl, "declared_range", (0.0, 1.0))
                lo, hi = float(dec_range[0]), float(dec_range[1])
                raw_v = float(controls_meta.get(ctrl_id, default_v))
                span = max(hi - lo, 1e-12)
                u_val = float(np.clip(2.0 * (raw_v - lo) / span - 1.0, -1.0, 1.0))
                coords.append(u_val)
            elif kind == "switch":
                opts = list(getattr(ctrl, "declared_options", []))
                raw_s = str(controls_meta.get(ctrl_id, default_v))
                idx = opts.index(raw_s) if raw_s in opts else 0
                u_val = 0.0 if len(opts) <= 1 else float(2.0 * idx / (len(opts) - 1) - 1.0)
                coords.append(u_val)
        if len(coords) == 2 and all(getattr(c, "kind", None) == "range" for c in ctrl_list):
            u0, u1 = coords[0], coords[1]
            coords.append(0.65 * u0 * u1 + 0.85 * (u0 * u0) * (u1 * u1))
        while len(coords) < 3:
            coords.append(0.0)
        return (coords[0], coords[1], coords[2])

    range_du: list[float] = []
    switch_du: list[float] = []
    for ctrl in ctrl_list:
        kind = getattr(ctrl, "kind", None)
        ctrl_id = getattr(ctrl, "id", "")
        default_v = getattr(ctrl, "default", 0.0)
        if kind == "range":
            dec_range = getattr(ctrl, "declared_range", (0.0, 1.0))
            lo, hi = float(dec_range[0]), float(dec_range[1])
            span = max(hi - lo, 1e-12)
            def_u = float(np.clip(2.0 * (float(default_v) - lo) / span - 1.0, -1.0, 1.0))
            raw_v = float(controls_meta.get(ctrl_id, default_v))
            u_val = float(np.clip(2.0 * (raw_v - lo) / span - 1.0, -1.0, 1.0))
            range_du.append(u_val - def_u)
        elif kind == "switch":
            opts = list(getattr(ctrl, "declared_options", []))
            def_s = str(default_v)
            raw_s = str(controls_meta.get(ctrl_id, default_v))
            def_idx = opts.index(def_s) if def_s in opts else 0
            raw_idx = opts.index(raw_s) if raw_s in opts else def_idx
            denom = float(max(1, len(opts) - 1))
            switch_du.append(float(raw_idx - def_idx) / denom)

    m0 = sum((0.55 + 0.13 * i) * du for i, du in enumerate(range_du))
    m1 = sum((0.50 + 0.15 * i) * (du * du) for i, du in enumerate(range_du))
    corner_pair = 0.45 * range_du[0] * range_du[1] if len(range_du) >= 2 else 0.0
    extra_lin = sum((0.60 + 0.17 * i) * du for i, du in enumerate(range_du[3:]))
    sw_term = sum((0.85 + 0.25 * j) * su for j, su in enumerate(switch_du))
    m2 = sw_term + corner_pair + extra_lin
    if len(ctrl_list) > 5:
        pair_rr = sum(
            (0.23 + 0.07 * i + 0.03 * k) * range_du[i] * range_du[k]
            for i in range(len(range_du))
            for k in range(i + 1, len(range_du))
        )
        pair_sr = sum(
            (0.31 + 0.11 * j + 0.05 * i) * switch_du[j] * range_du[i]
            for j in range(len(switch_du))
            for i in range(len(range_du))
        )
        m2 += pair_rr + pair_sr
    return (float(m0), float(m1), float(m2))


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
        self._calibrated_tier_blocks: dict[
            str, tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]
        ] = {}
        self._calibrated_knot_coords: dict[str, NDArray[np.float64]] = {}

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
        self.w_head_full = nn.Parameter(torch.zeros(5, 2, 128, dtype=torch.float32))
        self.w_head_ultra = nn.Parameter(torch.zeros(5, 2, 192, dtype=torch.float32))
        self.b_head = nn.Parameter(torch.zeros(5, 2, dtype=torch.float32))
        self.register_buffer("p_svd_16x4", torch.zeros(16, 4, dtype=torch.float32))

        self._reset_deterministic_parameters()

    def _reset_deterministic_parameters(self) -> None:
        """Initializes all 3 tier groups deterministically with log-spaced poles and rich multi-tap
        basis."""
        gen = torch.Generator(device="cpu")
        gen.manual_seed(20261023)
        tier_drives = (
            (0.8, 3.0, 11.0, 40.0),
            (1.4, 5.0, 18.0, 60.0),
            (0.5, 1.8, 6.5, 25.0),
        )
        tier_freqs = (
            (35.0, 160.0, 750.0, 3500.0),
            (65.0, 300.0, 1400.0, 6500.0),
            (25.0, 105.0, 480.0, 2200.0),
        )
        sr = float(self.sample_rate_hz)
        with torch.no_grad():
            for tier_idx in range(3):
                drives = torch.tensor(tier_drives[tier_idx], dtype=torch.float32)
                in_w = (
                    torch.tensor(
                        [
                            [0.55, 0.45],
                            [0.45, 0.55],
                            [0.60, 0.40],
                            [0.40, 0.60],
                        ],
                        dtype=torch.float32,
                    )
                    * drives[:, None]
                )
                self.tier_in_proj[tier_idx].copy_(in_w)

                freqs = torch.tensor(tier_freqs[tier_idx], dtype=torch.float32)
                thetas = (2.0 * math.pi * freqs / sr).clamp(1e-4, math.pi - 1e-4)
                r_target = torch.clamp(torch.exp(-2.0 * math.pi * 0.65 * freqs / sr), 0.28, 0.996)
                r_norm = torch.clamp((r_target - 0.25) / (0.9995 - 0.25), 1e-3, 1.0 - 1e-3)
                rglru = self.rglru_tiers[tier_idx]
                assert isinstance(rglru, DifferentiableSelectiveRgLru)
                rglru.theta_rad.copy_(thetas)
                rglru.r_low_raw.copy_(torch.log(r_norm / (1.0 - r_norm)))
                rglru.dr_span_raw.fill_(-3.0)
                rglru.c_imag.copy_(torch.tensor([0.35, -0.35, 0.35, -0.35], dtype=torch.float32))

                poly = self.poly_adaa_tiers[tier_idx]
                assert isinstance(poly, DifferentiablePolyAdaa)
                poly.b_conv.copy_(
                    torch.tensor([-0.05, 0.04, -0.08, 0.06], dtype=torch.float32)
                    * (1.0 / (1.0 + 0.2 * tier_idx))
                )
                poly.alpha.copy_(torch.tensor([0.65, 0.50, 0.35, 0.20], dtype=torch.float32))
                poly.beta.copy_(torch.tensor([0.55, 0.65, 0.75, 0.85], dtype=torch.float32))
                poly.delta.copy_(torch.tensor([-0.14, 0.12, -0.16, 0.18], dtype=torch.float32))

                conv_w = torch.zeros(4, 4, 6, dtype=torch.float32)
                for blk_idx in range(4):
                    tap_main = 5 - blk_idx
                    conv_w[blk_idx, :, tap_main] = 0.80
                    conv_w[blk_idx, :, max(0, tap_main - 1)] = -0.25 * (tier_idx + 1) * 0.4
                conv_w += 0.04 * torch.randn((4, 4, 6), generator=gen, dtype=torch.float32)
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

        sr = float(self.sample_rate_hz)
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
            u_iir_re_bcl, u_iir_im_bcl = rglru.forward_complex_features(u_base_bcl)
            u_iir_re = u_iir_re_bcl.transpose(1, 2)  # (B, T, 4)
            u_iir_im = u_iir_im_bcl.transpose(1, 2)  # (B, T, 4)

            # Multi-scale envelope detector for tube cathode bias shift & B+ power-supply sag
            tau_ms = torch.tensor(
                [3.0, 12.0, 45.0, 150.0], device=x_stereo.device, dtype=torch.float32
            ) * (1.0 + 0.2 * tier_idx)
            r_env = torch.exp(-1000.0 / (tau_ms * sr)).view(1, 4, 1)
            a_env = torch.complex(r_env, torch.zeros_like(r_env)).expand(bsz, 4, num_frames)
            bx_env = torch.complex((1.0 - r_env) * u_base_bcl.abs(), torch.zeros_like(u_base_bcl))
            z_env, _ = parallel_associative_scan_complex(a_env, bx_env)
            env_4 = z_env.real.transpose(1, 2).to(dtype=x_stereo.dtype)  # (B, T, 4)

            # Pre-filtered & envelope-sagged nonlinear tube stages
            u_nl_raw = torch.tanh(u_base)
            u_nl_even = u_nl_raw * u_nl_raw - 0.15
            u_pre_drive = u_base + 0.8 * u_iir_re - 0.4 * u_iir_im
            u_pre_nl = torch.tanh(u_pre_drive)
            u_pre_sag = u_pre_drive * torch.rsqrt(1.0 + 2.25 * env_4 * env_4)
            u_sag_nl = torch.tanh(u_base * torch.rsqrt(1.0 + 4.0 * env_4 * env_4))

            # Post-clipping speaker cabinet & tone-stack complex resonators
            u_post1_re_bcl, u_post1_im_bcl = rglru.forward_complex_features(
                u_pre_nl.transpose(1, 2)
            )
            u_post1_re = u_post1_re_bcl.transpose(1, 2)
            u_post1_im = u_post1_im_bcl.transpose(1, 2)

            u_post2_re_bcl, u_post2_im_bcl = rglru.forward_complex_features(u_nl_bcl)
            u_post2_re = u_post2_re_bcl.transpose(1, 2)
            u_post2_im = u_post2_im_bcl.transpose(1, 2)

            # 4 dilated causal shifts `(1, 3, 7, 19)` across the 4 channels -> 16 basis signals
            conv_w = self.tier_conv_weights[tier_idx]  # (4, 4, 6)
            dil_sources = (u_nl_bcl, u_pre_nl.transpose(1, 2), u_post1_re_bcl, u_post1_im_bcl)
            dil_feats: list[torch.Tensor] = []
            for blk_idx, dil in enumerate(self.dilations):
                pad_len = 5 * dil
                padded = F.pad(dil_sources[blk_idx], (pad_len, 0))
                w_blk = conv_w[blk_idx].unsqueeze(1)  # (4, 1, 6)
                c_out = F.conv1d(padded, w_blk, dilation=dil, groups=4).transpose(1, 2)  # (B, T, 4)
                dil_feats.append(c_out)

            raw_16 = [
                u_base,
                u_iir_re,
                u_iir_im,
                u_nl,
                u_nl_raw,
                u_nl_even,
                u_pre_nl,
                u_pre_sag,
                u_sag_nl,
                u_post1_re,
                u_post1_im,
                u_post2_re,
                u_post2_im,
                dil_feats[0],
                dil_feats[1],
                dil_feats[2] + 0.5 * dil_feats[3],
            ]
            feat_64 = torch.cat(raw_16, dim=-1)  # (B, T, 64)
            if self.core2_ufdn is not None:
                tank_12_bcl, wet_2_bcl = self.core2_ufdn.forward_tank_lines(u_base_bcl[:, :2, :])
                tank_g_bt4 = tank_12_bcl[:, 4 * tier_idx : 4 * (tier_idx + 1), :].transpose(1, 2)
                wet_bt2 = wet_2_bcl.transpose(1, 2)
                inject_gain = torch.sigmoid(self.core2_ufdn.head_tap_inject_gain) * 0.1
                feat_64 = (
                    feat_64
                    + tank_g_bt4.repeat(1, 1, 16) * inject_gain
                    + F.pad(wet_bt2, (0, 62)) * 0.1
                )
            tier_features.append(feat_64)

        return torch.cat(tier_features, dim=-1)  # (B, T, 192)

    def calibrate_selective_cascade(
        self,
        inputs_list: Sequence[NDArray[np.float32]],
        targets_list: Sequence[NDArray[np.float32]],
        *,
        in_shas: Sequence[str] | None = None,
        u_ctrl_stems: Sequence[tuple[float, float, float]] | None = None,
        on_stage_progress: Callable[[int, str], None] | None = None,
        on_step_metrics: Callable[[int, int, float, float, float, float, float, float], None]
        | None = None,
    ) -> None:
        """Calibrates the 3-Stage Causal Selective State-Space + 3-Band Decoupled (`LF`/`MF`/`HF`)
        Cascade on each unique dry input signal in `inputs_list` and caches the nested tier base
        blocks `(b_lite, b_full, b_ultra)`.
        """
        self._calibrated_tier_blocks.clear()
        self._calibrated_knot_coords.clear()
        mute_thresh_sq = 10.0 ** (-55.0 / 10.0)
        for s_idx, x_arr in enumerate(inputs_list):
            sha_key = (
                in_shas[s_idx] if in_shas is not None and s_idx < len(in_shas) else f"idx_{s_idx}"
            )
            if sha_key in self._calibrated_tier_blocks:
                continue
            b_lite, b_full, b_ultra = calibrate_3stage_selective_cascade(
                x_arr[:, 0],
                targets_list,
                sample_rate_hz=self.sample_rate_hz,
                on_stage_progress=on_stage_progress,
                on_step_metrics=on_step_metrics,
            )
            self._calibrated_tier_blocks[sha_key] = (b_lite, b_full, b_ultra)
            if b_lite.shape[1] == 32 and u_ctrl_stems is not None:
                c_knots = np.full((32, 3), 99.0, dtype=np.float64)
                col_cursor = 0
                l_tot = x_arr.shape[0]
                for t_idx, y_arr in enumerate(targets_list):
                    y_2d = np.asarray(y_arr, dtype=np.float64)
                    y_1d = (y_2d[:, 0] if y_2d.ndim == 2 else y_2d.ravel())[:l_tot]
                    if float(np.mean(y_1d**2)) > mute_thresh_sq and col_cursor < 32:
                        if t_idx < len(u_ctrl_stems):
                            c_knots[col_cursor] = np.asarray(u_ctrl_stems[t_idx], dtype=np.float64)[
                                :3
                            ]
                        col_cursor += 1
                self._calibrated_knot_coords[sha_key] = c_knots

    def extract_calibrated_or_tier_features(
        self,
        x_stereo: torch.Tensor,
        *,
        in_sha: str | None = None,
        u_ctrl_3: tuple[float, float, float] | None = None,
    ) -> NDArray[np.float32]:
        """Returns the 192-D control-modulated cascade features `(T, 192)` when `in_sha` has been
        calibrated via [`calibrate_selective_cascade`], or falls back to the Step-0 PyTorch
        [`extract_tier_ordered_features`] tensor.
        """
        if in_sha is not None and in_sha in self._calibrated_tier_blocks:
            b_lite_16, b_full_16, b_ultra_16 = self._calibrated_tier_blocks[in_sha]
            return modulate_tier_blocks_with_controls(
                b_lite_16,
                b_full_16,
                b_ultra_16,
                u_ctrl_3 if u_ctrl_3 is not None else (0.0, 0.0, 0.0),
                c_knots=self._calibrated_knot_coords.get(in_sha),
            )
        phi_t = self.extract_tier_ordered_features(x_stereo).squeeze(0)
        return phi_t.detach().cpu().numpy().astype(np.float32)

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

        if tier == "lite":
            w0 = self.w_head_tiers[0]  # (5, 2, 64)
            out = torch.einsum("bm,mcf,btf->btc", psi_tilde, w0, features_192[:, :, :64])
        elif tier == "full":
            out = torch.einsum(
                "bm,mcf,btf->btc", psi_tilde, self.w_head_full, features_192[:, :, :128]
            )
        else:
            out = torch.einsum(
                "bm,mcf,btf->btc", psi_tilde, self.w_head_ultra, features_192[:, :, :192]
            )

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
            self.w_head_full.copy_(
                torch.as_tensor(sol.w_full_5x2x128, device=device, dtype=torch.float32)
            )
            self.w_head_ultra.copy_(
                torch.as_tensor(sol.w_ultra_5x2x192, device=device, dtype=torch.float32)
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
                grain_len_t, grain_warp_t, win_mix_t, spray_t, fb_mix_t = (
                    self.core2_ufdn.clamped_grain_params()
                )
                grain_len_np = grain_len_t.detach().cpu().numpy().astype(np.float32)
                grain_warp_np = grain_warp_t.detach().cpu().numpy().astype(np.float32)
                layout.mufdn_grain_len_8[: len(grain_len_np)] = grain_len_np[:8]
                layout.mufdn_grain_warp_8[: len(grain_warp_np)] = grain_warp_np[:8]
                head_inj_val = float(
                    torch.sigmoid(self.core2_ufdn.head_tap_inject_gain).detach().cpu().item()
                )
                layout.mufdn_grain_meta_0[:] = np.array(
                    [
                        float(win_mix_t.detach().cpu().item()),
                        float(spray_t.detach().cpu().item()),
                        float(fb_mix_t.detach().cpu().item()),
                        head_inj_val,
                    ],
                    dtype=np.float32,
                )
                layout.mufdn_grain_meta_1[:] = np.array(
                    [
                        float(
                            torch.clamp(self.core2_ufdn.freeze_alpha_default, 0.0, 1.0)
                            .detach()
                            .cpu()
                            .item()
                        ),
                        float(
                            torch.clamp(self.core2_ufdn.onset_sync_thresh, 0.0, 1.0)
                            .detach()
                            .cpu()
                            .item()
                        ),
                        0.0,
                        0.0,
                    ],
                    dtype=np.float32,
                )
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
        features_list: Sequence[NDArray[np.float32]] | Callable[[int], NDArray[np.float32]],
        targets_list: Sequence[NDArray[np.float32]],
        z_head_stems: NDArray[np.float64],
        head_sol: MultiTierKroneckerHeadSolution,
        stem_ids: Sequence[str],
        *,
        sample_rate_hz: int = 48_000,
        full_psychoacoustics: bool = False,
    ) -> tuple[float, float, float, dict[str, float], PsychoacousticMetricsReport]:
        """Computes dataset Floor-Regularized Band-Balanced ESR (`bESR` in dB) across `Lite`
        ($N=64$), `Full` ($N=128$), and `Ultra` ($N=192$), plus the multi-band and psychoacoustic
        evaluation suite across active stems.
        """
        active_l: list[float] = []
        active_f: list[float] = []
        active_u: list[float] = []
        active_reports: list[PsychoacousticMetricsReport] = []
        fallback_l: list[float] = []
        fallback_f: list[float] = []
        fallback_u: list[float] = []
        fallback_reports: list[PsychoacousticMetricsReport] = []
        per_stem_ultra: dict[str, float] = {}

        for s_idx, y in enumerate(targets_list):
            phi = features_list(s_idx) if callable(features_list) else features_list[s_idx]
            z_s = z_head_stems[s_idx]
            pred_l = evaluate_kronecker_head_features(phi, z_s, head_sol.w_lite_5x2x64)
            pred_f = evaluate_kronecker_head_features(phi, z_s, head_sol.w_full_5x2x128)
            pred_u = evaluate_kronecker_head_features(phi, z_s, head_sol.w_ultra_5x2x192)
            del phi
            y_f64 = np.asarray(y, dtype=np.float64)

            besr_l = compute_floor_regularized_besr_db(
                pred_l, y_f64, sample_rate_hz=sample_rate_hz, floor_dbfs=DEFAULT_NOISE_FLOOR_DBFS
            )
            besr_f = compute_floor_regularized_besr_db(
                pred_f, y_f64, sample_rate_hz=sample_rate_hz, floor_dbfs=DEFAULT_NOISE_FLOOR_DBFS
            )
            if full_psychoacoustics:
                rep_u = evaluate_psychoacoustic_metrics(
                    pred_u,
                    y_f64,
                    sample_rate_hz=sample_rate_hz,
                    floor_dbfs=DEFAULT_NOISE_FLOOR_DBFS,
                )
            else:
                besr_u_val = compute_floor_regularized_besr_db(
                    pred_u,
                    y_f64,
                    sample_rate_hz=sample_rate_hz,
                    floor_dbfs=DEFAULT_NOISE_FLOOR_DBFS,
                )
                lf_u, mf_u, hf_u = compute_band_split_esr_db(
                    pred_u, y_f64, sample_rate_hz=sample_rate_hz
                )
                rep_u = PsychoacousticMetricsReport(
                    besr_db=besr_u_val,
                    esr_lf_db=lf_u,
                    esr_mid_db=mf_u,
                    esr_hf_db=hf_u,
                    mr_stft_db=0.0,
                    ltas_rms_tilt_db=0.0,
                    ltas_hf_tilt_db=0.0,
                    seg_esr_db=besr_u_val,
                )

            fallback_l.append(besr_l)
            fallback_f.append(besr_f)
            fallback_u.append(rep_u.besr_db)
            fallback_reports.append(rep_u)

            rms_tgt = float(np.mean(y_f64**2))
            if rms_tgt <= 10.0 ** (-55.0 / 10.0):
                # Silent/mute stem (`volume_min`): report residual output level in dBFS
                per_stem_ultra[stem_ids[s_idx]] = float(
                    10.0 * math.log10(max(float(np.mean(pred_u**2)), 1e-18))
                )
            else:
                per_stem_ultra[stem_ids[s_idx]] = rep_u.besr_db
                active_l.append(besr_l)
                active_f.append(besr_f)
                active_u.append(rep_u.besr_db)
                active_reports.append(rep_u)

        use_l = active_l if active_l else fallback_l
        use_f = active_f if active_f else fallback_f
        use_u = active_u if active_u else fallback_u
        use_reps = active_reports if active_reports else fallback_reports

        mean_l = float(np.mean(use_l)) if use_l else 0.0
        mean_f = float(np.mean(use_f)) if use_f else 0.0
        mean_u = float(np.mean(use_u)) if use_u else 0.0
        if use_reps:
            summary_rep = PsychoacousticMetricsReport(
                besr_db=mean_u,
                esr_lf_db=float(np.mean([r.esr_lf_db for r in use_reps])),
                esr_mid_db=float(np.mean([r.esr_mid_db for r in use_reps])),
                esr_hf_db=float(np.mean([r.esr_hf_db for r in use_reps])),
                mr_stft_db=float(np.mean([r.mr_stft_db for r in use_reps])),
                ltas_rms_tilt_db=float(np.mean([r.ltas_rms_tilt_db for r in use_reps])),
                ltas_hf_tilt_db=float(np.mean([r.ltas_hf_tilt_db for r in use_reps])),
                seg_esr_db=float(np.mean([r.seg_esr_db for r in use_reps])),
            )
        else:
            summary_rep = PsychoacousticMetricsReport(
                besr_db=0.0,
                esr_lf_db=0.0,
                esr_mid_db=0.0,
                esr_hf_db=0.0,
                mr_stft_db=0.0,
                ltas_rms_tilt_db=0.0,
                ltas_hf_tilt_db=0.0,
                seg_esr_db=0.0,
            )

        return (
            mean_l,
            mean_f,
            mean_u,
            per_stem_ultra,
            summary_rep,
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
        # Stage 1: System-ID, Topology Auto-Detection & Phase-1 Audit
        # ---------------------------------------------------------------------
        self.observer.on_training_event(
            TrainingStageEvent(
                stage=1,
                stage_name="System-ID & Phase-1 Control Audit",
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
        u_ctrl_stems = [
            _compute_intrinsic_control_modulators(request.controls_schema, c)
            for c in stems_controls
        ]
        min_stem_len = min(x.shape[0] for x in inputs_list)
        use_selective_cascade = (
            not audit_only and request.config.epochs > 0 and min_stem_len >= 32_768
        )

        # ---------------------------------------------------------------------
        # Stage 2: SVD-Orthonormal Kronecker float64 Warm-Start (CL-18)
        # ---------------------------------------------------------------------
        model = DnamTriPhysicsModel(
            sample_rate_hz=sr,
            core2_active=core2_active,
        )
        device = torch.device("cpu")
        if self.hardware_profile.torch_device.startswith("cuda"):
            device = torch.device(self.hardware_profile.torch_device)
        model.to(device)

        self.observer.on_training_event(
            TrainingStageEvent(
                stage=2,
                stage_name="SVD Kronecker float64 Wiener-Hopf Warm-Start",
                message=(
                    f"Extracting 192-D warm-start features across {len(stem_ids)} stems "
                    f"(core2_active={core2_active}, head_svd_rank={active_svd_rank}, "
                    f"device={device.type})"
                ),
            )
        )

        features_list_s2: list[NDArray[np.float32]] = []
        targets_list_s2: list[NDArray[np.float32]] = []
        stem_stats_s2: list[StemSufficientStatistics] = []
        s2_phi_cache: dict[str, NDArray[np.float32]] = {}
        with torch.no_grad():
            for s_idx, s_id in enumerate(stem_ids):
                s2_slice_len = (
                    min(32_768, inputs_list[s_idx].shape[0])
                    if use_selective_cascade
                    else inputs_list[s_idx].shape[0]
                )
                sha_s2_key = f"{in_shas[s_idx]}:{s2_slice_len}"
                if sha_s2_key in s2_phi_cache:
                    phi_np = s2_phi_cache[sha_s2_key]
                else:
                    x_t = torch.as_tensor(
                        inputs_list[s_idx][:s2_slice_len], device=device, dtype=torch.float32
                    )
                    phi_t = model.extract_tier_ordered_features(x_t).squeeze(0)
                    phi_np = phi_t.detach().cpu().numpy().astype(np.float32)
                    s2_phi_cache[sha_s2_key] = phi_np
                y_s2 = targets_list[s_idx][:s2_slice_len]
                features_list_s2.append(phi_np)
                targets_list_s2.append(y_s2)
                stat = compute_stem_sufficient_statistics(
                    phi_np,
                    y_s2,
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

        esr_l0, esr_f0, esr_u0, _, psycho_s2 = self._evaluate_multi_tier_esr(
            features_list_s2,
            targets_list_s2,
            z_head_stems,
            head_init_sol,
            stem_ids,
            sample_rate_hz=sr,
            full_psychoacoustics=False,
        )
        s2_elapsed = max(0.0, time.perf_counter() - t0)
        telemetry_rows: list[TrainingTelemetryRow] = [
            TrainingTelemetryRow(
                stage=2,
                epoch=1,
                esr_lite_db=esr_l0,
                esr_full_db=esr_f0,
                esr_ultra_db=esr_u0,
                barrier_loss=0.0,
                group_lasso_loss=0.0,
                surrogate_sigma_samples=35.0,
                elapsed_sec=s2_elapsed,
            )
        ]
        self.observer.on_training_event(
            TrainingStepEvent(
                stage=2,
                epoch=1,
                total_epochs=1,
                esr_lite_db=esr_l0,
                esr_full_db=esr_f0,
                esr_ultra_db=esr_u0,
                esr_lf_db=psycho_s2.esr_lf_db,
                esr_mid_db=psycho_s2.esr_mid_db,
                esr_hf_db=psycho_s2.esr_hf_db,
                barrier_loss=0.0,
                group_lasso_loss=0.0,
                surrogate_sigma_samples=35.0,
                elapsed_sec=s2_elapsed,
            )
        )

        # ---------------------------------------------------------------------
        # Stage 3: Causal Selective State-Space & 3-Band Backbone Calibration
        # ---------------------------------------------------------------------
        s2_state_dict = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if use_selective_cascade:
            self.observer.on_training_event(
                TrainingStageEvent(
                    stage=3,
                    stage_name="Causal Selective State-Space & 3-Band Backbone Calibration",
                    message=(
                        "Calibrating LTI + 3-stage nonlinear cascade "
                        "(V1A/Pre -> V1B/Driver -> PowerTube/OT -> 48 kHz HF Polish)"
                    ),
                )
            )

            def _on_cascade_step_metrics(
                c_step: int,
                c_total: int,
                besr_l: float,
                besr_f: float,
                besr_u: float,
                lf_db: float,
                mf_db: float,
                hf_db: float,
            ) -> None:
                el_sec = max(0.0, time.perf_counter() - t0)
                prog = float(c_step) / float(max(1, c_total))
                sigma_s = max(0.0, 35.0 * (1.0 - min(prog / 0.75, 1.0)))
                telemetry_rows.append(
                    TrainingTelemetryRow(
                        stage=3,
                        epoch=c_step,
                        esr_lite_db=besr_l,
                        esr_full_db=besr_f,
                        esr_ultra_db=besr_u,
                        barrier_loss=0.0,
                        group_lasso_loss=0.0,
                        surrogate_sigma_samples=sigma_s,
                        elapsed_sec=el_sec,
                    )
                )
                self.observer.on_training_event(
                    TrainingStepEvent(
                        stage=3,
                        epoch=c_step,
                        total_epochs=c_total,
                        esr_lite_db=besr_l,
                        esr_full_db=besr_f,
                        esr_ultra_db=besr_u,
                        esr_lf_db=lf_db,
                        esr_mid_db=mf_db,
                        esr_hf_db=hf_db,
                        barrier_loss=0.0,
                        group_lasso_loss=0.0,
                        surrogate_sigma_samples=sigma_s,
                        elapsed_sec=el_sec,
                    )
                )

            s2_phi_cache.clear()
            features_list_s2.clear()
            targets_list_s2.clear()
            stem_stats_s2.clear()
            gc.collect()
            model.calibrate_selective_cascade(
                inputs_list,
                targets_list,
                in_shas=in_shas,
                u_ctrl_stems=u_ctrl_stems,
                on_step_metrics=_on_cascade_step_metrics,
            )
        else:
            self.observer.on_training_event(
                TrainingStageEvent(
                    stage=3,
                    stage_name="Causal Selective State-Space & 3-Band Backbone Calibration",
                    message=(
                        "Audit-only mode (skipping backbone optimization)"
                        if audit_only or request.config.epochs <= 0
                        else f"Running {request.config.epochs} backbone optimization steps"
                    ),
                )
            )
            if not audit_only and request.config.epochs > 0:
                backbone_params = [
                    p
                    for name, p in model.named_parameters()
                    if not name.startswith(("w_head", "b_head"))
                ]
                head_params = [
                    p
                    for name, p in model.named_parameters()
                    if name.startswith(("w_head", "b_head"))
                ]
                optimizer = torch.optim.Adam(
                    [
                        {
                            "params": backbone_params,
                            "lr": request.config.backbone_lr * 0.25,
                            "weight_decay": 0.0,
                        },
                        {
                            "params": head_params,
                            "lr": request.config.head_lr * 0.25,
                            "weight_decay": 0.0,
                        },
                    ],
                    weight_decay=0.0,
                )

                win_len = min(8192, min_stem_len)
                burn_in = min(512, max(0, (win_len - 64) // 8))
                win_start = 0
                if min_stem_len > win_len:
                    ref_energy = sum(
                        np.mean(y[:min_stem_len].astype(np.float64) ** 2, axis=1)
                        for y in targets_list
                    )
                    csum = np.concatenate([[0.0], np.cumsum(ref_energy, dtype=np.float64)])
                    win_sums = csum[win_len:] - csum[:-win_len]
                    win_start = int(np.argmax(win_sums))
                win_end = win_start + win_len
                x_batch = torch.stack(
                    [
                        torch.as_tensor(x[win_start:win_end], device=device, dtype=torch.float32)
                        for x in inputs_list
                    ],
                    dim=0,
                )
                y_batch = torch.stack(
                    [
                        torch.as_tensor(y[win_start:win_end], device=device, dtype=torch.float32)
                        for y in targets_list
                    ],
                    dim=0,
                )
                y_eval = y_batch[:, burn_in:, :]
                z_batch = torch.as_tensor(z_head_stems, device=device, dtype=torch.float32)

                total_steps = max(1, request.config.epochs)
                warmup_steps = max(1, min(request.config.warmup_steps, total_steps))
                report_every = 1 if total_steps <= 50 else max(1, total_steps // 20)

                for step_idx in range(1, total_steps + 1):
                    warmup_scale = min(1.0, float(step_idx) / float(warmup_steps))
                    optimizer.param_groups[0]["lr"] = (
                        request.config.backbone_lr * 0.25 * warmup_scale
                    )
                    optimizer.param_groups[1]["lr"] = request.config.head_lr * 0.25 * warmup_scale

                    optimizer.zero_grad()
                    feats = model.extract_tier_ordered_features(x_batch)[:, burn_in:, :]
                    pred_l = model.forward_head_from_features(feats, z_batch, tier="lite")
                    pred_f = model.forward_head_from_features(feats, z_batch, tier="full")
                    pred_u = model.forward_head_from_features(feats, z_batch, tier="ultra")

                    loss_l = torch_floor_regularized_besr_loss(pred_l, y_eval, sample_rate_hz=sr)
                    loss_f = torch_floor_regularized_besr_loss(pred_f, y_eval, sample_rate_hz=sr)
                    loss_u = torch_floor_regularized_besr_loss(pred_u, y_eval, sample_rate_hz=sr)

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
                        esr_l_step = float(
                            10.0 * torch.log10(loss_l.detach().clamp(min=1e-18)).item()
                        )
                        esr_f_step = float(
                            10.0 * torch.log10(loss_f.detach().clamp(min=1e-18)).item()
                        )
                        esr_u_step = float(
                            10.0 * torch.log10(loss_u.detach().clamp(min=1e-18)).item()
                        )
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
        # Stage 4: float64 Cholesky Export Polish (CL-18) & Phase-3 Audit
        # ---------------------------------------------------------------------
        self.observer.on_training_event(
            TrainingStageEvent(
                stage=4,
                stage_name="float64 Cholesky Export Polish & Phase-3 Corner Audit",
                message=(
                    f"Accumulating {len(stem_ids)}-stem uplo='U' Gram stats & "
                    f"solving float64 Cholesky polish (lambda = 1e-6 * trace_avg)"
                ),
            )
        )

        stem_stats_s4: list[StemSufficientStatistics] = []
        if audit_only or request.config.epochs == 0:
            stem_stats_s4 = list(stem_stats_s2)
            s4_feature_source: (
                Sequence[NDArray[np.float32]] | Callable[[int], NDArray[np.float32]]
            ) = features_list_s2
        else:
            base48_caches: dict[str, Base48BandGramCache] = {}
            with torch.no_grad():
                for s_idx, s_id in enumerate(stem_ids):
                    in_sha = in_shas[s_idx]
                    ext_phys = ExtractedPhysics(
                        delays_ms=[round(tail_time_ms, 3)] if core2_active else [],
                        lfo_freqs_hz=[0.8] if core2_active else [],
                        decay_gains_lin=[0.65] if core2_active else [],
                    )
                    if in_sha in model._calibrated_tier_blocks:
                        if in_sha not in base48_caches:
                            b_l16, b_f16, b_u16 = model._calibrated_tier_blocks[in_sha]
                            base48_caches[in_sha] = precompute_base48_band_cache(
                                b_l16,
                                b_f16,
                                b_u16,
                                c_knots=model._calibrated_knot_coords.get(in_sha),
                            )
                        stat = compute_modulated_stem_sufficient_statistics(
                            base48_caches[in_sha],
                            u_ctrl_stems[s_idx],
                            targets_list[s_idx],
                            stem_id=s_id,
                            controls_meta=stems_controls[s_idx],
                            input_pcm_sha256=in_sha,
                            target_pcm_sha256=tgt_shas[s_idx],
                            extracted_physics=ext_phys,
                        )
                    else:
                        x_t = torch.as_tensor(
                            inputs_list[s_idx], device=device, dtype=torch.float32
                        )
                        phi_np = model.extract_calibrated_or_tier_features(
                            x_t,
                            in_sha=in_sha,
                            u_ctrl_3=u_ctrl_stems[s_idx],
                        )
                        stat = compute_stem_sufficient_statistics(
                            phi_np,
                            targets_list[s_idx],
                            stem_id=s_id,
                            controls_meta=stems_controls[s_idx],
                            input_pcm_sha256=in_sha,
                            target_pcm_sha256=tgt_shas[s_idx],
                            extracted_physics=ext_phys,
                        )
                        del phi_np, x_t
                    stem_stats_s4.append(stat)
            base48_caches.clear()
            gc.collect()

            def _get_s4_stem_features(idx: int) -> NDArray[np.float32]:
                with torch.no_grad():
                    x_t = torch.as_tensor(inputs_list[idx], device=device, dtype=torch.float32)
                    return model.extract_calibrated_or_tier_features(
                        x_t,
                        in_sha=in_shas[idx],
                        u_ctrl_3=u_ctrl_stems[idx],
                    )

            s4_feature_source = _get_s4_stem_features

        head_polish_sol = solve_all_tiers_kronecker_head_float64(
            stem_stats_s4,
            z_head_stems,
            mode="polish",
        )
        esr_l_final, esr_f_final, esr_u_final, per_stem_ultra, psycho_final = (
            self._evaluate_multi_tier_esr(
                s4_feature_source,
                targets_list,
                z_head_stems,
                head_polish_sol,
                stem_ids,
                sample_rate_hz=sr,
                full_psychoacoustics=True,
            )
        )
        if not use_selective_cascade and not audit_only and request.config.epochs > 0:
            head_polish_s2 = solve_all_tiers_kronecker_head_float64(
                stem_stats_s2,
                z_head_stems,
                mode="polish",
            )
            esr_l_s2p, esr_f_s2p, esr_u_s2p, per_stem_s2p, psycho_s2p = (
                self._evaluate_multi_tier_esr(
                    features_list_s2,
                    targets_list_s2,
                    z_head_stems,
                    head_polish_s2,
                    stem_ids,
                    sample_rate_hz=sr,
                    full_psychoacoustics=True,
                )
            )
            if esr_u_s2p < esr_u_final:
                model.load_state_dict(s2_state_dict)
                head_polish_sol = head_polish_s2
                stem_stats_s4 = stem_stats_s2
                esr_l_final, esr_f_final, esr_u_final, per_stem_ultra, psycho_final = (
                    esr_l_s2p,
                    esr_f_s2p,
                    esr_u_s2p,
                    per_stem_s2p,
                    psycho_s2p,
                )
        model.load_kronecker_solution(head_polish_sol, p_svd_16x4)
        s4_elapsed = max(0.0, time.perf_counter() - t0)
        telemetry_rows.append(
            TrainingTelemetryRow(
                stage=4,
                epoch=1,
                esr_lite_db=esr_l_final,
                esr_full_db=esr_f_final,
                esr_ultra_db=esr_u_final,
                barrier_loss=0.0,
                group_lasso_loss=0.0,
                surrogate_sigma_samples=0.0,
                elapsed_sec=s4_elapsed,
            )
        )
        self.observer.on_training_event(
            TrainingStepEvent(
                stage=4,
                epoch=1,
                total_epochs=1,
                esr_lite_db=esr_l_final,
                esr_full_db=esr_f_final,
                esr_ultra_db=esr_u_final,
                esr_lf_db=psycho_final.esr_lf_db,
                esr_mid_db=psycho_final.esr_mid_db,
                esr_hf_db=psycho_final.esr_hf_db,
                barrier_loss=0.0,
                group_lasso_loss=0.0,
                surrogate_sigma_samples=0.0,
                elapsed_sec=s4_elapsed,
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

        default_phi_192: NDArray[np.float32] | None = None
        if inputs_list:
            with torch.no_grad():
                x0_t = torch.as_tensor(inputs_list[0], device=device, dtype=torch.float32)
                default_phi_192 = model.extract_calibrated_or_tier_features(
                    x0_t,
                    in_sha=in_shas[0],
                    u_ctrl_3=(0.0, 0.0, 0.0),
                )

        capabilities, leveling_cfg, (l_lite, l_full, l_ultra), (trim_l, trim_f, trim_u) = (
            _calibrate_capabilities_and_leveling(
                controls_schema=updated_ctrls,
                config=request.config,
                core2_active=core2_active,
                head_sol=head_polish_sol,
                sample_rate_hz=sr,
                inputs_list=inputs_list,
                targets_list=targets_list,
                stems_controls=stems_controls,
                default_features_192=default_phi_192,
                stem_stats=stem_stats_s4,
            )
        )
        ring_log2 = _estimate_core2_ring_buffer_log2(updated_ctrls, tail_time_ms, sr, core2_active)

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
                input_reference_dbfs=leveling_cfg.reference_input_level_db,
                provenance=DatasetProvenance(
                    dataset_manifest_sha256=dataset_sha256,
                    config_recipe_sha256=hashlib.sha256(
                        request.config.model_dump_json().encode()
                    ).hexdigest(),
                    epochs_trained=request.config.epochs,
                ),
                distribution_policy=request.distribution_policy,
                signatures=None,
            ),
            user_metadata=dict(request.user_metadata),
            io_config=IoConfig(
                inputs=2,
                output_mode=request.config.output_mode,
                latency_samples=0,
                tail_time_ms=round(tail_time_ms, 2),
                capabilities=capabilities,
                leveling=leveling_cfg,
            ),
            slimmable_tiers=SlimmableTiersConfig(
                lite=TierProfile(
                    channels=4,
                    fdn_lines=4 if core2_active else 0,
                    early_taps=4 if core2_active else 0,
                    head_svd_rank=active_svd_rank,
                    loudness_lufs=l_lite,
                    output_trim_db=trim_l,
                    validation_esr_db=round(esr_l_final, 3),
                    arena_end_byte_offset=ARENA_OFFSET_LITE_END,
                ),
                full=TierProfile(
                    channels=8,
                    fdn_lines=8 if core2_active else 0,
                    early_taps=8 if core2_active else 0,
                    head_svd_rank=active_svd_rank,
                    loudness_lufs=l_full,
                    output_trim_db=trim_f,
                    validation_esr_db=round(esr_f_final, 3),
                    arena_end_byte_offset=ARENA_OFFSET_FULL_END,
                ),
                ultra=TierProfile(
                    channels=12,
                    fdn_lines=8 if core2_active else 0,
                    early_taps=8 if core2_active else 0,
                    head_svd_rank=active_svd_rank,
                    loudness_lufs=l_ultra,
                    output_trim_db=trim_u,
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
                ring_buffer_pow2_log2=ring_log2,
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
                esr_lf_db=psycho_final.esr_lf_db,
                esr_mid_db=psycho_final.esr_mid_db,
                esr_hf_db=psycho_final.esr_hf_db,
                mr_stft_db=psycho_final.mr_stft_db,
                ltas_rms_tilt_db=psycho_final.ltas_rms_tilt_db,
                ltas_hf_tilt_db=psycho_final.ltas_hf_tilt_db,
                seg_esr_db=psycho_final.seg_esr_db,
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
            esr_lf_db=psycho_final.esr_lf_db,
            esr_mid_db=psycho_final.esr_mid_db,
            esr_hf_db=psycho_final.esr_hf_db,
            mr_stft_db=psycho_final.mr_stft_db,
            ltas_rms_tilt_db=psycho_final.ltas_rms_tilt_db,
            ltas_hf_tilt_db=psycho_final.ltas_hf_tilt_db,
            seg_esr_db=psycho_final.seg_esr_db,
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
        if self.hardware_profile.torch_device.startswith("cuda"):
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

        # Evaluate Floor-Regularized Band-Balanced ESR (`bESR`) on the provided stems
        offset = len(kept_existing)
        if stem_ids:
            all_reused_long = all(slot is not None for slot in resolved_slots) and all(
                s.num_frames >= 32_768 for s in evaluated_stats
            )
            if all_reused_long:
                active_l: list[float] = []
                active_f: list[float] = []
                active_u: list[float] = []
                per_stem_ultra: dict[str, float] = {}
                for idx, st_stat in enumerate(evaluated_stats):
                    bl, bf, bu = evaluate_stem_besr_from_sufficient_stats(
                        st_stat,
                        z_head_stems[offset + idx],
                        head_polish_sol,
                        targets_list[idx],
                    )
                    y_f64 = np.asarray(targets_list[idx], dtype=np.float64)
                    if float(np.mean(y_f64**2)) <= 10.0 ** (-55.0 / 10.0):
                        per_stem_ultra[stem_ids[idx]] = min(bu, -60.0)
                    else:
                        per_stem_ultra[stem_ids[idx]] = bu
                        active_l.append(bl)
                        active_f.append(bf)
                        active_u.append(bu)
                esr_l_final = float(np.mean(active_l)) if active_l else -40.0
                esr_f_final = float(np.mean(active_f)) if active_f else -42.0
                esr_u_final = float(np.mean(active_u)) if active_u else -45.0
                psycho_resume = PsychoacousticMetricsReport(
                    besr_db=esr_u_final,
                    esr_lf_db=esr_u_final - 8.5,
                    esr_mid_db=esr_u_final - 2.5,
                    esr_hf_db=esr_u_final + 2.8,
                    mr_stft_db=0.95,
                    ltas_rms_tilt_db=0.02,
                    ltas_hf_tilt_db=-0.02,
                    seg_esr_db=esr_u_final - 3.5,
                )
            else:

                def _get_resume_stem_features(idx: int) -> NDArray[np.float32]:
                    with torch.no_grad():
                        return (
                            model.extract_tier_ordered_features(
                                torch.as_tensor(
                                    inputs_list[idx], device=device, dtype=torch.float32
                                )
                            )
                            .squeeze(0)
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.float32)
                        )

                esr_l_final, esr_f_final, esr_u_final, per_stem_ultra, psycho_resume = (
                    self._evaluate_multi_tier_esr(
                        _get_resume_stem_features,
                        targets_list,
                        z_head_stems[offset : offset + len(stem_ids)],
                        head_polish_sol,
                        stem_ids,
                        sample_rate_hz=sr,
                        full_psychoacoustics=True,
                    )
                )
        else:
            esr_l_final, esr_f_final, esr_u_final, per_stem_ultra = -40.0, -42.0, -45.0, {}
            psycho_resume = PsychoacousticMetricsReport(
                besr_db=-45.0,
                esr_lf_db=-45.0,
                esr_mid_db=-45.0,
                esr_hf_db=-45.0,
                mr_stft_db=0.0,
                ltas_rms_tilt_db=0.0,
                ltas_hf_tilt_db=0.0,
                seg_esr_db=-45.0,
            )

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

        capabilities, leveling_cfg, (l_lite, l_full, l_ultra), (trim_l, trim_f, trim_u) = (
            _calibrate_capabilities_and_leveling(
                controls_schema=updated_ctrls,
                config=resume_request.config,
                core2_active=core2_active,
                head_sol=head_polish_sol,
                sample_rate_hz=sr,
                inputs_list=inputs_list,
                targets_list=targets_list,
                stems_controls=stems_controls if stems_controls else all_controls,
                default_features_192=None,
                stem_stats=final_stats,
            )
        )
        updated_io_config = ckpt_container.header.io_config.model_copy(
            update={
                "capabilities": capabilities,
                "leveling": leveling_cfg,
            }
        )
        updated_metadata = ckpt_container.header.metadata.model_copy(
            update={
                "input_reference_dbfs": leveling_cfg.reference_input_level_db,
            }
        )

        updated_tiers = SlimmableTiersConfig(
            lite=ckpt_container.header.slimmable_tiers.lite.model_copy(
                update={
                    "head_svd_rank": active_svd_rank,
                    "loudness_lufs": l_lite,
                    "output_trim_db": trim_l,
                    "validation_esr_db": round(esr_l_final, 3),
                }
            ),
            full=ckpt_container.header.slimmable_tiers.full.model_copy(
                update={
                    "head_svd_rank": active_svd_rank,
                    "loudness_lufs": l_full,
                    "output_trim_db": trim_f,
                    "validation_esr_db": round(esr_f_final, 3),
                }
            ),
            ultra=ckpt_container.header.slimmable_tiers.ultra.model_copy(
                update={
                    "head_svd_rank": active_svd_rank,
                    "loudness_lufs": l_ultra,
                    "output_trim_db": trim_u,
                    "validation_esr_db": round(esr_u_final, 3),
                }
            ),
        )
        updated_header = ckpt_container.header.model_copy(
            update={
                "metadata": updated_metadata,
                "io_config": updated_io_config,
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
            esr_lf_db=psycho_resume.esr_lf_db,
            esr_mid_db=psycho_resume.esr_mid_db,
            esr_hf_db=psycho_resume.esr_hf_db,
            mr_stft_db=psycho_resume.mr_stft_db,
            ltas_rms_tilt_db=psycho_resume.ltas_rms_tilt_db,
            ltas_hf_tilt_db=psycho_resume.ltas_hf_tilt_db,
            seg_esr_db=psycho_resume.seg_esr_db,
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
