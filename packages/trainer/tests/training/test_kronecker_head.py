"""Unit tests for `dnam_trainer.training.kronecker_head` (`CL-18` Dual-Ridge `float64` Kronecker
Cholesky Solver, Orthonormal SVD Control Projector, and `Arena 3` `uplo='U'` packing)."""

from __future__ import annotations

import numpy as np
import pytest

from dnam_trainer.training.kronecker_head import (
    ARENA3_STEM_BYTE_LEN,
    StemSufficientStatistics,
    compute_stem_sufficient_statistics,
    compute_svd_control_projector,
    evaluate_kronecker_head_features,
    merge_stem_statistics_for_resume,
    solve_all_tiers_kronecker_head_float64,
)


def test_svd_control_projector_handles_collinear_controls() -> None:
    # 2 collinear columns (Master == 2 * Gain) -> active_rank == 1 (CL-18)
    u_gain = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float64)
    psi_collinear = np.column_stack([u_gain, 2.0 * u_gain, np.zeros_like(u_gain)])
    p_svd, p_svd_16x4, rank, z_head = compute_svd_control_projector(psi_collinear)
    assert rank == 1
    assert p_svd.shape == (3, 4)
    assert p_svd_16x4.shape == (16, 4)
    assert z_head.shape == (5, 4)
    # Columns 1..3 of z_head are identically zero
    assert float(np.max(np.abs(z_head[:, 1:]))) < 1e-12


def test_stem_sufficient_statistics_arena3_roundtrip_and_multi_tier_kronecker_solve() -> None:
    rng = np.random.default_rng(2026)
    num_frames = 2048
    phi_s1 = rng.standard_normal((num_frames, 192)).astype(np.float32) * 0.2
    phi_s2 = rng.standard_normal((num_frames, 192)).astype(np.float32) * 0.2

    # True linear head mapping on first 64 features + control modulation
    w_true_0 = rng.standard_normal((2, 192)).astype(np.float64) * 0.05
    w_true_1 = rng.standard_normal((2, 192)).astype(np.float64) * 0.02

    y_s1 = (phi_s1.astype(np.float64) @ (w_true_0 - 0.5 * w_true_1).T).astype(np.float32)
    y_s2 = (phi_s2.astype(np.float64) @ (w_true_0 + 0.5 * w_true_1).T).astype(np.float32)

    stat1 = compute_stem_sufficient_statistics(
        phi_s1,
        y_s1,
        stem_id="s1",
        controls_meta={"drive": 2.0},
        input_pcm_sha256="11" * 32,
        target_pcm_sha256="22" * 32,
        window_size=512,
    )
    stat2 = compute_stem_sufficient_statistics(
        phi_s2,
        y_s2,
        stem_id="s2",
        controls_meta={"drive": 8.0},
        input_pcm_sha256="11" * 32,
        target_pcm_sha256="33" * 32,
        window_size=512,
    )

    # Verify Arena 3 serialization and CRC32C verification
    entry1, payload1 = stat1.to_registry_entry(0)
    assert len(payload1) == ARENA3_STEM_BYTE_LEN
    fake_arena = b"\x00" * 36_224 + payload1
    restored1 = StemSufficientStatistics.from_checkpoint_entry(entry1, fake_arena)
    assert np.allclose(restored1.g_audio_uplo_u_f64, stat1.g_audio_uplo_u_f64)
    assert np.allclose(restored1.c_audio_tier_f64, stat1.c_audio_tier_f64)

    # Tampered Arena 3 raises CRC32C mismatch
    corrupted = bytearray(fake_arena)
    corrupted[36_224] ^= 0xFF
    with pytest.raises(ValueError, match="CRC32C mismatch"):
        StemSufficientStatistics.from_checkpoint_entry(entry1, bytes(corrupted))

    # Solve multi-tier Kronecker head in both 'init' and 'polish' modes
    z_stems = np.array([[-0.5, 0.0, 0.0, 0.0], [0.5, 0.0, 0.0, 0.0]], dtype=np.float64)
    sol_init = solve_all_tiers_kronecker_head_float64([stat1, stat2], z_stems, mode="init")
    sol_polish = solve_all_tiers_kronecker_head_float64([stat1, stat2], z_stems, mode="polish")

    assert sol_init.ridge_ultra > sol_polish.ridge_ultra
    pred_s1_polish = evaluate_kronecker_head_features(
        phi_s1, z_stems[0], sol_polish.w_ultra_5x2x192
    )
    rel_err = np.linalg.norm(pred_s1_polish - y_s1) / np.linalg.norm(y_s1)
    assert rel_err < 1e-4

    # Deduplication & merge for .dnam-ckpt resume
    slots, eval_indices = merge_stem_statistics_for_resume(
        [stat1, stat2],
        [
            ("s1", {"drive": 2.0}, "11" * 32, "22" * 32),  # unchanged -> reused!
            ("s3_new", {"drive": 10.0}, "11" * 32, "44" * 32),  # new -> needs eval!
        ],
    )
    assert slots[0] is stat1
    assert slots[1] is None
    assert eval_indices == [1]

    # Validation guards & mono (1D / (T, 1)) target inputs
    stat_1d = compute_stem_sufficient_statistics(
        phi_s1[:128],
        y_s1[:128, 0],
        stem_id="s_1d",
        controls_meta={},
        input_pcm_sha256="11" * 32,
        target_pcm_sha256="22" * 32,
    )
    stat_col1 = compute_stem_sufficient_statistics(
        phi_s1[:128],
        y_s1[:128, :1],
        stem_id="s_col1",
        controls_meta={},
        input_pcm_sha256="11" * 32,
        target_pcm_sha256="22" * 32,
    )
    assert np.allclose(stat_1d.c_audio_tier_f64, stat_col1.c_audio_tier_f64)

    with pytest.raises(ValueError, match="features_t_x_192 must have shape"):
        compute_stem_sufficient_statistics(
            np.zeros((16, 8)),
            np.zeros((16, 2)),
            stem_id="bad",
            controls_meta={},
            input_pcm_sha256="11" * 32,
            target_pcm_sha256="22" * 32,
        )
    with pytest.raises(ValueError, match="target_t_x_2 must have shape"):
        compute_stem_sufficient_statistics(
            phi_s1[:16],
            np.zeros((16, 4)),
            stem_id="bad2",
            controls_meta={},
            input_pcm_sha256="11" * 32,
            target_pcm_sha256="22" * 32,
        )
    with pytest.raises(ValueError, match="requires at least 1 stem"):
        solve_all_tiers_kronecker_head_float64([], np.zeros((0, 4)))
    with pytest.raises(ValueError, match="z_head_stems must have shape"):
        solve_all_tiers_kronecker_head_float64([stat1], np.zeros((1, 2)))

    # Trilimb f32 tensor round-trip + band_floors_dbfs + is_silent + device cache
    g_limbs, c_limbs = stat1.to_trilimb_f32_tensors()
    recomb = StemSufficientStatistics.from_trilimb_f32_tensors(
        stem_id=stat1.stem_id,
        controls_meta=stat1.controls_meta,
        input_pcm_sha256=stat1.input_pcm_sha256,
        target_pcm_sha256=stat1.target_pcm_sha256,
        num_frames=stat1.num_frames,
        g_trilimb=g_limbs,
        c_trilimb=c_limbs,
    )
    np.testing.assert_allclose(recomb.g_audio_uplo_u_f64, stat1.g_audio_uplo_u_f64)
    np.testing.assert_allclose(recomb.c_audio_tier_f64, stat1.c_audio_tier_f64)

    from dnam_trainer.training.accelerator import is_mps_available
    from dnam_trainer.training.kronecker_head import (
        compute_modulated_stem_sufficient_statistics,
        precompute_base48_band_cache,
    )

    dev_str = "mps" if is_mps_available() else "cpu"
    stat_sil = compute_stem_sufficient_statistics(
        phi_s1[:128],
        y_s1[:128],
        stem_id="s_sil",
        controls_meta={},
        input_pcm_sha256="11" * 32,
        target_pcm_sha256="22" * 32,
        band_floors_dbfs=(-90.0, -90.0, -90.0),
        is_silent=True,
        device=dev_str,
    )
    assert stat_sil.num_frames == 128

    b16 = rng.standard_normal((128, 16)).astype(np.float32) * 0.1
    b_cache = precompute_base48_band_cache(b16, b16, b16, device=dev_str)
    stat_mod_sil = compute_modulated_stem_sufficient_statistics(
        b_cache,
        (0.0, 0.0, 0.0),
        y_s1[:128],
        stem_id="sm_sil",
        controls_meta={},
        input_pcm_sha256="11" * 32,
        target_pcm_sha256="22" * 32,
        band_floors_dbfs=(-90.0, -90.0, -90.0),
        is_silent=True,
        device=dev_str,
    )
    assert stat_mod_sil.num_frames == 128
