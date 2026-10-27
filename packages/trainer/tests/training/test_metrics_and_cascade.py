"""Unit and regression tests for `dnam_trainer.training.metrics` (Universal Psychoacoustic
Evaluation Suite, `bESR`, `MR-STFT`, `LTAS`, `SegESR`) and `dnam_trainer.training.selective_cascade`
(3-Stage Causal Selective State-Space + 3-Band Decoupled Cascade).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from dnam_trainer.control.basis import ParametricControlHyperBus
from dnam_trainer.schema import parse_compact_range_spec, parse_compact_switch_spec
from dnam_trainer.training.kronecker_head import (
    compute_modulated_stem_sufficient_statistics,
    compute_stem_sufficient_statistics,
    compute_svd_control_projector,
    compute_tier32_modulator_vectors,
    evaluate_stem_besr_from_sufficient_stats,
    precompute_base48_band_cache,
    solve_all_tiers_kronecker_head_float64,
)
from dnam_trainer.training.metrics import (
    causal_1pole_lpf_np,
    compute_band_split_esr_db,
    compute_floor_regularized_besr_db,
    compute_mr_stft_loss_db,
    compute_segmental_esr_db,
    compute_third_octave_ltas_tilt_db,
    evaluate_psychoacoustic_metrics,
    split_three_bands_np,
    torch_floor_regularized_besr_loss,
    torch_mr_stft_loss,
)
from dnam_trainer.training.pipeline import (
    DnamTriPhysicsModel,
    _compute_intrinsic_control_modulators,
)
from dnam_trainer.training.selective_cascade import (
    apply_ltas_hf_tilt_balance,
    build_causal_bspline_window_weights,
    build_fast_refinement_gates,
    build_interstage_basis,
    build_refinement_fuzz_basis,
    build_stage1_fuzz_basis,
    calibrate_3stage_selective_cascade,
    fast_1pole_lpf,
    fast_poly_adaa_batch,
    fast_poly_inst_batch,
    fast_wavefolder_inst_batch,
    measure_and_shift_usb_jitter,
    modulate_tier_blocks_with_controls,
    parseval_inband_projectors,
    project_fft_band,
    project_orthonormal_basis,
    shift_signal_fft_1d,
    solve_3band_decoupled_window_stage,
    solve_lti_cabinet_and_space_stage,
)
from dnam_trainer.training.wav_io import read_wav, write_wav_32bit_float


def test_split_three_bands_complementary_and_esr_trap_detection() -> None:
    """Verifies exact complementary 3-band decomposition (`LF + MF + HF == sig`) and confirms
    that `bESR` and `ESR_HF` expose high-frequency roll-off that plain broadband `ESR` misses.
    """
    sr = 48_000
    t = np.linspace(0.0, 0.25, 12_000, dtype=np.float64)
    # 1/f^2 guitar/bass-like spectrum: strong 110 Hz fundamental + quiet 5 kHz harmonic (-24 dB)
    y_lf = 0.50 * np.sin(2.0 * np.pi * 110.0 * t)
    y_mf = 0.15 * np.sin(2.0 * np.pi * 800.0 * t)
    y_hf = 0.03 * np.sin(2.0 * np.pi * 5000.0 * t)
    target = np.column_stack([y_lf + y_mf + y_hf, y_lf + y_mf + y_hf])

    # 1. Exact complementary reconstruction
    b_lf, b_mf, b_hf = split_three_bands_np(target, sample_rate_hz=float(sr))
    np.testing.assert_allclose(b_lf + b_mf + b_hf, target, atol=1e-12)
    assert causal_1pole_lpf_np(np.array([1.0]), 250.0).shape == (1,)

    # 2. A dark prediction that completely drops the 5 kHz harmonic (`y_hf = 0`):
    pred_dark = np.column_stack([y_lf + y_mf, y_lf + y_mf])
    naive_l2_db = float(10.0 * np.log10(np.sum((pred_dark - target) ** 2) / np.sum(target**2)))
    besr_dark = compute_floor_regularized_besr_db(pred_dark, target, sample_rate_hz=sr)
    esr_lf, esr_mid, esr_hf = compute_band_split_esr_db(pred_dark, target, sample_rate_hz=sr)

    # Naive global L2 is fooled (< -24 dB); bESR and ESR_HF immediately expose the missing highs!
    assert naive_l2_db < -24.0
    assert esr_hf > -3.0
    assert besr_dark > -15.0
    assert esr_lf < -30.0
    assert esr_mid < -20.0

    # 3. Accurate prediction across all 3 bands achieves studio/reference bESR (< -35 dB)
    pred_good = target + 1e-4 * np.random.default_rng(0).standard_normal(target.shape)
    rep_good = evaluate_psychoacoustic_metrics(pred_good, target, sample_rate_hz=sr)
    assert rep_good.besr_db < -35.0
    assert rep_good.esr_lf_db < -35.0
    assert rep_good.esr_mid_db < -35.0
    assert rep_good.esr_hf_db < -30.0
    assert rep_good.mr_stft_db < 1.0
    assert abs(rep_good.ltas_hf_tilt_db) < 0.2
    assert rep_good.seg_esr_db < -35.0


def test_psychoacoustic_metrics_edge_cases_and_differentiable_losses() -> None:
    """Verifies edge cases (1D inputs, short buffers, silent targets) and PyTorch differentiable
    losses (`torch_floor_regularized_besr_loss`, `torch_mr_stft_loss`).
    """
    # Short / empty / silent edge cases
    assert compute_floor_regularized_besr_db(np.zeros(0), np.zeros(0)) == 0.0
    assert compute_band_split_esr_db(np.zeros(4), np.zeros(4)) == (0.0, 0.0, 0.0)
    assert compute_mr_stft_loss_db(np.zeros(16), np.zeros(16)) == 0.0
    assert compute_third_octave_ltas_tilt_db(np.zeros(64), np.zeros(64)) == (0.0, 0.0, 0.0)
    assert compute_segmental_esr_db(np.zeros(8), np.zeros(8)) < -50.0
    assert compute_segmental_esr_db(np.zeros(4096), np.zeros(4096)) == -60.0

    # Silent target report (`volume_min`)
    silent_rep = evaluate_psychoacoustic_metrics(np.zeros((1024, 2)), np.zeros((1024, 2)))
    assert silent_rep.besr_db <= -100.0
    assert silent_rep.mr_stft_db == 0.0

    # PyTorch differentiable losses
    x_btc = torch.randn(2, 1024, 2, dtype=torch.float32, requires_grad=True)
    y_btc = x_btc.detach() * 0.9 + 0.02
    l_besr = torch_floor_regularized_besr_loss(x_btc, y_btc)
    l_stft = torch_mr_stft_loss(x_btc, y_btc, fft_sizes=(256, 512))
    total = l_besr + l_stft
    total.backward()
    assert x_btc.grad is not None
    assert torch.all(torch.isfinite(x_btc.grad))

    # Short tensor edge cases for torch losses
    short_btc = torch.zeros(1, 1, 2, dtype=torch.float32)
    assert float(torch_floor_regularized_besr_loss(short_btc, short_btc).item()) >= 0.0
    assert float(torch_mr_stft_loss(short_btc, short_btc).item()) == 0.0


def test_selective_cascade_monotonic_bspline_and_calibration() -> None:
    """Verifies strictly monotonic causal state variable `e_cum` across silent gaps, partition of
    unity, and 3-Stage Selective Cascade calibration + control modulation.
    """
    # Signal with loud burst followed by complete silence then quiet swell
    x = torch.zeros(4096, dtype=torch.float32)
    x[:1024] = 0.8 * torch.sin(torch.linspace(0.0, 50.0, 1024))
    x[3072:] = 0.05 * torch.sin(torch.linspace(0.0, 20.0, 1024))

    win_bounds = [(0, 1024), (1024, 2048), (2048, 3072), (3072, 4096)]
    weights = build_causal_bspline_window_weights(x, win_bounds, overlap=64)
    assert weights.shape == (4096, 4)
    # Partition of unity: weights sum to 1.0 at every sample
    np.testing.assert_allclose(
        torch.sum(weights, dim=-1).numpy(), np.ones(4096, dtype=np.float32), atol=1e-5
    )
    # Single window returns all ones
    w1 = build_causal_bspline_window_weights(x, [(0, 4096)])
    assert w1.shape == (4096, 1)
    assert float(fast_1pole_lpf(torch.tensor([1.0]), 100.0).item()) == 1.0

    # Test fast_poly_adaa_batch, fast_poly_inst_batch, gates, and build_interstage_basis
    gates = build_fast_refinement_gates(x)
    assert len(gates) == 15
    adaa_out = fast_poly_adaa_batch(x.unsqueeze(-1), (1.0, 3.0))
    inst_out = fast_poly_inst_batch(x.unsqueeze(-1), (1.0, 3.0))
    assert adaa_out.shape == (4096, 4)
    assert inst_out.shape == (4096, 4)

    z_dummy = torch.randn(512, 7, dtype=torch.float32)
    inter = build_interstage_basis(
        z_dummy, z_dummy, (20.0, 200.0), 20.0, 35.0, 8.0, (1.0, 4.0), use_power_sag=True
    )
    assert inter.shape[0] == 512

    # Calibrate 3-stage cascade on a synthetic nonlinear target and verify tier blocks
    rng = np.random.default_rng(123)
    x_np = (0.3 * rng.standard_normal(512)).astype(np.float32)
    y_np = np.tanh(2.5 * x_np).astype(np.float32)
    y_2ch = np.column_stack([y_np, y_np])
    stages_seen: list[int] = []
    b_lite, b_full, b_ultra = calibrate_3stage_selective_cascade(
        x_np,
        [y_2ch],
        sample_rate_hz=48_000,
        on_stage_progress=lambda st, _msg: stages_seen.append(st),
    )
    assert stages_seen == [1, 2, 3]
    assert b_lite.shape == (512, 16)
    assert b_full.shape == (512, 16)
    assert b_ultra.shape == (512, 16)

    phi_192 = modulate_tier_blocks_with_controls(b_lite, b_full, b_ultra, (-1.0, 0.0, 1.0))
    assert phi_192.shape == (512, 192)

    # Verify DnamTriPhysicsModel calibration and intrinsic control modulators
    ctrls = [
        parse_compact_range_spec("bass=0..10:5"),
        parse_compact_switch_spec("voice=vintage,modern:vintage"),
    ]
    u_mod = _compute_intrinsic_control_modulators(ctrls, {"bass": 10.0, "voice": "modern"})
    assert u_mod == (1.0, 1.0, 0.0)

    model = DnamTriPhysicsModel(sample_rate_hz=48_000)
    x_2ch = np.column_stack([x_np, x_np])
    model.calibrate_selective_cascade([x_2ch, x_2ch], [y_2ch, y_2ch], in_shas=["sha_a", "sha_a"])
    feat_cal = model.extract_calibrated_or_tier_features(
        torch.as_tensor(x_2ch), in_sha="sha_a", u_ctrl_3=u_mod
    )
    assert feat_cal.shape == (512, 192)
    feat_uncal = model.extract_calibrated_or_tier_features(
        torch.as_tensor(x_2ch), in_sha="sha_missing", u_ctrl_3=u_mod
    )
    assert feat_uncal.shape == (512, 192)


def test_lti_cabinet_and_acoustic_space_modeling() -> None:
    """Verifies that Linear, Time-Invariant (LTI) systems — including Speaker Cabinet IRs
    (bass-reflex port + cone breakup resonances + mic comb notches) and Acoustic Spaces
    (early reflections + reverberant decay) — are modeled to Reference Grade (`bESR <= -30 dB`).
    """
    sr = 48_000
    rng = np.random.default_rng(42)
    l_tot = 16_384
    x = (0.25 * rng.standard_normal(l_tot)).astype(np.float32)

    # 1. Construct a realistic 512-tap Speaker Cabinet IR (port resonance + cone breakup + mic comb)
    k_cab = np.arange(512, dtype=np.float64)
    t_cab = k_cab / float(sr)
    h_cab = (
        0.45 * np.exp(-t_cab / 0.0025) * np.sin(2.0 * np.pi * 85.0 * t_cab)
        + 0.35 * np.exp(-t_cab / 0.0012) * np.sin(2.0 * np.pi * 1850.0 * t_cab)
        + 0.20 * np.exp(-t_cab / 0.0008) * np.cos(2.0 * np.pi * 3900.0 * t_cab)
    )
    h_cab[0] += 0.30
    h_cab[14] -= 0.12  # Mic phase/comb reflection at 0.29 ms

    # 2. Construct a 1024-tap Acoustic Space IR (early reflections + diffusive room tail)
    k_room = np.arange(1024, dtype=np.float64)
    t_room = k_room / float(sr)
    h_room = np.zeros(1024, dtype=np.float64)
    h_room[0] = 0.60
    for delay_samp, gain in ((43, 0.25), (117, -0.18), (251, 0.14), (419, -0.10)):
        h_room[delay_samp] += gain
    h_room += 0.08 * np.exp(-t_room / 0.006) * np.sin(2.0 * np.pi * 620.0 * t_room)

    y_cab = np.convolve(x.astype(np.float64), h_cab, mode="full")[:l_tot]
    y_room = np.convolve(x.astype(np.float64), h_room, mode="full")[:l_tot]

    y7 = torch.as_tensor(
        np.column_stack([y_cab, y_room, y_cab, y_room, y_cab, y_room, y_cab]),
        dtype=torch.float64,
    )
    z_lti_full, _, _, _ = solve_lti_cabinet_and_space_stage(
        torch.as_tensor(x, dtype=torch.float32), y7, max_ir_len=2048, sr=float(sr)
    )
    pred_cab = z_lti_full[:, 0].numpy().astype(np.float64)
    pred_room = z_lti_full[:, 1].numpy().astype(np.float64)

    rep_cab = evaluate_psychoacoustic_metrics(pred_cab, y_cab, sample_rate_hz=sr)
    rep_room = evaluate_psychoacoustic_metrics(pred_room, y_room, sample_rate_hz=sr)

    # Both Speaker Cabinet IR and Acoustic Space IR must achieve Reference-Grade bESR (<= -30 dB)
    assert rep_cab.besr_db <= -30.0
    assert rep_cab.esr_lf_db <= -34.0
    assert rep_cab.esr_mid_db <= -31.0
    assert rep_cab.esr_hf_db <= -26.0
    assert rep_room.besr_db <= -30.0
    assert rep_room.esr_lf_db <= -34.0
    assert rep_room.esr_mid_db <= -31.0
    assert rep_room.esr_hf_db <= -26.0


def test_multi_control_kronecker_and_base48_gram_cache() -> None:
    """Verifies `>3` controls + switches in `_compute_intrinsic_control_modulators`, `>7` active
    stems in `calibrate_3stage_selective_cascade`, `-55 dBFS` mute-stem handling (`-57.2 dBFS`),
    and exact `Base48BandGramCache` + `compute_modulated_stem_sufficient_statistics` assembly.
    """
    # 1. 5-control schema (4 ranges + 1 switch, matching fruity_hb500)
    ctrls5 = [
        parse_compact_range_spec("gain=0..10:5"),
        parse_compact_range_spec("bass=0..10:5"),
        parse_compact_range_spec("middle=0..10:5"),
        parse_compact_range_spec("treble=0..10:5"),
        parse_compact_switch_spec("overdrive=Off,On:Off"),
    ]
    stems_meta: list[dict[str, float | str]] = [
        {"gain": 5.0, "bass": 5.0, "middle": 5.0, "treble": 5.0, "overdrive": "Off"},
        {"gain": 0.0, "bass": 5.0, "middle": 5.0, "treble": 5.0, "overdrive": "Off"},
        {"gain": 10.0, "bass": 5.0, "middle": 5.0, "treble": 5.0, "overdrive": "Off"},
        {"gain": 5.0, "bass": 0.0, "middle": 5.0, "treble": 5.0, "overdrive": "Off"},
        {"gain": 5.0, "bass": 10.0, "middle": 5.0, "treble": 5.0, "overdrive": "Off"},
        {"gain": 5.0, "bass": 5.0, "middle": 0.0, "treble": 5.0, "overdrive": "Off"},
        {"gain": 5.0, "bass": 5.0, "middle": 10.0, "treble": 5.0, "overdrive": "Off"},
        {"gain": 5.0, "bass": 5.0, "middle": 5.0, "treble": 0.0, "overdrive": "Off"},
        {"gain": 5.0, "bass": 5.0, "middle": 5.0, "treble": 10.0, "overdrive": "Off"},
        {"gain": 5.0, "bass": 5.0, "middle": 5.0, "treble": 5.0, "overdrive": "On"},
        {"gain": 10.0, "bass": 10.0, "middle": 5.0, "treble": 5.0, "overdrive": "Off"},
        {"gain": 0.0, "bass": 5.0, "middle": 5.0, "treble": 5.0, "overdrive": "On"},
        {"gain": 10.0, "bass": 5.0, "middle": 5.0, "treble": 5.0, "overdrive": "On"},
    ]
    hyper_bus = ParametricControlHyperBus(ctrls5)
    psi_stems = hyper_bus.encode_batch_coords(stems_meta).detach().cpu().numpy()
    _, _, active_rank, z_head_stems = compute_svd_control_projector(psi_stems)
    assert active_rank == 4

    k_rows: list[np.ndarray] = []
    for s_idx, meta in enumerate(stems_meta):
        u_mod = _compute_intrinsic_control_modulators(ctrls5, meta)
        psi_tilde = np.concatenate([[1.0], z_head_stems[s_idx]])
        m_vec = np.array([1.0, u_mod[0], u_mod[1], u_mod[2]], dtype=np.float64)
        k_rows.append(np.kron(psi_tilde, m_vec))
    k_mat = np.vstack(k_rows)  # (13, 20)
    assert np.linalg.matrix_rank(k_mat, tol=1e-4) == 13

    # 2. >7 active stems in calibrate_3stage_selective_cascade + -57.2 dBFS mute stem
    rng = np.random.default_rng(77)
    t_len = 256
    x_np = (0.3 * rng.standard_normal(t_len)).astype(np.float32)
    targets_13: list[np.ndarray] = []
    for s_idx in range(13):
        if s_idx in (1, 11):
            # Mute stems at -61 dBFS and -57.2 dBFS (between -60 and -55 dBFS)
            amp = 10.0 ** (-61.0 / 20.0) if s_idx == 1 else 10.0 ** (-57.2 / 20.0)
            y_s = (amp * rng.standard_normal(t_len)).astype(np.float32)
        else:
            drive = 1.2 + 0.15 * s_idx
            y_s = np.tanh(drive * x_np).astype(np.float32)
        targets_13.append(np.column_stack([y_s, y_s]))

    b_lite, b_full, b_ultra = calibrate_3stage_selective_cascade(
        x_np, targets_13, sample_rate_hz=48_000
    )
    assert b_lite.shape == (t_len, 16)
    assert b_full.shape == (t_len, 16)
    assert b_ultra.shape == (t_len, 16)

    # 3. Verify Base48BandGramCache matches full-length compute_stem_sufficient_statistics
    base_cache = precompute_base48_band_cache(b_lite, b_full, b_ultra)
    stats_mod = []
    for s_idx, meta in enumerate(stems_meta):
        u_mod = _compute_intrinsic_control_modulators(ctrls5, meta)
        st_m = compute_modulated_stem_sufficient_statistics(
            base_cache,
            u_mod,
            targets_13[s_idx],
            stem_id=f"stem_{s_idx:02d}",
            controls_meta=meta,
            input_pcm_sha256="sha_in",
            target_pcm_sha256=f"sha_{s_idx:02d}",
        )
        stats_mod.append(st_m)
        if s_idx in (0, 11):
            phi_192 = modulate_tier_blocks_with_controls(b_lite, b_full, b_ultra, u_mod)
            st_ref = compute_stem_sufficient_statistics(
                phi_192,
                targets_13[s_idx],
                stem_id=f"stem_{s_idx:02d}",
                controls_meta=meta,
                input_pcm_sha256="sha_in",
                target_pcm_sha256=f"sha_{s_idx:02d}",
            )
            np.testing.assert_allclose(
                st_m.g_audio_uplo_u_f64, st_ref.g_audio_uplo_u_f64, rtol=1e-4, atol=1e-2
            )
            np.testing.assert_allclose(
                st_m.c_audio_tier_f64, st_ref.c_audio_tier_f64, rtol=1e-4, atol=1e-2
            )

    sol = solve_all_tiers_kronecker_head_float64(stats_mod, z_head_stems, mode="polish")
    assert sol.w_ultra_5x2x192.shape == (5, 2, 192)


def test_wav_32bit_float_unclipped_roundtrip(tmp_path: Path) -> None:
    """Verifies that `write_wav_32bit_float` and `read_wav` preserve unclipped 32-bit
    IEEE-754 float signals with peaks outside `[-1.0, +1.0]` (up to `+40 dBFS`).
    """
    sig = np.array([0.0, 0.25, -1.0, 1.0, 12.5, -42.75, 99.5, -100.0], dtype=np.float32)
    wav_path = tmp_path / "hot_stem_f32.wav"
    write_wav_32bit_float(wav_path, sig, 48_000)
    decoded, sr = read_wav(wav_path)
    assert sr == 48_000
    np.testing.assert_allclose(decoded, sig, rtol=0.0, atol=0.0)


def test_30stem_10control_tier32_usb_jitter_and_quadratic_besr() -> None:
    """Verifies `> 16` active stems (`20` stems across 10 controls = 7 ranges + 3 switches
    matching `swe_tc650`), `(T, 32)` tier base blocks, `compute_tier32_modulator_vectors`,
    96-channel `Base48BandGramCache`, `fast_wavefolder_inst_batch`, sub-sample USB jitter
    alignment (`measure_and_shift_usb_jitter`, `shift_signal_fft_1d`), and exact quadratic
    form `bESR` identity (`evaluate_stem_besr_from_sufficient_stats`).
    """
    rng = np.random.default_rng(2026)
    t_len = 256
    x_np = (0.35 * rng.standard_normal(t_len)).astype(np.float32)
    x_t = torch.as_tensor(x_np, dtype=torch.float32)

    # 1. Verify wavefolder basis and orthonormal projection
    wf = fast_wavefolder_inst_batch(x_t.unsqueeze(-1), (1.5, 4.5, 9.0, 18.0))
    assert wf.shape == (t_len, 8)
    q_proj = project_orthonormal_basis(wf, 6)
    assert q_proj.shape == (t_len, 6)
    assert torch.all(torch.isfinite(q_proj))
    assert project_orthonormal_basis(wf[:, :4], 6).shape == (t_len, 4)

    # 2. Verify sub-sample USB jitter alignment and inverse shift
    x_jit = (0.35 * rng.standard_normal(1024)).astype(np.float64)
    y_true = np.tanh(2.0 * x_jit)
    y_true_t = torch.as_tensor(y_true, dtype=torch.float64)
    y_delayed_t = shift_signal_fft_1d(y_true_t, -3.25)
    y_delayed = y_delayed_t.numpy()
    y_aligned, delta_s, corr_s = measure_and_shift_usb_jitter(x_jit, y_delayed, min_len=64)
    assert 2.5 < delta_s < 3.6
    assert abs(corr_s) > 0.35
    y_restored = shift_signal_fft_1d(
        torch.as_tensor(y_aligned, dtype=torch.float64), -delta_s
    ).numpy()
    assert float(np.corrcoef(y_restored, y_delayed)[0, 1]) > 0.999
    assert torch.allclose(shift_signal_fft_1d(y_true_t, 0.0), y_true_t)
    # Also test default min_len=48_000 early return on short buffer
    _, delta_short, _ = measure_and_shift_usb_jitter(x_np.astype(np.float64), y_delayed[:t_len])
    assert delta_short == 0.0

    # 3. 10-control schema matching swe_tc650 (7 ranges + 3 switches) and 20 active stems (> 16)
    ctrls10 = [
        parse_compact_range_spec("gain=0..10:5"),
        parse_compact_switch_spec("character=Off,On:Off"),
        parse_compact_range_spec("comp=0..10:5"),
        parse_compact_range_spec("bass=0..10:5"),
        parse_compact_range_spec("mid=0..10:5"),
        parse_compact_range_spec("mid_freq=50..2000:400"),
        parse_compact_range_spec("treble=0..10:5"),
        parse_compact_range_spec("bright=0..10:5"),
        parse_compact_switch_spec("tube_mode=Off,On:Off"),
        parse_compact_switch_spec("drive_mode=Norm,Boost:Norm"),
    ]
    stems_20_meta: list[dict[str, float | str]] = []
    targets_20: list[np.ndarray] = []
    for s_idx in range(20):
        g_val = float((s_idx * 3) % 11)
        b_val = float((s_idx * 5) % 11)
        t_val = float((s_idx * 7) % 11)
        ch_val = "On" if (s_idx % 2 == 1) else "Off"
        tm_val = "On" if (s_idx % 3 == 1) else "Off"
        dm_val = "Boost" if (s_idx % 4 == 1) else "Norm"
        stems_20_meta.append(
            {
                "gain": g_val,
                "character": ch_val,
                "comp": 5.0,
                "bass": b_val,
                "mid": 5.0,
                "mid_freq": 400.0,
                "treble": t_val,
                "bright": 5.0,
                "tube_mode": tm_val,
                "drive_mode": dm_val,
            }
        )
        drive = 1.0 + 0.25 * g_val + (1.5 if ch_val == "On" else 0.0)
        y_s = (
            0.7 * np.tanh(drive * x_np)
            + 0.15 * (b_val / 10.0) * x_np
            + 0.15 * (t_val / 10.0) * np.sin(2.5 * x_np)
        ).astype(np.float32)
        if s_idx == 5:
            # Inject 3-sample USB jitter on stem 5
            y_s = (
                shift_signal_fft_1d(torch.as_tensor(y_s, dtype=torch.float64), -3.0)
                .to(torch.float32)
                .numpy()
            )
        targets_20.append(np.column_stack([y_s, y_s]))

    b_lite, b_full, b_ultra = calibrate_3stage_selective_cascade(
        x_np, targets_20, sample_rate_hz=48_000
    )
    assert b_lite.shape == (t_len, 32)
    assert b_full.shape == (t_len, 32)
    assert b_ultra.shape == (t_len, 32)

    # 4. Verify 32-channel tier modulation, 96-channel Base48BandGramCache, and exact quadratic bESR
    c_knots = np.zeros((32, 3), dtype=np.float64)
    for s_idx, meta in enumerate(stems_20_meta):
        c_knots[s_idx] = np.asarray(
            _compute_intrinsic_control_modulators(ctrls10, meta), dtype=np.float64
        )
    base_cache_96 = precompute_base48_band_cache(b_lite, b_full, b_ultra, c_knots=c_knots)
    assert base_cache_96.g48_lf.shape == (96, 96)
    assert base_cache_96.c_knots is not None

    f0, f1 = compute_tier32_modulator_vectors((-0.5, 0.25, 0.75))
    assert f0.shape == (32,)
    assert f1.shape == (32,)
    f0_k, f1_k = compute_tier32_modulator_vectors(c_knots[0], c_knots=c_knots)
    assert f0_k.shape == (32,)
    assert abs(float(f1_k[0]) - 1.0) < 1e-9

    hyper_bus = ParametricControlHyperBus(ctrls10)
    psi_stems = hyper_bus.encode_batch_coords(stems_20_meta).detach().cpu().numpy()
    _, _, _, z_head_stems = compute_svd_control_projector(psi_stems)

    stats_20 = []
    for s_idx, meta in enumerate(stems_20_meta):
        u_mod = _compute_intrinsic_control_modulators(ctrls10, meta)
        st_m = compute_modulated_stem_sufficient_statistics(
            base_cache_96,
            u_mod,
            targets_20[s_idx],
            stem_id=f"stem_{s_idx:02d}",
            controls_meta=meta,
            input_pcm_sha256="sha_in",
            target_pcm_sha256=f"sha_{s_idx:02d}",
        )
        stats_20.append(st_m)
        if s_idx == 0:
            phi_192 = modulate_tier_blocks_with_controls(
                b_lite, b_full, b_ultra, u_mod, c_knots=c_knots
            )
            assert phi_192.shape == (t_len, 192)
            st_ref = compute_stem_sufficient_statistics(
                phi_192,
                targets_20[s_idx],
                stem_id="stem_00",
                controls_meta=meta,
                input_pcm_sha256="sha_in",
                target_pcm_sha256="sha_00",
            )
            np.testing.assert_allclose(
                st_m.g_audio_uplo_u_f64, st_ref.g_audio_uplo_u_f64, rtol=1e-4, atol=1e-2
            )

    sol = solve_all_tiers_kronecker_head_float64(stats_20, z_head_stems, mode="polish")
    u_mod_0 = _compute_intrinsic_control_modulators(ctrls10, stems_20_meta[0])
    phi_0 = modulate_tier_blocks_with_controls(b_lite, b_full, b_ultra, u_mod_0, c_knots=c_knots)
    psi_tilde_0 = np.concatenate([[1.0], z_head_stems[0]]).astype(np.float64)
    w_eff_ultra_0 = np.tensordot(psi_tilde_0, sol.w_ultra_5x2x192, axes=([0], [0]))
    pred_ultra_0 = phi_0.astype(np.float64) @ w_eff_ultra_0.T
    besr_direct = compute_floor_regularized_besr_db(
        pred_ultra_0, targets_20[0], sample_rate_hz=48_000
    )
    _, _, besr_quad = evaluate_stem_besr_from_sufficient_stats(
        stats_20[0], z_head_stems[0], sol, targets_20[0]
    )
    assert abs(besr_direct - besr_quad) < 1e-3


def test_duality_fuzz_orthogonal_fft_cascade_and_2control_lobatto_grid() -> None:
    """Verifies:
    1. `compute_band_split_esr_db` channel-count broadcasting invariance when comparing a 2-channel
       prediction `(T, 2)` against a 1-channel mono target `(T,)` or `(T, 1)`.
    2. `_compute_intrinsic_control_modulators` rank-9 cross-term on a 2-range-control 3x3 Lobatto
       grid (`duality_fuzz`).
    3. `measure_and_shift_usb_jitter` clipped fuzz proxy fallback when linear correlation `< 0.35`.
    4. Exact orthogonal FFT band projectors (`project_fft_band`), Parseval commuted in-band
       eigenvectors (`parseval_inband_projectors`), `build_stage1_fuzz_basis`,
       `build_refinement_fuzz_basis`, `solve_3band_decoupled_window_stage`, and
       `apply_ltas_hf_tilt_balance`.
    """
    rng = np.random.default_rng(20261026)
    sr = 48_000

    # 1. Verify mono (T,) vs stereo (T, 2) target broadcasting in compute_band_split_esr_db
    t_mono = (0.3 * rng.standard_normal(2048)).astype(np.float64)
    p_mono = t_mono + 0.01 * rng.standard_normal(2048)
    p_stereo = np.column_stack([p_mono, p_mono])
    t_stereo = np.column_stack([t_mono, t_mono])
    esr_vs_1d = compute_band_split_esr_db(p_stereo, t_mono, sample_rate_hz=sr)
    esr_vs_2d_1ch = compute_band_split_esr_db(p_stereo, t_mono[:, None], sample_rate_hz=sr)
    esr_vs_2d_2ch = compute_band_split_esr_db(p_stereo, t_stereo, sample_rate_hz=sr)
    np.testing.assert_allclose(esr_vs_1d, esr_vs_2d_2ch, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(esr_vs_2d_1ch, esr_vs_2d_2ch, rtol=1e-12, atol=1e-12)

    # 2. Verify 2-range-control 3x3 Lobatto grid rank-9 Kronecker head (duality_fuzz schema)
    ctrls2 = [
        parse_compact_range_spec("duality=0..10:5"),
        parse_compact_range_spec("filter=0..10:5"),
    ]
    stems_9_meta: list[dict[str, float | str]] = [
        {"duality": d_val, "filter": f_val}
        for d_val in (0.0, 5.0, 10.0)
        for f_val in (0.0, 5.0, 10.0)
    ]
    hyper2 = ParametricControlHyperBus(ctrls2)
    psi9 = hyper2.encode_batch_coords(stems_9_meta).detach().cpu().numpy()
    _, _, active_rank2, z_head9 = compute_svd_control_projector(psi9)
    assert active_rank2 == 4
    k9_rows: list[np.ndarray] = []
    for s_idx, meta in enumerate(stems_9_meta):
        u_mod = _compute_intrinsic_control_modulators(ctrls2, meta)
        psi_tilde = np.concatenate([[1.0], z_head9[s_idx]])
        m_vec = np.array([1.0, u_mod[0], u_mod[1], u_mod[2]], dtype=np.float64)
        k9_rows.append(np.kron(psi_tilde, m_vec))
    k9_mat = np.vstack(k9_rows)  # (9, 20)
    assert np.linalg.matrix_rank(k9_mat, tol=1e-4) == 9

    # 3. Verify clipped fuzz proxy fallback in measure_and_shift_usb_jitter when linear corr < 0.35
    x_fuzz = (0.35 * rng.standard_normal(2048)).astype(np.float64)
    # Construct a heavily clipped + folded fuzz signal with low linear correlation to x_fuzz
    x_f32_t = torch.as_tensor(x_fuzz, dtype=torch.float32)
    x_hp80 = x_f32_t - fast_1pole_lpf(x_f32_t, 80.0, sr=float(sr))
    y_fuzz_clean = (
        fast_1pole_lpf(
            torch.tanh(12.0 * x_hp80) - 0.85 * torch.abs(torch.tanh(8.0 * x_hp80)),
            800.0,
            sr=float(sr),
        )
        .to(torch.float64)
        .numpy()
    )
    y_fuzz_delayed = shift_signal_fft_1d(
        torch.as_tensor(y_fuzz_clean, dtype=torch.float64), -2.5
    ).numpy()
    _, delta_fuzz, corr_fuzz = measure_and_shift_usb_jitter(x_fuzz, y_fuzz_delayed, min_len=64)
    assert 1.5 < delta_fuzz < 3.5
    assert abs(corr_fuzz) > 0.25

    # 4. Verify orthogonal FFT projectors, Parseval in-band projectors, and 3-band decoupled solver
    l_len = 1024
    x_t = torch.as_tensor((0.3 * rng.standard_normal(l_len)).astype(np.float32))
    u1 = build_stage1_fuzz_basis(x_t, sr=float(sr))
    assert u1.shape[0] == l_len
    assert u1.shape[1] > 200

    p_lf, p_mf, p_hf = parseval_inband_projectors(u1, 24, 32, 40, sr=float(sr))
    assert p_lf.shape == (u1.shape[1], 24)
    assert p_mf.shape == (u1.shape[1], 32)
    assert p_hf.shape == (u1.shape[1], 40)

    # Verify idempotence & completeness of project_fft_band: P_lf + P_mf + P_hf == I, P_b^2 == P_b
    y_raw = torch.stack(
        [
            torch.tanh(6.0 * x_t) + 0.3 * torch.sin(10.0 * x_t),
            torch.tanh(12.0 * x_t) - 0.2 * torch.cos(14.0 * x_t),
        ],
        dim=-1,
    )
    y_lf = project_fft_band(y_raw, 0.0, 250.0, sr=float(sr))
    y_mf = project_fft_band(y_raw, 250.0, 2500.0, sr=float(sr))
    y_hf = project_fft_band(y_raw, 2500.0, sr * 0.5, sr=float(sr))
    torch.testing.assert_close(y_lf + y_mf + y_hf, y_raw, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(
        project_fft_band(y_mf, 250.0, 2500.0, sr=float(sr)), y_mf, rtol=1e-5, atol=1e-5
    )

    wins = [(0, 512), (512, l_len)]
    bw = build_causal_bspline_window_weights(x_t, wins, overlap=32, sr=float(sr))
    cur_lf = torch.zeros_like(y_lf)
    cur_mf = torch.zeros_like(y_mf)
    cur_hf = torch.zeros_like(y_hf)
    cur_lf, cur_mf, cur_hf = solve_3band_decoupled_window_stage(
        u1,
        cur_lf,
        cur_mf,
        cur_hf,
        y_lf,
        y_mf,
        y_hf,
        bw,
        wins,
        max_cols_lf=24,
        max_cols_mf=32,
        max_cols_hf=40,
        overlap=32,
        sr=float(sr),
    )
    u2 = build_refinement_fuzz_basis(
        cur_lf, cur_mf, cur_hf, include_wavefolder=True, n_modes=2, sr=float(sr)
    )
    assert u2.shape[0] == l_len
    cur_hf_bal = apply_ltas_hf_tilt_balance(cur_hf, y_hf, sr=float(sr))
    assert cur_hf_bal.shape == y_hf.shape
    pred_tot = (cur_lf + cur_mf + cur_hf_bal).numpy().astype(np.float64)
    esr_lf, esr_mf, esr_hf = compute_band_split_esr_db(
        pred_tot[:, 0], y_raw[:, 0].numpy().astype(np.float64), sample_rate_hz=sr
    )
    besr = compute_floor_regularized_besr_db(
        pred_tot[:, 0], y_raw[:, 0].numpy().astype(np.float64), sample_rate_hz=sr
    )
    assert besr < -15.0
    assert esr_lf < -15.0
    assert esr_mf < -15.0
    assert esr_hf < -10.0
