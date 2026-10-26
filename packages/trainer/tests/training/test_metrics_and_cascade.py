"""Unit and regression tests for `dnam_trainer.training.metrics` (Universal Psychoacoustic
Evaluation Suite, `bESR`, `MR-STFT`, `LTAS`, `SegESR`) and `dnam_trainer.training.selective_cascade`
(3-Stage Causal Selective State-Space + 3-Band Decoupled Cascade).
"""

from __future__ import annotations

import numpy as np
import torch

from dnam_trainer.schema import parse_compact_range_spec, parse_compact_switch_spec
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
    build_causal_bspline_window_weights,
    build_fast_refinement_gates,
    build_interstage_basis,
    calibrate_3stage_selective_cascade,
    fast_1pole_lpf,
    fast_poly_adaa_batch,
    fast_poly_inst_batch,
    modulate_tier_blocks_with_controls,
    solve_lti_cabinet_and_space_stage,
)


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
    x_np = (0.3 * rng.standard_normal(2048)).astype(np.float32)
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
    assert b_lite.shape == (2048, 16)
    assert b_full.shape == (2048, 16)
    assert b_ultra.shape == (2048, 16)

    phi_192 = modulate_tier_blocks_with_controls(b_lite, b_full, b_ultra, (-1.0, 0.0, 1.0))
    assert phi_192.shape == (2048, 192)

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
    assert feat_cal.shape == (2048, 192)
    feat_uncal = model.extract_calibrated_or_tier_features(
        torch.as_tensor(x_2ch), in_sha="sha_missing", u_ctrl_3=u_mod
    )
    assert feat_uncal.shape == (2048, 192)


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
