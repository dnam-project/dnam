"""End-to-end architectural verification suite (`Tests 6-8H`) from
`docs/verification_and_guardrails.md`."""

from __future__ import annotations

import numpy as np
import torch

from dnam_engine import provenance
from dnam_engine.schema import WatermarkParams
from dnam_trainer.control.audit import (
    compute_phase2_corner_barrier_loss,
    run_phase1_preflight_audit,
    run_phase3_export_certification,
)
from dnam_trainer.control.basis import (
    compute_lobatto_degree_and_anchors,
    compute_step0_variance_gates,
    eval_node_anchored_newton_basis,
    fit_order_statistic_delay_transport,
    normalize_and_clamp_range,
)
from dnam_trainer.dsp.modulated_ufdn import dual_pointer_switch_crossfade_torch
from dnam_trainer.dsp.selective_rglru import log_octave_pole_init
from dnam_trainer.schema import (
    ResumeJobRequest,
    TrainingConfig,
    TrainingJobRequest,
    TrainingStem,
    parse_compact_range_spec,
    parse_compact_switch_spec,
)
from dnam_trainer.training.kronecker_head import (
    compute_stem_sufficient_statistics,
    compute_svd_control_projector,
    evaluate_kronecker_head_features,
    solve_all_tiers_kronecker_head_float64,
)
from dnam_trainer.training.pipeline import DnamTrainer, compute_esr_db
from dnam_trainer.training.surrogate_delay import (
    _sample_cubic_lagrange_1d,
    optimize_delay_with_surrogate,
)


def test_verification_test6_and_test7_and_test7b() -> None:
    """Verifies Test 6 (Surrogate Delay), Test 7 (Log-octave init), and Test 7B (4-Stage
    Wiener-Hopf + Polish)."""
    # Test 6: Surrogate Delay locks within < 0.05 samples from a 55-sample initial offset
    t = torch.linspace(0.0, 1.0, 2048, dtype=torch.float32)
    sig = (
        torch.exp(-((t - 0.4) ** 2) / (2.0 * 0.03**2)) * torch.sin(2.0 * 3.14159265 * 190.0 * t)
    ).unsqueeze(0)
    target, _ = _sample_cubic_lagrange_1d(sig, torch.tensor([[78.42]], dtype=torch.float32))
    est_tau = optimize_delay_with_surrogate(sig, target, init_delay_samples=23.0, num_steps=140)
    assert abs(est_tau - 78.42) < 0.05

    # Test 7: Log-octave pole initialization spans [25 Hz, 16 kHz]
    freqs, _ = log_octave_pole_init(12, sample_rate_hz=48_000.0)
    assert abs(float(freqs[0].item()) - 25.0) < 1e-4
    assert abs(float(freqs[-1].item()) - 16_000.0) < 1e-2

    # Test 7B: Dual-Ridge float64 Kronecker solve drops ESR below -25 dB at init and < -45 dB at
    # polish
    rng = np.random.default_rng(702)
    phi = rng.standard_normal((2048, 192)).astype(np.float32) * 0.25
    w_true = rng.standard_normal((2, 192)).astype(np.float64) * 0.08
    y_true = (phi.astype(np.float64) @ w_true.T).astype(np.float32)
    stat = compute_stem_sufficient_statistics(
        phi,
        y_true,
        stem_id="s0",
        controls_meta={},
        input_pcm_sha256="00" * 32,
        target_pcm_sha256="11" * 32,
    )
    z0 = np.zeros((1, 4), dtype=np.float64)
    sol_init = solve_all_tiers_kronecker_head_float64([stat], z0, mode="init")
    sol_pol = solve_all_tiers_kronecker_head_float64([stat], z0, mode="polish")
    esr_init = compute_esr_db(
        evaluate_kronecker_head_features(phi, z0[0], sol_init.w_ultra_5x2x192), y_true
    )
    esr_pol = compute_esr_db(
        evaluate_kronecker_head_features(phi, z0[0], sol_pol.w_ultra_5x2x192), y_true
    )
    assert esr_init < -25.0
    assert esr_pol < -45.0
    assert esr_pol < esr_init


def test_verification_tests_8_8b_8c_8d_8e_8f() -> None:
    """Verifies Tests 8, 8B, 8C, 8D, 8E, and 8F across the parametric control bus and audit
    registry."""
    # Test 8 & 8B: 3-Phase Non-Stop Control-Space Clamping + Stop-Gradient Corner Barrier
    ctrls = [
        parse_compact_range_spec("gain=0..10:5"),
        parse_compact_switch_spec("channel=clean,lead,ultra:clean"),
    ]
    p1 = run_phase1_preflight_audit(
        ctrls,
        [
            {"controls": {"gain": 2.0, "channel": "clean"}},
            {"controls": {"gain": 6.0, "channel": "lead"}},
        ],
        corner_pole_radii={"channel=lead": 0.9999},
    )
    assert p1.updated_controls[0].safe_clamped_range == (2.0, 6.0)
    assert p1.updated_controls[1].active_options == ["clean", "lead"]
    assert "channel=lead" in p1.updated_controls[0].conditional_max

    t_main = torch.tensor([0.98], dtype=torch.float64, requires_grad=True)
    dt_fm = torch.tensor([0.04], dtype=torch.float64, requires_grad=True)
    b_loss = compute_phase2_corner_barrier_loss(t_main, dt_fm, 0.9990)
    b_loss.backward()
    assert t_main.grad is None or float(torch.abs(t_main.grad).item()) == 0.0

    # Test 8C: 128-sample raised-cosine dual-pointer delay switch crossfade
    sig = torch.sin(torch.linspace(0.0, 20.0, 512, dtype=torch.float64))
    y_xf = dual_pointer_switch_crossfade_torch(sig, 12.0, 36.0, trig_idx=128, xfade_len=128)
    assert torch.all(torch.isfinite(y_xf))

    # Test 8D: Order-statistic delay transport + Step-0 physical variance gating
    inter, slp = fit_order_statistic_delay_transport([-1.0, 1.0], [[30.0, 10.0], [20.0, 60.0]])
    assert inter[0] < inter[1]
    assert np.all(slp > 0.0)
    gates = compute_step0_variance_gates([[-1.0, -1.0], [1.0, 1.0]], [[-2.0, 0.0], [2.0, 0.0]])
    assert gates[0, 1] == 0.0

    # Test 8E: Fixed declared_range coordinate normalization + Lobatto cardinality degree
    _, u_n1, _ = normalize_and_clamp_range(5.0, (0.0, 10.0), (2.0, 6.0))
    _, u_n2, _ = normalize_and_clamp_range(5.0, (0.0, 10.0), (0.0, 10.0))
    assert u_n1 == u_n2 == 0.0

    # Test 8F: SVD collinear control projection + 32-pt FFT head spectral gain audit
    u_col = np.linspace(-1.0, 1.0, 6)
    _, _, active_rank, _ = compute_svd_control_projector(np.column_stack([u_col, u_col * 3.0]))
    assert active_rank == 1
    _, trust_r, p3_rep = run_phase3_export_certification(
        ctrls,
        head_weights_corners=np.ones((2, 16), dtype=np.float64) * 8.0,
        max_train_stem_spec_peak=1.0,
    )
    assert trust_r < 2.5
    assert any("[CLAMP-HEAD-GAIN]" in c for c in p3_rep.phase3_clamps)


def test_verification_test8g_and_test8h(
    synthetic_training_stems: list[TrainingStem],
) -> None:
    """Verifies Test 8G (`.dnam-ckpt` iterative stem expansion parity) and Test 8H (Provenance &
    Watermarking)."""
    ctrls = [
        parse_compact_range_spec("drive=0..10:5"),
        parse_compact_switch_spec("attack=cut,flat,boost:flat"),
    ]
    trainer = DnamTrainer()
    seed_hex = "55" * 32

    # 1. Train with 2 stems (`epochs=0` pure closed-form Kronecker solve)
    req_2 = TrainingJobRequest(
        model_id="verify_8g",
        model_name="Verify 8G",
        controls_schema=ctrls,
        config=TrainingConfig(epochs=1, topology="core1_only"),
        stems=synthetic_training_stems[:2],
    )
    _, _, ckpt_2 = trainer.fit_job(req_2, creator_seed_hex=seed_hex, audit_only=True)

    # 2. Resume with 3rd stem (`stem_02_high`) vs full 3-stem solve from scratch (`audit_only=True`)
    req_3_full = TrainingJobRequest(
        model_id="verify_8g",
        model_name="Verify 8G",
        controls_schema=ctrls,
        config=TrainingConfig(epochs=1, topology="core1_only"),
        stems=synthetic_training_stems,
    )
    _, dnam_full, _ = trainer.fit_job(req_3_full, creator_seed_hex=seed_hex, audit_only=True)

    resume_req = ResumeJobRequest(
        checkpoint_bytes=ckpt_2.to_bytes(),
        new_stems=synthetic_training_stems,
        config=TrainingConfig(epochs=1, topology="core1_only"),
    )
    _, dnam_resumed, _, res_info = trainer.resume_from_checkpoint(
        resume_req,
        controls_schema=ctrls,
        creator_seed_hex=seed_hex,
    )
    assert res_info.reused_stem_count == 2
    assert res_info.evaluated_stem_count == 1
    # Exact Arena 0..2 byte parity between Tier-1 resume and full 3-stem solve!
    assert bytes(dnam_resumed.arena) == bytes(dnam_full.arena)

    # Also verify Node-Anchored Newton basis zero perturbation on old anchors
    deg, anchors = compute_lobatto_degree_and_anchors([-0.6, 0.0, 0.6], default_u=0.0)
    assert (
        float(torch.max(torch.abs(eval_node_anchored_newton_basis(0.0, anchors, deg))).item())
        < 1e-15
    )

    # Test 8H: Ed25519 provenance + Dual-Mechanism Buyer Watermarking + 160-D structural hash
    assert str(dnam_resumed.verification) == "CreatorVerified"
    code = provenance.generate_tardos_code(
        b"\xaa" * 32,
        model_id="verify_8g",
        license_id="license-01",
    )
    wm = dnam_resumed.with_watermark(
        WatermarkParams(
            tardos_code_4608=tuple(float(x) for x in code),
            tardos_epsilon=2e-4,
        )
    )
    z_score = provenance.score_tardos_watermark(
        bytes(dnam_resumed.arena),
        bytes(wm.arena),
        code_4608=code,
    )
    assert z_score > 30.0
    assert (
        provenance.structural_cosine_similarity(
            dnam_resumed.structural_hash(),
            wm.structural_hash(),
        )
        > 0.999999
    )
