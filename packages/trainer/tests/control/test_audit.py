"""Unit tests for `dnam_trainer.control.audit` (`CL-15`..`CL-17`: 3-Phase Non-Stop Control-Space
Audit, Stop-Gradient Virtual Corner Barrier, and Consolidated Stem Prescription Report)."""

from __future__ import annotations

import numpy as np
import torch

from dnam_trainer.control.audit import (
    compute_phase2_corner_barrier_loss,
    format_consolidated_prescription_report,
    run_phase1_preflight_audit,
    run_phase2_rail_lockup_audit,
    run_phase3_export_certification,
)
from dnam_trainer.dsp.poly_adaa import DifferentiablePolyAdaa
from dnam_trainer.schema import parse_compact_range_spec, parse_compact_switch_spec


def test_phase1_preflight_audit_range_switch_and_stability_clamps() -> None:
    ctrls = [
        parse_compact_range_spec("drive=0..10:5"),
        parse_compact_switch_spec("voice=clean,crunch,lead:clean"),
    ]
    # Only record stems in drive in [2.0, 7.0] and voice in {'clean', 'crunch'}
    stems = [
        {"controls": {"drive": 2.0, "voice": "clean"}},
        {"controls": {"drive": 5.0, "voice": "clean"}},
        {"controls": {"drive": 7.0, "voice": "crunch"}},
    ]
    res = run_phase1_preflight_audit(
        ctrls,
        stems,
        corner_pole_radii={"voice=crunch": 0.9998},
    )
    drive_up = res.updated_controls[0]
    voice_up = res.updated_controls[1]

    # CL-15: [CLAMP-RANGE] and [CLAMP-STABILITY] and [CLAMP-SWITCH]
    assert drive_up.safe_clamped_range == (2.0, 7.0)
    assert drive_up.lobatto_degree == 2
    assert "voice=crunch" in drive_up.conditional_max
    assert voice_up.active_options == ["clean", "crunch"]
    assert any("[CLAMP-RANGE]" in c for c in res.audit_report.phase1_clamps)
    assert any("[CLAMP-SWITCH]" in c for c in res.audit_report.phase1_clamps)
    assert any("[CLAMP-STABILITY]" in c for c in res.audit_report.phase1_clamps)
    assert len(res.structured_prescriptions) >= 3


def test_phase2_stop_gradient_corner_barrier_and_rail_lockup() -> None:
    # CL-16: Stop-gradient on theta_main ensures 0.000000 gradient leakage onto 1st-order main
    # effects
    theta_main = torch.tensor([0.95, 0.98], dtype=torch.float64, requires_grad=True)
    delta_fm = torch.tensor([0.08, 0.05], dtype=torch.float64, requires_grad=True)
    loss = compute_phase2_corner_barrier_loss(
        theta_main,
        delta_fm,
        0.9990,
        theta_safe_min=0.25,
        lambda_bar=10.0,
    )
    loss.backward()
    assert theta_main.grad is None or float(torch.max(torch.abs(theta_main.grad)).item()) == 0.0
    assert delta_fm.grad is not None
    assert float(torch.max(torch.abs(delta_fm.grad)).item()) > 0.0

    # Rail lockup audit ([CLAMP-RAIL])
    layer = DifferentiablePolyAdaa(num_channels=2)
    # Drive hard into rails so rail lockup fraction > 0.95
    _, _ = layer(torch.full((1, 2, 128), 50.0, dtype=torch.float32))
    assert layer.last_rail_lockup_fraction > 0.95

    ctrls = [parse_compact_range_spec("drive=0..10:5")]
    up_ctrls, warnings = run_phase2_rail_lockup_audit([layer], ctrls, threshold=0.95)
    assert len(warnings) == 1
    assert "[CLAMP-RAIL]" in warnings[0]
    assert up_ctrls[0].safe_clamped_range == (0.0, 9.0)


def test_phase3_export_certification_and_prescription_report() -> None:
    ctrls = [
        parse_compact_range_spec("drive=0..10:5"),
        parse_compact_switch_spec("voice=clean,lead:clean"),
    ]
    # Trigger both [CLAMP-DYNAMIC] (delay < 4.0) and [CLAMP-HEAD-GAIN] (> +6 dB corner peak)
    hot_head_corners = np.ones((4, 16), dtype=np.float64) * 5.0
    up_ctrls, trust_r, report = run_phase3_export_certification(
        ctrls,
        min_delay_samples_at_trough=2.5,
        max_pole_radius_at_peak=0.9998,
        head_weights_corners=hot_head_corners,
        max_train_stem_spec_peak=1.0,
        initial_trust_region_radius=2.5,
    )
    assert trust_r < 2.5
    assert any("[CLAMP-DYNAMIC]" in c for c in report.phase3_clamps)
    assert any("[CLAMP-HEAD-GAIN]" in c for c in report.phase3_clamps)

    formatted = format_consolidated_prescription_report(
        "dist/model.dnam",
        "models/m1/checkpoint.dnam-ckpt",
        report,
        up_ctrls,
    )
    assert "DNAM NON-STOP 3-PHASE CONTROL-SPACE AUDIT" in formatted
    assert "[CLAMP-DYNAMIC]" in formatted
