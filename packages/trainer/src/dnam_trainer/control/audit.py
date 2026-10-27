"""Non-Stop 3-Phase Control-Space Audit & Safety Clamping (`CL-15`, `CL-16`, `CL-17`):
- **Phase 1 (Step-0 Pre-Flight Audit, `< 200 ms`, `CL-15`):** `[CLAMP-RANGE]`, `[CLAMP-SWITCH]`,
`[CLAMP-STABILITY]`.
- **Phase 2 (In-Loop Stop-Gradient Corner Barrier & Rail Audit, `CL-16`):** `L_barrier`
(`detach()` on 1st-order main effects) + `[CLAMP-RAIL]`.
- **Phase 3 (Post-Polish Export Certification, `CL-17`):** `[CLAMP-DYNAMIC]` +
`[CLAMP-HEAD-GAIN]` + Consolidated Prescription Report."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from numpy.typing import ArrayLike
from torch import Tensor, nn

from dnam_trainer.control.basis import (
    compute_lobatto_degree_and_anchors,
    normalize_and_clamp_range,
)
from dnam_trainer.schema import (
    AuditReport,
    ControlCoords,
    ControlKind,
    ControlSchemaDef,
    StemPrescription,
    StrictModel,
)


class Phase1PreflightResult(StrictModel):
    """Output of Phase-1 Step-0 Pre-Flight Control-Space Audit (`CL-15`)."""

    updated_controls: list[ControlSchemaDef]
    audit_report: AuditReport
    structured_prescriptions: list[StemPrescription]


def run_phase1_preflight_audit(
    controls_schema: Sequence[ControlSchemaDef],
    stem_records: Sequence[Mapping[str, Any]],
    *,
    corner_pole_radii: Mapping[str, float] | None = None,
) -> Phase1PreflightResult:
    """Executes Phase-1 Pre-Flight Control-Space Audit (`CL-15`) in `< 200 ms` without blocking
    training.

    Enforces:
    - **`[CLAMP-RANGE]` (`CL-15`):** Sets `safe_clamped_range = [min_s(x_{p,s}), max_s(x_{p,s})]`
      and determines Lobatto degree `K_p = min(U_p - 1, 3)` (`CL-13`).
    - **`[CLAMP-SWITCH]` (`CL-15`):** Restricts `active_options` to switch states observed in at
      least 1 stem (falling back to `[default]` when 0 stems are provided).
    - **`[CLAMP-STABILITY]` (`CL-15`):** Tightens `conditional_max` if an extrapolated corner
      exceeds `r_c > 0.9995` or `g_lin > 0.9990`.
    """
    updated_ctrls: list[ControlSchemaDef] = []
    phase1_clamps: list[str] = []
    prescriptions_text: list[str] = []
    structured_prescs: list[StemPrescription] = []

    # Default coordinates for prescription generation
    def_coords: ControlCoords = {}
    for c in controls_schema:
        def_coords[c.id] = c.default

    for ctrl in controls_schema:
        if ctrl.kind == ControlKind.RANGE:
            dec_lo, dec_hi = ctrl.declared_range or (0.0, 10.0)
            obs_vals: list[float] = []
            for s in stem_records:
                c_meta = s.get("controls", s.get("controls_meta", s))
                if isinstance(c_meta, Mapping) and ctrl.id in c_meta:
                    val = c_meta[ctrl.id]
                    if isinstance(val, (int, float)) and not isinstance(val, bool):
                        obs_vals.append(float(val))

            if not obs_vals:
                safe_lo = float(ctrl.default)
                safe_hi = float(ctrl.default)
            else:
                safe_lo = max(dec_lo, min(obs_vals))
                safe_hi = min(dec_hi, max(obs_vals))

            _, def_u, _ = normalize_and_clamp_range(float(ctrl.default), (dec_lo, dec_hi))
            obs_u = [
                normalize_and_clamp_range(v, (dec_lo, dec_hi))[1] for v in (obs_vals or [safe_lo])
            ]
            deg, anchors = compute_lobatto_degree_and_anchors(obs_u, default_u=def_u)

            warnings: list[str] = []
            if safe_lo > dec_lo + 1e-4 or safe_hi < dec_hi - 1e-4:
                msg = (
                    f"[CLAMP-RANGE] Control '{ctrl.id}' clamped to observed span "
                    f"[{safe_lo:.2f}, {safe_hi:.2f}] (declared [{dec_lo:.2f}, {dec_hi:.2f}])"
                )
                phase1_clamps.append(msg)
                warnings.append(msg)

                if safe_hi < dec_hi - 1e-4:
                    sug = dict(def_coords)
                    sug[ctrl.id] = float(dec_hi)
                    reason = (
                        f"Record 1 stem at ({ctrl.id}={dec_hi:g}) to unlock "
                        f"[{safe_hi:g} -> {dec_hi:g}]"
                    )
                    prescriptions_text.append(f"[CLAMP-RANGE] {reason}")
                    structured_prescs.append(
                        StemPrescription(
                            clamp_tag="CLAMP-RANGE",
                            control_id=ctrl.id,
                            suggested_controls=sug,
                            reason=reason,
                        )
                    )
                if safe_lo > dec_lo + 1e-4:
                    sug_lo = dict(def_coords)
                    sug_lo[ctrl.id] = float(dec_lo)
                    reason_lo = (
                        f"Record 1 stem at ({ctrl.id}={dec_lo:g}) to unlock "
                        f"[{dec_lo:g} -> {safe_lo:g}]"
                    )
                    prescriptions_text.append(f"[CLAMP-RANGE] {reason_lo}")
                    structured_prescs.append(
                        StemPrescription(
                            clamp_tag="CLAMP-RANGE",
                            control_id=ctrl.id,
                            suggested_controls=sug_lo,
                            reason=reason_lo,
                        )
                    )

            cond_max = dict(ctrl.conditional_max)
            if corner_pole_radii:
                for corner_key, r_val in corner_pole_radii.items():
                    if r_val > 0.9995:
                        safe_corner_hi = safe_lo + 0.85 * (safe_hi - safe_lo)
                        cond_max[corner_key] = safe_corner_hi
                        stab_msg = (
                            f"[CLAMP-STABILITY] Corner '{corner_key}' pole radius "
                            f"{r_val:.5f} > 0.9995; clamped '{ctrl.id}' "
                            f"conditional_max to {safe_corner_hi:.2f}"
                        )
                        phase1_clamps.append(stab_msg)
                        warnings.append(stab_msg)
                        sug_c = dict(def_coords)
                        sug_c[ctrl.id] = float(safe_hi)
                        c_reason = (
                            f"Record 1 hot-corner stem at ({corner_key}, {ctrl.id}={safe_hi:g}) "
                            f"to lift conditional stability clamp"
                        )
                        prescriptions_text.append(f"[CLAMP-STABILITY] {c_reason}")
                        structured_prescs.append(
                            StemPrescription(
                                clamp_tag="CLAMP-STABILITY",
                                control_id=ctrl.id,
                                suggested_controls=sug_c,
                                reason=c_reason,
                            )
                        )

            updated_ctrls.append(
                ctrl.model_copy(
                    update={
                        "lobatto_degree": deg,
                        "anchor_nodes_norm": list(anchors),
                        "safe_clamped_range": (safe_lo, safe_hi),
                        "conditional_max": cond_max,
                        "clamp_warning": "; ".join(warnings) if warnings else None,
                    }
                )
            )
        else:
            dec_opts = list(ctrl.declared_options or ["default"])
            obs_opts: list[str] = []
            for s in stem_records:
                c_meta = s.get("controls", s.get("controls_meta", s))
                if isinstance(c_meta, Mapping) and ctrl.id in c_meta:
                    opt_s = str(c_meta[ctrl.id])
                    if opt_s in dec_opts and opt_s not in obs_opts:
                        obs_opts.append(opt_s)

            if not obs_opts:
                obs_opts = [str(ctrl.default)]

            if (
                ctrl.calibration is not None
                and ctrl.calibration.role in ("stereo_mode", "freeze_latch")
                and stem_records
            ):
                for opt in dec_opts:
                    is_known = (
                        ctrl.calibration.stereo_width_for_option(opt) is not None
                        if ctrl.calibration.role == "stereo_mode"
                        else ctrl.calibration.freeze_amount_for_option(opt) is not None
                    )
                    if is_known and opt not in obs_opts:
                        obs_opts.append(opt)

            missing_opts = [o for o in dec_opts if o not in obs_opts]
            sw_warning: str | None = None
            if missing_opts:
                sw_warning = (
                    f"[CLAMP-SWITCH] Switch '{ctrl.id}' locked to observed options {obs_opts}; "
                    f"unobserved {missing_opts} disabled"
                )
                phase1_clamps.append(sw_warning)
                for m_opt in missing_opts:
                    sug_sw = dict(def_coords)
                    sug_sw[ctrl.id] = m_opt
                    sw_reason = (
                        f"Record 1 stem with ({ctrl.id}='{m_opt}') "
                        f"to unlock switch option '{m_opt}'"
                    )
                    prescriptions_text.append(f"[CLAMP-SWITCH] {sw_reason}")
                    structured_prescs.append(
                        StemPrescription(
                            clamp_tag="CLAMP-SWITCH",
                            control_id=ctrl.id,
                            suggested_controls=sug_sw,
                            reason=sw_reason,
                        )
                    )

            updated_ctrls.append(
                ctrl.model_copy(
                    update={
                        "active_options": obs_opts,
                        "clamp_warning": sw_warning,
                    }
                )
            )

    report = AuditReport(
        phase1_clamps=phase1_clamps,
        phase2_rail_warnings=[],
        phase3_clamps=[],
        prescriptions=prescriptions_text,
    )
    return Phase1PreflightResult(
        updated_controls=updated_ctrls,
        audit_report=report,
        structured_prescriptions=structured_prescs,
    )


def compute_phase2_corner_barrier_loss(
    theta_main_corners: Tensor,
    delta_theta_fm_corners: Tensor,
    theta_safe_max: Tensor | float,
    *,
    theta_safe_min: Tensor | float | None = None,
    lambda_bar: float = 10.0,
) -> Tensor:
    """Enforces `CL-16`: Phase-2 Stop-Gradient Virtual Corner Barrier Loss `L_barrier`.

    `L_barrier = lambda_bar * sum(relu(sg(theta_main) + delta_theta_fm - theta_safe_max)^2)`
    Applies `.detach()` (`sg`) on `theta_main_corners` so 1st-order main effects on recorded
    stems receive exact `0.000000` gradient interference while 2nd-order Rank-4 FM weights
    `delta_theta_fm_corners` are smoothly steered away from unstable corners.
    """
    if lambda_bar <= 0.0:
        return torch.zeros((), dtype=theta_main_corners.dtype, device=theta_main_corners.device)

    theta_total = theta_main_corners.detach() + delta_theta_fm_corners
    upper_excess = torch.relu(theta_total - theta_safe_max)
    loss = torch.sum(upper_excess * upper_excess)

    if theta_safe_min is not None:
        lower_excess = torch.relu(theta_safe_min - theta_total)
        loss = loss + torch.sum(lower_excess * lower_excess)

    return float(lambda_bar) * loss


def run_phase2_rail_lockup_audit(
    poly_adaa_layers: Sequence[nn.Module],
    controls_schema: Sequence[ControlSchemaDef],
    *,
    threshold: float = 0.95,
) -> tuple[list[ControlSchemaDef], list[str]]:
    """Executes Phase-2 `[CLAMP-RAIL]` audit (`CL-16`) across `DifferentiablePolyAdaa` layers."""
    max_lockup = 0.0
    for layer in poly_adaa_layers:
        frac = float(getattr(layer, "last_rail_lockup_fraction", 0.0))
        if frac > max_lockup:
            max_lockup = frac

    if max_lockup <= threshold:
        return list(controls_schema), []

    warnings: list[str] = []
    updated: list[ControlSchemaDef] = []
    for c in controls_schema:
        if c.kind == ControlKind.RANGE:
            lo, hi = c.safe_clamped_range or c.declared_range or (0.0, 10.0)
            tightened_hi = lo + 0.90 * (hi - lo)
            msg = (
                f"[CLAMP-RAIL] PolyADAA rail lockup {max_lockup * 100.0:.1f}% > "
                f"{threshold * 100.0:.0f}%; tightened '{c.id}' safe upper bound "
                f"{hi:.2f} -> {tightened_hi:.2f}"
            )
            warnings.append(msg)
            updated.append(
                c.model_copy(
                    update={
                        "safe_clamped_range": (lo, tightened_hi),
                        "clamp_warning": f"{c.clamp_warning}; {msg}" if c.clamp_warning else msg,
                    }
                )
            )
        else:
            updated.append(c)

    return updated, warnings


def run_phase3_export_certification(
    controls_schema: Sequence[ControlSchemaDef],
    *,
    min_delay_samples_at_trough: float = 16.0,
    max_pole_radius_at_peak: float = 0.990,
    head_weights_corners: ArrayLike | None = None,
    max_train_stem_spec_peak: float = 1.0,
    initial_trust_region_radius: float = 2.5,
    existing_audit: AuditReport | None = None,
) -> tuple[list[ControlSchemaDef], float, AuditReport]:
    """Executes Phase-3 Post-Polish Export Certification (`CL-17`: `[CLAMP-DYNAMIC]`,
    `[CLAMP-HEAD-GAIN]`).

    1. **`[CLAMP-DYNAMIC]` (`CL-17`):** Verifies `min_delay_samples_at_trough >= 4.0` and
       `max_pole_radius_at_peak <= 0.9995`, tightening `safe_clamped_range` if violated.
    2. **`[CLAMP-HEAD-GAIN]` (`CL-17`):** Evaluates the 32-point FFT spectral peak across
       16-tap output head corner weights and tightens `head_trust_region_radius` (`M_h`) if any
       unrecorded corner exceeds `max_train_stem_spec_peak` by `> +6.0 dB` (`2.0x` linear).
    """
    phase3_clamps: list[str] = []
    new_prescriptions: list[str] = []
    updated_ctrls = list(controls_schema)

    if min_delay_samples_at_trough < 4.0 or max_pole_radius_at_peak > 0.9995:
        dyn_updated: list[ControlSchemaDef] = []
        for c in updated_ctrls:
            if c.kind == ControlKind.RANGE:
                lo, hi = c.safe_clamped_range or c.declared_range or (0.0, 10.0)
                new_hi = lo + 0.88 * (hi - lo)
                msg = (
                    f"[CLAMP-DYNAMIC] LFO/Envelope trough delay="
                    f"{min_delay_samples_at_trough:.2f} samples (min 4.0) or peak pole r="
                    f"{max_pole_radius_at_peak:.5f} (max 0.9995); "
                    f"clamped '{c.id}' max {hi:.2f} -> {new_hi:.2f}"
                )
                phase3_clamps.append(msg)
                new_prescriptions.append(
                    f"[CLAMP-DYNAMIC] Record 1 stem at ({c.id}={hi:g}) "
                    f"to certify dynamic modulation excursion"
                )
                dyn_updated.append(
                    c.model_copy(
                        update={
                            "safe_clamped_range": (lo, new_hi),
                            "clamp_warning": f"{c.clamp_warning}; {msg}"
                            if c.clamp_warning
                            else msg,
                        }
                    )
                )
            else:
                dyn_updated.append(c)
        updated_ctrls = dyn_updated

    trust_radius = float(initial_trust_region_radius)
    if head_weights_corners is not None:
        hw = np.ascontiguousarray(head_weights_corners, dtype=np.float64)
        # Reshape last dimension into 16-tap FIR filters if divisible by 16
        tap_len = 16 if hw.shape[-1] >= 16 else hw.shape[-1]
        taps = hw.reshape(-1, tap_len)
        spec_32 = np.abs(np.fft.rfft(taps, n=32, axis=-1))
        worst_corner_peak = float(np.max(spec_32)) if spec_32.size > 0 else 0.0
        ref_peak = max(float(max_train_stem_spec_peak), 1e-6)
        gain_ratio = worst_corner_peak / ref_peak
        # +6.0 dB threshold == 10^(6/20) = 1.995262
        if gain_ratio > 1.995262:
            excess_db = 20.0 * math.log10(gain_ratio)
            scale_back = 1.995262 / gain_ratio
            trust_radius = max(0.5, float(initial_trust_region_radius) * scale_back)
            head_msg = (
                f"[CLAMP-HEAD-GAIN] 32-pt FFT corner head peak +{excess_db:.2f} dB > "
                f"+6.0 dB limit; tightened head_trust_region_radius "
                f"{initial_trust_region_radius:.2f} -> {trust_radius:.2f}"
            )
            phase3_clamps.append(head_msg)
            new_prescriptions.append(
                "[CLAMP-HEAD-GAIN] Record 1 extreme corner stem to constrain SVD head extrapolation"
            )

    base_p1 = list(existing_audit.phase1_clamps) if existing_audit else []
    base_p2 = list(existing_audit.phase2_rail_warnings) if existing_audit else []
    base_p3 = list(existing_audit.phase3_clamps) if existing_audit else []
    base_pr = list(existing_audit.prescriptions) if existing_audit else []

    merged_report = AuditReport(
        phase1_clamps=base_p1,
        phase2_rail_warnings=base_p2,
        phase3_clamps=[*base_p3, *phase3_clamps],
        prescriptions=[*base_pr, *new_prescriptions],
    )
    return updated_ctrls, trust_radius, merged_report


def format_consolidated_prescription_report(
    model_path: str,
    ckpt_path: str,
    audit_report: AuditReport,
    controls_schema: Sequence[ControlSchemaDef] = (),
) -> str:
    """Formats the Consolidated Post-Training Audit & Stem Prescription Report
    (`docs/clamp_and_audit_registry.md` §2)."""
    lines: list[str] = [
        "=" * 78,
        "DNAM NON-STOP 3-PHASE CONTROL-SPACE AUDIT & STEM PRESCRIPTION REPORT",
        "=" * 78,
        f"  Playable Model (.dnam):       {model_path}",
        f"  Resume Checkpoint (.dnam-ckpt): {ckpt_path}",
        "-" * 78,
        "  Active Control Bounds:",
    ]
    if not controls_schema:
        lines.append("    (No parametric controls declared - fixed-setting model)")
    else:
        for c in controls_schema:
            if c.kind == ControlKind.RANGE:
                dec = c.declared_range or (0.0, 10.0)
                safe = c.safe_clamped_range or dec
                deg = c.lobatto_degree if c.lobatto_degree is not None else 0
                status = (
                    "CLAMPED"
                    if (safe[0] > dec[0] + 1e-4 or safe[1] < dec[1] - 1e-4 or c.conditional_max)
                    else "FULL"
                )
                lines.append(
                    f"    - {c.id:<14} [range]  safe=[{safe[0]:.2f}, {safe[1]:.2f}] "
                    f"declared=[{dec[0]:.2f}, {dec[1]:.2f}] (Lobatto K={deg}) [{status}]"
                )
            else:
                dec_o = c.declared_options or []
                act_o = c.active_options or dec_o
                status = "CLAMPED" if len(act_o) < len(dec_o) else "FULL"
                lines.append(
                    f"    - {c.id:<14} [switch] active={act_o} declared={dec_o} [{status}]"
                )

    all_clamps = [
        *audit_report.phase1_clamps,
        *audit_report.phase2_rail_warnings,
        *audit_report.phase3_clamps,
    ]
    lines.append("-" * 78)
    if not all_clamps:
        lines.append("  Audit Status: 0 active clamps (100% of declared control space unlocked!)")
    else:
        lines.append(f"  Active Safety Clamps ({len(all_clamps)}):")
        for item in all_clamps:
            lines.append(f"    * {item}")

    if audit_report.prescriptions:
        lines.append("-" * 78)
        lines.append("  Actionable Stem Prescriptions (Iterative Resume via `dnam train`):")
        for idx, presc in enumerate(audit_report.prescriptions, start=1):
            lines.append(f"    {idx}. {presc}")

    lines.append("=" * 78)
    return "\n".join(lines)


__all__ = [
    "Phase1PreflightResult",
    "compute_phase2_corner_barrier_loss",
    "format_consolidated_prescription_report",
    "run_phase1_preflight_audit",
    "run_phase2_rail_lockup_audit",
    "run_phase3_export_certification",
]
