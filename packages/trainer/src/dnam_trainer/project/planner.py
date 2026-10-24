"""Star + Switch + Extreme Corner + `.dnam-ckpt` Prescription capture planner (`planner.py`)."""

from __future__ import annotations

import contextlib
from typing import Literal

from dnam_trainer.schema import (
    AuditReport,
    ControlCoords,
    ControlKind,
    ControlValue,
    ProjectModelSpec,
    StemPrescription,
    StrictModel,
)

type PlanCategory = Literal["reference", "star_range", "switch", "corner", "prescribed"]


class PlannedStemCapture(StrictModel):
    """Single planned stem capture coordinate in the Star + Switch + Corner + Prescription
    schedule."""

    stem_id: str
    category: PlanCategory
    controls: ControlCoords
    is_recorded: bool
    reason: str


def _coords_match(a: ControlCoords, b: ControlCoords, tol: float = 1e-4) -> bool:
    if set(a.keys()) != set(b.keys()):
        return False
    for k, va in a.items():
        vb = b[k]
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            if abs(float(va) - float(vb)) > tol:
                return False
        elif str(va) != str(vb):
            return False
    return True


def _default_coords(spec: ProjectModelSpec) -> ControlCoords:
    coords: ControlCoords = {}
    for c in spec.controls:
        if c.kind == ControlKind.RANGE:
            lo, hi = c.declared_range or (0.0, 10.0)
            coords[c.id] = float(c.default) if c.default is not None else 0.5 * (lo + hi)
        else:
            opts = c.declared_options or ["default"]
            coords[c.id] = str(c.default) if c.default is not None else opts[0]
    return coords


def plan_model_capture_schedule(
    model_spec: ProjectModelSpec,
    *,
    ckpt_audit_report: AuditReport | None = None,
    structured_prescriptions: list[StemPrescription] | None = None,
    include_corners: bool = True,
) -> list[PlannedStemCapture]:
    """Generates the ordered Star + Switch + Corner + Prescription capture schedule for
    `model_spec`."""
    recorded_coords = [s.controls for s in model_spec.stems]
    plan: list[PlannedStemCapture] = []

    def _add_candidate(
        stem_id: str,
        category: PlanCategory,
        coords: ControlCoords,
        reason: str,
    ) -> None:
        if any(_coords_match(p.controls, coords) for p in plan):
            return
        is_rec = any(_coords_match(rc, coords) for rc in recorded_coords)
        plan.append(
            PlannedStemCapture(
                stem_id=stem_id,
                category=category,
                controls=dict(coords),
                is_recorded=is_rec,
                reason=reason,
            )
        )

    base = _default_coords(model_spec)
    _add_candidate(
        "stem_00_reference",
        "reference",
        base,
        "Noon / default reference stem anchoring Lobatto node u_{p,1}",
    )

    range_ctrls = [c for c in model_spec.controls if c.kind == ControlKind.RANGE]
    switch_ctrls = [c for c in model_spec.controls if c.kind == ControlKind.SWITCH]

    # 1. Star ("1-knob-at-a-time") min and max endpoints for every range control
    for rc in range_ctrls:
        lo, hi = rc.declared_range or (0.0, 10.0)
        c_min = dict(base)
        c_min[rc.id] = float(lo)
        _add_candidate(
            f"stem_star_{rc.id}_min",
            "star_range",
            c_min,
            f"1D Star minimum endpoint for '{rc.id}' ({lo}) to unlock full declared_range",
        )
        c_max = dict(base)
        c_max[rc.id] = float(hi)
        _add_candidate(
            f"stem_star_{rc.id}_max",
            "star_range",
            c_max,
            f"1D Star maximum endpoint for '{rc.id}' ({hi}) to unlock full declared_range",
        )

    # 2. Switch states at reference range coordinates
    for sc in switch_ctrls:
        for opt in sc.declared_options or []:
            c_sw = dict(base)
            c_sw[sc.id] = opt
            _add_candidate(
                f"stem_switch_{sc.id}_{opt}",
                "switch",
                c_sw,
                f"Switch option '{sc.id}={opt}' at reference knob coordinates",
            )

    # 3. Extreme all-min / all-max corners for 2nd-order Rank-4 FM interaction calibration
    if include_corners and len(range_ctrls) >= 2:
        all_max = dict(base)
        all_min = dict(base)
        for rc in range_ctrls:
            lo, hi = rc.declared_range or (0.0, 10.0)
            all_min[rc.id] = float(lo)
            all_max[rc.id] = float(hi)
        _add_candidate(
            "stem_corner_all_max",
            "corner",
            all_max,
            "All-maximum hot corner to calibrate 2nd-order Rank-4 FM & lift conditional clamps",
        )
        _add_candidate(
            "stem_corner_all_min",
            "corner",
            all_min,
            "All-minimum corner for full-span multi-knob coverage",
        )

    # 4. Append any active `.dnam-ckpt` StemPrescriptions
    if structured_prescriptions:
        for idx, presc in enumerate(structured_prescriptions, start=1):
            c_presc = dict(base)
            for k, v in presc.suggested_controls.items():
                if isinstance(v, (int, float)):
                    c_presc[k] = float(v)
                else:
                    c_presc[k] = v
            _add_candidate(
                f"stem_prescribed_{idx:02d}",
                "prescribed",
                c_presc,
                f"[{presc.clamp_tag}] {presc.reason}",
            )
    elif ckpt_audit_report is not None:
        for idx, text_p in enumerate(ckpt_audit_report.prescriptions, start=1):
            # Parse "key=val" tokens from prescription string if present
            c_presc = dict(base)
            for token in text_p.replace(",", " ").replace("(", " ").replace(")", " ").split():
                if "=" in token:
                    k, v_str = token.split("=", 1)
                    k = k.strip()
                    v_str = v_str.strip().strip("'\"")
                    if any(rc.id == k for rc in range_ctrls):
                        with contextlib.suppress(ValueError):
                            c_presc[k] = float(v_str)
                    elif any(sc.id == k for sc in switch_ctrls):
                        c_presc[k] = v_str
            _add_candidate(
                f"stem_prescribed_{idx:02d}",
                "prescribed",
                c_presc,
                text_p,
            )

    return plan


def compute_control_deltas(
    prev_controls: ControlCoords | None,
    next_controls: ControlCoords,
) -> dict[str, tuple[ControlValue | None, ControlValue]]:
    """Computes per-control `(previous_value, target_value)` deltas for interactive capture
    prompts."""
    deltas: dict[str, tuple[ControlValue | None, ControlValue]] = {}
    for k, target_val in next_controls.items():
        prev_val = prev_controls.get(k) if prev_controls is not None else None
        deltas[k] = (prev_val, target_val)
    return deltas


__all__ = [
    "PlanCategory",
    "PlannedStemCapture",
    "compute_control_deltas",
    "plan_model_capture_schedule",
]
