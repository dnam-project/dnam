"""Unit tests for `dnam_trainer.project.planner` (Star + Switch + Corner + Prescription planner)."""

from __future__ import annotations

from dnam_trainer.project.planner import compute_control_deltas, plan_model_capture_schedule
from dnam_trainer.project.workspace import parse_compact_range_spec, parse_compact_switch_spec
from dnam_trainer.schema import AuditReport, ProjectModelSpec, StemPrescription


def test_plan_model_capture_schedule_and_deltas() -> None:
    spec = ProjectModelSpec(
        model_id="test_amp",
        name="Test Amp",
        controls=[
            parse_compact_range_spec("drive=0..10:5"),
            parse_compact_range_spec("tone=0..10:5"),
            parse_compact_switch_spec("voice=vintage,modern:vintage"),
        ],
    )
    prescs = [
        StemPrescription(
            clamp_tag="CLAMP-RANGE",
            control_id="drive",
            suggested_controls={"drive": 9.5, "tone": 5.0, "voice": "modern"},
            reason="Unlock upper drive in modern voice",
        )
    ]
    plan = plan_model_capture_schedule(
        spec,
        structured_prescriptions=prescs,
        include_corners=True,
    )
    categories = [p.category for p in plan]
    assert "reference" in categories
    assert "star_range" in categories
    assert "switch" in categories
    assert "corner" in categories
    assert "prescribed" in categories

    # Also test text-based ckpt_audit_report prescription parsing
    audit = AuditReport(
        prescriptions=["[CLAMP-RANGE] Record 1 stem at (drive=8.5, voice='modern')"]
    )
    plan2 = plan_model_capture_schedule(spec, ckpt_audit_report=audit, include_corners=False)
    assert any(p.category == "prescribed" for p in plan2)

    deltas = compute_control_deltas(
        {"drive": 5.0, "voice": "vintage"},
        {"drive": 10.0, "voice": "vintage"},
    )
    assert deltas["drive"] == (5.0, 10.0)
    assert deltas["voice"] == ("vintage", "vintage")
