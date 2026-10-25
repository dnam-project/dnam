"""Unit tests for `dnam_trainer.schema` and `dnam_trainer.events`."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from dnam_engine.exceptions import DnamValidationError
from dnam_trainer.events import (
    AuditPhaseEvent,
    CalibrationStepEvent,
    NullObserver,
    StemCaptureEvent,
    TrainingCompleteEvent,
    TrainingStageEvent,
    TrainingStepEvent,
    format_training_event_sse,
    parse_capture_event_json,
    parse_training_event_json,
)
from dnam_trainer.schema import (
    AuditReport,
    InMemoryPcmRef,
    LocalWavPathRef,
    RemoteBlobUriRef,
    ResumeJobRequest,
    StemPrescription,
    TrainingConfig,
    TrainingJobRequest,
    TrainingJobResult,
    TrainingStem,
    TrainingTelemetryRow,
    coerce_audio_f32,
    compute_pcm_sha256,
    parse_compact_range_spec,
    parse_compact_switch_spec,
)


def test_coerce_audio_f32_and_pcm_sha256() -> None:
    mono = np.linspace(-0.5, 0.5, 128, dtype=np.float64)
    c_mono = coerce_audio_f32(mono)
    assert c_mono.dtype == np.float32
    assert c_mono.shape == (128,)

    # (L, 2) auto-transposed to (2, L)
    stereo_l2 = np.column_stack([mono, -mono])
    c_st = coerce_audio_f32(stereo_l2, allow_stereo=True)
    assert c_st.shape == (2, 128)

    # (1, L) squeezed to (L,)
    c_1l = coerce_audio_f32(mono[None, :], allow_stereo=True)
    assert c_1l.shape == (128,)

    with pytest.raises(ValueError, match="at least 1 sample"):
        coerce_audio_f32(np.array([], dtype=np.float32))

    with pytest.raises(ValueError, match="Expected 1D mono"):
        coerce_audio_f32(stereo_l2, allow_stereo=False)

    sha = compute_pcm_sha256(c_mono)
    assert len(sha) == 64


def test_training_stem_validation_and_wire_descriptor() -> None:
    x = np.sin(np.linspace(0.0, 10.0, 256, dtype=np.float32))
    y = np.stack([(x * 0.8).astype(np.float32), (x * 0.7).astype(np.float32)], axis=0)
    stem = TrainingStem(
        stem_id="s1",
        controls_meta={"drive": 5.0},
        input_audio=x,
        target_audio=y,
        sample_rate_hz=48_000,
    )
    assert stem.num_samples == 256
    assert stem.is_stereo_target
    assert stem.target_stereo_2d().shape == (2, 256)
    assert len(stem.input_pcm_sha256) == 64
    assert len(stem.target_pcm_sha256) == 64

    mono_stem = TrainingStem(
        stem_id="s_mono",
        input_audio=x,
        target_audio=(x * 0.5).astype(np.float32),
    )
    assert not mono_stem.is_stereo_target
    assert mono_stem.target_stereo_2d().shape == (2, 256)

    wire = stem.to_wire_descriptor()
    assert isinstance(wire.input_ref, InMemoryPcmRef)
    assert isinstance(wire.target_ref, InMemoryPcmRef)
    assert wire.target_ref.channels == 2

    with pytest.raises((ValidationError, DnamValidationError)):
        TrainingStem(
            stem_id="bad",
            input_audio=x,
            target_audio=x[:64],
        )


def test_stem_audio_ref_discriminated_union_and_job_schemas() -> None:
    sha = "a" * 64
    r_mem = InMemoryPcmRef(pcm_sha256=sha, num_samples=1024)
    r_loc = LocalWavPathRef(path="stems/s1.wav", pcm_sha256=sha, num_samples=1024)
    r_rem = RemoteBlobUriRef(uri="gs://bucket/s1.pcm", pcm_sha256=sha, num_samples=1024)
    assert r_mem.ref_kind == "memory"
    assert r_loc.ref_kind == "local_path"
    assert r_rem.ref_kind == "remote_uri"

    presc = StemPrescription(
        clamp_tag="CLAMP-RANGE",
        control_id="drive",
        suggested_controls={"drive": 10.0},
        reason="Unlock max",
    )
    assert presc.control_id == "drive"

    req = TrainingJobRequest(
        model_id="m1",
        model_name="Model 1",
        config=TrainingConfig(epochs=10),
    )
    dumped = req.model_dump_json()
    restored = TrainingJobRequest.model_validate_json(dumped)
    assert restored.model_id == "m1"

    res_req = ResumeJobRequest(checkpoint_bytes=b"DNCK", force_tier2=True)
    assert res_req.force_tier2

    res = TrainingJobResult(
        model_id="m1",
        dnam_bytes=b"DNAM",
        checkpoint_bytes=b"DNCK",
        audit_report=AuditReport(),
        telemetry=[
            TrainingTelemetryRow(
                stage=2,
                epoch=0,
                esr_lite_db=-20.0,
                esr_full_db=-22.0,
                esr_ultra_db=-25.0,
                barrier_loss=0.0,
                group_lasso_loss=0.0,
                surrogate_sigma_samples=35.0,
                elapsed_sec=0.1,
            )
        ],
    )
    assert len(res.telemetry) == 1


def test_events_json_and_sse_roundtrip() -> None:
    e_cal = CalibrationStepEvent(
        step=1,
        backend_id="software",
        send_gain_db=0.0,
        unity_ref_dbfs=0.0,
        measured_peak_dbfs=-1.0,
        measured_rms_dbfs=-12.0,
    )
    parsed_cal = parse_capture_event_json(e_cal.model_dump_json())
    assert isinstance(parsed_cal, CalibrationStepEvent)
    assert parsed_cal.step == 1

    e_stem = StemCaptureEvent(
        status="completed",
        model_id="m1",
        stem_id="s1",
        stem_index=1,
        total_stems=2,
    )
    parsed_stem = parse_capture_event_json(e_stem.model_dump_json())
    assert isinstance(parsed_stem, StemCaptureEvent)

    for ev in (
        TrainingStageEvent(stage=1, stage_name="S1", message="msg"),
        TrainingStepEvent(
            stage=3,
            epoch=1,
            total_epochs=10,
            esr_lite_db=-20.0,
            esr_full_db=-22.0,
            esr_ultra_db=-24.0,
        ),
        AuditPhaseEvent(phase=1, audit_report=AuditReport()),
        TrainingCompleteEvent(
            model_id="m1",
            esr_lite_db=-30.0,
            esr_full_db=-35.0,
            esr_ultra_db=-40.0,
            core2_active=False,
            lfo_env_active=True,
        ),
    ):
        sse = format_training_event_sse(ev)
        assert sse.startswith(f"event: {ev.event_type}\ndata: ")
        data_line = next(line for line in sse.splitlines() if line.startswith("data:"))
        restored = parse_training_event_json(data_line)
        assert restored.event_type == ev.event_type

    obs = NullObserver()
    obs.on_capture_event(e_cal)
    obs.on_training_event(TrainingStageEvent(stage=1, stage_name="S1", message="ok"))


def test_compact_range_and_switch_parsers() -> None:
    r1 = parse_compact_range_spec("drive=0..10:6.5")
    assert r1.id == "drive"
    assert r1.declared_range == (0.0, 10.0)
    assert r1.default == 6.5

    r2 = parse_compact_range_spec("blend=-12..12")
    assert r2.default == 0.0

    with pytest.raises(ValueError, match="Invalid compact range"):
        parse_compact_range_spec("drive")
    with pytest.raises(ValueError, match="Empty control id"):
        parse_compact_range_spec("=0..10")
    with pytest.raises(ValueError, match="strictly less"):
        parse_compact_range_spec("drive=10..0")
    with pytest.raises(ValueError, match="outside declared range"):
        parse_compact_range_spec("drive=0..10:15")

    s1 = parse_compact_switch_spec("attack=cut,flat,boost:boost")
    assert s1.id == "attack"
    assert s1.declared_options == ["cut", "flat", "boost"]
    assert s1.default == "boost"

    with pytest.raises(ValueError, match="Invalid compact switch"):
        parse_compact_switch_spec("attack")
    with pytest.raises(ValueError, match="Empty control id"):
        parse_compact_switch_spec("=a,b")
    with pytest.raises(ValueError, match="at least 2 options"):
        parse_compact_switch_spec("attack=only_one")
    with pytest.raises(ValueError, match="not in declared options"):
        parse_compact_switch_spec("attack=cut,boost:invalid")
