"""Unit tests for the headless `dnam-train` worker CLI (`dnam_trainer.cli`)."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest
from rich.console import Console

from dnam_engine import DnamContainer
from dnam_trainer.cli import (
    JsonlTrainingObserver,
    RichTrainingObserver,
    SseTrainingObserver,
    create_training_observer,
    get_trainer_status_payload,
    main,
    run_worker_job,
)
from dnam_trainer.events import (
    AuditPhaseEvent,
    CacheEvent,
    PlannerEvent,
    RuntimeEvent,
    SolverEvent,
    TrainingCompleteEvent,
    TrainingStageEvent,
    TrainingStepEvent,
    parse_training_event_json,
)
from dnam_trainer.schema import (
    AuditReport,
    LocalWavPathRef,
    ResumeJobRequest,
    TrainingConfig,
    TrainingJobRequest,
    TrainingStem,
    WireStemDescriptor,
    parse_compact_range_spec,
    parse_compact_switch_spec,
)
from dnam_trainer.training.wav_io import read_wav, write_wav_24bit


def test_wav_io_16_24_32bit_and_stereo_roundtrip(tmp_path: Path) -> None:
    import wave

    # 24-bit mono and stereo roundtrip
    mono = np.linspace(-0.5, 0.5, 256, dtype=np.float32)
    p_mono = write_wav_24bit(tmp_path / "m24.wav", mono, sample_rate_hz=48_000)
    r_mono, sr = read_wav(p_mono)
    assert sr == 48_000
    assert r_mono.shape == (256,)
    assert np.allclose(r_mono, mono, atol=1e-4)

    stereo = np.stack([mono, -mono], axis=0)
    p_st = write_wav_24bit(tmp_path / "s24.wav", stereo, sample_rate_hz=48_000)
    r_st, sr_st = read_wav(p_st)
    assert sr_st == 48_000
    assert r_st.shape == (2, 256)

    # 16-bit and 32-bit read support
    p16 = tmp_path / "m16.wav"
    with wave.open(str(p16), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(48_000)
        wf.writeframes((mono * 32767.0).astype("<i2").tobytes())
    r16, _ = read_wav(p16)
    assert r16.shape == (256,)

    p32 = tmp_path / "m32.wav"
    with wave.open(str(p32), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(4)
        wf.setframerate(48_000)
        wf.writeframes((mono * 2147483647.0).astype("<i4").tobytes())
    r32, _ = read_wav(p32)
    assert r32.shape == (256,)

    with pytest.raises(FileNotFoundError):
        read_wav(tmp_path / "missing.wav")
    with pytest.raises(ValueError, match="Invalid sample_rate_hz"):
        write_wav_24bit(tmp_path / "bad.wav", mono, sample_rate_hz=1000)


def test_training_observers_rich_jsonl_and_sse() -> None:
    buf_jsonl = io.StringIO()
    obs_jsonl = create_training_observer("jsonl", stream=buf_jsonl)
    assert isinstance(obs_jsonl, JsonlTrainingObserver)

    buf_sse = io.StringIO()
    obs_sse = create_training_observer("sse", stream=buf_sse)
    assert isinstance(obs_sse, SseTrainingObserver)

    con = Console(quiet=True)
    obs_rich = create_training_observer("rich", console=con)
    assert isinstance(obs_rich, RichTrainingObserver)

    events = [
        TrainingStageEvent(stage=1, stage_name="System-ID", message="Starting"),
        TrainingStepEvent(
            stage=3,
            epoch=1,
            total_epochs=2,
            esr_lite_db=-20.0,
            esr_full_db=-22.0,
            esr_ultra_db=-25.0,
            esr_lf_db=-24.0,
            esr_mid_db=-26.0,
            esr_hf_db=-21.0,
        ),
        AuditPhaseEvent(
            phase=1,
            audit_report=AuditReport(phase1_clamps=["[CLAMP-RANGE] drive clamped"]),
        ),
        AuditPhaseEvent(
            phase=2,
            audit_report=AuditReport(phase2_rail_warnings=["[WARN-RAIL] rail warning"]),
        ),
        TrainingCompleteEvent(
            model_id="b7k",
            esr_lite_db=-25.0,
            esr_full_db=-30.0,
            esr_ultra_db=-35.0,
            esr_lf_db=-28.0,
            esr_mid_db=-32.0,
            esr_hf_db=-24.0,
            mr_stft_db=0.45,
            ltas_rms_tilt_db=0.5,
            ltas_hf_tilt_db=0.8,
            seg_esr_db=-26.0,
            core2_active=False,
            lfo_env_active=False,
            resume_tier_executed=1,
        ),
        RuntimeEvent(
            device_kind="mps",
            torch_device="mps:0",
            contraction_dtype="bfloat16",
            bf16_native=True,
            fp32_matmul_exact=True,
            fast_math_off=True,
            budget_bytes=8 * 1024**3,
            max_tensor_bytes=1 * 1024**3,
            unverified_hardware=True,
            warning="test warning",
        ),
        PlannerEvent(
            mode="STREAMED",
            num_samples=48000,
            num_stems=2,
            block_rows=16384,
            stem_group_size=1,
            budget_bytes=8 * 1024**3,
            estimated_peak_bytes=2 * 1024**3,
            regimes=((0, 24000), (24000, 48000)),
        ),
        CacheEvent(
            tier="sidecar",
            action="hit",
            key="stage1",
            bytes_count=1024 * 1024,
            elapsed_ms=1.2,
            message="ok",
        ),
        SolverEvent(
            pass_name="stage2",
            num_solves=4,
            matrix_dim=64,
            rhs_cols=8,
            transfer_ms=0.5,
            solve_ms=1.1,
            lstsq_fallbacks=1,
            warning="lstsq fallback",
        ),
    ]
    for ev in events:
        obs_jsonl.on_training_event(ev)
        obs_sse.on_training_event(ev)
        obs_rich.on_training_event(ev)

    jsonl_lines = [ln for ln in buf_jsonl.getvalue().splitlines() if ln.strip()]
    assert len(jsonl_lines) == 9
    assert parse_training_event_json(jsonl_lines[0]).event_type == "training_stage"
    assert "event: training_complete" in buf_sse.getvalue()


def test_dnam_train_worker_cli_status_train_and_ckpt_resume(
    synthetic_training_stems: list[TrainingStem],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # 1. --status-json
    main(["--status-json"])
    out_status = capsys.readouterr().out.strip()
    status_obj = json.loads(out_status)
    assert status_obj["installed"] is True
    assert "torch_version" in status_obj
    assert get_trainer_status_payload()["installed"] is True

    with pytest.raises(ValueError, match="Provide `--request-json"):
        main([])

    # 2. Write local WAV stems and build a file-backed TrainingJobRequest
    descriptors: list[WireStemDescriptor] = []
    for idx, stem in enumerate(synthetic_training_stems):
        in_wav = write_wav_24bit(
            tmp_path / f"in_{idx}.wav",
            stem.input_audio,
            sample_rate_hz=stem.sample_rate_hz,
        )
        tgt_wav = write_wav_24bit(
            tmp_path / f"tgt_{idx}.wav",
            stem.target_audio,
            sample_rate_hz=stem.sample_rate_hz,
        )
        descriptors.append(
            WireStemDescriptor(
                stem_id=stem.stem_id,
                controls_meta=dict(stem.controls_meta),
                input_ref=LocalWavPathRef(
                    path=str(in_wav),
                    pcm_sha256=stem.input_pcm_sha256,
                    sample_rate_hz=stem.sample_rate_hz,
                    channels=1,
                    num_samples=stem.num_samples,
                ),
                target_ref=LocalWavPathRef(
                    path=str(tgt_wav),
                    pcm_sha256=stem.target_pcm_sha256,
                    sample_rate_hz=stem.sample_rate_hz,
                    channels=2,
                    num_samples=stem.num_samples,
                ),
                noise_floor_dbfs=stem.noise_floor_dbfs,
            )
        )

    ctrls = [
        parse_compact_range_spec("drive=0..10:5"),
        parse_compact_switch_spec("attack=cut,flat,boost:flat"),
    ]
    req = TrainingJobRequest(
        model_id="b7k_worker",
        model_name="B7K Worker",
        author="Studio Test",
        controls_schema=ctrls,
        config=TrainingConfig(epochs=1, topology="core1_only"),
        stem_descriptors=descriptors[:2],
    )
    req_path = tmp_path / "job_req.json"
    req_path.write_text(req.model_dump_json(indent=2), encoding="utf-8")

    out_dnam = tmp_path / "dist" / "b7k_worker.dnam"
    out_ckpt = tmp_path / "models" / "b7k_worker" / "checkpoint.dnam-ckpt"
    reports_dir = tmp_path / "models" / "b7k_worker" / "reports"
    res_json = tmp_path / "result_summary.json"
    seed_hex = "42" * 32

    # 3. Initial train via `main` with `--event-format jsonl`
    main(
        [
            "--request-json",
            str(req_path),
            "--output-dnam",
            str(out_dnam),
            "--output-ckpt",
            str(out_ckpt),
            "--reports-dir",
            str(reports_dir),
            "--result-json",
            str(res_json),
            "--creator-seed-hex",
            seed_hex,
            "--event-format",
            "jsonl",
        ]
    )
    stream_out = capsys.readouterr().out
    events = [parse_training_event_json(ln) for ln in stream_out.splitlines() if ln.strip()]
    assert any(e.event_type == "training_complete" for e in events)
    assert out_dnam.is_file()
    assert out_ckpt.is_file()
    assert (reports_dir / "telemetry.parquet").is_file()
    assert (reports_dir / "telemetry.html").is_file()
    assert (reports_dir / "audit_report.json").is_file()
    assert res_json.is_file()

    c_dnam = DnamContainer.from_file(out_dnam)
    assert str(c_dnam.verification) == "CreatorVerified"

    # 4. Two-Tier `.dnam-ckpt` resume using `--checkpoint-in` and all 3 stems
    req_all = req.model_copy(update={"stem_descriptors": descriptors})
    req_all_path = tmp_path / "job_req_resume.json"
    req_all_path.write_text(req_all.model_dump_json(indent=2), encoding="utf-8")

    res_resume = run_worker_job(
        req_all_path,
        checkpoint_in=out_ckpt,
        output_dnam=out_dnam,
        output_ckpt=out_ckpt,
        epochs=1,
        topology="core1_only",
        device="cpu",
        precision="fp32",
        creator_seed_hex=seed_hex,
        event_format="rich",
        console=Console(quiet=True),
    )
    assert res_resume.resume_tier_executed == 1
    c_ckpt = DnamContainer.from_file(out_ckpt)
    assert c_ckpt.is_checkpoint
    assert c_ckpt.preamble.num_ckpt_stems == 3

    # 5. Also test explicit ResumeJobRequest payload JSON and env seed fallbacks
    explicit_resume = ResumeJobRequest(
        checkpoint_bytes=out_ckpt.read_bytes(),
        new_stem_descriptors=descriptors,
        config=TrainingConfig(epochs=1, topology="core1_only"),
    )
    exp_path = tmp_path / "explicit_resume.json"
    exp_path.write_text(explicit_resume.model_dump_json(), encoding="utf-8")
    res_exp = run_worker_job(exp_path, unsigned=True, event_format="sse")
    assert res_exp.resume_tier_executed == 1

    # 6. Test TrainingJobRequest with epochs/topology/device/precision overrides & DNAM_SIGNING_KEY
    key_file = tmp_path / "creator.key"
    key_file.write_text(seed_hex + "\n", encoding="utf-8")
    import os

    os.environ["DNAM_SIGNING_KEY"] = str(key_file)
    try:
        res_ovr = run_worker_job(
            req_path,
            from_scratch=True,
            audit_only=True,
            epochs=1,
            topology="core1_only",
            device="cpu",
            precision="fp32",
            no_cache=True,
            cache_max_gb=0.5,
            event_format="jsonl",
        )
        assert res_ovr.model_id == "b7k_worker"
    finally:
        os.environ.pop("DNAM_SIGNING_KEY", None)

    # 7. Test default_user_config_dir and _resolve_creator_seed_hex
    import dnam_trainer.cli.worker as worker_mod
    from dnam_trainer.cli.worker import _resolve_creator_seed_hex, default_user_config_dir

    os.environ["DNAM_CONFIG_DIR"] = str(tmp_path / "cfg_override")
    try:
        assert default_user_config_dir() == tmp_path / "cfg_override"
        user_key = default_user_config_dir() / "keys" / "creator_ed25519.key"
        user_key.parent.mkdir(parents=True, exist_ok=True)
        user_key.write_text(seed_hex + "\n", encoding="utf-8")
        assert _resolve_creator_seed_hex(None, False) == seed_hex
        user_key.write_text("short\n", encoding="utf-8")
        assert _resolve_creator_seed_hex(None, False) is None
    finally:
        os.environ.pop("DNAM_CONFIG_DIR", None)

    os.environ["DNAM_CREATOR_SEED_HEX"] = seed_hex
    try:
        assert _resolve_creator_seed_hex(None, False) == seed_hex
    finally:
        os.environ.pop("DNAM_CREATOR_SEED_HEX", None)
    assert isinstance(default_user_config_dir(), Path)

    orig_plat = worker_mod.sys.platform
    try:
        worker_mod.sys.platform = "win32"
        os.environ["APPDATA"] = str(tmp_path / "appdata")
        assert default_user_config_dir() == tmp_path / "appdata" / "dnam"
        os.environ.pop("APPDATA", None)
        assert default_user_config_dir().name == "dnam"

        worker_mod.sys.platform = "linux"
        os.environ["XDG_CONFIG_HOME"] = str(tmp_path / "xdg")
        assert default_user_config_dir() == tmp_path / "xdg" / "dnam"
        os.environ.pop("XDG_CONFIG_HOME", None)
        assert default_user_config_dir().name == "dnam"
    finally:
        worker_mod.sys.platform = orig_plat
