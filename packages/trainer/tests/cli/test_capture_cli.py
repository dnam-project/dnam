"""Unit tests for `dnam_trainer.cli.capture` and `dnam_trainer.cli.console`."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console

from dnam_trainer.cli import main
from dnam_trainer.cli.console import (
    CliCaptureObserver,
    prompt_calibration_step1,
    prompt_calibration_step2,
    prompt_next_stem_action,
)
from dnam_trainer.events import CalibrationStepEvent, StemCaptureEvent
from dnam_trainer.project.planner import PlannedStemCapture
from dnam_trainer.project.workspace import DnamProject


def test_prompt_next_stem_action_record_retake_skip() -> None:
    con = Console(quiet=True)
    planned = PlannedStemCapture(
        stem_id="stem_01",
        category="star_range",
        controls={"drive": 10.0, "attack": "boost"},
        is_recorded=False,
        reason="Unlock max drive",
    )

    assert (
        prompt_next_stem_action(
            planned,
            stem_index=1,
            total_stems=3,
            prev_controls={"drive": 5.0, "attack": "flat"},
            auto_yes=True,
            console=con,
        )
        == "record"
    )
    assert (
        prompt_next_stem_action(
            planned,
            stem_index=1,
            total_stems=3,
            countdown_sec=0.01,
            console=con,
        )
        == "record"
    )
    assert (
        prompt_next_stem_action(
            planned,
            stem_index=1,
            total_stems=3,
            console=con,
            input_fn=lambda _: "s",
        )
        == "skip"
    )
    assert (
        prompt_next_stem_action(
            planned,
            stem_index=1,
            total_stems=3,
            console=con,
            input_fn=lambda _: "r",
        )
        == "retake"
    )
    assert (
        prompt_next_stem_action(
            planned,
            stem_index=1,
            total_stems=3,
            console=con,
            input_fn=lambda _: "",
        )
        == "record"
    )

    obs = CliCaptureObserver(console=con)
    obs.on_capture_event(
        StemCaptureEvent(
            status="skipped",
            model_id="m1",
            stem_id="stem_01",
            stem_index=1,
            total_stems=3,
        )
    )

    # Interactive calibration step 1 & step 2 prompts
    prompt_calibration_step1(
        "Connect USB In/Out",
        auto_yes=False,
        console=con,
        input_fn=lambda _: "",
    )
    ev1 = CalibrationStepEvent(
        step=1,
        backend_id="usb",
        send_gain_db=0.0,
        unity_ref_dbfs=0.0,
        measured_peak_dbfs=-1.0,
        measured_rms_dbfs=-12.0,
        calibrated_latency_samples=64,
        noise_floor_dbfs=-96.0,
    )
    prompt_calibration_step2(
        ev1,
        auto_yes=False,
        console=con,
        input_fn=lambda _: "",
    )


def test_cli_capture_subcommands(
    initialized_project: DnamProject,
    tmp_path: Path,
) -> None:
    proj_dir = initialized_project.root_dir
    gen_wav = tmp_path / "gen_dry.wav"

    main(["capture", "backends"])
    main(["capture", "devices", "--backend", "software"])
    main(["capture", "generate", str(gen_wav), "--duration-sec", "1.5"])
    assert gen_wav.is_file()

    # Running record before calibrate automatically runs probe calibration & synthesizes input.wav
    main(
        [
            "capture",
            "record",
            "b7k_drive",
            "--backend",
            "software",
            "--no-include-corners",
            "--yes",
            "--project-dir",
            str(proj_dir),
        ]
    )
    reloaded = DnamProject.load(proj_dir)
    assert reloaded.manifest.capture.excitation_profile is not None
    assert len(reloaded.list_stems("b7k_drive")) >= 5

    # Explicit calibrate with --send-gain-db also updates excitation_profile and input.wav
    main(
        [
            "capture",
            "calibrate",
            "--backend",
            "software",
            "--send-gain-db",
            "0.0",
            "--project-dir",
            str(proj_dir),
        ]
    )
    main(["capture", "plan", "b7k_drive", "--project-dir", str(proj_dir)])

    # Second run when all planned stems are already recorded exits cleanly
    main(
        [
            "capture",
            "record",
            "b7k_drive",
            "--backend",
            "software",
            "--no-include-corners",
            "--yes",
            "--project-dir",
            str(proj_dir),
        ]
    )
