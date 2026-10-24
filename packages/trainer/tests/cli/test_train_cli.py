"""Unit tests for `dnam_trainer.cli.train` (`dnam train` with smart `.dnam-ckpt` auto-resume,
`--from-scratch`, and report generation)."""

from __future__ import annotations

from dnam_engine import DnamContainer
from dnam_trainer.cli import main
from dnam_trainer.project.workspace import DnamProject


def test_cli_train_and_smart_ckpt_auto_resume(initialized_project: DnamProject) -> None:
    proj = initialized_project
    proj_dir = proj.root_dir

    # Record stems using software backend
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

    # 1. Initial train (creates dist/b7k_drive.dnam, checkpoint.dnam-ckpt, telemetry.parquet,
    # telemetry.html)
    main(
        [
            "train",
            "b7k_drive",
            "--epochs",
            "1",
            "--topology",
            "core1_only",
            "--project-dir",
            str(proj_dir),
        ]
    )
    dist_path = proj.dist_dnam_path("b7k_drive")
    ckpt_path = proj.model_checkpoint_path("b7k_drive")
    reports_dir = proj.model_reports_dir("b7k_drive")

    assert dist_path.is_file()
    assert ckpt_path.is_file()
    assert (reports_dir / "telemetry.parquet").is_file()
    assert (reports_dir / "telemetry.html").is_file()
    assert (reports_dir / "audit_report.json").is_file()

    c_dnam = DnamContainer.from_file(dist_path)
    assert str(c_dnam.verification) == "CreatorVerified"

    # 2. Second `dnam train` automatically resumes from checkpoint.dnam-ckpt (`Tier 1`)
    main(
        [
            "train",
            "--epochs",
            "1",
            "--project-dir",
            str(proj_dir),
        ]
    )
    c_ckpt = DnamContainer.from_file(ckpt_path)
    assert c_ckpt.is_checkpoint
    assert c_ckpt.preamble.num_ckpt_stems >= 5
