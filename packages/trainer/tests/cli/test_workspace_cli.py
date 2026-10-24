"""Unit tests for `dnam_trainer.cli.workspace` (`init`, `status`, `inspect`, `render`,
`model`, `control`, `preset`, and `stem` CLI subcommands)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dnam_engine import DnamContainer
from dnam_trainer.capture.excitation import write_wav_24bit
from dnam_trainer.cli import main, parse_key_value_controls


def test_parse_key_value_controls() -> None:
    coords = parse_key_value_controls(["drive=7.5,blend=5.0", "attack=boost"])
    assert coords == {"drive": 7.5, "blend": 5.0, "attack": "boost"}
    with pytest.raises(ValueError, match="Expected 'key=value'"):
        parse_key_value_controls(["bad_token"])


def test_cli_init_status_inspect_render_and_crud(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DNAM_CONFIG_DIR", str(tmp_path / "cfg"))
    proj_dir = tmp_path / "cli_proj"

    # 1. dnam init & status
    main(
        [
            "init",
            str(proj_dir),
            "--name",
            "CLI Project",
            "--author",
            "CLI User",
            "--dry-duration-sec",
            "1.5",
            "--model-id",
            "m_init",
        ]
    )
    main(["status", "--project-dir", str(proj_dir)])

    # 2. dnam model add / list / show / update / remove
    main(
        [
            "model",
            "add",
            "m_cli",
            "--name",
            "CLI Drive",
            "--range",
            "drive=0..10:5",
            "--switch",
            "attack=cut,flat,boost:flat",
            "--project-dir",
            str(proj_dir),
        ]
    )
    main(["model", "list", "--project-dir", str(proj_dir)])
    main(["model", "show", "m_cli", "--project-dir", str(proj_dir)])
    main(["model", "update", "m_cli", "--name", "CLI Drive v2", "--project-dir", str(proj_dir)])
    main(["model", "remove", "m_init", "--project-dir", str(proj_dir)])

    # 3. dnam control add / list / update / remove
    main(["control", "add", "m_cli", "--range", "tone=0..10:5", "--project-dir", str(proj_dir)])
    main(["control", "list", "m_cli", "--project-dir", str(proj_dir)])
    main(
        [
            "control",
            "update",
            "m_cli",
            "tone",
            "--label",
            "Tone Knob",
            "--min-val",
            "0.0",
            "--max-val",
            "12.0",
            "--default",
            "6.0",
            "--project-dir",
            str(proj_dir),
        ]
    )
    main(["control", "remove", "m_cli", "tone", "--project-dir", str(proj_dir)])

    # 4. dnam preset add / list / update / remove
    main(
        [
            "preset",
            "add",
            "m_cli",
            "p_noon",
            "--name",
            "Noon",
            "--set",
            "drive=5.0",
            "--set",
            "attack=flat",
            "--project-dir",
            str(proj_dir),
        ]
    )
    main(["preset", "list", "m_cli", "--project-dir", str(proj_dir)])
    main(
        [
            "preset",
            "update",
            "m_cli",
            "p_noon",
            "--set",
            "drive=6.0",
            "--project-dir",
            str(proj_dir),
        ]
    )
    main(["preset", "remove", "m_cli", "p_noon", "--project-dir", str(proj_dir)])

    # 5. dnam stem add / list / update / remove
    wet_wav = tmp_path / "wet_test.wav"
    sig = np.sin(np.linspace(0.0, 20.0, int(1.5 * 48_000), dtype=np.float32)) * 0.25
    write_wav_24bit(wet_wav, sig, sample_rate_hz=48_000)

    main(
        [
            "stem",
            "add",
            "m_cli",
            "s_00",
            str(wet_wav),
            "--set",
            "drive=5.0",
            "--no-auto-align",
            "--project-dir",
            str(proj_dir),
        ]
    )
    main(["stem", "list", "m_cli", "--project-dir", str(proj_dir)])
    main(["stem", "update", "m_cli", "s_00", "--set", "drive=7.0", "--project-dir", str(proj_dir)])
    main(["stem", "remove", "m_cli", "s_00", "--project-dir", str(proj_dir)])

    # 6. dnam inspect & render
    ref_path = DnamContainer.reference(core2_active=False).save(tmp_path / "ref.dnam")
    main(["inspect", str(ref_path)])
    out_render = tmp_path / "rendered.wav"
    main(["render", str(ref_path), str(wet_wav), str(out_render)])
    assert out_render.is_file()
