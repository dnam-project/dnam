"""Unit tests for `dnam_trainer.project.workspace` (`DnamProject` CRUD, TOML persistence, and
Ed25519 keys)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dnam_trainer.capture.excitation import write_wav_24bit
from dnam_trainer.project.workspace import (
    DnamProject,
    default_user_config_dir,
    dumps_toml,
    ensure_creator_ed25519_keypair,
    parse_compact_range_spec,
    parse_compact_switch_spec,
)
from dnam_trainer.schema import PresetDef


def test_default_user_config_dir_and_ed25519_keypair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DNAM_CONFIG_DIR", str(tmp_path / "cfg_override"))
    assert default_user_config_dir() == (tmp_path / "cfg_override").resolve()

    key_file, seed1, pub1 = ensure_creator_ed25519_keypair()
    assert key_file.is_file()
    assert len(seed1) == 32
    assert len(pub1) == 64

    # Second call reuses existing keypair
    _, seed2, pub2 = ensure_creator_ed25519_keypair()
    assert seed1 == seed2
    assert pub1 == pub2

    # Invalid seed length raises ValueError
    bad_key = tmp_path / "bad.key"
    bad_key.write_text("001122\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected 32 bytes"):
        ensure_creator_ed25519_keypair(bad_key)

    # Test default_user_config_dir without env override and DNAM_SIGNING_KEY env override
    monkeypatch.delenv("DNAM_CONFIG_DIR", raising=False)
    assert default_user_config_dir().name == "dnam"
    monkeypatch.setenv("DNAM_SIGNING_KEY", str(key_file))
    p_env, _, pub_env = ensure_creator_ed25519_keypair()
    assert p_env == key_file
    assert pub_env == pub1


def test_compact_range_and_switch_parsers() -> None:
    r1 = parse_compact_range_spec("drive=0..10:6.5")
    assert r1.id == "drive"
    assert r1.declared_range == (0.0, 10.0)
    assert r1.default == 6.5

    r2 = parse_compact_range_spec("blend=-12..12")
    assert r2.default == 0.0

    with pytest.raises(ValueError, match="Invalid compact range"):
        parse_compact_range_spec("drive")
    with pytest.raises(ValueError, match="strictly less"):
        parse_compact_range_spec("drive=10..0")
    with pytest.raises(ValueError, match="outside declared range"):
        parse_compact_range_spec("drive=0..10:15")

    s1 = parse_compact_switch_spec("attack=cut,flat,boost:boost")
    assert s1.id == "attack"
    assert s1.declared_options == ["cut", "flat", "boost"]
    assert s1.default == "boost"

    with pytest.raises(ValueError, match="at least 2 options"):
        parse_compact_switch_spec("attack=only_one")
    with pytest.raises(ValueError, match="not in declared options"):
        parse_compact_switch_spec("attack=cut,boost:invalid")


def test_dnam_project_crud_lifecycle(
    initialized_project: DnamProject,
    tmp_path: Path,
) -> None:
    proj = initialized_project
    assert proj.dnam_toml_path.is_file()
    assert proj.dry_input_wav_path.is_file()

    # Load from directory or dnam.toml path
    reloaded = DnamProject.load(proj.root_dir)
    assert reloaded.manifest.project_name == "Test Pedal Pack"
    assert len(reloaded.list_models()) == 1

    # Add second model, update, and remove
    m2 = proj.add_model("alpha_omega", name="Alpha Omega", topology="core1_only")
    assert m2.topology == "core1_only"
    with pytest.raises(ValueError, match="already exists"):
        proj.add_model("alpha_omega")

    m2_up = proj.update_model("alpha_omega", name="Alpha Omega Ultra", topology="full")
    assert m2_up.name == "Alpha Omega Ultra"
    assert m2_up.topology == "full"
    proj.remove_model("alpha_omega", delete_files=True)
    with pytest.raises(KeyError):
        proj.get_model("alpha_omega")

    # Control CRUD on b7k_drive
    proj.update_control(
        "b7k_drive", "drive", label="Drive Gain", declared_range=(0.0, 12.0), default=6.0
    )
    ctrls = proj.list_controls("b7k_drive")
    drive_c = next(c for c in ctrls if c.id == "drive")
    assert drive_c.name == "Drive Gain"
    assert drive_c.declared_range == (0.0, 12.0)

    proj.add_control("b7k_drive", parse_compact_range_spec("level=0..10:5"))
    proj.remove_control("b7k_drive", "level")
    with pytest.raises(KeyError):
        proj.remove_control("b7k_drive", "nonexistent")

    # Preset CRUD
    proj.update_preset(
        "b7k_drive", "default_noon", name="Noon Updated", controls={"drive": 6.0, "attack": "boost"}
    )
    assert proj.list_presets("b7k_drive")[0].name == "Noon Updated"
    proj.add_preset("b7k_drive", PresetDef(id="clean", name="Clean", controls={"drive": 1.0}))
    proj.remove_preset("b7k_drive", "clean")
    with pytest.raises(KeyError):
        proj.remove_preset("b7k_drive", "missing")

    # Stem CRUD (register external WAV, update, load_training_stems, remove)
    from dnam_trainer.capture.excitation import read_wav

    dry_pcm, dry_sr = read_wav(proj.dry_input_wav_path)
    ext_wav = tmp_path / "external_wet.wav"
    write_wav_24bit(ext_wav, (dry_pcm * 0.8).astype(np.float32), sample_rate_hz=dry_sr)

    entry = proj.register_stem(
        "b7k_drive",
        stem_id="stem_00_noon",
        wet_wav_path=ext_wav,
        controls={"drive": 6.0, "attack": "flat"},
        auto_align=True,
    )
    assert entry.stem_id == "stem_00_noon"
    assert len(proj.list_stems("b7k_drive")) == 1

    up_entry = proj.update_stem(
        "b7k_drive", "stem_00_noon", controls={"drive": 7.0, "attack": "boost"}
    )
    assert up_entry.controls["drive"] == 7.0

    loaded_stems = proj.load_training_stems("b7k_drive")
    assert len(loaded_stems) == 1
    assert loaded_stems[0].stem_id == "stem_00_noon"

    proj.remove_stem("b7k_drive", "stem_00_noon", delete_files=True)
    assert len(proj.list_stems("b7k_drive")) == 0

    with pytest.raises(ValueError, match="Cannot serialize non-finite float"):
        dumps_toml({"bad": float("nan")})

    # Additional TOML serialization, error paths, and CRUD edge cases
    toml_str = dumps_toml(
        {
            "flag_true": True,
            "flag_false": False,
            "nums": [1, 2, 3],
            "empty_list": [],
            "nested": {"k with space": "v", "sub": {"x": 1.5}},
            "items": [{"id": "a", "meta": {"m": 1}}],
        }
    )
    assert "flag_true = true" in toml_str
    assert "flag_false = false" in toml_str
    with pytest.raises(TypeError, match="Unsupported TOML value type"):
        dumps_toml({"bad_obj": object()})

    assert proj.audio_dir.is_dir()
    assert proj.model_renders_dir("b7k_drive").is_dir()

    with pytest.raises(FileNotFoundError, match="DNAM project manifest not found"):
        DnamProject.load(tmp_path / "nonexistent_dir")

    with pytest.raises(ValueError, match="Empty control id"):
        parse_compact_range_spec("=0..10")
    with pytest.raises(ValueError, match="Invalid compact switch spec"):
        parse_compact_switch_spec("no_equals")
    with pytest.raises(ValueError, match="Empty control id"):
        parse_compact_switch_spec("=a,b")

    with pytest.raises(ValueError, match="Invalid model_id"):
        proj.add_model("bad/id")

    from dnam_trainer.schema import CaptureBackendConfig, TrainingConfig

    proj.update_model(
        "b7k_drive",
        description="Updated desc",
        output_mode="mono",
        capture_config=CaptureBackendConfig(),
        training_config=TrainingConfig(),
    )
    assert proj.get_model("b7k_drive").output_mode == "mono"

    proj.add_model("temp_no_del")
    proj.remove_model("temp_no_del", delete_files=False)

    with pytest.raises(ValueError, match="already exists"):
        proj.add_control("b7k_drive", parse_compact_range_spec("drive=0..10:5"))
    proj.update_control("b7k_drive", "attack", options=["cut", "flat", "boost"], default="cut")
    with pytest.raises(KeyError, match="not found"):
        proj.update_control("b7k_drive", "nonexistent", label="X")

    with pytest.raises(ValueError, match="already exists"):
        proj.add_preset(
            "b7k_drive", PresetDef(id="default_noon", name="Noon Updated", controls={"drive": 5.0})
        )
    proj.update_preset("b7k_drive", "default_noon", description="New preset desc")
    with pytest.raises(KeyError, match="not found"):
        proj.update_preset("b7k_drive", "nonexistent", name="X")

    with pytest.raises(ValueError, match="0 recorded stems"):
        proj.load_training_stems("b7k_drive")
    with pytest.raises(KeyError, match="not found"):
        proj.update_stem("b7k_drive", "missing_stem", controls={"drive": 5.0})
    with pytest.raises(KeyError, match="not found"):
        proj.remove_stem("b7k_drive", "missing_stem")

    c_atk = next(c for c in proj.list_controls("b7k_drive") if c.id == "attack")
    assert c_atk.active_options == ["cut", "flat", "boost"]
