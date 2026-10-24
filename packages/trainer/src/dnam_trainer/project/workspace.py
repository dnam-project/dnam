"""Multi-model DNAM project workspace manager (`DnamProject`), atomic TOML persistence,
OS-appropriate `Ed25519` creator keypair provisioning, and compact control spec parsers."""

from __future__ import annotations

import json
import os
import secrets
import shutil
import sys
import tomllib
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dnam_engine import provenance
from dnam_trainer.capture.alignment import (
    align_hardware_latency,
    analyze_hot_corner_nonlinearity,
    measure_silent_window_noise_floor,
)
from dnam_trainer.capture.excitation import generate_dry_excitation, read_wav, write_wav_24bit
from dnam_trainer.schema import (
    CaptureBackendConfig,
    ControlCoords,
    ControlKind,
    ControlSchemaDef,
    ControlUiHints,
    ControlValue,
    DnamProjectManifest,
    OutputMode,
    PresetDef,
    ProjectModelSpec,
    ProjectStemEntry,
    TopologyMode,
    TrainingConfig,
    TrainingStem,
    compute_pcm_sha256,
)

_GITIGNORE_CONTENT = """# DNAM Project Ignore Rules
*.dnam-ckpt
audio/input.wav
models/*/renders/
models/*/reports/*.parquet
models/*/reports/*.html
.cache/
"""


def _utc_now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def default_user_config_dir() -> Path:
    """Returns the OS-appropriate user configuration directory for `dnam`.

    - Respects `DNAM_CONFIG_DIR` environment variable when set.
    - macOS (`darwin`): `~/Library/Application Support/dnam`
    - Windows (`win32`): `%APPDATA%\\dnam` (or `~/AppData/Roaming/dnam`)
    - Linux / POSIX: `$XDG_CONFIG_HOME/dnam` (or `~/.config/dnam`)
    """
    env_override = os.environ.get("DNAM_CONFIG_DIR", "").strip()
    if env_override:
        return Path(env_override).expanduser().resolve()

    home = Path.home()
    if sys.platform == "darwin":
        return home / "Library" / "Application Support" / "dnam"
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA", "").strip()
        if appdata:
            return Path(appdata) / "dnam"
        return home / "AppData" / "Roaming" / "dnam"
    xdg = os.environ.get("XDG_CONFIG_HOME", "").strip()
    if xdg:
        return Path(xdg) / "dnam"
    return home / ".config" / "dnam"


def ensure_creator_ed25519_keypair(
    key_path: str | Path | None = None,
) -> tuple[Path, bytes, str]:
    """Loads or generates the reusable 32-byte creator `Ed25519` keypair.

    Returns `(resolved_key_path, seed_32_bytes, pubkey_ed25519_hex)`.
    """
    env_key = os.environ.get("DNAM_SIGNING_KEY", "").strip()
    if key_path is not None:
        resolved = Path(key_path).expanduser().resolve()
    elif env_key:
        resolved = Path(env_key).expanduser().resolve()
    else:
        resolved = default_user_config_dir() / "keys" / "creator_ed25519.key"

    if resolved.is_file():
        raw_text = resolved.read_text(encoding="utf-8").strip()
        seed_32 = bytes.fromhex(raw_text)
        if len(seed_32) != 32:
            raise ValueError(
                f"Invalid Ed25519 secret seed in {resolved}: expected 32 bytes (64 hex chars), "
                f"got {len(seed_32)} bytes"
            )
    else:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        seed_32 = secrets.token_bytes(32)
        resolved.write_text(seed_32.hex() + "\n", encoding="utf-8")
        if os.name != "nt":
            resolved.chmod(0o600)

    pubkey_hex = provenance.ed25519_pubkey_hex(seed_32)
    return resolved, seed_32, pubkey_hex


def _format_toml_scalar(val: Any) -> str:
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, int):
        return str(val)
    if isinstance(val, float):
        if val != val or val == float("inf") or val == float("-inf"):
            raise ValueError(f"Cannot serialize non-finite float {val} to TOML")
        text = repr(val)
        if "." not in text and "e" not in text and "E" not in text:
            text += ".0"
        return text
    if isinstance(val, str):
        return json.dumps(val)
    if isinstance(val, (list, tuple)):
        return "[" + ", ".join(_format_toml_scalar(x) for x in val) + "]"
    if isinstance(val, Mapping):
        items = [
            f"{_format_toml_key(str(k))} = {_format_toml_scalar(v)}"
            for k, v in val.items()
            if v is not None
        ]
        return "{ " + ", ".join(items) + " }"
    raise TypeError(f"Unsupported TOML value type: {type(val)!r}")


def _format_toml_key(key: str) -> str:
    if key and all(c.isalnum() or c in "_-" for c in key):
        return key
    return json.dumps(key)


def _emit_toml_table(prefix: str, table: Mapping[str, Any], lines: list[str]) -> None:
    scalars: list[tuple[str, Any]] = []
    subtables: list[tuple[str, Mapping[str, Any]]] = []
    array_tables: list[tuple[str, list[Mapping[str, Any]]]] = []

    for k, v in table.items():
        if v is None:
            continue
        if isinstance(v, Mapping):
            subtables.append((k, v))
        elif isinstance(v, list) and v and all(isinstance(item, Mapping) for item in v):
            array_tables.append((k, v))
        elif isinstance(v, list) and not v:
            scalars.append((k, []))
        else:
            scalars.append((k, v))

    if prefix and (scalars or not (subtables or array_tables)):
        lines.append(f"[{prefix}]")
    for k, v in scalars:
        lines.append(f"{_format_toml_key(k)} = {_format_toml_scalar(v)}")
    if (prefix and scalars) or (not prefix and scalars):
        lines.append("")

    for k, sub in subtables:
        sub_prefix = f"{prefix}.{_format_toml_key(k)}" if prefix else _format_toml_key(k)
        _emit_toml_table(sub_prefix, sub, lines)

    for k, arr in array_tables:
        arr_prefix = f"{prefix}.{_format_toml_key(k)}" if prefix else _format_toml_key(k)
        for item in arr:
            lines.append(f"[[{arr_prefix}]]")
            for ik, iv in item.items():
                if iv is None:
                    continue
                lines.append(f"{_format_toml_key(ik)} = {_format_toml_scalar(iv)}")
            lines.append("")


def dumps_toml(data: Mapping[str, Any]) -> str:
    """Serializes a nested mapping to canonical UTF-8 TOML."""
    lines: list[str] = []
    _emit_toml_table("", data, lines)
    return "\n".join(lines).rstrip() + "\n"


def atomic_write_toml(path: Path, data: Mapping[str, Any]) -> None:
    """Atomically writes `data` as TOML to `path` via a temporary sibling file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    content = dumps_toml(data)
    # Verify round-trip parse before committing
    _ = tomllib.loads(content)
    tmp_path.write_text(content, encoding="utf-8")
    tmp_path.replace(path)


def parse_compact_range_spec(spec: str) -> ControlSchemaDef:
    """Parses a compact range spec `'id=min..max[:default]'` (e.g. `'drive=0..10:5'`)."""
    if "=" not in spec or ".." not in spec:
        raise ValueError(
            f"Invalid compact range spec '{spec}'. Expected 'id=min..max[:default]' "
            f"(e.g. 'drive=0..10:5')"
        )
    ctrl_id, rhs = spec.split("=", 1)
    ctrl_id = ctrl_id.strip()
    if not ctrl_id:
        raise ValueError(f"Empty control id in '{spec}'")

    if ":" in rhs:
        bounds_part, def_part = rhs.split(":", 1)
        default_val: float | None = float(def_part.strip())
    else:
        bounds_part = rhs
        default_val = None

    min_str, max_str = bounds_part.split("..", 1)
    lo = float(min_str.strip())
    hi = float(max_str.strip())
    if lo >= hi:
        raise ValueError(f"Range min ({lo}) must be strictly less than max ({hi}) in '{spec}'")

    resolved_default = 0.5 * (lo + hi) if default_val is None else default_val
    if resolved_default < lo or resolved_default > hi:
        raise ValueError(
            f"Default value {resolved_default} is outside declared range [{lo}, {hi}] in '{spec}'"
        )

    label = ctrl_id.replace("_", " ").title()
    return ControlSchemaDef(
        id=ctrl_id,
        name=label,
        kind=ControlKind.RANGE,
        declared_range=(lo, hi),
        safe_clamped_range=(lo, hi),
        default=resolved_default,
        ui_hints=ControlUiHints(),
    )


def parse_compact_switch_spec(spec: str) -> ControlSchemaDef:
    """Parses a compact switch spec `'id=opt1,opt2,...[:default]'` (e.g.
    `'attack=cut,flat,boost:flat'`)."""
    if "=" not in spec:
        raise ValueError(f"Invalid compact switch spec '{spec}'. Expected 'id=opt1,opt2[:default]'")
    ctrl_id, rhs = spec.split("=", 1)
    ctrl_id = ctrl_id.strip()
    if not ctrl_id:
        raise ValueError(f"Empty control id in '{spec}'")

    if ":" in rhs:
        opts_part, def_part = rhs.split(":", 1)
        default_opt: str | None = def_part.strip()
    else:
        opts_part = rhs
        default_opt = None

    options = [o.strip() for o in opts_part.split(",") if o.strip()]
    if len(options) < 2:
        raise ValueError(f"Switch control '{ctrl_id}' must declare at least 2 options in '{spec}'")

    resolved_default = options[0] if default_opt is None else default_opt
    if resolved_default not in options:
        raise ValueError(
            f"Default option '{resolved_default}' is not in declared options "
            f"{options} for '{ctrl_id}'"
        )

    label = ctrl_id.replace("_", " ").title()
    return ControlSchemaDef(
        id=ctrl_id,
        name=label,
        kind=ControlKind.SWITCH,
        declared_options=options,
        active_options=list(options),
        default=resolved_default,
    )


class DnamProject:
    """
    Multi-model DNAM project workspace backed by `dnam.toml` and `models/<model_id>/model.toml`.
    """

    def __init__(self, root_dir: Path, manifest: DnamProjectManifest) -> None:
        self.root_dir = root_dir.resolve()
        self.manifest = manifest

    @property
    def dnam_toml_path(self) -> Path:
        return self.root_dir / "dnam.toml"

    @property
    def audio_dir(self) -> Path:
        return self.root_dir / "audio"

    @property
    def dist_dir(self) -> Path:
        return self.root_dir / "dist"

    @property
    def models_dir(self) -> Path:
        return self.root_dir / "models"

    @property
    def dry_input_wav_path(self) -> Path:
        return self.manifest.resolve_dry_input(self.root_dir)

    def model_dir(self, model_id: str) -> Path:
        return self.models_dir / model_id

    def model_toml_path(self, model_id: str) -> Path:
        return self.model_dir(model_id) / "model.toml"

    def model_stems_dir(self, model_id: str) -> Path:
        return self.model_dir(model_id) / "stems"

    def model_reports_dir(self, model_id: str) -> Path:
        return self.model_dir(model_id) / "reports"

    def model_renders_dir(self, model_id: str) -> Path:
        return self.model_dir(model_id) / "renders"

    def model_checkpoint_path(self, model_id: str) -> Path:
        return self.model_dir(model_id) / "checkpoint.dnam-ckpt"

    def dist_dnam_path(self, model_id: str) -> Path:
        return self.dist_dir / f"{model_id}.dnam"

    def save_manifest(self) -> None:
        """Persists `self.manifest` atomically to `<root_dir>/dnam.toml`."""
        self.manifest = self.manifest.model_copy(update={"updated_at": _utc_now_iso()})
        atomic_write_toml(self.dnam_toml_path, self.manifest.model_dump(mode="json"))

    @classmethod
    def init(
        cls,
        project_dir: str | Path,
        *,
        project_name: str = "DNAM Studio Project",
        author: str = "DNAM Creator",
        sample_rate_hz: int = 48_000,
        generate_dry: bool = True,
        dry_duration_sec: float = 4.0,
        signing_key_path: str | Path | None = None,
        initial_model_id: str | None = None,
    ) -> DnamProject:
        """Initializes a new multi-model DNAM project directory."""
        root = Path(project_dir).resolve()
        root.mkdir(parents=True, exist_ok=True)
        (root / "audio").mkdir(parents=True, exist_ok=True)
        (root / "dist").mkdir(parents=True, exist_ok=True)
        (root / "models").mkdir(parents=True, exist_ok=True)

        gitignore_path = root / ".gitignore"
        if not gitignore_path.exists():
            gitignore_path.write_text(_GITIGNORE_CONTENT, encoding="utf-8")

        _, _, pubkey_hex = ensure_creator_ed25519_keypair(signing_key_path)
        now_iso = _utc_now_iso()
        manifest = DnamProjectManifest(
            project_name=project_name,
            author=author,
            signing_key_path=str(signing_key_path) if signing_key_path is not None else None,
            author_pubkey_ed25519=pubkey_hex,
            sample_rate_hz=sample_rate_hz,
            dry_input_path="audio/input.wav",
            capture=CaptureBackendConfig(sample_rate_hz=sample_rate_hz),
            training=TrainingConfig(sample_rate_train_hz=sample_rate_hz),
            created_at=now_iso,
            updated_at=now_iso,
        )
        proj = cls(root, manifest)
        proj.save_manifest()

        if generate_dry and not proj.dry_input_wav_path.exists():
            generate_dry_excitation(
                proj.dry_input_wav_path,
                duration_sec=dry_duration_sec,
                sample_rate_hz=sample_rate_hz,
            )

        if initial_model_id:
            proj.add_model(initial_model_id, name=initial_model_id.replace("_", " ").title())

        return proj

    @classmethod
    def load(cls, project_dir: str | Path = ".") -> DnamProject:
        """Loads an existing DNAM project from `project_dir` (or `<project_dir>/dnam.toml`)."""
        p = Path(project_dir).resolve()
        toml_path = p if p.is_file() and p.name == "dnam.toml" else p / "dnam.toml"
        if not toml_path.is_file():
            raise FileNotFoundError(
                f"DNAM project manifest not found at {toml_path}. Run `dnam init` first."
            )
        raw = tomllib.loads(toml_path.read_text(encoding="utf-8"))
        manifest = DnamProjectManifest.model_validate(raw)
        return cls(toml_path.parent, manifest)

    # ---- Model CRUD ----

    def list_models(self) -> list[ProjectModelSpec]:
        """Lists all models in `models/*/model.toml` sorted by `model_id`."""
        if not self.models_dir.is_dir():
            return []
        specs: list[ProjectModelSpec] = []
        for sub in sorted(self.models_dir.iterdir()):
            m_toml = sub / "model.toml"
            if sub.is_dir() and m_toml.is_file():
                raw = tomllib.loads(m_toml.read_text(encoding="utf-8"))
                specs.append(ProjectModelSpec.model_validate(raw))
        return specs

    def get_model(self, model_id: str) -> ProjectModelSpec:
        """Loads `models/<model_id>/model.toml` or raises `KeyError`."""
        m_toml = self.model_toml_path(model_id)
        if not m_toml.is_file():
            raise KeyError(f"Model '{model_id}' not found at {m_toml}")
        raw = tomllib.loads(m_toml.read_text(encoding="utf-8"))
        return ProjectModelSpec.model_validate(raw)

    def save_model(self, spec: ProjectModelSpec) -> ProjectModelSpec:
        """Persists `spec` atomically to `models/<spec.model_id>/model.toml`."""
        m_dir = self.model_dir(spec.model_id)
        (m_dir / "stems").mkdir(parents=True, exist_ok=True)
        (m_dir / "reports").mkdir(parents=True, exist_ok=True)
        (m_dir / "renders").mkdir(parents=True, exist_ok=True)
        atomic_write_toml(self.model_toml_path(spec.model_id), spec.model_dump(mode="json"))
        return spec

    def add_model(
        self,
        model_id: str,
        *,
        name: str | None = None,
        description: str = "",
        output_mode: OutputMode = "stereo",
        topology: TopologyMode = "auto",
        controls: list[ControlSchemaDef] | None = None,
        presets: list[PresetDef] | None = None,
    ) -> ProjectModelSpec:
        """Creates a new model under `models/<model_id>/model.toml`."""
        if not model_id or "/" in model_id or "\\" in model_id:
            raise ValueError(f"Invalid model_id '{model_id}'")
        if self.model_toml_path(model_id).exists():
            raise ValueError(f"Model '{model_id}' already exists in project")

        spec = ProjectModelSpec(
            model_id=model_id,
            name=name or model_id.replace("_", " ").title(),
            description=description,
            output_mode=output_mode,
            topology=topology,
            controls=list(controls or []),
            presets=list(presets or []),
        )
        return self.save_model(spec)

    def update_model(
        self,
        model_id: str,
        *,
        name: str | None = None,
        description: str | None = None,
        output_mode: OutputMode | None = None,
        topology: TopologyMode | None = None,
        capture_config: CaptureBackendConfig | None = None,
        training_config: TrainingConfig | None = None,
    ) -> ProjectModelSpec:
        """Updates metadata or config overrides for `model_id`."""
        spec = self.get_model(model_id)
        updates: dict[str, Any] = {}
        if name is not None:
            updates["name"] = name
        if description is not None:
            updates["description"] = description
        if output_mode is not None:
            updates["output_mode"] = output_mode
        if topology is not None:
            updates["topology"] = topology
        if capture_config is not None:
            updates["capture_config"] = capture_config
        if training_config is not None:
            updates["training_config"] = training_config
        updated = spec.model_copy(update=updates)
        return self.save_model(updated)

    def remove_model(self, model_id: str, *, delete_files: bool = True) -> None:
        """Removes `model_id` from the project."""
        _ = self.get_model(model_id)
        m_dir = self.model_dir(model_id)
        if delete_files and m_dir.exists():
            shutil.rmtree(m_dir)
        else:
            self.model_toml_path(model_id).unlink(missing_ok=True)
        self.dist_dnam_path(model_id).unlink(missing_ok=True)

    # ---- Control CRUD ----

    def list_controls(self, model_id: str) -> list[ControlSchemaDef]:
        return list(self.get_model(model_id).controls)

    def add_control(self, model_id: str, control: ControlSchemaDef) -> ProjectModelSpec:
        spec = self.get_model(model_id)
        if any(c.id == control.id for c in spec.controls):
            raise ValueError(f"Control '{control.id}' already exists on model '{model_id}'")
        updated = spec.model_copy(update={"controls": [*spec.controls, control]})
        return self.save_model(updated)

    def update_control(
        self,
        model_id: str,
        control_id: str,
        *,
        label: str | None = None,
        declared_range: tuple[float, float] | None = None,
        options: list[str] | None = None,
        default: ControlValue | None = None,
    ) -> ProjectModelSpec:
        spec = self.get_model(model_id)
        new_ctrls: list[ControlSchemaDef] = []
        found = False
        for c in spec.controls:
            if c.id == control_id:
                found = True
                updates: dict[str, Any] = {}
                if label is not None:
                    updates["name"] = label
                if declared_range is not None:
                    updates["declared_range"] = declared_range
                    updates["safe_clamped_range"] = declared_range
                if options is not None:
                    updates["declared_options"] = options
                    updates["active_options"] = list(options)
                if default is not None:
                    updates["default"] = default
                new_ctrls.append(c.model_copy(update=updates))
            else:
                new_ctrls.append(c)
        if not found:
            raise KeyError(f"Control '{control_id}' not found on model '{model_id}'")
        return self.save_model(spec.model_copy(update={"controls": new_ctrls}))

    def remove_control(self, model_id: str, control_id: str) -> ProjectModelSpec:
        spec = self.get_model(model_id)
        remaining = [c for c in spec.controls if c.id != control_id]
        if len(remaining) == len(spec.controls):
            raise KeyError(f"Control '{control_id}' not found on model '{model_id}'")
        return self.save_model(spec.model_copy(update={"controls": remaining}))

    # ---- Preset CRUD ----

    def list_presets(self, model_id: str) -> list[PresetDef]:
        return list(self.get_model(model_id).presets)

    def add_preset(self, model_id: str, preset: PresetDef) -> ProjectModelSpec:
        spec = self.get_model(model_id)
        if any(p.id == preset.id or p.name == preset.name for p in spec.presets):
            raise ValueError(f"Preset '{preset.name}' already exists on model '{model_id}'")
        return self.save_model(spec.model_copy(update={"presets": [*spec.presets, preset]}))

    def update_preset(
        self,
        model_id: str,
        preset_id_or_name: str,
        *,
        name: str | None = None,
        description: str | None = None,
        controls: ControlCoords | None = None,
    ) -> ProjectModelSpec:
        spec = self.get_model(model_id)
        new_presets: list[PresetDef] = []
        found = False
        for p in spec.presets:
            if p.id == preset_id_or_name or p.name == preset_id_or_name:
                found = True
                updates: dict[str, Any] = {}
                if name is not None:
                    updates["name"] = name
                if description is not None:
                    updates["description"] = description
                if controls is not None:
                    updates["controls"] = dict(controls)
                new_presets.append(p.model_copy(update=updates))
            else:
                new_presets.append(p)
        if not found:
            raise KeyError(f"Preset '{preset_id_or_name}' not found on model '{model_id}'")
        return self.save_model(spec.model_copy(update={"presets": new_presets}))

    def remove_preset(self, model_id: str, preset_id_or_name: str) -> ProjectModelSpec:
        spec = self.get_model(model_id)
        remaining = [
            p for p in spec.presets if p.id != preset_id_or_name and p.name != preset_id_or_name
        ]
        if len(remaining) == len(spec.presets):
            raise KeyError(f"Preset '{preset_id_or_name}' not found on model '{model_id}'")
        return self.save_model(spec.model_copy(update={"presets": remaining}))

    # ---- Stem CRUD ----

    def list_stems(self, model_id: str) -> list[ProjectStemEntry]:
        return list(self.get_model(model_id).stems)

    def register_stem(
        self,
        model_id: str,
        *,
        stem_id: str,
        wet_wav_path: str | Path,
        controls: ControlCoords | None = None,
        dry_wav_path: str | Path | None = None,
        auto_align: bool = True,
        overwrite: bool = False,
    ) -> ProjectStemEntry:
        """Imports a wet WAV, optionally aligns latency against dry input, and registers in
        `model.toml`."""
        spec = self.get_model(model_id)
        if not overwrite and any(s.stem_id == stem_id for s in spec.stems):
            raise ValueError(f"Stem '{stem_id}' already exists on model '{model_id}'")

        dry_path = Path(dry_wav_path).resolve() if dry_wav_path else self.dry_input_wav_path
        dry_audio, dry_fs = read_wav(dry_path)
        wet_audio, wet_fs = read_wav(wet_wav_path)
        if dry_fs != wet_fs:
            raise ValueError(
                f"Sample rate mismatch between dry ({dry_fs} Hz) and wet ({wet_fs} Hz) WAVs"
            )

        if auto_align:
            align_res = align_hardware_latency(dry_audio, wet_audio, sample_rate_hz=wet_fs)
            aligned_wet = align_res.aligned_audio
            latency_offset = align_res.latency_offset_samples
        else:
            aligned_wet = wet_audio
            latency_offset = 0

        cap_cfg = spec.capture_config or self.manifest.capture
        hot_stats = analyze_hot_corner_nonlinearity(
            dry_audio,
            aligned_wet,
            unity_ref_dbfs=cap_cfg.unity_ref_dbfs,
        )
        noise_db = measure_silent_window_noise_floor(aligned_wet, sample_rate_hz=wet_fs)

        dest_wav = self.model_stems_dir(model_id) / f"{stem_id}.wav"
        if Path(wet_wav_path).resolve() != dest_wav.resolve() or auto_align:
            write_wav_24bit(dest_wav, aligned_wet, sample_rate_hz=wet_fs)

        entry = ProjectStemEntry(
            stem_id=stem_id,
            wet_wav=f"stems/{stem_id}.wav",
            controls=dict(controls or {}),
            send_gain_db=cap_cfg.send_gain_db,
            unity_ref_dbfs=cap_cfg.unity_ref_dbfs,
            measured_return_peak_dbfs=hot_stats.measured_return_peak_dbfs,
            measured_return_rms_dbfs=hot_stats.measured_return_rms_dbfs,
            net_gain_over_unity_db=hot_stats.net_gain_over_unity_db,
            noise_floor_dbfs=noise_db,
            clipped_sample_fraction=hot_stats.clipped_sample_fraction,
            latency_offset_samples=latency_offset,
            input_pcm_sha256=compute_pcm_sha256(dry_audio),
            target_pcm_sha256=compute_pcm_sha256(aligned_wet),
            captured_at=_utc_now_iso(),
        )
        self.upsert_stem_entry(model_id, entry)
        return entry

    def upsert_stem_entry(self, model_id: str, entry: ProjectStemEntry) -> ProjectModelSpec:
        """Inserts or replaces `entry` in `models/<model_id>/model.toml`."""
        spec = self.get_model(model_id)
        stems = [s for s in spec.stems if s.stem_id != entry.stem_id]
        stems.append(entry)
        return self.save_model(spec.model_copy(update={"stems": stems}))

    def update_stem(
        self,
        model_id: str,
        stem_id: str,
        *,
        controls: ControlCoords | None = None,
    ) -> ProjectStemEntry:
        spec = self.get_model(model_id)
        new_stems: list[ProjectStemEntry] = []
        updated_entry: ProjectStemEntry | None = None
        for s in spec.stems:
            if s.stem_id == stem_id:
                updated_entry = s.model_copy(
                    update={"controls": dict(controls) if controls is not None else s.controls}
                )
                new_stems.append(updated_entry)
            else:
                new_stems.append(s)
        if updated_entry is None:
            raise KeyError(f"Stem '{stem_id}' not found on model '{model_id}'")
        self.save_model(spec.model_copy(update={"stems": new_stems}))
        return updated_entry

    def remove_stem(
        self,
        model_id: str,
        stem_id: str,
        *,
        delete_files: bool = False,
    ) -> ProjectModelSpec:
        spec = self.get_model(model_id)
        target: ProjectStemEntry | None = None
        remaining: list[ProjectStemEntry] = []
        for s in spec.stems:
            if s.stem_id == stem_id:
                target = s
            else:
                remaining.append(s)
        if target is None:
            raise KeyError(f"Stem '{stem_id}' not found on model '{model_id}'")
        if delete_files:
            wav_file = self.model_dir(model_id) / target.wet_wav
            wav_file.unlink(missing_ok=True)
        return self.save_model(spec.model_copy(update={"stems": remaining}))

    def load_training_stems(self, model_id: str) -> list[TrainingStem]:
        """Loads all registered stems for `model_id` into validated `TrainingStem` instances."""
        spec = self.get_model(model_id)
        if not spec.stems:
            raise ValueError(f"Model '{model_id}' has 0 recorded stems")

        dry_audio, dry_fs = read_wav(self.dry_input_wav_path)
        loaded: list[TrainingStem] = []
        for s in spec.stems:
            wet_path = self.model_dir(model_id) / s.wet_wav
            wet_audio, wet_fs = read_wav(wet_path)
            if wet_fs != dry_fs:
                raise ValueError(
                    f"Stem '{s.stem_id}' sample rate ({wet_fs} Hz) does not match "
                    f"dry input ({dry_fs} Hz)"
                )
            loaded.append(
                TrainingStem(
                    stem_id=s.stem_id,
                    controls_meta=dict(s.controls),
                    input_audio=dry_audio,
                    target_audio=wet_audio,
                    sample_rate_hz=wet_fs,
                    noise_floor_dbfs=s.noise_floor_dbfs,
                    input_pcm_sha256=s.input_pcm_sha256,
                    target_pcm_sha256=s.target_pcm_sha256,
                )
            )
        return loaded


__all__ = [
    "DnamProject",
    "atomic_write_toml",
    "default_user_config_dir",
    "dumps_toml",
    "ensure_creator_ed25519_keypair",
    "parse_compact_range_spec",
    "parse_compact_switch_spec",
]
