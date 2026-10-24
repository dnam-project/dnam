"""CLI commands for project initialization (`init`), status (`status`), container inspection
(`inspect`), offline rendering (`render`), and full CRUD sub-apps (`model`, `control`, `preset`,
`stem`)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import cyclopts
from cyclopts import Parameter
from rich.console import Console

from dnam_trainer.cli.console import (
    render_controls_table,
    render_models_table,
    render_presets_table,
    render_stems_table,
)
from dnam_trainer.project.runner import inspect_dnam_container, render_with_dnam_engine
from dnam_trainer.project.workspace import (
    DnamProject,
    parse_compact_range_spec,
    parse_compact_switch_spec,
)
from dnam_trainer.schema import (
    ControlCoords,
    ControlSchemaDef,
    ControlValue,
    OutputMode,
    PresetDef,
    Tier,
    TopologyMode,
)


def parse_key_value_controls(pairs: list[str] | tuple[str, ...] | None) -> ControlCoords:
    """Parses `['drive=7.5', 'attack=boost']` tokens into typed `ControlCoords`."""
    coords: ControlCoords = {}
    if not pairs:
        return coords
    for item in pairs:
        for token in item.split(","):
            token_s = token.strip()
            if not token_s:
                continue
            if "=" not in token_s:
                raise ValueError(f"Expected 'key=value' control assignment, got '{token_s}'")
            k, v_raw = token_s.split("=", 1)
            k = k.strip()
            v_str = v_raw.strip()
            try:
                coords[k] = float(v_str)
            except ValueError:
                coords[k] = v_str
    return coords


model_app = cyclopts.App(
    name="model", help="Manage models in the DNAM project (add/list/show/update/remove)."
)
control_app = cyclopts.App(
    name="control", help="Manage controls on a model (add/list/update/remove)."
)
preset_app = cyclopts.App(name="preset", help="Manage presets on a model (add/list/update/remove).")
stem_app = cyclopts.App(
    name="stem", help="Manage recorded stems on a model (add/list/update/remove)."
)


@model_app.command(name="add")
def model_add(
    model_id: str,
    *,
    name: str | None = None,
    description: str = "",
    output: OutputMode = "stereo",
    topology: TopologyMode = "auto",
    range_spec: Annotated[list[str] | None, Parameter(name=["--range", "-r"])] = None,
    switch_spec: Annotated[list[str] | None, Parameter(name=["--switch", "-s"])] = None,
    project_dir: Path = Path("."),
) -> None:
    """Add a new model (`models/<model_id>/model.toml`) with optional `--range` and `--switch`
    controls."""
    proj = DnamProject.load(project_dir)
    ctrls: list[ControlSchemaDef] = []
    for r_str in range_spec or []:
        ctrls.append(parse_compact_range_spec(r_str))
    for s_str in switch_spec or []:
        ctrls.append(parse_compact_switch_spec(s_str))
    spec = proj.add_model(
        model_id,
        name=name,
        description=description,
        output_mode=output,
        topology=topology,
        controls=ctrls,
    )
    Console().print(
        f"[green]✓ Added model[/green] [bold]{spec.model_id}[/bold] ({spec.name}) "
        f"with {len(spec.controls)} controls."
    )


@model_app.command(name="list")
def model_list(*, project_dir: Path = Path(".")) -> None:
    """List all models in the DNAM project workspace."""
    proj = DnamProject.load(project_dir)
    render_models_table(proj.list_models())


@model_app.command(name="show")
def model_show(model_id: str, *, project_dir: Path = Path(".")) -> None:
    """Show detailed controls, presets, and recorded stems for `model_id`."""
    proj = DnamProject.load(project_dir)
    spec = proj.get_model(model_id)
    con = Console()
    render_models_table([spec], console=con)
    render_controls_table(model_id, spec.controls, console=con)
    render_presets_table(model_id, spec.presets, console=con)
    render_stems_table(model_id, spec.stems, console=con)


@model_app.command(name="update")
def model_update(
    model_id: str,
    *,
    name: str | None = None,
    description: str | None = None,
    output: OutputMode | None = None,
    topology: TopologyMode | None = None,
    project_dir: Path = Path("."),
) -> None:
    """Update model metadata, output mode, or topology (`auto | core1_only | full`)."""
    proj = DnamProject.load(project_dir)
    spec = proj.update_model(
        model_id,
        name=name,
        description=description,
        output_mode=output,
        topology=topology,
    )
    Console().print(f"[green]✓ Updated model[/green] [bold]{spec.model_id}[/bold].")


@model_app.command(name="remove")
def model_remove(
    model_id: str,
    *,
    delete_files: bool = True,
    project_dir: Path = Path("."),
) -> None:
    """Remove `model_id` from the project workspace."""
    proj = DnamProject.load(project_dir)
    proj.remove_model(model_id, delete_files=delete_files)
    Console().print(f"[green]✓ Removed model[/green] [bold]{model_id}[/bold].")


@control_app.command(name="add")
def control_add(
    model_id: str,
    *,
    range_spec: Annotated[str | None, Parameter(name=["--range", "-r"])] = None,
    switch_spec: Annotated[str | None, Parameter(name=["--switch", "-s"])] = None,
    project_dir: Path = Path("."),
) -> None:
    """Add a range (`--range drive=0..10:5`) or switch (`--switch attack=cut,flat,boost:flat`)
    control."""
    proj = DnamProject.load(project_dir)
    if (range_spec is None) == (switch_spec is None):
        raise ValueError("Provide exactly one of `--range` or `--switch`")
    ctrl = (
        parse_compact_range_spec(range_spec)
        if range_spec is not None
        else parse_compact_switch_spec(switch_spec or "")
    )
    proj.add_control(model_id, ctrl)
    Console().print(f"[green]✓ Added control[/green] [bold]{ctrl.id}[/bold] to '{model_id}'.")


@control_app.command(name="list")
def control_list(model_id: str, *, project_dir: Path = Path(".")) -> None:
    """List controls declared on `model_id`."""
    proj = DnamProject.load(project_dir)
    render_controls_table(model_id, proj.list_controls(model_id))


@control_app.command(name="update")
def control_update(
    model_id: str,
    control_id: str,
    *,
    label: str | None = None,
    min_val: float | None = None,
    max_val: float | None = None,
    options: str | None = None,
    default: str | None = None,
    project_dir: Path = Path("."),
) -> None:
    """Update a control's label, range bounds, switch options, or default value."""
    proj = DnamProject.load(project_dir)
    dec_range: tuple[float, float] | None = None
    if min_val is not None and max_val is not None:
        dec_range = (float(min_val), float(max_val))
    opts_list = [o.strip() for o in options.split(",") if o.strip()] if options else None
    parsed_def: ControlValue | None = None
    if default is not None:
        try:
            parsed_def = float(default)
        except ValueError:
            parsed_def = default
    proj.update_control(
        model_id,
        control_id,
        label=label,
        declared_range=dec_range,
        options=opts_list,
        default=parsed_def,
    )
    Console().print(f"[green]✓ Updated control[/green] [bold]{control_id}[/bold] on '{model_id}'.")


@control_app.command(name="remove")
def control_remove(
    model_id: str,
    control_id: str,
    *,
    project_dir: Path = Path("."),
) -> None:
    """Remove `control_id` from `model_id`."""
    proj = DnamProject.load(project_dir)
    proj.remove_control(model_id, control_id)
    Console().print(
        f"[green]✓ Removed control[/green] [bold]{control_id}[/bold] from '{model_id}'."
    )


@preset_app.command(name="add")
def preset_add(
    model_id: str,
    preset_id: str,
    *,
    name: str | None = None,
    description: str | None = None,
    set_controls: Annotated[list[str] | None, Parameter(name=["--set", "-c"])] = None,
    project_dir: Path = Path("."),
) -> None:
    """Add a preset to `model_id` (`--set drive=7.5 --set attack=boost`)."""
    proj = DnamProject.load(project_dir)
    coords = parse_key_value_controls(set_controls)
    preset = PresetDef(
        id=preset_id,
        name=name or preset_id.replace("_", " ").title(),
        description=description,
        controls=coords,
    )
    proj.add_preset(model_id, preset)
    Console().print(f"[green]✓ Added preset[/green] [bold]{preset.id}[/bold] to '{model_id}'.")


@preset_app.command(name="list")
def preset_list(model_id: str, *, project_dir: Path = Path(".")) -> None:
    """List presets declared on `model_id`."""
    proj = DnamProject.load(project_dir)
    render_presets_table(model_id, proj.list_presets(model_id))


@preset_app.command(name="update")
def preset_update(
    model_id: str,
    preset_id: str,
    *,
    name: str | None = None,
    description: str | None = None,
    set_controls: Annotated[list[str] | None, Parameter(name=["--set", "-c"])] = None,
    project_dir: Path = Path("."),
) -> None:
    """Update a preset on `model_id`."""
    proj = DnamProject.load(project_dir)
    coords = parse_key_value_controls(set_controls) if set_controls else None
    proj.update_preset(
        model_id,
        preset_id,
        name=name,
        description=description,
        controls=coords,
    )
    Console().print(f"[green]✓ Updated preset[/green] [bold]{preset_id}[/bold] on '{model_id}'.")


@preset_app.command(name="remove")
def preset_remove(
    model_id: str,
    preset_id: str,
    *,
    project_dir: Path = Path("."),
) -> None:
    """Remove `preset_id` from `model_id`."""
    proj = DnamProject.load(project_dir)
    proj.remove_preset(model_id, preset_id)
    Console().print(f"[green]✓ Removed preset[/green] [bold]{preset_id}[/bold] from '{model_id}'.")


@stem_app.command(name="add")
def stem_add(
    model_id: str,
    stem_id: str,
    wet_wav: Path,
    *,
    set_controls: Annotated[list[str] | None, Parameter(name=["--set", "-c"])] = None,
    dry_wav: Path | None = None,
    auto_align: bool = True,
    overwrite: bool = False,
    project_dir: Path = Path("."),
) -> None:
    """Import an external or pre-recorded wet WAV stem into `models/<model_id>/model.toml`."""
    proj = DnamProject.load(project_dir)
    coords = parse_key_value_controls(set_controls)
    entry = proj.register_stem(
        model_id,
        stem_id=stem_id,
        wet_wav_path=wet_wav,
        controls=coords,
        dry_wav_path=dry_wav,
        auto_align=auto_align,
        overwrite=overwrite,
    )
    Console().print(
        f"[green]✓ Registered stem[/green] [bold]{entry.stem_id}[/bold] "
        f"(peak={entry.measured_return_peak_dbfs:+.2f} dBFS, "
        f"latency={entry.latency_offset_samples} samples)."
    )


@stem_app.command(name="list")
def stem_list(model_id: str, *, project_dir: Path = Path(".")) -> None:
    """List recorded stems for `model_id`."""
    proj = DnamProject.load(project_dir)
    render_stems_table(model_id, proj.list_stems(model_id))


@stem_app.command(name="update")
def stem_update(
    model_id: str,
    stem_id: str,
    *,
    set_controls: Annotated[list[str] | None, Parameter(name=["--set", "-c"])] = None,
    project_dir: Path = Path("."),
) -> None:
    """Update control coordinates for an existing stem."""
    proj = DnamProject.load(project_dir)
    coords = parse_key_value_controls(set_controls)
    proj.update_stem(model_id, stem_id, controls=coords)
    Console().print(f"[green]✓ Updated stem[/green] [bold]{stem_id}[/bold] on '{model_id}'.")


@stem_app.command(name="remove")
def stem_remove(
    model_id: str,
    stem_id: str,
    *,
    delete_files: bool = False,
    project_dir: Path = Path("."),
) -> None:
    """Remove `stem_id` from `model_id`."""
    proj = DnamProject.load(project_dir)
    proj.remove_stem(model_id, stem_id, delete_files=delete_files)
    Console().print(f"[green]✓ Removed stem[/green] [bold]{stem_id}[/bold] from '{model_id}'.")


def register_workspace_commands(app: cyclopts.App) -> None:
    """Registers top-level workspace commands (`init`, `status`, `inspect`, `render`) and CRUD
    sub-apps."""

    @app.command(name="init")
    def cmd_init(
        project_dir: Path = Path("."),
        *,
        name: str = "DNAM Studio Project",
        author: str = "DNAM Creator",
        sample_rate_hz: int = 48_000,
        dry_duration_sec: float = 4.0,
        model_id: str | None = None,
    ) -> None:
        """Initialize a new multi-model DNAM project workspace (`dnam.toml`, `audio/input.wav`,
        `Ed25519` keypair)."""
        proj = DnamProject.init(
            project_dir,
            project_name=name,
            author=author,
            sample_rate_hz=sample_rate_hz,
            dry_duration_sec=dry_duration_sec,
            initial_model_id=model_id,
        )
        pub_short = (proj.manifest.author_pubkey_ed25519 or "")[:16]
        Console().print(
            f"[bold green]✓ Initialized DNAM project[/bold green] at [cyan]{proj.root_dir}[/cyan] "
            f"(pubkey={pub_short}...)"
        )

    @app.command(name="status")
    def cmd_status(*, project_dir: Path = Path(".")) -> None:
        """Show project manifest summary and all models in the workspace."""
        proj = DnamProject.load(project_dir)
        con = Console()
        con.print(
            f"[bold cyan]{proj.manifest.project_name}[/bold cyan] | "
            f"author={proj.manifest.author} | sr={proj.manifest.sample_rate_hz} Hz | "
            f"pubkey={proj.manifest.author_pubkey_ed25519 or 'unsigned'}"
        )
        render_models_table(proj.list_models(), console=con)

    @app.command(name="inspect")
    def cmd_inspect(container_path: Path) -> None:
        """
        Inspect a `.dnam` or `.dnam-ckpt` container's preamble, tiers, controls, and provenance.
        """
        info = inspect_dnam_container(container_path)
        con = Console()
        con.print(
            f"[bold cyan]{info.model_name}[/bold cyan] ({info.magic}) | "
            f"author={info.author} | sr={info.sample_rate_train_hz} Hz | "
            f"verification={info.verification_status} | "
            f"core2={'ON' if info.core2_active else 'BYPASS'} | "
            f"arena={info.tensor_arena_byte_len:,} B"
        )
        render_controls_table(info.model_name, info.controls, console=con)
        if info.presets:
            render_presets_table(info.model_name, info.presets, console=con)

    @app.command(name="render")
    def cmd_render(
        container_path: Path,
        input_wav: Path,
        output_wav: Path,
        *,
        tier: Tier = Tier.ULTRA,
        preset: str | None = None,
        set_controls: Annotated[list[str] | None, Parameter(name=["--set", "-c"])] = None,
    ) -> None:
        """Render `input_wav` through the Rust SIMD `DnamEngine` and write a 24-bit WAV."""
        coords = parse_key_value_controls(set_controls)
        out = render_with_dnam_engine(
            container_path,
            input_wav,
            output_wav,
            tier=tier,
            preset=preset,
            controls=coords,
        )
        Console().print(f"[green]✓ Rendered 24-bit WAV to[/green] [bold]{out}[/bold].")

    app.command(model_app)
    app.command(control_app)
    app.command(preset_app)
    app.command(stem_app)
