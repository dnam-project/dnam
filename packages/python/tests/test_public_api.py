"""Tests for public API curation, docstrings, legacy Tri-Physics removal, and README quickstart."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

import dnam

PUBLIC_MODULES: tuple[ModuleType, ...] = (
    dnam,
    dnam.errors,
    dnam.model,
    dnam.engine,
    dnam.controls,
    dnam.schema,
    dnam.provenance,
    dnam.dsp,
)

FORBIDDEN_LEGACY_NAMES: tuple[str, ...] = (
    "SelectiveRgLru",
    "SelectiveRgLruParams",
    "SelectiveRgLruState",
    "ModulatedUfdn",
    "MufdnParams",
    "MufdnState",
    "Core1Config",
    "Core2Config",
    "core2_active",
)


def test_curated_exports_docstrings_and_no_native_leakage() -> None:
    """verify[sdk.python.layout.package-namespace] verify[sdk.python.layout.curated-exports]
    verify[sdk.python.api.core-types]
    Verifies that every public module defines `__all__`, every exported symbol has a docstring,
    no `_native` or `native_*` symbols leak into `__all__`, and all legacy Tri-Physics names
    are absent.
    """
    for mod in PUBLIC_MODULES:
        exported = getattr(mod, "__all__", None)
        assert isinstance(exported, list)
        assert len(exported) > 0
        assert "_native" not in exported
        for name in exported:
            assert not name.startswith("native_"), (
                f"Native symbol '{name}' leaked into {mod.__name__}.__all__"
            )
            obj = getattr(mod, name)
            if name != "__version__":
                doc = getattr(obj, "__doc__", None)
                assert isinstance(doc, str)
                assert doc.strip()

        for legacy in FORBIDDEN_LEGACY_NAMES:
            assert not hasattr(mod, legacy), (
                f"Legacy Tri-Physics symbol '{legacy}' still present on {mod.__name__}"
            )


def test_readme_quickstart_workflow() -> None:
    """verify[sdk.python.api.core-types] verify[sdk.python.api.engine-facets]
    verify[sdk.python.engine.pedalboard-call]
    Executes the README quickstart workflow end-to-end.
    """
    model = dnam.Model.reference()
    assert model.max_tier == dnam.Tier.ULTRA

    engine = dnam.Engine(model, sample_rate=48_000, tier="full")
    engine.controls["gain"] = 7.5
    engine.controls.snap(gain=7.5, boost="on")
    engine.presets.apply("warm_slapback")
    engine.tempo.sync(bpm=120, subdivision="1/8.")

    audio = np.zeros((2, 256), dtype=np.float32)
    wet = engine(audio, 48_000)
    assert wet.shape == (2, 256)
    assert wet.dtype == np.float32

    with engine.bypassed(trails=True):
        tail = engine(audio, 48_000, reset=False)
        assert tail.shape == (2, 256)
    with engine.frozen():
        pad = engine(audio, 48_000, reset=False)
        assert pad.shape == (2, 256)

    bounced = engine.bounce(audio)
    assert bounced.shape == (2, 256)

    ckpt = dnam.Checkpoint.reference()
    lite_model = ckpt.export_model().with_tier_limit("lite")
    assert lite_model.max_tier == dnam.Tier.LITE


@pytest.mark.parametrize(
    "example_name",
    [
        "realtime_audio_and_bounce.py",
        "controls_presets_and_tempo.py",
        "model_slimming_and_checkpoint.py",
        "marketplace_provenance_and_watermark.py",
        "reference_dsp_operators.py",
    ],
)
def test_runnable_examples_execute_cleanly(example_name: str) -> None:
    """verify[sdk.python.api.core-types] verify[sdk.python.api.engine-facets]
    Verifies that every runnable script in `packages/python/examples/` executes cleanly.
    """
    example_path = Path(__file__).resolve().parent.parent / "examples" / example_name
    assert example_path.is_file()
    spec = importlib.util.spec_from_file_location(example_path.stem, example_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    main_fn = getattr(module, "main", None)
    assert callable(main_fn)
    main_fn()
