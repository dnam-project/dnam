"""Trainer-package AST & numerical precision guardrails (`G-01`, `G-06`) for `packages/trainer/`.

For monorepo-wide documentation, specification, clamp registry, and progressive-disclosure
guardrail tests (`G-02`, `G-03`, `G-04`, `G-05`, `G-08`), see `tests/test_guardrails.py`.
"""

from __future__ import annotations

import ast
from pathlib import Path

TRAINER_PKG_ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_IMPORTS = {"scipy", "pandas", "matplotlib", "soundfile"}


def test_zero_forbidden_imports_and_zero_type_ignore() -> None:
    """Verifies (`G-01`) zero forbidden packages (`scipy`, `pandas`, `matplotlib`, `soundfile`)
    and zero `# type: ignore` comments across `packages/trainer/src/` and `packages/trainer/tests/`.
    """
    py_files = [
        *list((TRAINER_PKG_ROOT / "src").rglob("*.py")),
        *list((TRAINER_PKG_ROOT / "tests").rglob("*.py")),
    ]
    assert len(py_files) >= 15

    for py_path in py_files:
        source = py_path.read_text(encoding="utf-8")
        if py_path.name != "test_guardrails.py":
            assert "# type: ignore" not in source, f"Forbidden '# type: ignore' in {py_path}"
            assert "# pyrefly: ignore" not in source, f"Forbidden '# pyrefly: ignore' in {py_path}"

        tree = ast.parse(source, filename=str(py_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top_mod = alias.name.split(".")[0]
                    assert top_mod not in FORBIDDEN_IMPORTS, (
                        f"Forbidden import '{alias.name}' in {py_path}"
                    )
            elif isinstance(node, ast.ImportFrom) and node.module:
                top_mod = node.module.split(".")[0]
                assert top_mod not in FORBIDDEN_IMPORTS, (
                    f"Forbidden import from '{node.module}' in {py_path}"
                )


def test_device_32bit_only_and_no_legacy_window_tables() -> None:
    """Enforces (`G-06`) the 32-bit-only device compute policy and absence of hard-coded 500k
    sample tables across `dsp/` and cascade solver modules.
    """
    device_only_modules = [
        TRAINER_PKG_ROOT / "src" / "dnam_trainer" / "dsp" / "poly_adaa.py",
        TRAINER_PKG_ROOT / "src" / "dnam_trainer" / "dsp" / "selective_rglru.py",
        TRAINER_PKG_ROOT / "src" / "dnam_trainer" / "dsp" / "modulated_ufdn.py",
        TRAINER_PKG_ROOT / "src" / "dnam_trainer" / "training" / "cascade_orthogonal_fft.py",
        TRAINER_PKG_ROOT / "src" / "dnam_trainer" / "training" / "cascade_selective_stages.py",
        TRAINER_PKG_ROOT / "src" / "dnam_trainer" / "training" / "selective_cascade.py",
    ]
    forbidden_tokens = (
        "torch.float64",
        "torch.complex128",
        "torch.int64",
        "torch.long",
        ".double()",
        "np.linalg",
        "np.fft",
        "500_000",
    )
    for mod_path in device_only_modules:
        text = mod_path.read_text(encoding="utf-8")
        for tok in forbidden_tokens:
            assert tok not in text, f"Forbidden token '{tok}' found in {mod_path}"
