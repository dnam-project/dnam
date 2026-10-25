"""Architectural guardrails and progressive-disclosure checks for `packages/trainer/`."""

from __future__ import annotations

import ast
from pathlib import Path

TRAINER_PKG_ROOT = Path(__file__).resolve().parents[1]
MONOREPO_ROOT = TRAINER_PKG_ROOT.parents[1]
FORBIDDEN_IMPORTS = {"scipy", "pandas", "matplotlib", "soundfile"}


def test_zero_forbidden_imports_and_zero_type_ignore() -> None:
    """Verifies zero forbidden packages (`scipy`, `pandas`, `matplotlib`, `soundfile`)
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


def test_progressive_disclosure_agents_md_byte_limits() -> None:
    """Enforces `< 5,000 B` on root `AGENTS.md` and `< 2,500 B` on all scoped `AGENTS.md` files."""
    root_agents = MONOREPO_ROOT / "AGENTS.md"
    assert root_agents.is_file()
    assert root_agents.stat().st_size < 5_000

    scoped_agents = [
        *list(TRAINER_PKG_ROOT.rglob("AGENTS.md")),
        MONOREPO_ROOT / "packages" / "ui" / "AGENTS.md",
        MONOREPO_ROOT / "packages" / "studio" / "AGENTS.md",
    ]
    assert len(scoped_agents) >= 8
    for agent_file in scoped_agents:
        assert agent_file.is_file(), f"Missing scoped AGENTS.md: {agent_file}"
        size = agent_file.stat().st_size
        assert size < 2_500, f"{agent_file} exceeds 2,500 B limit ({size} B)"


def test_clamp_registry_cl01_to_cl18_documented() -> None:
    """Verifies that `docs/clamp_and_audit_registry.md` documents `CL-01` through `CL-18`."""
    reg_doc = MONOREPO_ROOT / "docs" / "clamp_and_audit_registry.md"
    text = reg_doc.read_text(encoding="utf-8")
    for idx in range(1, 19):
        tag = f"CL-{idx:02d}"
        assert tag in text, f"Missing clamp ID {tag} in {reg_doc}"
