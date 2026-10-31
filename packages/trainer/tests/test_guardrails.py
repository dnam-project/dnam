"""Architectural guardrails and progressive-disclosure checks for `packages/trainer/`."""

from __future__ import annotations

import ast
import re
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
        p for p in (MONOREPO_ROOT / "packages").rglob("AGENTS.md") if "node_modules" not in p.parts
    ]
    assert len(scoped_agents) >= 18
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


def test_no_inline_latex_math_in_markdown_tables() -> None:
    """Verifies that no Markdown table row contains inline LaTeX math (`$...$` outside backticks),
    preventing GitHub MathJax horizontal scrollbars inside table cells.
    """
    md_files = [
        MONOREPO_ROOT / "README.md",
        MONOREPO_ROOT / "AGENTS.md",
        MONOREPO_ROOT / "CONTRIBUTING.md",
        *list((MONOREPO_ROOT / "docs").rglob("*.md")),
        *[
            p
            for p in (MONOREPO_ROOT / "packages").rglob("*.md")
            if "node_modules" not in p.parts and "target" not in p.parts and "dist" not in p.parts
        ],
    ]
    assert len(md_files) >= 30

    for md_path in md_files:
        for line_no, line in enumerate(md_path.read_text(encoding="utf-8").splitlines(), start=1):
            if line.lstrip().startswith("|"):
                without_code_spans = re.sub(r"`[^`]*`", "", line)
                assert "$" not in without_code_spans, (
                    f"Inline LaTeX math ('$') inside Markdown table at {md_path}:{line_no}: {line}"
                )


def test_valid_github_latex_math_across_docs() -> None:
    """Verifies that no `$...$` or `$$...$$` math block contains CommonMark-stripped punctuation
    escapes (`\\_`, `\\;`, `\\,`, `\\!`, `\\{`, `\\}`, `\\|`, `\\%`, `\\\\`), disallowed GitHub
    macros (`\\operatorname`, `\\multicolumn`, `\\texttt`), or underscores inside `\\text{...}`.
    """
    md_files = [
        MONOREPO_ROOT / "README.md",
        MONOREPO_ROOT / "AGENTS.md",
        MONOREPO_ROOT / "CONTRIBUTING.md",
        *list((MONOREPO_ROOT / "docs").rglob("*.md")),
        *[
            p
            for p in (MONOREPO_ROOT / "packages").rglob("*.md")
            if "node_modules" not in p.parts and "target" not in p.parts and "dist" not in p.parts
        ],
    ]
    forbidden_macros = ("\\operatorname", "\\multicolumn", "\\texttt", "\\text{--")

    for md_path in md_files:
        content = md_path.read_text(encoding="utf-8")
        no_fenced = re.sub(
            r"```.*?```",
            lambda m: "\n" * m.group(0).count("\n"),
            content,
            flags=re.DOTALL,
        )
        no_code = re.sub(r"`[^`\n]*`", "", no_fenced)
        for m in re.finditer(r"\$\$(.*?)\$\$|\$([^\$\n]+?)\$", no_code, flags=re.DOTALL):
            math_str = m.group(1) if m.group(1) is not None else m.group(2)
            line_no = content[: m.start()].count("\n") + 1
            punct_escapes = re.findall(r"\\[^a-zA-Z0-9\s]", math_str)
            assert not punct_escapes, (
                f"Punctuation escape {punct_escapes} inside math at {md_path}:{line_no}: {math_str}"
            )
            for bad in forbidden_macros:
                assert bad not in math_str, (
                    f"Forbidden macro '{bad}' inside math at {md_path}:{line_no}: {math_str}"
                )
            for tm in re.finditer(r"\\(?:text|mathrm)\{([^}]*)\}", math_str):
                assert "_" not in tm.group(1), (
                    f"Underscore inside {tm.group(0)} at {md_path}:{line_no}: {math_str}"
                )
            assert math_str.count("{") == math_str.count("}"), (
                f"Unbalanced braces in math at {md_path}:{line_no}: {math_str}"
            )

        for m in re.finditer(r"```mermaid\n(.*?)```", content, flags=re.DOTALL):
            mermaid_block = m.group(1)
            line_no = content[: m.start()].count("\n") + 1
            if mermaid_block.lstrip().startswith("sequenceDiagram"):
                assert ";" not in mermaid_block, (
                    f"Semicolon ';' in Mermaid sequenceDiagram at {md_path}:{line_no}"
                )


def test_device_32bit_only_and_no_legacy_window_tables() -> None:
    """Enforces the 32-bit-only device compute policy and absence of hard-coded 500k sample tables
    across `dsp/` and cascade solver modules.
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
