"""Monorepo-wide documentation, specification, clamp registry, and progressive-disclosure
guardrail tests (`G-02`, `G-03`, `G-04`, `G-05`, `G-08`).
"""

from __future__ import annotations

import re
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
MONOREPO_ROOT = TESTS_DIR.parent


def _all_repository_markdown_files() -> list[Path]:
    """Returns all tracked Markdown files across the root, `docs/`, `packages/`, and `tests/`."""
    return [
        MONOREPO_ROOT / "README.md",
        MONOREPO_ROOT / "AGENTS.md",
        MONOREPO_ROOT / "CONTRIBUTING.md",
        MONOREPO_ROOT / "CLA.md",
        *list((MONOREPO_ROOT / "docs").rglob("*.md")),
        *list((MONOREPO_ROOT / "tests").rglob("*.md")),
        *[
            p
            for p in (MONOREPO_ROOT / "packages").rglob("*.md")
            if "node_modules" not in p.parts and "target" not in p.parts and "dist" not in p.parts
        ],
    ]


def test_progressive_disclosure_agents_md_byte_limits() -> None:
    """Enforces `< 5,000 B` on root `AGENTS.md` and `< 2,500 B` on scoped `AGENTS.md` (`G-02`)."""
    root_agents = MONOREPO_ROOT / "AGENTS.md"
    assert root_agents.is_file()
    assert root_agents.stat().st_size < 5_000

    scoped_agents = [
        *[
            p
            for p in (MONOREPO_ROOT / "packages").rglob("AGENTS.md")
            if "node_modules" not in p.parts
        ],
        *list((MONOREPO_ROOT / "tests").rglob("AGENTS.md")),
    ]
    assert len(scoped_agents) >= 19
    for agent_file in scoped_agents:
        assert agent_file.is_file(), f"Missing scoped AGENTS.md: {agent_file}"
        size = agent_file.stat().st_size
        assert size < 2_500, f"{agent_file} exceeds 2,500 B limit ({size} B)"


def test_clamp_and_requirement_registries_documented() -> None:
    """Verifies (`G-03`) that:
    1. `docs/CONTRIBUTING.md` exists and documents specification authoring standards.
    2. `docs/training/clamp_and_audit_registry.md` documents `CL-01`..`CL-20` with existing target
       module paths on disk.
    3. `docs/overview/system_goals_and_requirements.md` documents all 40 requirements
       (`REQ-1.1`..`REQ-8.8`, including complex full-rig modeling `REQ-1.5`, MIDI/host tempo sync
       & rhythmic subdivision `REQ-2.5`, single-run slimmable tier training `REQ-5.1`..`REQ-5.6`,
       mono/true-stereo channel topologies `REQ-7.4`, and model/marketplace security
       `REQ-8.1`..`REQ-8.8`) with zero implementation-specific tokens.
    """
    docs_contributing = MONOREPO_ROOT / "docs" / "CONTRIBUTING.md"
    assert docs_contributing.is_file(), f"Missing documentation guide: {docs_contributing}"

    reg_doc = MONOREPO_ROOT / "docs" / "training" / "clamp_and_audit_registry.md"
    text = reg_doc.read_text(encoding="utf-8")
    for idx in range(1, 21):
        tag = f"CL-{idx:02d}"
        assert tag in text, f"Missing clamp ID {tag} in {reg_doc}"
        row_match = re.search(rf"^\|\s*\*\*`{tag}`\*\*.*$", text, flags=re.MULTILINE)
        assert row_match is not None, f"Missing table row for {tag} in {reg_doc}"
        mod_matches = re.findall(r"`(packages/[^`]+\.(?:py|rs))`", row_match.group(0))
        assert mod_matches, f"Missing target module path for {tag} in {reg_doc}"
        assert any((MONOREPO_ROOT / m).is_file() for m in mod_matches), (
            f"None of target modules {mod_matches} for {tag} exist on disk"
        )

    goals_doc = MONOREPO_ROOT / "docs" / "overview" / "system_goals_and_requirements.md"
    goals_text = goals_doc.read_text(encoding="utf-8")
    expected_counts = {1: 8, 2: 5, 3: 3, 4: 3, 5: 6, 6: 3, 7: 4, 8: 8}
    for goal_idx, max_sub in expected_counts.items():
        for sub_idx in range(1, max_sub + 1):
            req_tag = f"REQ-{goal_idx}.{sub_idx}"
            req_row = re.search(rf"^\|\s*\*\*`{req_tag}`\*\*.*$", goals_text, flags=re.MULTILINE)
            assert req_row is not None, (
                f"Missing canonical table row for requirement {req_tag} in {goals_doc}"
            )

    prose_without_link_targets = re.sub(r"\]\([^)]+\)", "]()", goals_text)
    forbidden_impl_tokens = (
        ".dnam-ckpt",
        ".dnam-cache",
        ".dnam-seal",
        ".dnce",
        ".dnam",
        "max_supported_tier",
        "user_metadata",
        "royalty_splits",
        "allowed_protection_profiles",
        "allowed_oem_vendors",
        "output head",
        "output-head",
        "SIMD lane",
        "Hann-windowed",
        "JSON header",
        "mono_only",
        "CL-0",
        "CL-1",
        "CL-2",
    )
    for impl_tok in forbidden_impl_tokens:
        assert impl_tok not in prose_without_link_targets, (
            f"Implementation-specific token '{impl_tok}' leaked into implementation-agnostic "
            f"requirements prose in {goals_doc}"
        )


def test_no_inline_latex_math_in_markdown_tables() -> None:
    """Verifies (`G-04`) that no Markdown table row contains inline LaTeX math (`$...$` outside
    backticks), preventing GitHub MathJax horizontal scrollbars inside table cells.
    """
    md_files = _all_repository_markdown_files()
    assert len(md_files) >= 32

    for md_path in md_files:
        for line_no, line in enumerate(md_path.read_text(encoding="utf-8").splitlines(), start=1):
            if line.lstrip().startswith("|"):
                without_code_spans = re.sub(r"`[^`]*`", "", line)
                assert "$" not in without_code_spans, (
                    f"Inline LaTeX math ('$') inside Markdown table at {md_path}:{line_no}: {line}"
                )


def test_valid_github_latex_math_across_docs() -> None:
    """Verifies (`G-05`) that no `$...$` or `$$...$$` math block contains CommonMark-stripped
    punctuation escapes (`\\_`, `\\;`, `\\,`, `\\!`, `\\{`, `\\}`, `\\|`, `\\%`, `\\\\`),
    disallowed GitHub macros (`\\operatorname`, `\\multicolumn`, `\\texttt`), or underscores
    inside `\\text{...}`.
    """
    md_files = _all_repository_markdown_files()
    forbidden_macros = ("\\operatorname", "\\multicolumn", "\\texttt", "\\text{--")

    for md_path in md_files:
        content = md_path.read_text(encoding="utf-8")
        no_fenced = re.sub(
            r"(?m)^```[^\n]*\n.*?\n^```[^\n]*$",
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
            stripped_block = mermaid_block.lstrip()
            if stripped_block.startswith("sequenceDiagram"):
                assert ";" not in mermaid_block, (
                    f"Semicolon ';' in Mermaid sequenceDiagram at {md_path}:{line_no}"
                )
            elif stripped_block.startswith("flowchart"):
                unquoted_labels = re.findall(
                    r"\b[A-Za-z0-9_]+\[(?!\")([^\]]*[():][^\]]*)\]", mermaid_block
                )
                assert not unquoted_labels, (
                    f"Unquoted special characters in Mermaid flowchart label {unquoted_labels} "
                    f"at {md_path}:{line_no}"
                )
                invalid_bidir_labels = re.findall(
                    r"<(?:--|==)\s+[^->=\n]+\s+(?:-->|==>)", mermaid_block
                )
                assert not invalid_bidir_labels, (
                    f"Invalid inline label in bidirectional Mermaid arrow {invalid_bidir_labels} "
                    f"(use '<-->|\"...\"|' or '<==>|\"...\"|') at {md_path}:{line_no}"
                )


def _github_heading_slugs(md_path: Path) -> set[str]:
    """Computes GitHub-flavored Markdown heading anchor slugs for `md_path`,
    including link-label extraction and `-1`, `-2` deduplication suffixes.
    """
    slugs: set[str] = set()
    counts: dict[str, int] = {}
    content = md_path.read_text(encoding="utf-8")
    no_fenced = re.sub(
        r"(?m)^```[^\n]*\n.*?\n^```[^\n]*$",
        "",
        content,
        flags=re.DOTALL,
    )
    for line in no_fenced.splitlines():
        m = re.match(r"^(#{1,6})\s+(.*)$", line.strip())
        if not m:
            continue
        heading = m.group(2).strip()
        heading = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", heading)
        heading = re.sub(r"`([^`]*)`", r"\1", heading)
        heading = re.sub(r"\$([^\$]*)\$", r"\1", heading)
        heading = heading.lower()
        base_slug = re.sub(r"[^\w\s-]", "", heading)
        base_slug = re.sub(r"\s", "-", base_slug)
        occ = counts.get(base_slug, 0)
        counts[base_slug] = occ + 1
        slug = base_slug if occ == 0 else f"{base_slug}-{occ}"
        slugs.add(slug)
    return slugs


def test_relative_markdown_links_resolve() -> None:
    """Verifies (`G-08`) that all relative Markdown file links and `#section-anchor` fragments
    across `README.md`, `AGENTS.md`, `CONTRIBUTING.md`, `CLA.md`, `docs/**/*.md`, `tests/**/*.md`,
    and `packages/**/*.md` resolve to an existing file/directory and heading slug.
    """
    md_files = _all_repository_markdown_files()
    slug_cache: dict[Path, set[str]] = {}

    for md_path in md_files:
        content = md_path.read_text(encoding="utf-8")
        no_fenced = re.sub(
            r"(?m)^```[^\n]*\n.*?\n^```[^\n]*$",
            lambda m: "\n" * m.group(0).count("\n"),
            content,
            flags=re.DOTALL,
        )
        for m in re.finditer(r"\[[^\]]+\]\(([^)]+)\)", no_fenced):
            raw_target = m.group(1).strip()
            if (
                raw_target.startswith(("http://", "https://", "mailto:", "file://"))
                or not raw_target
            ):
                continue
            parts = raw_target.split("#", 1)
            path_part = parts[0]
            anchor_part = parts[1] if len(parts) > 1 else None
            resolved = (
                md_path.resolve() if not path_part else (md_path.parent / path_part).resolve()
            )
            line_no = content[: m.start()].count("\n") + 1
            assert resolved.exists(), (
                f"Broken relative Markdown link '{raw_target}' at {md_path}:{line_no} "
                f"(resolved to {resolved})"
            )
            if anchor_part and resolved.is_file() and resolved.suffix == ".md":
                if resolved not in slug_cache:
                    slug_cache[resolved] = _github_heading_slugs(resolved)
                assert anchor_part in slug_cache[resolved], (
                    f"Broken Markdown heading anchor '#{anchor_part}' in link '{raw_target}' "
                    f"at {md_path}:{line_no} (target file: {resolved})"
                )
