"""Monorepo-wide documentation, specification, and progressive-disclosure guardrail tests
(`verify[repo.docs.agents-md-budgets]`, `verify[repo.docs.requirements-layer]`,
`verify[repo.docs.markdown-table-math]`, `verify[repo.docs.github-latex-and-mermaid]`,
`verify[repo.docs.relative-links-and-anchors]`, `verify[repo.docs.spec-rules-and-adrs]`).
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
    """verify[repo.docs.agents-md-budgets] Enforces `< 5,000 B` on root `AGENTS.md` and
    `< 2,500 B` on scoped `AGENTS.md`.
    """
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


def test_requirement_registry_and_agnostic_prose() -> None:
    """verify[repo.docs.requirements-layer] Verifies that:
    1. `docs/CONTRIBUTING.md` exists and documents specification authoring standards.
    2. `docs/requirements/system_goals_and_requirements.md` documents all 40 requirements
       (`REQ-1.1`..`REQ-8.8`, including complex full-rig modeling `REQ-1.5`, MIDI/host tempo sync
       & rhythmic subdivision `REQ-2.5`, single-run slimmable tier training `REQ-5.1`..`REQ-5.6`,
       mono/true-stereo channel topologies `REQ-7.4`, and model/marketplace security
       `REQ-8.1`..`REQ-8.8`) with zero implementation-specific tokens.
    """
    docs_contributing = MONOREPO_ROOT / "docs" / "CONTRIBUTING.md"
    assert docs_contributing.is_file(), f"Missing documentation guide: {docs_contributing}"

    goals_doc = MONOREPO_ROOT / "docs" / "requirements" / "system_goals_and_requirements.md"
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
    """verify[repo.docs.markdown-table-math] Verifies that no Markdown table row contains inline
    LaTeX math (`$...$` outside backticks), preventing GitHub MathJax horizontal scrollbars.
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
    """verify[repo.docs.github-latex-and-mermaid] Verifies that no `$...$` or `$$...$$` math block
    contains CommonMark-stripped punctuation escapes (`\\_`, `\\;`, `\\,`, `\\!`, `\\{`, `\\}`,
    `\\|`, `\\%`, `\\\\`), disallowed GitHub macros (`\\operatorname`, `\\multicolumn`, `\\texttt`),
    or underscores inside `\\text{...}`.
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
    """verify[repo.docs.relative-links-and-anchors] Verifies that all relative Markdown file links
    and `#section-anchor` fragments across `README.md`, `AGENTS.md`, `CONTRIBUTING.md`, `CLA.md`,
    `docs/**/*.md`, `tests/**/*.md`, and `packages/**/*.md` resolve on disk.
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
        no_code = re.sub(r"`[^`\n]*`", lambda m: " " * len(m.group(0)), no_fenced)
        for m in re.finditer(r"\[[^\]]+\]\(([^)]+)\)", no_code):
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


def _parse_frontmatter(md_path: Path) -> tuple[dict[str, str | list[str]], str]:
    """Parses YAML frontmatter (`key: value` and `- item` lists) at the start of `md_path`."""
    content = md_path.read_text(encoding="utf-8")
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", content, flags=re.DOTALL)
    assert m is not None, f"Missing YAML frontmatter ('--- ... ---') at top of {md_path}"
    raw_yaml, body = m.group(1), m.group(2)
    meta: dict[str, str | list[str]] = {}
    current_list_key: str | None = None
    for line in raw_yaml.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("- "):
            assert current_list_key is not None, f"Orphan list item in frontmatter of {md_path}"
            item = stripped[2:].strip().strip("\"'")
            val = meta[current_list_key]
            assert isinstance(val, list)
            val.append(item)
            continue
        assert ":" in stripped, f"Invalid frontmatter line '{line}' in {md_path}"
        key, raw_val = stripped.split(":", 1)
        key = key.strip()
        raw_val = raw_val.strip()
        if not raw_val or raw_val == "[]":
            meta[key] = []
            current_list_key = key if not raw_val else None
        else:
            meta[key] = raw_val.strip("\"'")
            current_list_key = None
    return meta, body


def test_spec_rule_identifiers_adr_structure_and_no_spec_text_parsing() -> None:
    """verify[repo.docs.spec-rules-and-adrs] Verifies that:
    1. Canonical templates exist under `docs/templates/` (`requirement_template.md`,
       `spec_template.md`, `adr_template.md`) with valid YAML frontmatter.
    2. All files in `docs/adr/` follow `NNNN-kebab-case.md` MADR naming and YAML frontmatter,
       and no normative spec (`docs/specs/`) or requirement (`docs/requirements/`) references
       a superseded ADR (`status: superseded`).
    3. Every normative spec file under `docs/specs/**/*.md` has YAML frontmatter, defines
       globally unique `r[<domain>.<subdomain>.<rule>]` identifiers matching its declared
       `rule_prefixes`, and covers every leaf normative section with at least one `r[...]` rule.
    4. No test file outside `tests/test_guardrails.py` reads or parses Markdown files from `docs/`.
    """
    docs_dir = MONOREPO_ROOT / "docs"
    assert not (docs_dir / "verification").exists(), (
        "docs/verification/ has been superseded; rules belong in their respective docs/specs/ files"
    )
    templates_dir = docs_dir / "templates"
    for tmpl_name in ("requirement_template.md", "spec_template.md", "adr_template.md"):
        tmpl_path = templates_dir / tmpl_name
        assert tmpl_path.is_file(), f"Missing canonical template: {tmpl_path}"
        tmpl_meta, _ = _parse_frontmatter(tmpl_path)
        assert {"id", "title", "status"} <= set(tmpl_meta)

    req_doc = docs_dir / "requirements" / "system_goals_and_requirements.md"
    req_meta, _ = _parse_frontmatter(req_doc)
    assert req_meta.get("status") == "canonical"
    assert req_meta.get("layer") == "requirements"

    adr_dir = docs_dir / "adr"
    adr_files = sorted(adr_dir.glob("*.md"))
    assert len(adr_files) >= 10
    active_adrs: set[str] = set()
    superseded_adr_ids: set[str] = set()
    superseded_adr_filenames: set[str] = set()
    superseded_targets: list[tuple[Path, str]] = []

    for adr_path in adr_files:
        assert re.fullmatch(r"\d{4}-[a-z0-9-]+\.md", adr_path.name), (
            f"ADR filename '{adr_path.name}' must follow MADR 'NNNN-kebab-case.md'"
        )
        adr_meta, _ = _parse_frontmatter(adr_path)
        adr_id = adr_meta.get("id")
        status = adr_meta.get("status")
        assert isinstance(adr_id, str), f"Missing string id in {adr_path}"
        assert adr_id == f"ADR-{adr_path.name[:4]}", (
            f"ADR frontmatter id '{adr_id}' does not match filename prefix in {adr_path}"
        )
        assert status in {"proposed", "accepted", "superseded"}, (
            f"Invalid ADR status '{status}' in {adr_path}"
        )
        if status == "superseded":
            superseded_by = adr_meta.get("superseded_by")
            assert isinstance(superseded_by, str), (
                f"Superseded ADR {adr_path} must declare 'superseded_by: ADR-NNNN'"
            )
            assert superseded_by.startswith("ADR-"), (
                f"Superseded ADR {adr_path} must declare 'superseded_by: ADR-NNNN'"
            )
            superseded_adr_ids.add(adr_id)
            superseded_adr_filenames.add(adr_path.name)
            superseded_targets.append((adr_path, superseded_by))
        else:
            active_adrs.add(adr_id)

    for adr_path, target_id in superseded_targets:
        assert target_id in active_adrs, (
            f"Superseded ADR {adr_path} points to non-active ADR '{target_id}'"
        )

    spec_files = sorted((docs_dir / "specs").rglob("*.md"))
    assert len(spec_files) == 16

    rule_def_pattern = re.compile(r"`r\[([a-z0-9]+(?:\.[a-z0-9_-]+){2,})\]`")
    rule_ref_pattern = re.compile(r"\bref\[([a-z0-9]+(?:\.[a-z0-9_-]+){2,})\]")
    rule_verify_pattern = re.compile(r"\bverify\[([a-z0-9]+(?:\.[a-z0-9_-]+){2,})\]")
    seen_rules: dict[str, Path] = {}

    for doc_path in [req_doc, *spec_files, *adr_files]:
        content = doc_path.read_text(encoding="utf-8")
        if doc_path.name != "project_and_documentation_quality.md":
            test_links = re.findall(
                r"\[[^\]]*\]\([^\)]*(?:test_|_spec_references|_verification)[^\)]*\)",
                content,
            )
            assert not test_links, (
                f"Document {doc_path} must not hardcode links to test files {test_links}; "
                f"tests reference specs via verify[...]"
            )
        if doc_path not in adr_files:
            assert re.search(r"\bCL-\d{2}\b", content) is None, (
                f"Legacy CL-XX identifier found in {doc_path}; use r[...]/ref[...] rules instead"
            )
            for sup_id in superseded_adr_ids:
                assert sup_id not in content, (
                    f"Normative document {doc_path} must not reference superseded ADR '{sup_id}'"
                )
            for sup_file in superseded_adr_filenames:
                assert sup_file not in content, (
                    f"Normative document {doc_path} must not link to superseded ADR '{sup_file}'"
                )

    for spec_path in spec_files:
        spec_meta, spec_body = _parse_frontmatter(spec_path)
        assert "clamps" not in spec_meta, (
            f"Legacy 'clamps' frontmatter property found in {spec_path}"
        )
        assert spec_meta.get("status") == "canonical", f"Invalid status in {spec_path}"
        prefixes = spec_meta.get("rule_prefixes")
        assert isinstance(prefixes, list), (
            f"Missing 'rule_prefixes' list in frontmatter of {spec_path}"
        )
        assert len(prefixes) > 0, f"Empty 'rule_prefixes' list in {spec_path}"
        adrs_field = spec_meta.get("adrs")
        assert isinstance(adrs_field, list), (
            f"Spec {spec_path} must declare justifying 'adrs' list in frontmatter"
        )
        assert len(adrs_field) > 0, f"Empty 'adrs' list in {spec_path}"
        for cited_adr in adrs_field:
            assert cited_adr in active_adrs, (
                f"Spec {spec_path} cites unknown or superseded ADR '{cited_adr}'"
            )

        no_fenced = re.sub(
            r"(?m)^```[^\n]*\n.*?\n^```[^\n]*$",
            "",
            spec_body,
            flags=re.DOTALL,
        )
        rules_in_file = [f"r[{m}]" for m in rule_def_pattern.findall(no_fenced)]
        assert rules_in_file, f"No r[...] specification rules found in {spec_path}"
        for rule_id in rules_in_file:
            assert any(rule_id.startswith(pref) for pref in prefixes), (
                f"Rule '{rule_id}' in {spec_path} does not match declared prefixes {prefixes}"
            )
            assert rule_id not in seen_rules, (
                f"Duplicate specification rule '{rule_id}' in {spec_path} "
                f"(already defined in {seen_rules[rule_id]})"
            )
            seen_rules[rule_id] = spec_path

        sections = re.split(r"(?m)^(#{2,4}\s+.*)$", no_fenced)
        for idx in range(1, len(sections), 2):
            heading_line = sections[idx].strip()
            section_text = sections[idx + 1]
            level = len(heading_line.split(" ", 1)[0])
            next_level = (
                len(sections[idx + 2].strip().split(" ", 1)[0]) if idx + 2 < len(sections) else 0
            )
            is_container_heading = next_level > level and not section_text.strip()
            if not is_container_heading:
                assert rule_def_pattern.search(section_text) is not None, (
                    f"Normative section '{heading_line}' in {spec_path} is missing a "
                    f"canonical `r[...]` spec reference"
                )

    allowed_def_files = {*spec_files, templates_dir / "spec_template.md"}
    template_rules = {
        f"r[{m}]"
        for m in rule_def_pattern.findall(
            (templates_dir / "spec_template.md").read_text(encoding="utf-8")
        )
    }
    valid_target_rules = set(seen_rules) | template_rules

    all_checked_files = [
        *_all_repository_markdown_files(),
        *list(TESTS_DIR.glob("*.py")),
        *[
            p
            for p in (MONOREPO_ROOT / "packages").rglob("*")
            if p.is_file()
            and p.suffix in {".rs", ".py", ".ts", ".tsx"}
            and "node_modules" not in p.parts
            and "target" not in p.parts
            and "dist" not in p.parts
        ],
    ]
    for path in all_checked_files:
        text = path.read_text(encoding="utf-8")
        if path.name != "test_guardrails.py":
            assert re.search(r"\bG-0\d\b", text) is None, (
                f"Legacy G-0X guardrail identifier found in {path}; use ref[...] instead"
            )
        if path not in allowed_def_files and path.name != "test_guardrails.py":
            stray_defs = [f"r[{m}]" for m in rule_def_pattern.findall(text)]
            assert not stray_defs, (
                f"Rule definition(s) {stray_defs} found outside docs/specs/ in {path}; "
                f"use ref[...] or verify[...] to reference an existing rule"
            )
        ver_matches = rule_verify_pattern.findall(text)
        if path.suffix == ".md":
            assert not ver_matches, (
                f"verify[...] citation(s) {ver_matches} found in Markdown file {path}; "
                f"use ref[...] in docs and verify[...] only in test source files"
            )
        if path.name not in {"CONTRIBUTING.md", "test_guardrails.py"}:
            for ref_inner in rule_ref_pattern.findall(text):
                target_rule = f"r[{ref_inner}]"
                assert target_rule in valid_target_rules, (
                    f"Unresolved rule reference 'ref[{ref_inner}]' in {path} "
                    f"(no matching '{target_rule}' definition in docs/specs/)"
                )
            for ver_inner in ver_matches:
                target_rule = f"r[{ver_inner}]"
                assert target_rule in valid_target_rules, (
                    f"Unresolved verification reference 'verify[{ver_inner}]' in {path} "
                    f"(no matching '{target_rule}' definition in docs/specs/)"
                )

    test_files = [
        *[p for p in TESTS_DIR.glob("test_*.py") if p.name != "test_guardrails.py"],
        *list((MONOREPO_ROOT / "packages").glob("*/tests/**/*.py")),
        *list((MONOREPO_ROOT / "packages").glob("*/tests/**/*.rs")),
    ]
    for test_path in test_files:
        test_src = test_path.read_text(encoding="utf-8")
        assert 'parents[1] / "docs/' not in test_src, (
            f"Test file {test_path} must not read Markdown files from docs/"
        )
        assert 'split("```text' not in test_src, (
            f"Test file {test_path} must not parse Markdown text blocks from docs/"
        )


def test_spec_rules_verified_in_tests() -> None:
    """verify[repo.docs.spec-rules-and-adrs] Report spec r[...] rules missing verify[...]."""
    spec_files = sorted((MONOREPO_ROOT / "docs" / "specs").rglob("*.md"))
    rule_def_pattern = re.compile(r"`r\[([a-z0-9]+(?:\.[a-z0-9_-]+){2,})\]`")
    rule_verify_pattern = re.compile(r"\bverify\[([a-z0-9]+(?:\.[a-z0-9_-]+){2,})\]")

    defined_rules: dict[str, Path] = {}
    for spec_path in spec_files:
        _, spec_body = _parse_frontmatter(spec_path)
        no_fenced = re.sub(
            r"(?m)^```[^\n]*\n.*?\n^```[^\n]*$",
            "",
            spec_body,
            flags=re.DOTALL,
        )
        for match in rule_def_pattern.findall(no_fenced):
            defined_rules[f"r[{match}]"] = spec_path

    test_files = [
        *list(TESTS_DIR.glob("test_*.py")),
        *list((MONOREPO_ROOT / "packages").glob("*/tests/**/*.py")),
        *list((MONOREPO_ROOT / "packages").glob("*/tests/**/*.rs")),
        *list((MONOREPO_ROOT / "packages").glob("*/src/**/*.test.ts")),
        *list((MONOREPO_ROOT / "packages").glob("*/src/**/*.test.tsx")),
    ]
    verified_in_tests: set[str] = set()
    for test_path in test_files:
        text = test_path.read_text(encoding="utf-8")
        for ver_inner in rule_verify_pattern.findall(text):
            verified_in_tests.add(f"r[{ver_inner}]")

    missing = sorted(set(defined_rules) - verified_in_tests)
    missing_report = "\n".join(
        f"  - {rule_id} ({defined_rules[rule_id].relative_to(MONOREPO_ROOT)})"
        for rule_id in missing
    )
    assert not missing, (
        f"{len(missing)}/{len(defined_rules)} specification rules in docs/specs/ "
        f"are missing a verify[...] citation in test files:\n{missing_report}"
    )
