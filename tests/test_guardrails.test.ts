/**
 * Repository-wide architectural, documentation, and specification guardrail tests.
 *
 * Enforces the invariants defined in:
 * - docs/specs/architecture/project_and_documentation_quality.md
 * - docs/CONTRIBUTING.md
 * - CONTRIBUTING.md
 * - AGENTS.md
 *
 * verify[repo.code.type-safety-and-imports]
 * verify[repo.docs.agents-md-budgets]
 * verify[repo.docs.requirements-layer]
 * verify[repo.docs.spec-rules-and-adrs]
 * verify[repo.docs.markdown-table-math]
 * verify[repo.docs.github-latex-and-mermaid]
 * verify[repo.docs.relative-links-and-anchors]
 * verify[arch.index.subsystems]
 */

import * as fs from "node:fs";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

const THIS_DIR = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(THIS_DIR, "..");
const DOCS_DIR = path.join(REPO_ROOT, "docs");
const SPECS_DIR = path.join(DOCS_DIR, "specs");
const ADR_DIR = path.join(DOCS_DIR, "adr");
const REQ_FILE = path.join(DOCS_DIR, "requirements", "system_goals_and_requirements.md");
const PACKAGES_DIR = path.join(REPO_ROOT, "packages");
const TESTS_DIR = path.join(REPO_ROOT, "tests");

const EXCLUDED_DIRS = new Set([
  ".git",
  ".hg",
  ".svn",
  ".venv",
  "venv",
  "node_modules",
  "target",
  "dist",
  "build",
  "coverage",
  ".moon",
  ".benchmarks",
  ".vitest",
  ".vitepress",
  "cosmos-export",
  "playwright-report",
  "test-results",
  "scratch",
]);

const FRONTMATTER_RE = /^---\n([\s\S]*?)\n---\n/;
const FENCED_BLOCK_RE = /^(`{3,}|~{3,})[^\n]*\n[\s\S]*?^\1[ \t]*$/gm;
const INLINE_CODE_RE = /`+[^`\n]+`+/g;

const SPEC_RULE_DEF_RE =
  /`r\[([a-z0-9-]+(?:\.[a-z0-9-]+){2,})\]`|^[ \t]*r\[([a-z0-9-]+(?:\.[a-z0-9-]+){2,})\][ \t]*$/gm;
const SPEC_REF_TOKEN_RE = /\bref\[([a-z0-9-]+(?:\.[a-z0-9-]+){2,})\]/g;
const SPEC_VERIFY_TOKEN_RE = /\bverify\[([a-z0-9-]+(?:\.[a-z0-9-]+){2,})\]/g;
const MALFORMED_SPEC_MARKER_RE = /(?:^|[^a-zA-Z0-9_])(r|ref|verify)\[([^\]]+)\]/gm;
const VALID_RULE_ID_RE = /^[a-z0-9-]+(?:\.[a-z0-9-]+){2,}$/;

const REQ_TABLE_RE = /^\|\s*\*\*`(REQ-\d+\.\d+)`\*\*\s*\|/gm;
const REQ_TOKEN_RE = /\bREQ-\d+\.\d+\b/g;
const ADR_TOKEN_RE = /\bADR-\d{4}\b/g;

const MD_LINK_RE = /(?<!!)\[([^\]]+)\]\(([^)\s]+)(?:\s+"[^"]*")?\)/g;
const HEADING_RE = /^(#{1,6})\s+(.+?)\s*$/gm;
const DISPLAY_MATH_RE = /\$\$([\s\S]*?)\$\$/g;
const INLINE_MATH_RE = /(?<!\\)\$(?!\$)([^\n$]+?)(?<!\\)\$(?!\$)/g;
const BACKSLASH_PUNCT_ESCAPE_RE = /\\[^a-zA-Z0-9\s]/;
const FORBIDDEN_MATH_MACROS = ["\\operatorname", "\\multicolumn", "\\texttt", "\\text{--"] as const;
const TEXT_UNDERSCORE_RE = /\\(?:text|mathrm)\{[^}]*_[^}]*\}/;
const LEGACY_ID_RE = /\bCL-\d{2}\b/;

function walkMarkdownFiles(root: string): string[] {
  const results: string[] = [];
  function recurse(currentDir: string): void {
    const entries = fs.readdirSync(currentDir, { withFileTypes: true });
    entries.sort((a, b) => a.name.localeCompare(b.name));
    for (const entry of entries) {
      if (EXCLUDED_DIRS.has(entry.name)) {
        continue;
      }
      const fullPath = path.join(currentDir, entry.name);
      if (entry.isDirectory()) {
        recurse(fullPath);
      } else if (entry.isFile() && entry.name.endsWith(".md")) {
        results.push(fullPath);
      }
    }
  }
  recurse(root);
  return results.sort();
}

function walkFilesByExtensions(root: string, extensions: Set<string>): string[] {
  const results: string[] = [];
  if (!fs.existsSync(root)) {
    return results;
  }
  function recurse(currentDir: string): void {
    const entries = fs.readdirSync(currentDir, { withFileTypes: true });
    entries.sort((a, b) => a.name.localeCompare(b.name));
    for (const entry of entries) {
      if (EXCLUDED_DIRS.has(entry.name)) {
        continue;
      }
      const fullPath = path.join(currentDir, entry.name);
      if (entry.isDirectory()) {
        recurse(fullPath);
      } else if (entry.isFile()) {
        const ext = path.extname(entry.name);
        if (extensions.has(ext)) {
          results.push(fullPath);
        }
      }
    }
  }
  recurse(root);
  return results.sort();
}

function relPath(filePath: string): string {
  return path.relative(REPO_ROOT, filePath);
}

function stripFencedCodeBlocks(text: string): string {
  return text.replace(FENCED_BLOCK_RE, (match) => "\n".repeat((match.match(/\n/g) ?? []).length));
}

function stripCode(text: string): string {
  const noFences = stripFencedCodeBlocks(text);
  return noFences.replace(INLINE_CODE_RE, " ");
}

function parseFrontmatter(text: string): Record<string, string | string[]> | null {
  const match = FRONTMATTER_RE.exec(text);
  if (!match || match[1] === undefined) {
    return null;
  }
  const raw = match[1];
  const data: Record<string, string | string[]> = {};
  let currentListKey: string | null = null;
  for (const line of raw.split("\n")) {
    const stripped = line.trim();
    if (!stripped || stripped.startsWith("#")) {
      continue;
    }
    if (stripped.startsWith("- ") && currentListKey !== null) {
      const val = stripped
        .slice(2)
        .trim()
        .replace(/^['"]|['"]$/g, "");
      const existing = data[currentListKey];
      if (Array.isArray(existing)) {
        existing.push(val);
      }
    } else if (stripped.includes(":")) {
      const colonIdx = stripped.indexOf(":");
      const key = stripped.slice(0, colonIdx).trim();
      const val = stripped
        .slice(colonIdx + 1)
        .trim()
        .replace(/^['"]|['"]$/g, "");
      if (val === "" || val === "[]") {
        data[key] = [];
        currentListKey = val === "" ? key : null;
      } else {
        data[key] = val;
        currentListKey = null;
      }
    }
  }
  return data;
}

function githubSlugify(headingText: string): string {
  let text = headingText.trim();
  text = text.replace(/\[([^\]]+)\]\([^)]+\)/g, "$1");
  text = text.replace(/`/g, "");
  text = text.toLowerCase();
  const kept: string[] = [];
  for (const ch of text) {
    if (/[\p{L}\p{N}_-]/u.test(ch)) {
      kept.push(ch);
    } else if (ch === " " || ch === "\t") {
      kept.push(" ");
    }
  }
  const joined = kept.join("");
  return joined.replace(/[ \t]/g, "-");
}

function extractHeadingAnchors(mdText: string): Set<string> {
  const clean = stripFencedCodeBlocks(mdText);
  const anchors = new Set<string>();
  const counts = new Map<string, number>();
  for (const match of clean.matchAll(HEADING_RE)) {
    const headingBody = match[2] ?? "";
    const base = githubSlugify(headingBody);
    if (!base) {
      continue;
    }
    if (!counts.has(base)) {
      counts.set(base, 0);
      anchors.add(base);
    } else {
      const nextCount = (counts.get(base) ?? 0) + 1;
      counts.set(base, nextCount);
      anchors.add(`${base}-${nextCount}`);
    }
  }
  return anchors;
}

function collectSpecRules(): Map<string, [string, number][]> {
  const rules = new Map<string, [string, number][]>();
  for (const specPath of walkMarkdownFiles(SPECS_DIR)) {
    const text = fs.readFileSync(specPath, "utf8");
    const clean = stripFencedCodeBlocks(text);
    const lines = clean.split("\n");
    for (let idx = 0; idx < lines.length; idx++) {
      const line = lines[idx] ?? "";
      for (const m of line.matchAll(SPEC_RULE_DEF_RE)) {
        const ruleId = m[1] ?? m[2] ?? "";
        const list = rules.get(ruleId) ?? [];
        list.push([relPath(specPath), idx + 1]);
        rules.set(ruleId, list);
      }
    }
  }
  return rules;
}

function collectSourceRefTokens(): [string, string, number][] {
  const refs: [string, string, number][] = [];
  const exts = new Set([".rs", ".ts", ".tsx", ".mjs", ".md"]);
  const templateRel = relPath(path.join(DOCS_DIR, "templates"));
  for (const root of [PACKAGES_DIR, DOCS_DIR, TESTS_DIR]) {
    for (const filePath of walkFilesByExtensions(root, exts)) {
      const rel = relPath(filePath);
      if (path.basename(filePath) === "test_guardrails.test.ts" || rel.startsWith(templateRel)) {
        continue;
      }
      const text = fs.readFileSync(filePath, "utf8");
      const lines = text.split("\n");
      for (let idx = 0; idx < lines.length; idx++) {
        const line = lines[idx] ?? "";
        for (const m of line.matchAll(SPEC_REF_TOKEN_RE)) {
          refs.push([m[1] ?? "", relPath(filePath), idx + 1]);
        }
      }
    }
  }
  return refs;
}

function collectTestVerifyTokens(): [string, string, number][] {
  const verifies: [string, string, number][] = [];
  const exts = new Set([".rs", ".ts", ".tsx"]);
  for (const root of [TESTS_DIR, PACKAGES_DIR]) {
    for (const filePath of walkFilesByExtensions(root, exts)) {
      const text = fs.readFileSync(filePath, "utf8");
      const lines = text.split("\n");
      for (let idx = 0; idx < lines.length; idx++) {
        const line = lines[idx] ?? "";
        for (const m of line.matchAll(SPEC_VERIFY_TOKEN_RE)) {
          verifies.push([m[1] ?? "", relPath(filePath), idx + 1]);
        }
      }
    }
  }
  return verifies;
}

interface SpecScopeRuleEntry {
  scope: string;
  reason?: string;
}

function _parseSpecScopeToml(text: string): Map<string, SpecScopeRuleEntry> {
  const rules = new Map<string, SpecScopeRuleEntry>();
  let currentRuleId: string | null = null;
  let currentEntry: SpecScopeRuleEntry | null = null;

  for (const rawLine of text.split("\n")) {
    const line = rawLine.trim();
    if (!line || line.startsWith("#")) {
      continue;
    }
    const headerMatch = /^\[rules\."([^"]+)"\]$/.exec(line);
    if (headerMatch && headerMatch[1] !== undefined) {
      if (currentRuleId !== null && currentEntry !== null) {
        rules.set(currentRuleId, currentEntry);
      }
      currentRuleId = headerMatch[1];
      currentEntry = { scope: "" };
      continue;
    }
    if (currentEntry === null) {
      continue;
    }
    const scopeMatch = /^scope\s*=\s*"([^"]+)"/.exec(line);
    if (scopeMatch && scopeMatch[1] !== undefined) {
      currentEntry.scope = scopeMatch[1];
      continue;
    }
    const reasonMatch = /^reason\s*=\s*"([^"]+)"/.exec(line);
    if (reasonMatch && reasonMatch[1] !== undefined) {
      currentEntry.reason = reasonMatch[1];
    }
  }
  if (currentRuleId !== null && currentEntry !== null) {
    rules.set(currentRuleId, currentEntry);
  }
  return rules;
}

function normalizeSpecRelPath(rawSpecRef: string): string {
  const trimmed = rawSpecRef.trim();
  if (trimmed.startsWith("docs/")) {
    return trimmed.slice("docs/".length);
  }
  if (trimmed.startsWith("../")) {
    return trimmed.slice("../".length);
  }
  return trimmed;
}

describe("DNAM Monorepo Architectural & Documentation Guardrails", () => {
  it("test_three_layer_docs_directory_structure", () => {
    expect(fs.existsSync(path.join(DOCS_DIR, "index.md"))).toBe(true);
    expect(fs.existsSync(path.join(DOCS_DIR, "CONTRIBUTING.md"))).toBe(true);
    expect(fs.existsSync(REQ_FILE)).toBe(true);
    expect(fs.existsSync(ADR_DIR) && fs.statSync(ADR_DIR).isDirectory()).toBe(true);
    expect(fs.existsSync(SPECS_DIR) && fs.statSync(SPECS_DIR).isDirectory()).toBe(true);
    expect(fs.existsSync(path.join(DOCS_DIR, "verification"))).toBe(false);

    const templatesDir = path.join(DOCS_DIR, "templates");
    expect(fs.existsSync(templatesDir) && fs.statSync(templatesDir).isDirectory()).toBe(true);
    for (const tplName of ["requirement_template.md", "spec_template.md", "adr_template.md"]) {
      const tplPath = path.join(templatesDir, tplName);
      expect(fs.existsSync(tplPath), `Missing docs/templates/${tplName}`).toBe(true);
      expect(
        parseFrontmatter(fs.readFileSync(tplPath, "utf8")),
        `Template ${tplName} is missing YAML frontmatter`,
      ).not.toBeNull();
    }

    const expectedDomains = new Set([
      "architecture",
      "dsp",
      "containers",
      "training",
      "security",
      "sdks",
    ]);
    const actualDomains = new Set(
      fs
        .readdirSync(SPECS_DIR, { withFileTypes: true })
        .filter((d) => d.isDirectory())
        .map((d) => d.name),
    );
    expect(actualDomains).toEqual(expectedDomains);
  });

  it("test_all_spec_files_have_valid_yaml_frontmatter", () => {
    const requiredKeys = [
      "id",
      "title",
      "status",
      "version",
      "domain",
      "rule_prefixes",
      "requirements",
      "adrs",
    ];
    const specFiles = walkMarkdownFiles(SPECS_DIR);
    expect(specFiles.length).toBe(17);

    const seenSpecIds = new Map<string, string>();
    for (const specPath of specFiles) {
      const text = fs.readFileSync(specPath, "utf8");
      const fm = parseFrontmatter(text);
      expect(fm, `${relPath(specPath)} is missing YAML frontmatter`).not.toBeNull();
      if (!fm) {
        continue;
      }
      expect("clamps" in fm, `${relPath(specPath)} has legacy 'clamps' frontmatter key`).toBe(
        false,
      );
      for (const key of requiredKeys) {
        expect(
          key in fm && fm[key] !== "" && (Array.isArray(fm[key]) ? fm[key].length > 0 : true),
          `${relPath(specPath)} frontmatter is missing non-empty '${key}'`,
        ).toBe(true);
      }
      const specId = String(fm["id"]);
      expect(specId.startsWith("spec.")).toBe(true);
      expect(seenSpecIds.has(specId)).toBe(false);
      seenSpecIds.set(specId, relPath(specPath));
      expect(fm["status"]).toBe("canonical");
      expect(Array.isArray(fm["rule_prefixes"])).toBe(true);
      expect(Array.isArray(fm["requirements"])).toBe(true);
      expect(Array.isArray(fm["adrs"])).toBe(true);
    }
  });

  it("test_all_adr_files_have_valid_yaml_frontmatter", () => {
    const requiredKeys = [
      "id",
      "title",
      "status",
      "supersedes",
      "superseded_by",
      "specs",
      "requirements",
    ];
    const adrFiles = fs
      .readdirSync(ADR_DIR)
      .filter((n) => n.endsWith(".md") && n !== "README.md")
      .sort()
      .map((n) => path.join(ADR_DIR, n));
    expect(adrFiles.length).toBeGreaterThanOrEqual(18);

    const seenAdrIds = new Map<string, string>();
    const adrStatuses = new Map<string, string>();
    const adrSupersededBy = new Map<string, string>();
    const adrFilenames = new Map<string, string>();
    for (const adrPath of adrFiles) {
      const text = fs.readFileSync(adrPath, "utf8");
      const fm = parseFrontmatter(text);
      expect(fm, `${relPath(adrPath)} is missing YAML frontmatter`).not.toBeNull();
      if (!fm) {
        continue;
      }
      for (const key of requiredKeys) {
        expect(key in fm, `${relPath(adrPath)} frontmatter is missing '${key}'`).toBe(true);
      }
      const adrId = String(fm["id"]);
      expect(adrId).toMatch(/^ADR-\d{4}$/);
      expect(path.basename(adrPath).startsWith(adrId.slice(4))).toBe(true);
      expect(seenAdrIds.has(adrId)).toBe(false);
      seenAdrIds.set(adrId, relPath(adrPath));
      adrFilenames.set(adrId, path.basename(adrPath));
      const status = String(fm["status"]);
      adrStatuses.set(adrId, status);
      expect(["proposed", "accepted", "superseded"].includes(status)).toBe(true);
      expect(Array.isArray(fm["specs"])).toBe(true);
      expect(Array.isArray(fm["requirements"])).toBe(true);
      if (status === "accepted") {
        expect((fm["specs"] as string[]).length).toBeGreaterThan(0);
      }
      if (status === "superseded") {
        const supBy = String(fm["superseded_by"]);
        expect(supBy).toMatch(/^ADR-\d{4}$/);
        adrSupersededBy.set(adrId, supBy);
      }
    }

    for (const [adrId, supBy] of adrSupersededBy.entries()) {
      expect(adrStatuses.has(supBy), `${adrId} superseded_by '${supBy}' does not exist`).toBe(true);
      expect(
        adrStatuses.get(supBy) !== "superseded",
        `${adrId} superseded_by '${supBy}' points to another superseded ADR`,
      ).toBe(true);
    }

    const supersededIds = Array.from(adrStatuses.entries())
      .filter(([, st]) => st === "superseded")
      .map(([aid]) => aid);
    for (const docFile of [REQ_FILE, ...walkMarkdownFiles(SPECS_DIR)]) {
      const rawText = fs.readFileSync(docFile, "utf8");
      const cleanBody = stripCode(rawText);
      const fm = parseFrontmatter(rawText);
      const fmAdrs = Array.isArray(fm?.["adrs"]) ? fm["adrs"] : [];
      for (const supId of supersededIds) {
        const supFilename = adrFilenames.get(supId) ?? "";
        expect(
          fmAdrs.includes(supId),
          `${relPath(docFile)} frontmatter cites superseded ${supId}`,
        ).toBe(false);
        expect(
          cleanBody.includes(supId) || (supFilename !== "" && rawText.includes(supFilename)),
          `${relPath(docFile)} cites superseded ${supId} (${supFilename})`,
        ).toBe(false);
      }
    }
  });

  it("test_requirement_registry_and_agnostic_prose", () => {
    const text = fs.readFileSync(REQ_FILE, "utf8");
    const fm = parseFrontmatter(text);
    expect(fm).not.toBeNull();
    if (!fm) {
      return;
    }
    expect(fm["id"]).toBe("req.system-goals");
    expect(fm["status"]).toBe("canonical");
    expect(fm["layer"]).toBe("requirements");

    const reqIds = Array.from(text.matchAll(REQ_TABLE_RE), (m) => m[1] ?? "");
    expect(reqIds.length).toBe(40);
    expect(reqIds.length).toBe(new Set(reqIds).size);

    const expectedCounts: Record<number, number> = {
      1: 8,
      2: 5,
      3: 3,
      4: 3,
      5: 6,
      6: 3,
      7: 4,
      8: 8,
    };
    for (const [goalStr, count] of Object.entries(expectedCounts)) {
      const goal = Number(goalStr);
      for (let sub = 1; sub <= count; sub++) {
        expect(reqIds).toContain(`REQ-${goal}.${sub}`);
      }
    }

    for (const specPath of walkMarkdownFiles(SPECS_DIR)) {
      const specFm = parseFrontmatter(fs.readFileSync(specPath, "utf8"));
      const reqList = specFm?.["requirements"];
      if (Array.isArray(reqList)) {
        for (const reqRef of reqList) {
          expect(
            reqIds.includes(reqRef),
            `${relPath(specPath)} frontmatter cites unknown requirement '${reqRef}'`,
          ).toBe(true);
        }
      }
    }

    const clean = stripFencedCodeBlocks(text);
    const proseOnly = clean.replace(/\]\([^)]+\)/g, "]()");
    const forbiddenFieldNames = [
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
      ".dnam",
      ".dnam-ckpt",
      ".dnam-cache",
      ".dnam-seal",
      ".dnce",
      "mono_only",
      "CL-0",
      "CL-1",
      "CL-2",
    ];
    for (const field of forbiddenFieldNames) {
      expect(proseOnly).not.toContain(field);
    }
  });

  it("test_layer_separation_no_adr_narratives_in_canonical_specs", () => {
    const forbiddenHeadings = [
      "rejected alternatives",
      "counterexamples and failure modes",
      "why alternatives lose",
      "candidate comparison",
    ];
    for (const specPath of walkMarkdownFiles(SPECS_DIR)) {
      const clean = stripFencedCodeBlocks(fs.readFileSync(specPath, "utf8"));
      for (const match of clean.matchAll(HEADING_RE)) {
        const headingText = (match[2] ?? "").toLowerCase();
        for (const phrase of forbiddenHeadings) {
          expect(headingText).not.toContain(phrase);
        }
      }
    }
  });

  it("test_spec_rule_identifiers_adr_structure_and_no_spec_text_parsing", () => {
    const rules = collectSpecRules();
    expect(rules.size).toBeGreaterThanOrEqual(150);

    const duplicates: string[] = [];
    for (const [ruleId, locations] of rules.entries()) {
      if (locations.length > 1) {
        const locStr = locations.map(([p, ln]) => `${p}:${ln}`).join(", ");
        duplicates.push(`r[${ruleId}] defined multiple times: ${locStr}`);
      }
    }
    expect(duplicates).toEqual([]);

    const missingLeafRules: string[] = [];
    const sectionHeadingRe = /^(#{2,4})\s+(.+?)\s*$/gm;
    for (const specPath of walkMarkdownFiles(SPECS_DIR)) {
      const text = fs.readFileSync(specPath, "utf8");
      const fm = parseFrontmatter(text);
      const prefixes = Array.isArray(fm?.["rule_prefixes"]) ? fm["rule_prefixes"] : [];
      const clean = stripFencedCodeBlocks(text);
      const fileRules = Array.from(clean.matchAll(SPEC_RULE_DEF_RE), (m) => m[1] ?? m[2] ?? "");
      expect(fileRules.length, `${relPath(specPath)} defines zero r[...] rules`).toBeGreaterThan(0);
      for (const ruleId of fileRules) {
        const fullToken = `r[${ruleId}]`;
        expect(
          prefixes.some((prefix) => fullToken.startsWith(prefix)),
          `${relPath(specPath)}: rule '${fullToken}' does not match any frontmatter rule_prefixes`,
        ).toBe(true);
      }

      const headings = Array.from(clean.matchAll(sectionHeadingRe));
      for (let idx = 0; idx < headings.length; idx++) {
        const h = headings[idx];
        if (!h || h.index === undefined) {
          continue;
        }
        const level = (h[1] ?? "").length;
        const title = (h[2] ?? "").trim();
        const start = h.index + h[0].length;
        const nextH = headings[idx + 1];
        const end = nextH && nextH.index !== undefined ? nextH.index : clean.length;
        const nextLevel = nextH ? (nextH[1] ?? "").length : 0;
        const sectionSlice = clean.slice(start, end);
        const isParentWithImmediateChild =
          idx + 1 < headings.length && nextLevel > level && !SPEC_RULE_DEF_RE.test(sectionSlice);
        SPEC_RULE_DEF_RE.lastIndex = 0;
        if (isParentWithImmediateChild) {
          continue;
        }
        if (!SPEC_RULE_DEF_RE.test(sectionSlice)) {
          missingLeafRules.push(
            `${relPath(specPath)}: leaf section '${"#".repeat(level)} ${title}' has no r[...] rule`,
          );
        }
        SPEC_RULE_DEF_RE.lastIndex = 0;
      }
    }
    expect(missingLeafRules).toEqual([]);

    const strayRules: string[] = [];
    const templateRel = relPath(path.join(DOCS_DIR, "templates"));
    for (const mdPath of walkMarkdownFiles(REPO_ROOT)) {
      const rel = relPath(mdPath);
      if (
        mdPath.startsWith(SPECS_DIR) ||
        rel.startsWith(templateRel) ||
        rel === "docs/CONTRIBUTING.md"
      ) {
        continue;
      }
      const clean = stripFencedCodeBlocks(fs.readFileSync(mdPath, "utf8"));
      const lines = clean.split("\n");
      for (let idx = 0; idx < lines.length; idx++) {
        const line = lines[idx] ?? "";
        for (const m of line.matchAll(SPEC_RULE_DEF_RE)) {
          const rid = m[1] ?? m[2] ?? "";
          strayRules.push(`${rel}:${idx + 1} defines r[${rid}] outside docs/specs/`);
        }
      }
    }
    expect(strayRules).toEqual([]);

    const malformed: string[] = [];
    for (const mdPath of walkMarkdownFiles(DOCS_DIR)) {
      const rel = relPath(mdPath);
      if (rel.startsWith(templateRel) || rel === "docs/CONTRIBUTING.md") {
        continue;
      }
      const clean = stripFencedCodeBlocks(fs.readFileSync(mdPath, "utf8"));
      const lines = clean.split("\n");
      for (let idx = 0; idx < lines.length; idx++) {
        const line = lines[idx] ?? "";
        for (const m of line.matchAll(MALFORMED_SPEC_MARKER_RE)) {
          const kind = m[1] ?? "";
          const inner = m[2] ?? "";
          if (
            (inner.endsWith(".") && kind === "r") ||
            inner.includes("...") ||
            inner.includes("*") ||
            inner.includes("<") ||
            inner.includes(">")
          ) {
            continue;
          }
          if (!VALID_RULE_ID_RE.test(inner)) {
            malformed.push(`${rel}:${idx + 1}: malformed '${kind}[${inner}]'`);
          }
        }
      }
    }
    expect(malformed).toEqual([]);

    const verifyInDocs: string[] = [];
    for (const mdPath of walkMarkdownFiles(REPO_ROOT)) {
      const rel = relPath(mdPath);
      if (
        rel.startsWith(templateRel) ||
        rel === "docs/CONTRIBUTING.md" ||
        rel === "docs/specs/architecture/project_and_documentation_quality.md"
      ) {
        continue;
      }
      const text = fs.readFileSync(mdPath, "utf8");
      const lines = text.split("\n");
      for (let idx = 0; idx < lines.length; idx++) {
        const line = lines[idx] ?? "";
        for (const m of line.matchAll(SPEC_VERIFY_TOKEN_RE)) {
          verifyInDocs.push(`${rel}:${idx + 1}: verify[${m[1] ?? ""}]`);
        }
      }
    }
    expect(verifyInDocs).toEqual([]);

    const legacyHits: string[] = [];
    for (const filePath of [REQ_FILE, ...walkMarkdownFiles(SPECS_DIR)]) {
      const text = fs.readFileSync(filePath, "utf8");
      const lines = text.split("\n");
      for (let idx = 0; idx < lines.length; idx++) {
        const line = lines[idx] ?? "";
        const m = LEGACY_ID_RE.exec(line);
        if (m) {
          legacyHits.push(`${relPath(filePath)}:${idx + 1}: ${m[0]}`);
        }
      }
    }
    const legacyGuardrailRe = /\bG-0\d\b/;
    for (const root of [DOCS_DIR, PACKAGES_DIR, TESTS_DIR]) {
      for (const filePath of walkFilesByExtensions(root, new Set([".md", ".rs", ".ts", ".tsx"]))) {
        if (path.basename(filePath) === "test_guardrails.test.ts") {
          continue;
        }
        const lines = fs.readFileSync(filePath, "utf8").split("\n");
        for (let idx = 0; idx < lines.length; idx++) {
          const line = lines[idx] ?? "";
          const m = legacyGuardrailRe.exec(line);
          if (m) {
            legacyHits.push(`${relPath(filePath)}:${idx + 1}: ${m[0]}`);
          }
        }
      }
    }
    expect(legacyHits).toEqual([]);

    const docToTestLinks: string[] = [];
    const docRoots = [
      ...walkMarkdownFiles(DOCS_DIR),
      path.join(REPO_ROOT, "README.md"),
      path.join(REPO_ROOT, "CONTRIBUTING.md"),
      path.join(REPO_ROOT, "AGENTS.md"),
    ];
    for (const mdPath of docRoots) {
      const rel = relPath(mdPath);
      if (rel === "docs/specs/architecture/project_and_documentation_quality.md") {
        continue;
      }
      const clean = stripCode(fs.readFileSync(mdPath, "utf8"));
      const lines = clean.split("\n");
      for (let idx = 0; idx < lines.length; idx++) {
        const line = lines[idx] ?? "";
        for (const m of line.matchAll(MD_LINK_RE)) {
          const target = m[2] ?? "";
          if (
            (mdPath.startsWith(SPECS_DIR) || mdPath.startsWith(ADR_DIR) || mdPath === REQ_FILE) &&
            (target.includes("test_") ||
              target.includes("_spec_references") ||
              target.includes("_verification") ||
              target.includes("/tests/"))
          ) {
            docToTestLinks.push(`${rel}:${idx + 1} -> ${target}`);
          }
        }
      }
    }
    expect(docToTestLinks).toEqual([]);

    const unresolvedRefs: string[] = [];
    for (const [ruleId, pathStr, lineno] of collectSourceRefTokens()) {
      if (!rules.has(ruleId)) {
        unresolvedRefs.push(`${pathStr}:${lineno} -> ref[${ruleId}]`);
      }
    }
    expect(unresolvedRefs).toEqual([]);

    const verifyTokens = collectTestVerifyTokens();
    expect(verifyTokens.length).toBeGreaterThan(0);
    const unresolvedVerifies: string[] = [];
    for (const [ruleId, pathStr, lineno] of verifyTokens) {
      if (!rules.has(ruleId)) {
        unresolvedVerifies.push(`${pathStr}:${lineno} -> verify[${ruleId}]`);
      }
    }
    expect(unresolvedVerifies).toEqual([]);

    const exts = new Set([".rs", ".ts", ".tsx"]);
    const forbiddenPatterns = [/docs\/specs\//, /docs\/adr\//, /docs\/requirements\//];
    const violations: string[] = [];
    for (const root of [TESTS_DIR, PACKAGES_DIR]) {
      for (const filePath of walkFilesByExtensions(root, exts)) {
        if (path.basename(filePath) === "test_guardrails.test.ts") {
          continue;
        }
        const rel = relPath(filePath);
        if (!rel.includes("test") && !rel.includes("tests")) {
          continue;
        }
        const text = fs.readFileSync(filePath, "utf8");
        const lines = text.split("\n");
        for (let idx = 0; idx < lines.length; idx++) {
          const line = lines[idx] ?? "";
          const stripped = line.trim();
          if (
            stripped.startsWith("#") ||
            stripped.startsWith("//") ||
            stripped.startsWith("*") ||
            stripped.startsWith("/*")
          ) {
            continue;
          }
          for (const pat of forbiddenPatterns) {
            if (pat.test(line)) {
              violations.push(`${rel}:${idx + 1}: ${stripped}`);
            }
          }
        }
      }
    }
    expect(violations).toEqual([]);

    const modRsFiles = walkFilesByExtensions(PACKAGES_DIR, new Set([".rs"]))
      .filter((p) => path.basename(p) === "mod.rs")
      .map((p) => relPath(p));
    expect(modRsFiles).toEqual([]);

    const proptestDirs = [
      path.join(PACKAGES_DIR, "engine", "src", "math"),
      path.join(PACKAGES_DIR, "engine", "src", "operators"),
      path.join(PACKAGES_DIR, "engine", "src", "stages"),
      path.join(PACKAGES_DIR, "engine", "src", "control"),
      path.join(PACKAGES_DIR, "engine", "src", "container"),
      path.join(PACKAGES_DIR, "engine", "src", "provenance"),
      path.join(PACKAGES_DIR, "engine", "src", "schema"),
      path.join(PACKAGES_DIR, "engine", "tests"),
      path.join(PACKAGES_DIR, "trainer"),
    ];
    for (const dirPath of proptestDirs) {
      const rsFiles = walkFilesByExtensions(dirPath, new Set([".rs"]));
      const hasProptest = rsFiles.some((f) => fs.readFileSync(f, "utf8").includes("proptest!"));
      expect(hasProptest, `${relPath(dirPath)} must contain at least one proptest! suite`).toBe(
        true,
      );
    }
  });

  it("test_spec_rules_verified_in_tests", () => {
    const rules = collectSpecRules();
    const verifiedIds = new Set(collectTestVerifyTokens().map(([ruleId]) => ruleId));
    const unverified = Array.from(rules.keys())
      .filter((r) => !verifiedIds.has(r))
      .sort();
    expect(unverified).toEqual([]);
  });

  it("test_engine_spec_scope_manifest", () => {
    const scopePath = path.join(PACKAGES_DIR, "engine", "spec_scope.toml");
    expect(fs.existsSync(scopePath)).toBe(true);

    const rulesMap = _parseSpecScopeToml(fs.readFileSync(scopePath, "utf8"));
    const canonicalRules = collectSpecRules();
    const canonicalIds = new Set(canonicalRules.keys());
    const manifestIds = new Set(rulesMap.keys());

    const missingInManifest = Array.from(canonicalIds)
      .filter((id) => !manifestIds.has(id))
      .sort();
    const staleInManifest = Array.from(manifestIds)
      .filter((id) => !canonicalIds.has(id))
      .sort();
    expect(missingInManifest).toEqual([]);
    expect(staleInManifest).toEqual([]);

    const engineDir = path.join(PACKAGES_DIR, "engine");
    const engineRsFiles = walkFilesByExtensions(engineDir, new Set([".rs"]));
    const actualEngineVerifies = new Map<string, Set<string>>();
    for (const rsPath of engineRsFiles) {
      const relToEngine = path.relative(engineDir, rsPath).split(path.sep).join("/");
      const text = fs.readFileSync(rsPath, "utf8");
      for (const m of text.matchAll(SPEC_VERIFY_TOKEN_RE)) {
        const rid = m[1] ?? "";
        const set = actualEngineVerifies.get(rid) ?? new Set<string>();
        set.add(relToEngine);
        actualEngineVerifies.set(rid, set);
      }
    }

    const errors: string[] = [];
    for (const [ruleId, entry] of Array.from(rulesMap.entries()).sort(([a], [b]) =>
      a.localeCompare(b),
    )) {
      const scope = entry.scope;
      if (scope !== "engine" && scope !== "out-of-scope") {
        errors.push(`r[${ruleId}]: invalid scope '${scope}'`);
        continue;
      }
      if (scope === "out-of-scope") {
        const reason = (entry.reason ?? "").trim();
        if (!reason) {
          errors.push(`r[${ruleId}]: scope='out-of-scope' requires a non-empty 'reason'`);
        }
        if (actualEngineVerifies.has(ruleId)) {
          const foundIn = Array.from(actualEngineVerifies.get(ruleId) ?? []).sort();
          errors.push(
            `r[${ruleId}]: marked 'out-of-scope' but verified in packages/engine: ${foundIn.join(", ")}`,
          );
        }
      } else {
        const actualFiles = actualEngineVerifies.get(ruleId) ?? new Set<string>();
        if (actualFiles.size === 0) {
          errors.push(
            `r[${ruleId}]: scope='engine', but no 'verify[${ruleId}]' found in packages/engine/**/*.rs`,
          );
        }
      }
    }
    expect(errors).toEqual([]);
  });

  it("test_progressive_disclosure_agents_md_byte_limits", () => {
    const allAgents = walkMarkdownFiles(REPO_ROOT).filter((p) => path.basename(p) === "AGENTS.md");
    const rootAgents = path.join(REPO_ROOT, "AGENTS.md");
    expect(allAgents.includes(rootAgents)).toBe(true);

    const rootSize = fs.statSync(rootAgents).size;
    expect(rootSize).toBeLessThan(5000);

    const scopedAgents = allAgents.filter((p) => p !== rootAgents);
    expect(scopedAgents.length).toBeGreaterThanOrEqual(19);

    for (const agentsPath of allAgents) {
      const size = fs.statSync(agentsPath).size;
      if (agentsPath !== rootAgents) {
        expect(size, `${relPath(agentsPath)} is ${size} B (must be < 2,500 B)`).toBeLessThan(2500);
      }
      const text = fs.readFileSync(agentsPath, "utf8");
      const clean = stripFencedCodeBlocks(text);
      expect(clean).not.toContain("$$");
      expect(clean.match(SPEC_RULE_DEF_RE)).toBeNull();
    }
  });

  it("test_no_inline_latex_math_in_markdown_tables", () => {
    const violations: string[] = [];
    for (const mdPath of walkMarkdownFiles(REPO_ROOT)) {
      const clean = stripFencedCodeBlocks(fs.readFileSync(mdPath, "utf8"));
      const lines = clean.split("\n");
      for (let idx = 0; idx < lines.length; idx++) {
        const line = lines[idx] ?? "";
        if (!line.trimStart().startsWith("|")) {
          continue;
        }
        const withoutInlineCode = line.replace(INLINE_CODE_RE, "");
        if (withoutInlineCode.includes("$")) {
          violations.push(`${relPath(mdPath)}:${idx + 1}: ${line.trim().slice(0, 100)}`);
        }
      }
    }
    expect(violations).toEqual([]);
  });

  it("test_valid_github_latex_math_across_docs", () => {
    const violations: string[] = [];
    const unquotedMermaidLabelRe = /\b[A-Za-z0-9_]+\[(?!")([^\]]*[():][^\]]*)\]/;
    const bidirMermaidArrowRe = /<(?:--|==)\s+[^->=\n]+\s+(?:-->|==>)/;

    function checkMathBody(rel: string, kind: string, body: string): void {
      if (BACKSLASH_PUNCT_ESCAPE_RE.test(body)) {
        violations.push(`${rel} (${kind} backslash escape): ${body.trim().slice(0, 80)}`);
      }
      for (const macro of FORBIDDEN_MATH_MACROS) {
        if (body.includes(macro)) {
          violations.push(
            `${rel} (${kind} forbidden macro '${macro}'): ${body.trim().slice(0, 80)}`,
          );
        }
      }
      if (TEXT_UNDERSCORE_RE.test(body)) {
        violations.push(
          `${rel} (${kind} unescaped '_' inside \\text/\\mathrm): ${body.trim().slice(0, 80)}`,
        );
      }
      const openCount = (body.match(/\{/g) ?? []).length;
      const closeCount = (body.match(/\}/g) ?? []).length;
      if (openCount !== closeCount) {
        violations.push(`${rel} (${kind} unbalanced braces): ${body.trim().slice(0, 80)}`);
      }
    }

    for (const mdPath of walkMarkdownFiles(REPO_ROOT)) {
      const rel = relPath(mdPath);
      const rawText = fs.readFileSync(mdPath, "utf8");
      for (const m of rawText.matchAll(/```mermaid\n([\s\S]*?)```/g)) {
        const block = m[1] ?? "";
        const strippedBlock = block.trimStart();
        if (strippedBlock.startsWith("sequenceDiagram") && block.includes(";")) {
          violations.push(`${rel}: semicolon inside Mermaid sequenceDiagram`);
        } else if (strippedBlock.startsWith("flowchart")) {
          if (unquotedMermaidLabelRe.test(block)) {
            violations.push(`${rel}: unquoted special chars in Mermaid flowchart node label`);
          }
          if (bidirMermaidArrowRe.test(block)) {
            violations.push(`${rel}: invalid bidirectional arrow in Mermaid flowchart`);
          }
        }
      }
      const clean = stripCode(rawText);
      for (const m of clean.matchAll(DISPLAY_MATH_RE)) {
        checkMathBody(rel, "display math", m[1] ?? "");
      }
      const noDisplay = clean.replace(DISPLAY_MATH_RE, " ");
      for (const m of noDisplay.matchAll(INLINE_MATH_RE)) {
        checkMathBody(rel, "inline math", m[1] ?? "");
      }
    }
    expect(violations).toEqual([]);
  });

  it("test_relative_markdown_links_resolve", () => {
    const anchorCache = new Map<string, Set<string>>();
    const brokenLinks: string[] = [];

    for (const mdPath of walkMarkdownFiles(REPO_ROOT)) {
      const clean = stripCode(fs.readFileSync(mdPath, "utf8"));
      const lines = clean.split("\n");
      for (let idx = 0; idx < lines.length; idx++) {
        const line = lines[idx] ?? "";
        for (const m of line.matchAll(MD_LINK_RE)) {
          const rawTarget = m[2] ?? "";
          if (
            rawTarget.startsWith("http://") ||
            rawTarget.startsWith("https://") ||
            rawTarget.startsWith("mailto:")
          ) {
            continue;
          }
          let targetFile: string;
          let anchor: string;
          if (rawTarget.startsWith("#")) {
            targetFile = mdPath;
            anchor = rawTarget.slice(1);
          } else if (rawTarget.includes("#")) {
            const hashIdx = rawTarget.indexOf("#");
            const filePart = rawTarget.slice(0, hashIdx);
            anchor = rawTarget.slice(hashIdx + 1);
            targetFile = path.resolve(path.dirname(mdPath), filePart);
          } else {
            targetFile = path.resolve(path.dirname(mdPath), rawTarget);
            anchor = "";
          }

          if (!fs.existsSync(targetFile)) {
            brokenLinks.push(
              `${relPath(mdPath)}:${idx + 1} -> target '${rawTarget}' does not exist`,
            );
            continue;
          }
          if (anchor && targetFile.endsWith(".md") && fs.statSync(targetFile).isFile()) {
            let anchors = anchorCache.get(targetFile);
            if (!anchors) {
              anchors = extractHeadingAnchors(fs.readFileSync(targetFile, "utf8"));
              anchorCache.set(targetFile, anchors);
            }
            if (!anchors.has(anchor)) {
              brokenLinks.push(
                `${relPath(mdPath)}:${idx + 1} -> anchor '#${anchor}' not found in ${relPath(targetFile)}`,
              );
            }
          }
        }
      }
    }
    expect(brokenLinks).toEqual([]);
  });

  it("test_docs_index_links_all_specs_and_adrs", () => {
    const indexText = fs.readFileSync(path.join(DOCS_DIR, "index.md"), "utf8");
    for (const specPath of walkMarkdownFiles(SPECS_DIR)) {
      const relFromDocs = path.relative(DOCS_DIR, specPath).split(path.sep).join("/");
      expect(indexText).toContain(relFromDocs);
    }
    for (const entry of fs.readdirSync(ADR_DIR).sort()) {
      if (!entry.endsWith(".md") || entry === "README.md") {
        continue;
      }
      const relFromDocs = `adr/${entry}`;
      expect(indexText).toContain(relFromDocs);
    }
  });

  it("test_root_readme_stays_focused_on_musician_and_oem_personas", () => {
    const readmeText = fs.readFileSync(path.join(REPO_ROOT, "README.md"), "utf8");
    expect(readmeText).toContain("Musicians");
    expect(readmeText).toContain("Tone Creators");
    expect(readmeText).toContain("docs/index.md");
    expect(readmeText).toContain("CONTRIBUTING.md");
    expect(readmeText).not.toContain("$$");
    expect(readmeText.match(SPEC_RULE_DEF_RE)).toBeNull();
  });

  it("test_bidirectional_req_and_adr_traceability_across_specs", () => {
    const reqText = fs.readFileSync(REQ_FILE, "utf8");
    const definedReqs = new Set(Array.from(reqText.matchAll(REQ_TABLE_RE), (m) => m[1] ?? ""));

    const adrStatuses = new Map<string, string>();
    const adrEnforcesSpecs = new Map<string, string[]>();
    for (const entry of fs.readdirSync(ADR_DIR).sort()) {
      if (!entry.endsWith(".md") || entry === "README.md") {
        continue;
      }
      const adrPath = path.join(ADR_DIR, entry);
      const fm = parseFrontmatter(fs.readFileSync(adrPath, "utf8"));
      if (!fm) {
        continue;
      }
      const adrId = String(fm["id"]);
      adrStatuses.set(adrId, String(fm["status"]));
      const rawSpecs = Array.isArray(fm["specs"]) ? fm["specs"] : [];
      adrEnforcesSpecs.set(
        adrId,
        rawSpecs.map((s) => normalizeSpecRelPath(s)),
      );
    }

    const supersededAdrs = new Set(
      Array.from(adrStatuses.entries())
        .filter(([, st]) => st === "superseded")
        .map(([aid]) => aid),
    );

    const specCitedReqs = new Set<string>();
    const specCitedAdrs = new Set<string>();
    const specFileSet = new Set<string>();
    for (const specPath of walkMarkdownFiles(SPECS_DIR)) {
      const relFromDocs = path.relative(DOCS_DIR, specPath).split(path.sep).join("/");
      specFileSet.add(relFromDocs);
      const text = fs.readFileSync(specPath, "utf8");
      const fm = parseFrontmatter(text);
      if (Array.isArray(fm?.["requirements"])) {
        for (const r of fm["requirements"]) {
          specCitedReqs.add(r);
        }
      }
      for (const m of text.matchAll(REQ_TOKEN_RE)) {
        specCitedReqs.add(m[0]);
      }
      if (Array.isArray(fm?.["adrs"])) {
        for (const a of fm["adrs"]) {
          expect(
            supersededAdrs.has(a),
            `${relPath(specPath)} frontmatter cites superseded ADR '${a}'`,
          ).toBe(false);
          specCitedAdrs.add(a);
        }
      }
      for (const m of text.matchAll(ADR_TOKEN_RE)) {
        specCitedAdrs.add(m[0]);
      }
    }

    for (const [aid, enforcedList] of adrEnforcesSpecs.entries()) {
      if (adrStatuses.get(aid) === "accepted") {
        for (const specRel of enforcedList) {
          if (specFileSet.has(specRel)) {
            specCitedAdrs.add(aid);
          }
        }
      }
      for (const specRel of enforcedList) {
        expect(
          specFileSet.has(specRel),
          `${aid} frontmatter specs cites non-existent '${specRel}'`,
        ).toBe(true);
      }
    }

    const uncitedReqs = Array.from(definedReqs)
      .filter((r) => !specCitedReqs.has(r))
      .sort();
    expect(uncitedReqs).toEqual([]);

    const acceptedAdrs = new Set(
      Array.from(adrStatuses.entries())
        .filter(([, st]) => st === "accepted")
        .map(([aid]) => aid),
    );
    const uncitedAdrs = Array.from(acceptedAdrs)
      .filter((a) => !specCitedAdrs.has(a))
      .sort();
    expect(uncitedAdrs).toEqual([]);
  });

  it("test_type_safety_and_import_boundaries", () => {
    const tsViolations: string[] = [];
    const forbiddenTsPatterns = [
      /@ts-ignore\b/,
      /@ts-expect-error\b/,
      /:\s*any\b/,
      /\bas\s+any\b/,
      /<any>/,
    ];
    for (const root of [PACKAGES_DIR, TESTS_DIR]) {
      for (const tsPath of walkFilesByExtensions(root, new Set([".ts", ".tsx"]))) {
        if (path.basename(tsPath) === "test_guardrails.test.ts") {
          continue;
        }
        const lines = fs.readFileSync(tsPath, "utf8").split("\n");
        for (let idx = 0; idx < lines.length; idx++) {
          const line = lines[idx] ?? "";
          const trimmed = line.trim();
          if (trimmed.startsWith("//") || trimmed.startsWith("*")) {
            continue;
          }
          for (const pat of forbiddenTsPatterns) {
            if (pat.test(line)) {
              tsViolations.push(`${relPath(tsPath)}:${idx + 1}: ${trimmed}`);
            }
          }
        }
      }
    }
    expect(tsViolations).toEqual([]);

    for (const rustLibRel of [
      "packages/engine/src/lib.rs",
      "packages/trainer/src/lib.rs",
      "packages/js/rust/lib.rs",
      "packages/studio/rust/lib.rs",
    ]) {
      const libPath = path.join(REPO_ROOT, rustLibRel);
      expect(fs.existsSync(libPath), `${rustLibRel} must exist`).toBe(true);
      const libText = fs.readFileSync(libPath, "utf8");
      expect(
        libText.includes("forbid(unsafe_code)") || libText.includes("deny(unsafe_code)"),
        `${rustLibRel} must declare forbid(unsafe_code) or deny(unsafe_code)`,
      ).toBe(true);
    }

    const pyFiles = walkFilesByExtensions(REPO_ROOT, new Set([".py"]));
    expect(pyFiles).toEqual([]);
  });

  it("test_zero_todo_placeholders_in_docs_and_source", () => {
    const todoRe = /\b(?:TODO|FIXME|XXX|HACK)\b/;
    const violations: string[] = [];
    for (const root of [DOCS_DIR, PACKAGES_DIR, TESTS_DIR]) {
      for (const filePath of walkFilesByExtensions(root, new Set([".md", ".rs", ".ts", ".tsx"]))) {
        if (path.basename(filePath) === "test_guardrails.test.ts") {
          continue;
        }
        const text = fs.readFileSync(filePath, "utf8");
        const lines = text.split("\n");
        for (let idx = 0; idx < lines.length; idx++) {
          const line = lines[idx] ?? "";
          if (todoRe.test(line)) {
            if (line.includes("`TODO`") || line.includes("`FIXME`") || line.includes("todo =")) {
              continue;
            }
            violations.push(`${relPath(filePath)}:${idx + 1}: ${line.trim().slice(0, 100)}`);
          }
        }
      }
    }
    expect(violations).toEqual([]);
  });

  it("test_license_and_cla_parity_across_manifests", () => {
    const licensePath = path.join(REPO_ROOT, "LICENSE");
    const claPath = path.join(REPO_ROOT, "CLA.md");
    expect(fs.existsSync(licensePath)).toBe(true);
    expect(fs.existsSync(claPath)).toBe(true);

    const licenseText = fs.readFileSync(licensePath, "utf8");
    expect(licenseText).toContain("DNAM Source-Available & Branding License 1.0.0");
    expect(licenseText).toContain("Distribution & Hardware/Plugin Linking License");

    const cargoToml = fs.readFileSync(path.join(REPO_ROOT, "Cargo.toml"), "utf8");
    expect(cargoToml).toContain('license = "LicenseRef-DNAM-1.0.0"');

    for (const pkgJsonRel of [
      "package.json",
      "packages/js/package.json",
      "packages/ui/package.json",
      "packages/studio/package.json",
    ]) {
      const pkgText = fs.readFileSync(path.join(REPO_ROOT, pkgJsonRel), "utf8");
      expect(pkgText).toContain('"license": "LicenseRef-DNAM-1.0.0"');
    }
  });
});
