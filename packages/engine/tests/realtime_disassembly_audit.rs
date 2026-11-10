//! Automated Release-Binary Disassembly Audit for Goal 6 (`REQ-6.2`, `REQ-6.3`).
//!
//! - `verify[repo.code.realtime-disassembly-audit]`
//! - `verify[arch.simd.zero-stall-lowering]`
//! - `verify[dsp.op-m.rotor.angle-wrap-and-retune]`
//! - `verify[dsp.op-t.guard.incremental-counters]`
//!
//! Disassembles the optimized `libdnam_engine.rlib` release artifact via `objdump -dr --demangle`
//! and verifies:
//! 1. **Crate-Wide `libm` Ban (`REQ-6.2`):** Zero external relocations to `_fmodf`, `_fmod`,
//!    `sinf`, `cosf`, `tanf`, `expf`, `logf`, `powf`, `sqrtf`, or `f64` `libm` equivalents.
//! 2. **Real-Time Loop Instruction & Call Audit (`REQ-6.2`, `REQ-6.3`):** Across all compiled
//!    `DnamEngine<*>::process_block*`, `process_mono*`, `step_frame_tora*`, `step_control_tick_3khz*`,
//!    `DnamHostHandle::process_*`, and C-ABI `_dnam_engine_process_*` functions:
//!    - Zero hardware floating-point or integer division/sqrt instructions (`fdiv`, `fsqrt`,
//!      `udiv`, `sdiv` on `aarch64`; `divss`/`divps`/`divsd`/`divpd`/`vdiv*`,
//!      `sqrtss`/`sqrtps`/`sqrtsd`/`sqrtpd`/`vsqrt*`, `idiv*`, `divl`/`divq` on `x86_64`).
//!    - Zero calls to 128-bit software division (`__udivti3`, `__divti3`, `__umodti3`, `__modti3`),
//!      heap allocation (`__rust_alloc`, `__rust_dealloc`, `__rust_realloc`), or uncached OS
//!      feature syscalls (`sysctlbyname`, `getauxval`).

#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::Command;

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root")
        .to_path_buf()
}

fn current_target_triple() -> &'static str {
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    {
        "aarch64-apple-darwin"
    }
    #[cfg(all(target_arch = "x86_64", target_os = "macos"))]
    {
        "x86_64-apple-darwin"
    }
    #[cfg(all(target_arch = "aarch64", target_os = "linux"))]
    {
        "aarch64-unknown-linux-gnu"
    }
    #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
    {
        "x86_64-unknown-linux-gnu"
    }
}

fn rlib_matches_current_arch(rlib_path: &Path) -> bool {
    if !rlib_path.is_file() {
        return false;
    }
    let Ok(out) = Command::new("objdump").arg("-t").arg(rlib_path).output() else {
        return false;
    };
    let text = String::from_utf8_lossy(&out.stdout);
    if !text.contains("step_frame_tora") {
        return false;
    }
    #[cfg(target_arch = "aarch64")]
    {
        text.contains("arm64") || text.contains("aarch64")
    }
    #[cfg(target_arch = "x86_64")]
    {
        text.contains("x86-64") || text.contains("x86_64")
    }
}

fn ensure_release_rlib() -> PathBuf {
    let root = workspace_root();
    let triple = current_target_triple();
    let audit_target_dir = root.join("target").join("release-audit");
    let status = Command::new("cargo")
        .current_dir(&root)
        .env_remove("RUSTFLAGS")
        .env_remove("CARGO_ENCODED_RUSTFLAGS")
        .env_remove("LLVM_PROFILE_FILE")
        .args([
            "build",
            "--package",
            "dnam-engine",
            "--release",
            "--target",
            triple,
            "--target-dir",
        ])
        .arg(&audit_target_dir)
        .status()
        .expect("failed to invoke cargo build --release for disassembly audit");
    assert!(status.success(), "cargo build --release failed");

    let built = audit_target_dir
        .join(triple)
        .join("release")
        .join("libdnam_engine.rlib");
    assert!(
        rlib_matches_current_arch(&built),
        "expected release rlib at {}",
        built.display()
    );
    built
}

fn is_realtime_root_symbol(sym: &str) -> bool {
    const RT_OWNER_PATTERNS: &[&str] = &[
        "DnamEngine",
        "DynamicDnamEngine",
        "ControlBus",
        "StagePipeline",
        "Stage1PreDynamics",
        "dnam_engine_process_block",
        "dnam_engine_process_mono",
    ];
    const RT_METHOD_PATTERNS: &[&str] = &[
        "::process_block",
        "::process_block_in_place",
        "::process_mono",
        "::process_mono_to_stereo",
        "::step_frame_tora",
        "::step_frame_4x_tora",
        "::step_frame_4port_tora",
        "::step_frame_4x_4port_tora",
        "::step_frame_with_feature_bus",
        "::step_control_tick_3khz",
        "::refresh_collapsed_head_weights",
        "::tick_3khz",
        "::tick_3khz_with_detectors",
        "::retune_kautz_formant_3khz",
        "::retune_op_m_pre_3khz",
        "dnam_engine_process_block",
        "dnam_engine_process_mono",
    ];
    RT_OWNER_PATTERNS.iter().any(|p| sym.contains(p))
        && RT_METHOD_PATTERNS.iter().any(|m| sym.contains(m))
}

fn clean_reloc_target(raw: &str) -> &str {
    let s = raw.trim_start_matches('_');
    let s = s.split('@').next().unwrap_or(s);
    let s = s.split('+').next().unwrap_or(s);
    s.split('-').next().unwrap_or(s)
}

fn is_forbidden_rt_instruction(mnemonic: &str) -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        matches!(mnemonic, "fdiv" | "fsqrt" | "udiv" | "sdiv")
    }
    #[cfg(target_arch = "x86_64")]
    {
        matches!(
            mnemonic,
            "divss"
                | "divps"
                | "divsd"
                | "divpd"
                | "vdivss"
                | "vdivps"
                | "vdivsd"
                | "vdivpd"
                | "sqrtss"
                | "sqrtps"
                | "sqrtsd"
                | "sqrtpd"
                | "vsqrtss"
                | "vsqrtps"
                | "vsqrtsd"
                | "vsqrtpd"
                | "idiv"
                | "idivl"
                | "idivq"
                | "div"
                | "divl"
                | "divq"
        )
    }
}

fn is_forbidden_crate_wide_reloc(target: &str) -> bool {
    let clean = clean_reloc_target(target);
    matches!(
        clean,
        "fmodf"
            | "fmod"
            | "sinf"
            | "cosf"
            | "tanf"
            | "asinf"
            | "acosf"
            | "atanf"
            | "atan2f"
            | "sinhf"
            | "coshf"
            | "tanhf"
            | "expf"
            | "exp2f"
            | "logf"
            | "log2f"
            | "log10f"
            | "powf"
            | "sqrtf"
            | "cbrtf"
            | "hypotf"
            | "sin"
            | "cos"
            | "tan"
            | "asin"
            | "acos"
            | "atan"
            | "atan2"
            | "sinh"
            | "cosh"
            | "tanh"
            | "exp"
            | "exp2"
            | "log"
            | "log2"
            | "log10"
            | "pow"
            | "sqrt"
            | "cbrt"
            | "hypot"
    )
}

fn is_forbidden_rt_reloc(target: &str) -> bool {
    if is_forbidden_crate_wide_reloc(target) {
        return true;
    }
    let clean = clean_reloc_target(target);
    matches!(
        clean,
        "udivti3"
            | "divti3"
            | "umodti3"
            | "modti3"
            | "rust_alloc"
            | "rust_dealloc"
            | "rust_realloc"
            | "rust_alloc_zeroed"
            | "sysctlbyname"
            | "getauxval"
    ) || clean.ends_with("___rust_alloc")
        || clean.ends_with("___rust_dealloc")
        || clean.ends_with("___rust_realloc")
        || clean.ends_with("___rust_alloc_zeroed")
}

#[derive(Default)]
struct FnDisasm {
    demangled: String,
    instructions: Vec<(String, String)>,
    relocs: Vec<(String, String)>,
    callees: BTreeSet<String>,
}

/// verify[dsp.op-t.zero-fdiv]
#[test]
fn test_release_disassembly_zero_pipeline_stalls_and_nondeterministic_calls() {
    let rlib = ensure_release_rlib();
    let out_raw = Command::new("objdump")
        .args(["-dr"])
        .arg(&rlib)
        .output()
        .expect("failed to execute objdump -dr");
    assert!(
        out_raw.status.success(),
        "objdump -dr failed on {}",
        rlib.display()
    );
    let text_raw = String::from_utf8_lossy(&out_raw.stdout);

    let out_demangled = Command::new("objdump")
        .args(["-dr", "--demangle"])
        .arg(&rlib)
        .output()
        .expect("failed to execute objdump -dr --demangle");
    assert!(
        out_demangled.status.success(),
        "objdump -dr --demangle failed on {}",
        rlib.display()
    );
    let text_demangled = String::from_utf8_lossy(&out_demangled.stdout);

    let demangled_headers: Vec<String> = text_demangled
        .lines()
        .filter_map(|raw_line| {
            let line = raw_line.trim();
            if line.ends_with(">:")
                && let Some(open) = line.find('<')
            {
                Some(line[open + 1..line.len() - 2].to_string())
            } else {
                None
            }
        })
        .collect();

    let mut fns: BTreeMap<String, FnDisasm> = BTreeMap::new();
    let mut current_fn = String::new();
    let mut header_idx = 0usize;

    for raw_line in text_raw.lines() {
        let line = raw_line.trim();
        if line.ends_with(">:")
            && let Some(open) = line.find('<')
        {
            let raw_sym = &line[open + 1..line.len() - 2];
            current_fn = clean_reloc_target(raw_sym).to_string();
            let demangled = demangled_headers
                .get(header_idx)
                .cloned()
                .unwrap_or_else(|| current_fn.clone());
            header_idx += 1;
            let entry = fns.entry(current_fn.clone()).or_default();
            entry.demangled = demangled;
            continue;
        }
        if current_fn.is_empty() {
            continue;
        }

        if line.contains("RELOC") || line.contains("R_AARCH64") || line.contains("R_X86_64") {
            if let Some(target) = line.split_whitespace().last() {
                let entry = fns.entry(current_fn.clone()).or_default();
                entry.relocs.push((target.to_string(), line.to_string()));
                let callee_key = clean_reloc_target(target).to_string();
                if callee_key != current_fn {
                    entry.callees.insert(callee_key);
                }
            }
            continue;
        }

        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() >= 2 {
            let insn_col = parts[parts.len() - 1].trim();
            let mnemonic = insn_col.split_whitespace().next().unwrap_or("");
            if !mnemonic.is_empty() {
                let entry = fns.entry(current_fn.clone()).or_default();
                entry
                    .instructions
                    .push((mnemonic.to_string(), line.to_string()));
                if matches!(mnemonic, "bl" | "b" | "call" | "callq" | "jmp")
                    && let (Some(open), Some(close)) = (insn_col.find('<'), insn_col.rfind('>'))
                    && open + 1 < close
                {
                    let callee = &insn_col[open + 1..close];
                    let callee_base = clean_reloc_target(callee);
                    if callee_base != current_fn {
                        entry.callees.insert(callee_base.to_string());
                    }
                }
            }
        }
    }

    // Compute transitive call-graph closure from all real-time roots
    let mut rt_closure: BTreeSet<String> = fns
        .iter()
        .filter(|(_, info)| is_realtime_root_symbol(&info.demangled))
        .map(|(k, _)| k.clone())
        .collect();
    let root_count = rt_closure.len();
    let mut worklist: Vec<String> = rt_closure.iter().cloned().collect();
    while let Some(sym) = worklist.pop() {
        if let Some(info) = fns.get(&sym) {
            for callee in &info.callees {
                if fns.contains_key(callee) && rt_closure.insert(callee.clone()) {
                    worklist.push(callee.clone());
                }
            }
        }
    }

    let mut crate_reloc_violations = Vec::new();
    let mut rt_violations = Vec::new();

    for (fn_key, info) in &fns {
        let fn_name = &info.demangled;
        for (target, line) in &info.relocs {
            if is_forbidden_crate_wide_reloc(target) {
                crate_reloc_violations.push(format!("{fn_name}: {line}"));
            }
        }
        if rt_closure.contains(fn_key) {
            for (target, line) in &info.relocs {
                if is_forbidden_rt_reloc(target) {
                    rt_violations.push(format!("{fn_name}: {line}"));
                }
            }
            for (mnemonic, line) in &info.instructions {
                if is_forbidden_rt_instruction(mnemonic) {
                    rt_violations.push(format!("{fn_name}: {line}"));
                }
            }
        }
    }

    assert!(
        root_count >= 15 && rt_closure.len() > root_count,
        "Expected >= 15 real-time root functions and non-empty transitive callees in {}, found {} roots and {} total in closure",
        rlib.display(),
        root_count,
        rt_closure.len()
    );
    assert!(
        crate_reloc_violations.is_empty(),
        "REQ-6.2 violation: forbidden libm/fmod relocations in {}:\n{}",
        rlib.display(),
        crate_reloc_violations.join("\n")
    );
    assert!(
        rt_violations.is_empty(),
        "REQ-6.2 / REQ-6.3 violation: pipeline-stalling instructions or nondeterministic calls in {}:\n{}",
        rlib.display(),
        rt_violations.join("\n")
    );
}
