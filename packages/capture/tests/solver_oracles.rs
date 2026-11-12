#![forbid(unsafe_code)]
//! Step 2.3 Structural Isomorphism Oracles (`Oracle-01`, `Oracle-02`, `Oracle-03`) for the
//! `< 25 ms` 5-Stage Closed-Form + `VarPro` Solver (`dnam_capture::solver`).
//!
//! Canonical References:
//! - `docs/specs/training/four_stage_pipeline.md` (`r[train.pipeline.*]`, `r[train.gram.*]`)
//! - `docs/specs/containers/project_container_format.md` (`r[container.proj.*]`)

use std::collections::BTreeMap;

use dnam_capture::metrics::{evaluate_besr_from_band_energies, split_bezout_3band_energies};
use dnam_capture::proj::{
    DnamProjManifest, DnamProjView, PROJ_FLAG_COMPACTED_CAPSULES, PROJ_FLAG_HAS_RAW_STEMS,
    PROJ_STAT_BYTES_ULTRA, ProjChunkKind, StemProvenance, StmcTrajectoryWindow, VaultCaptureSpec,
    add_capture, add_capture_with_options, build_dnam_proj_container,
    build_dnam_proj_container_with_cp1, build_proj_inpc_capsule, compact_to_capsules,
    export_runtime_dnam, parse_proj_manifest_range, remove_capture, replace_capture,
    set_capture_enabled, solve_captures_batch_with_options,
};
use dnam_capture::solver::stage_a::CASCADE_W_BASES;
use dnam_capture::solver::stage_b_c::{HEAD_BASES, ULTRA_MODES_5_TO_8_BASE};
use dnam_capture::solver::{
    SolverOptions, StageAConfig, Wide192Accum, compute_role_b_multipliers,
    harvest_star_center_256d, lift_stage_a_backbone, solve_stage_c_nested_schur_4tier,
    star_center_control_values, write_effective_mode0_weights, zero_all_role_b_cascade_weights,
};
use dnam_capture::{CORESET_FP16_FRAMES_PER_STEM, f32_to_fp16_bits, fp16_bits_to_f32};
use dnam_engine::container::{
    ARENA_OFFSET_ULTRA_END, DnamContainerView, build_dnam_container_with_cp1,
    strip_container_to_tier,
};
use dnam_engine::control::Cp1ContributorSlot;
use dnam_engine::math::DefaultMathBackend;
use dnam_engine::provenance::{
    DefaultCryptoProvider, STAGE3_KAUTZ_BLOCKS_PER_TIER, apply_cascade_head_gauge,
    apply_isometric_gauge_watermark, extract_structural_hash_224d, kautz_soa_byte_offset,
    structural_cosine_similarity,
};
use dnam_engine::schema::{ControlKind, ControlSchemaDef, DnamJsonHeader};
use dnam_engine::{DnamEngine, Tier};
use sha2::{Digest, Sha256};

fn f32_slice_to_fp16_bytes(samples: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(samples.len() * 2);
    for &s in samples {
        out.extend_from_slice(&f32_to_fp16_bits(s).to_le_bytes());
    }
    out
}

fn fp16_bytes_to_f32_vec(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|&c| fp16_bits_to_f32(u16::from_le_bytes(c)))
        .collect()
}

fn pcm_sha256_hex(samples: &[f32]) -> String {
    let mut h = Sha256::new();
    for &s in samples {
        h.update((if s == 0.0 { 0.0_f32 } else { s }).to_le_bytes());
    }
    let digest: [u8; 32] = h.finalize().into();
    let mut out = String::with_capacity(64);
    for b in digest {
        use std::fmt::Write;
        let _ = write!(&mut out, "{b:02x}");
    }
    out
}

fn write_f32(buf: &mut [u8], off: usize, v: f32) {
    buf[off..off + 4].copy_from_slice(&v.to_le_bytes());
}

fn pack_dnam(hdr: &DnamJsonHeader, arena: &[u8], slots: &[Cp1ContributorSlot]) -> Vec<u8> {
    let mut h = hdr.clone();
    build_dnam_container_with_cp1(&mut h, arena, arena, false, slots).unwrap()
}

fn ctrl_meta(gain: f64, channel: &str) -> BTreeMap<String, serde_json::Value> {
    BTreeMap::from([
        ("gain".into(), serde_json::json!(gain)),
        ("channel".into(), serde_json::json!(channel)),
    ])
}

fn make_range_ctrl(id: &str, lo: f32, hi: f32, def: f32) -> ControlSchemaDef {
    serde_json::from_value(serde_json::json!({
        "id": id, "name": id, "type": "range", "lobatto_degree": 2,
        "declared_range": [lo, hi], "safe_clamped_range": [lo, hi], "default": def
    }))
    .unwrap()
}

fn make_switch_ctrl(id: &str, opts: &[&str], def_idx: usize) -> ControlSchemaDef {
    serde_json::from_value(serde_json::json!({
        "id": id, "name": id, "type": "switch",
        "declared_options": opts, "active_options": opts, "default": opts[def_idx]
    }))
    .unwrap()
}

fn stmc_win(control_id: u16, ctrl_kind: u8, active_opt: u8, u_phys: f32) -> StmcTrajectoryWindow {
    StmcTrajectoryWindow {
        control_id,
        ctrl_kind,
        active_opt,
        u_phys,
        window_weight: 1.0,
        emr_velocity: 0.0,
    }
}

fn make_vault_spec(
    stem_id: &str,
    is_star_center: bool,
    meta: BTreeMap<String, serde_json::Value>,
    dry: &[f32],
    wet_interleaved: &[f32],
    windows: Vec<StmcTrajectoryWindow>,
    optional_spcm: Option<Vec<u8>>,
) -> VaultCaptureSpec {
    VaultCaptureSpec {
        stem_id: stem_id.to_string(),
        is_star_center,
        enabled: true,
        pad_gain_f32: 1.0,
        fractional_lag_samples_f32: 0.0,
        controls_meta: meta,
        provenance: StemProvenance {
            input_pcm_sha256: pcm_sha256_hex(dry),
            target_pcm_sha256: pcm_sha256_hex(wet_interleaved),
            num_frames: dry.len() as u64,
        },
        user_metadata: BTreeMap::new(),
        extracted_physics: None,
        windows,
        band_psd_summary_88: [0u8; 88],
        n_comb_bins: 32,
        spectral_summary_4096: [0u8; 4_096],
        wet_coreset_fp16: f32_slice_to_fp16_bytes(wet_interleaved),
        optional_spcm,
    }
}

fn synthesize_rich_coreset_dry(n: usize) -> Vec<f32> {
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    (0..n)
        .map(|i| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let noise = ((state >> 40) as f32 / ((1u64 << 24) as f32)) * 2.0 - 1.0;
            let t = (i as f32) / 48_000.0;
            let env = 0.25 + 0.65 * ((i as f32 * 0.0017).sin().abs());
            let tones = 0.45 * (std::f32::consts::TAU * 110.0 * t).sin()
                + 0.30 * (std::f32::consts::TAU * 440.0 * t).cos()
                + 0.20 * (std::f32::consts::TAU * 1_760.0 * t).sin();
            (env * (0.65 * tones + 0.35 * noise)).clamp(-0.85, 0.85)
        })
        .collect()
}

/// Builds a ground-truth `64,000 B` TORA arena with populated Stage I, Stage II (`Role A` + non-zero
/// `Role B`), Stage III (Early FIR + shared-pole Kautz across all 4 tiers), Stage IV coprime tank,
/// and decaying 4-tier Mode-0 stereo output heads.
fn build_oracle01_teacher_arena() -> Vec<u8> {
    let mut arena = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
    write_f32(&mut arena, 124, 0.9993);
    write_f32(&mut arena, 7_520, 10.0);
    write_f32(&mut arena, 10_560, 10.0);
    write_f32(&mut arena, 10_688, 0.0625);
    for t in 0..16 {
        write_f32(&mut arena, 10_624 + t * 4, 1.0);
    }

    for lane in 0..4 {
        let r = 0.90_f32 + 0.015 * (lane as f32);
        write_f32(&mut arena, 256 + lane * 4, r);
        write_f32(&mut arena, 256 + 16 + lane * 4, r + 0.02);
        write_f32(&mut arena, 256 + 32 + lane * 4, 0.08 + 0.07 * (lane as f32));
        write_f32(&mut arena, 256 + 48 + lane * 4, (1.0 - r * r).sqrt());
        write_f32(&mut arena, 256 + 64 + lane * 4, 0.28);
        write_f32(&mut arena, 256 + 96 + lane * 4, -0.19);
    }

    for (t, &cw_base) in CASCADE_W_BASES.iter().enumerate() {
        let k_t = 48 * (t + 1);
        for q in 0..16 {
            let role_a = 0.045_f32 * (((q * 5 + t * 3 + 1) % 7) as f32 - 3.0);
            write_f32(&mut arena, cw_base + q * 4, role_a);
        }
        for b in 1..k_t {
            let blk = cw_base + b * 64;
            for q in 0..16 {
                let role_b = 0.008_f32 * (((b * 3 + q * 7 + t) % 9) as f32 - 4.0);
                write_f32(&mut arena, blk + q * 4, role_b);
            }
        }
    }
    for s in 0..16 {
        write_f32(&mut arena, 4_096 + s * 4, 0.45 + 0.025 * (s as f32));
        write_f32(&mut arena, 4_160 + s * 4, 0.02 + 0.005 * (s as f32));
        write_f32(&mut arena, 4_224 + s * 4, 1.15 + 0.04 * (s as f32));
    }

    let fir_bases = [4_288_usize, 17_408, 31_296, 48_384];
    for (t, &f_base) in fir_bases.iter().enumerate() {
        for tap in 0..32 {
            let h = 0.14 * (-0.11 * (tap as f32)).exp() * ((tap + t + 1) as f32 * 0.63).cos();
            write_f32(&mut arena, f_base + tap * 4, h);
        }
        for blk in 0..STAGE3_KAUTZ_BLOCKS_PER_TIER[t] {
            let k_off = kautz_soa_byte_offset(t, blk);
            for lane in 0..4 {
                let r_lo = 0.89_f32 + 0.018 * (lane as f32);
                let theta = 0.065_f32 + 0.19_f32 * ((t * 2 + blk + lane) as f32);
                write_f32(&mut arena, k_off + lane * 4, r_lo);
                write_f32(&mut arena, k_off + 16 + lane * 4, 0.02);
                write_f32(&mut arena, k_off + 32 + lane * 4, theta);
                write_f32(
                    &mut arena,
                    k_off + 48 + lane * 4,
                    (1.0 - r_lo * r_lo).sqrt(),
                );
                write_f32(&mut arena, k_off + 64 + lane * 4, 0.24 - 0.03 * (t as f32));
                write_f32(&mut arena, k_off + 80 + lane * 4, 0.15);
                write_f32(&mut arena, k_off + 96 + lane * 4, 0.21 - 0.03 * (t as f32));
                write_f32(&mut arena, k_off + 112 + lane * 4, -0.14);
            }
        }
    }

    for (l, &(d, ap)) in [
        (53.0_f32, 0.45_f32),
        (89.0, -0.41),
        (137.0, 0.37),
        (199.0, -0.33),
    ]
    .iter()
    .enumerate()
    {
        write_f32(&mut arena, 4_672 + l * 4, d);
        write_f32(&mut arena, 4_700 + l * 4, ap);
    }
    for &(off, v) in &[(4_688, 0.38_f32), (4_692, 1.0), (4_696, 0.34)] {
        write_f32(&mut arena, off, v);
    }
    for (base, delays) in [
        (17_792_usize, &[257.0_f32, 313.0, 389.0, 463.0][..]),
        (34_496, &[523.0, 593.0, 661.0, 743.0][..]),
        (
            48_896,
            &[773.0, 809.0, 839.0, 877.0, 907.0, 941.0, 971.0, 997.0][..],
        ),
    ] {
        for (l, &d) in delays.iter().enumerate() {
            write_f32(&mut arena, base + l * 4, d);
        }
    }

    let tier_scales = [0.55_f32, 0.20, 0.06, 0.015];
    for (t, &h_base) in HEAD_BASES.iter().enumerate() {
        let sc = tier_scales[t];
        for i in 0..64 {
            let wl = sc * (((i * 11 + t * 5 + 3) % 13) as f32 - 6.0) / 6.0;
            let wr = sc * (((i * 7 + t * 9 + 2) % 13) as f32 - 6.0) / 6.0;
            write_f32(&mut arena, h_base + i * 4, wl);
            write_f32(&mut arena, h_base + 256 + i * 4, wr);
        }
    }
    arena
}

fn render_stereo_dnam(
    dnam_bytes: &[u8],
    tier: Tier,
    dry: &[f32],
    ctrls: &[(usize, f32)],
) -> Vec<[f32; 2]> {
    let mut eng =
        DnamEngine::<DefaultMathBackend>::from_bytes_owned(dnam_bytes, tier, 48_000).unwrap();
    for &(idx, val) in ctrls {
        eng.set_control_immediate(idx, val);
    }
    dry.iter()
        .map(|&x| eng.step_frame_tora(x, x, false))
        .collect()
}

fn esr_db_stereo(target: &[[f32; 2]], pred: &[[f32; 2]]) -> f64 {
    let skip = 128.min(target.len() / 8);
    let mut err = 0.0_f64;
    let mut sig = 0.0_f64;
    for (&[tl, tr], &[pl, pr]) in target[skip..].iter().zip(&pred[skip..]) {
        let dl = f64::from(tl) - f64::from(pl);
        let dr = f64::from(tr) - f64::from(pr);
        err += dl * dl + dr * dr;
        sig += f64::from(tl) * f64::from(tl) + f64::from(tr) * f64::from(tr);
    }
    10.0 * (err / sig.max(1.0e-15)).max(1.0e-15).log10()
}

fn besr_3band_stereo_db(target: &[[f32; 2]], pred: &[[f32; 2]]) -> (f64, [f64; 3]) {
    let ch_rep = |ch: usize| {
        let t: Vec<f64> = target.iter().map(|v| f64::from(v[ch])).collect();
        let p: Vec<f64> = pred.iter().map(|v| f64::from(v[ch])).collect();
        let (s, e, n) = split_bezout_3band_energies(&t, &p, None, 48_000.0, 512);
        evaluate_besr_from_band_energies(s, e, n)
    };
    let (rl, rr) = (ch_rep(0), ch_rep(1));
    (
        rl.besr_db.max(rr.besr_db),
        std::array::from_fn(|b| rl.band_esr_db[b].max(rr.band_esr_db[b])),
    )
}

fn interleave_lr(frames: &[[f32; 2]]) -> Vec<f32> {
    let mut out = Vec::with_capacity(frames.len() * 2);
    for &[l, r] in frames {
        out.push(l);
        out.push(r);
    }
    out
}

fn assert_0_ulp_stereo(a: &[[f32; 2]], b: &[[f32; 2]], msg: &str) {
    assert_eq!(a.len(), b.len(), "{msg}: length mismatch");
    for (x, y) in a.iter().zip(b) {
        assert_eq!(x[0].to_bits(), y[0].to_bits(), "{msg}: L ULP mismatch");
        assert_eq!(x[1].to_bits(), y[1].to_bits(), "{msg}: R ULP mismatch");
    }
}

#[test]
fn test_oracle_01_256d_schur_recovery_prefix_strip_and_gauge_invariance() {
    // verify[train.pipeline.stage-b-harvest]
    // verify[train.pipeline.stage-c-schur]
    // verify[train.pipeline.isomorphism-parity]
    // verify[train.gram.accuracy-contract]
    // verify[train.gram.wide192]
    let crypto = DefaultCryptoProvider::default();
    let teacher_arena = build_oracle01_teacher_arena();
    let mut teacher_hdr = DnamJsonHeader::default_reference();
    teacher_hdr.io_config.output_mode = "stereo".to_string();
    let teacher_dnam = pack_dnam(&teacher_hdr, &teacher_arena, &[]);

    // 1. Wide192Accum cancellation immunity: [1e8, 2^-30, -1e8] repeated 4,096 times
    let tiny = 2.0_f64.powi(-30);
    let mut wide = Wide192Accum::default();
    let mut naive = 0.0_f64;
    for _ in 0..4_096 {
        wide.add_f64(1.0e8);
        wide.add_f64(tiny);
        wide.add_f64(-1.0e8);
        naive = ((naive + 1.0e8) + tiny) - 1.0e8;
    }
    let expected_sum = 4_096.0 * tiny;
    assert!((wide.to_f64() - expected_sum).abs() <= 1.0e-18);
    assert!((naive - expected_sum).abs() > 1.0e-7);

    // 2. Render 12,000-frame coreset through teacher Ultra model
    let dry_fp16 =
        f32_slice_to_fp16_bytes(&synthesize_rich_coreset_dry(CORESET_FP16_FRAMES_PER_STEM));
    let dry = fp16_bytes_to_f32_vec(&dry_fp16);
    let wet_ultra = render_stereo_dnam(&teacher_dnam, Tier::Ultra, &dry, &[]);
    let dry_lr: Vec<[f32; 2]> = dry.iter().map(|&x| [x, x]).collect();

    // Exact f32 Stage B + Stage C recovery (zero fp16 target quantization): must beat -85 dB
    let mut student_arena = teacher_arena.clone();
    for &h_base in &HEAD_BASES {
        student_arena[h_base..h_base + 2_560].fill(0);
    }
    student_arena[ULTRA_MODES_5_TO_8_BASE..ULTRA_MODES_5_TO_8_BASE + 8_192].fill(0);
    let mut manifest = DnamProjManifest::default_reference();
    manifest.base.io_config.output_mode = "stereo".to_string();
    let role_b = compute_role_b_multipliers(&student_arena);
    let harvest = harvest_star_center_256d(&manifest, &student_arena, &dry_lr, &wet_ultra, &[], 1)
        .expect("Stage B 256-D harvest");
    let stage_c = solve_stage_c_nested_schur_4tier(
        &mut student_arena,
        Some(&harvest),
        &harvest.packed_gram_256,
        &harvest.cross_256,
        &role_b,
    )
    .expect("Stage C nested Schur solve");
    assert_eq!(stage_c.stat_bytes.len(), PROJ_STAT_BYTES_ULTRA);
    let exact_student_dnam = pack_dnam(&manifest.base, &student_arena, &[]);
    let pred_exact_ultra = render_stereo_dnam(&exact_student_dnam, Tier::Ultra, &dry, &[]);
    let esr_exact_db = esr_db_stereo(&wet_ultra, &pred_exact_ultra);
    let (besr_exact_db, besr_exact_bands) = besr_3band_stereo_db(&wet_ultra, &pred_exact_ultra);
    assert!(
        esr_exact_db < -85.0
            && besr_exact_db < -85.0
            && besr_exact_bands.iter().all(|&b| b < -85.0),
        "Oracle-01 exact f32 256-D recovery must exceed -85 dB (ESR={esr_exact_db:.2}, bESR={besr_exact_db:.2}, bands={besr_exact_bands:?})"
    );

    // 3. End-to-end .dnam-proj fp16 capsule solve: Ultra bESR < -60 dB + strict 4-tier monotonicity
    let mut seed_arena = teacher_arena.clone();
    for &h_base in &HEAD_BASES {
        seed_arena[h_base..h_base + 2_560].fill(0);
    }
    let seed_proj =
        build_dnam_proj_container_with_cp1(&mut manifest.clone(), &[], &seed_arena, &[]).unwrap();
    let wet_inter = interleave_lr(&wet_ultra);
    let inpc = build_proj_inpc_capsule(1, &[0x11; 256], &[0x22; 320], &dry_fp16).unwrap();
    let star_spec = make_vault_spec(
        "oracle01_star",
        true,
        BTreeMap::new(),
        &dry,
        &wet_inter,
        vec![stmc_win(0, 0, 0xFF, 0.0)],
        None,
    );
    let solved_proj = add_capture_with_options(
        &seed_proj,
        star_spec,
        Some(&inpc),
        SolverOptions::preserve_backbone(),
    )
    .unwrap();

    let dnam_ultra = export_runtime_dnam(&solved_proj, Some(Tier::Ultra)).unwrap();
    let dnam_pro = export_runtime_dnam(&solved_proj, Some(Tier::Pro)).unwrap();
    let dnam_full = export_runtime_dnam(&solved_proj, Some(Tier::Full)).unwrap();
    let dnam_lite = export_runtime_dnam(&solved_proj, Some(Tier::Lite)).unwrap();

    let solved_view = DnamProjView::from_bytes(&solved_proj, &crypto).unwrap();
    let wght_slice = solved_view.chunk_slice(ProjChunkKind::Wght, 0).unwrap();
    let mut untrimmed_arena = DnamContainerView::from_bytes(wght_slice, &crypto)
        .unwrap()
        .arena_storage
        .as_bytes()
        .to_vec();
    for t in 0..4 {
        write_f32(&mut untrimmed_arena, 10_624 + t * 4, 1.0);
    }
    let untrimmed_dnam = pack_dnam(&manifest.base, &untrimmed_arena, &[]);
    let tier_esrs = [Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra].map(|tier| {
        let pred = render_stereo_dnam(&untrimmed_dnam, tier, &dry, &[]);
        esr_db_stereo(&wet_ultra, &pred)
    });
    assert!(
        tier_esrs[3] < -60.0 && tier_esrs.windows(2).all(|w| w[1] < w[0]),
        "Oracle-01 fp16 4-tier ESR must beat -60 dB and be monotone: {tier_esrs:?}"
    );
    let tiers = [Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra];
    let tier_dnams = [&dnam_lite, &dnam_full, &dnam_pro, &dnam_ultra];
    let mut tier_lufs = [0.0_f32; 4];
    let tier_besr_full: [(f64, [f64; 3]); 4] = std::array::from_fn(|i| {
        let pred = render_stereo_dnam(tier_dnams[i], tiers[i], &dry, &[]);
        tier_lufs[i] = dnam_capture::solver::stage_e::compute_integrated_lufs(&pred[512..]);
        besr_3band_stereo_db(&wet_ultra, &pred)
    });
    let tier_besrs = tier_besr_full.map(|x| x.0);
    let ub = tier_besr_full[3].1;
    println!(
        "[CERT-ORACLE-01] f32={besr_exact_db:.3} {besr_exact_bands:.3?} | fp16={tier_besrs:.3?} {ub:.3?} lufs={tier_lufs:.3?}"
    );
    assert!(
        tier_besrs[3] < -60.0 && tier_besrs.windows(2).all(|w| w[1] < w[0]),
        "Oracle-01 4-tier 3-band Bezout bESR must beat -60 dB and be monotone: {tier_besrs:?}"
    );

    // 4. 0.000000 ULP keyless prefix stripping invariance: Ultra -> Pro -> Full -> Lite
    let stripped_pro = strip_container_to_tier(&dnam_ultra, Tier::Pro).unwrap();
    let stripped_full = strip_container_to_tier(&stripped_pro, Tier::Full).unwrap();
    let stripped_lite = strip_container_to_tier(&stripped_full, Tier::Lite).unwrap();
    for (tier, stripped, direct) in [
        (Tier::Pro, &stripped_pro, &dnam_pro),
        (Tier::Full, &stripped_full, &dnam_full),
        (Tier::Lite, &stripped_lite, &dnam_lite),
    ] {
        let y_s = render_stereo_dnam(stripped, tier, &dry[..512], &[]);
        let y_d = render_stereo_dnam(direct, tier, &dry[..512], &[]);
        assert_0_ulp_stereo(&y_s, &y_d, &format!("prefix strip {tier:?}"));
    }

    // 5. Dyadic modal + Role B cascade gauge (0.000000 ULP) and continuous SO(4) gauge (< 1e-5)
    let mut kappa = [1.0_f32; 32];
    for (i, k) in kappa.iter_mut().enumerate() {
        *k = [0.5_f32, 2.0, 1.0][i % 3];
    }
    let q_ident: [[f32; 4]; 4] =
        std::array::from_fn(|r| std::array::from_fn(|c| if r == c { 1.0 } else { 0.0 }));
    let mut zero_rb_arena = teacher_arena.clone();
    zero_all_role_b_cascade_weights(&mut zero_rb_arena);
    let mut zero_rb_gauged = zero_rb_arena.clone();
    apply_isometric_gauge_watermark(&mut zero_rb_gauged, &kappa, &q_ident).unwrap();

    // Structural 224-D hash is exactly invariant under Mechanism A (modal kappa + SO(4) head gauge)
    let h_orig = extract_structural_hash_224d(&zero_rb_arena).unwrap();
    let h_gaug = extract_structural_hash_224d(&zero_rb_gauged).unwrap();
    assert!(1.0 - structural_cosine_similarity(&h_orig, &h_gaug) < 1.0e-12);

    // Also apply dyadic 2x Role B cascade gauge (+1.0 -> scale_ratio = 0.5) and verify 0.000000 ULP audio
    apply_cascade_head_gauge(&mut zero_rb_gauged, 0, &[[1.0_f32; 4]; 32]).unwrap();
    let d_orig = pack_dnam(&manifest.base, &zero_rb_arena, &[]);
    let d_gaug = pack_dnam(&manifest.base, &zero_rb_gauged, &[]);
    let y0 = render_stereo_dnam(&d_orig, Tier::Ultra, &dry[..256], &[]);
    let y1 = render_stereo_dnam(&d_gaug, Tier::Ultra, &dry[..256], &[]);
    assert_0_ulp_stereo(&y0, &y1, "dyadic modal + Role B gauge");

    let (c, s) = (0.6_f32, 0.8_f32);
    let q_so4 = [
        [c, -s, 0.0, 0.0],
        [s, c, 0.0, 0.0],
        [0.0, 0.0, c, -s],
        [0.0, 0.0, s, c],
    ];
    let mut so4_arena = teacher_arena.clone();
    apply_isometric_gauge_watermark(&mut so4_arena, &[1.0; 32], &q_so4).unwrap();
    let d_so4 = pack_dnam(&manifest.base, &so4_arena, &[]);
    let y_ref = render_stereo_dnam(&teacher_dnam, Tier::Ultra, &dry[..256], &[]);
    let y_so4 = render_stereo_dnam(&d_so4, Tier::Ultra, &dry[..256], &[]);
    for (a, b) in y_ref.iter().zip(&y_so4) {
        assert!((a[0] - b[0]).abs() < 1.0e-5 && (a[1] - b[1]).abs() < 1.0e-5);
    }

    let y_zero_rb = render_stereo_dnam(&d_orig, Tier::Ultra, &dry[..256], &[]);
    let rb_diff: f32 = y_ref
        .iter()
        .zip(&y_zero_rb)
        .map(|(a, b)| (a[0] - b[0]).abs() + (a[1] - b[1]).abs())
        .sum();
    assert!(rb_diff > 1.0e-2, "7,616 Role B floats must be load-bearing");

    let mut fresh_arena = teacher_arena.clone();
    let rep_a = lift_stage_a_backbone(
        &manifest,
        &mut fresh_arena,
        &dry_lr[..2_048],
        &wet_ultra[..2_048],
        None,
        StageAConfig {
            preserve_existing_backbone: false,
            vp_gn_steps: 1,
        },
    );
    assert!(rep_a.role_b_zeroed && !rep_a.preserved_backbone);
    let rb_after = compute_role_b_multipliers(&fresh_arena);
    assert!(
        rb_after
            .iter()
            .all(|m| (m[0] - 1.0).abs() == 0.0 && (m[1] - 1.0).abs() == 0.0)
    );
    write_effective_mode0_weights(&mut fresh_arena, &vec![[0.0; 2]; 256], &rb_after);
}

fn run_oracle_02_for_p(num_controls: usize) {
    let crypto = DefaultCryptoProvider::default();
    let mut arena = build_oracle01_teacher_arena();
    for (t, &h_base) in HEAD_BASES.iter().enumerate() {
        for mode_k in 1..=4 {
            let off = h_base + mode_k * 512;
            let sc = 0.14_f32 / ((t + 1) as f32);
            for i in 0..64 {
                let wl = sc * (((i * (mode_k * 3 + 1) + t) % 11) as f32 - 5.0) / 5.0;
                let wr = sc * (((i * (mode_k * 5 + 2) + t * 3) % 11) as f32 - 5.0) / 5.0;
                write_f32(&mut arena, off + i * 4, wl);
                write_f32(&mut arena, off + 256 + i * 4, wr);
            }
        }
    }
    write_f32(&mut arena, 7_552 + 3 * 4, 0.85);

    let mut manifest = DnamProjManifest::default_reference();
    manifest.base.io_config.output_mode = "stereo".to_string();
    let mut schema = vec![
        make_range_ctrl("drive_asym_neg", -1.0, 1.0, -0.35),
        make_range_ctrl("tone_asym_pos", -1.0, 1.0, 0.40),
        make_switch_ctrl("voice_sw0", &["Clean", "Crunch", "Lead"], 0),
        make_switch_ctrl("bright_sw1", &["Dark", "Normal", "Bright"], 1),
    ];
    for p in 4..num_controls {
        schema.push(make_range_ctrl(&format!("knob_{p}"), 0.0, 10.0, 5.0));
    }
    manifest.base.controls_schema = schema;

    let range_slot = |c_idx: u16, d: usize, l0: f32, l1: f32, vb0: f32| {
        let mut s = Cp1ContributorSlot::zero();
        s.lanes = 2;
        s.control_idx = c_idx;
        s.option_idx = 0xFF;
        s.main_base[0][d] = l0;
        s.main_base[1][d] = l1;
        s.emb = [1.0, 0.0, 0.0];
        s.v_base[0] = vb0;
        s
    };
    let mut teacher_slots = vec![
        range_slot(0, 0, 0.90, 0.35, 0.90),
        range_slot(1, 1, -0.80, 0.30, 0.80),
    ];
    for &(c_idx, opt_idx, m2) in &[
        (2u16, 1u8, 0.75_f32),
        (2, 2, -0.65),
        (3, 0, -0.55),
        (3, 2, 0.70),
    ] {
        let mut sw = Cp1ContributorSlot::zero();
        sw.lanes = 1;
        sw.ctrl_kind = 1;
        sw.control_idx = c_idx;
        sw.option_idx = opt_idx;
        sw.main_base[0][2] = m2;
        sw.emb = [1.0, 0.0, 0.0];
        teacher_slots.push(sw);
    }
    teacher_slots.push(range_slot((num_controls - 1) as u16, 2, 0.60, -0.25, 0.0));

    let teacher_dnam = pack_dnam(&manifest.base, &arena, &teacher_slots);

    let dry_fp16 =
        f32_slice_to_fp16_bytes(&synthesize_rich_coreset_dry(CORESET_FP16_FRAMES_PER_STEM));
    let dry = fp16_bytes_to_f32_vec(&dry_fp16);
    let inpc = build_proj_inpc_capsule(1, &[0x22; 256], &[0x33; 320], &dry_fp16).unwrap();

    let mut seed_arena = arena.clone();
    for &h_base in &HEAD_BASES {
        seed_arena[h_base..h_base + 2_560].fill(0);
    }
    seed_arena[ULTRA_MODES_5_TO_8_BASE..ULTRA_MODES_5_TO_8_BASE + 8_192].fill(0);
    seed_arena[7_552..7_680].fill(0);
    let seed_proj =
        build_dnam_proj_container_with_cp1(&mut manifest.clone(), &[], &seed_arena, &[]).unwrap();

    let star_vals = star_center_control_values(&manifest);
    assert!((star_vals[0] - (-0.35)).abs() < 1.0e-6 && (star_vals[1] - 0.40).abs() < 1.0e-6);
    assert!((star_vals[2] - 0.0).abs() < 1.0e-6 && (star_vals[3] - 1.0).abs() < 1.0e-6);

    let make_stem = |stem_id: &str,
                     is_star: bool,
                     overrides: &[(usize, f32)],
                     windows: Vec<StmcTrajectoryWindow>| {
        let mut full_ctrls: Vec<(usize, f32)> = star_vals.iter().copied().enumerate().collect();
        let mut meta = BTreeMap::new();
        for (idx, ctrl_def) in manifest.base.controls_schema.iter().enumerate() {
            let v = overrides
                .iter()
                .find_map(|&(i, val)| (i == idx).then_some(val))
                .unwrap_or(star_vals[idx]);
            full_ctrls[idx] = (idx, v);
            match ctrl_def.kind {
                ControlKind::Range => {
                    meta.insert(ctrl_def.id.clone(), serde_json::json!(f64::from(v)));
                }
                ControlKind::Switch => {
                    let name = &ctrl_def.declared_options.as_ref().unwrap()[v.round() as usize];
                    meta.insert(ctrl_def.id.clone(), serde_json::json!(name));
                }
            }
        }
        let wet_lr = render_stereo_dnam(&teacher_dnam, Tier::Ultra, &dry, &full_ctrls);
        make_vault_spec(
            stem_id,
            is_star,
            meta,
            &dry,
            &interleave_lr(&wet_lr),
            windows,
            None,
        )
    };

    let star_spec = make_stem("star_0", true, &[], vec![stmc_win(0, 0, 0xFF, -0.35)]);
    let proj_star_only = solve_captures_batch_with_options(
        &seed_proj,
        vec![star_spec.clone()],
        Some(&inpc),
        SolverOptions::preserve_backbone(),
    )
    .unwrap();
    let dnam_star_only = export_runtime_dnam(&proj_star_only, Some(Tier::Ultra)).unwrap();
    let star_pairs: Vec<(usize, f32)> = star_vals.iter().copied().enumerate().collect();
    let star_ref_audio = render_stereo_dnam(&dnam_star_only, Tier::Ultra, &dry[..512], &star_pairs);

    let mut batch = vec![star_spec];
    for &(sid, p_idx, val) in &[
        ("c0_lo", 0usize, -1.0_f32),
        ("c0_hi", 0, 1.0),
        ("c1_lo", 1, -1.0),
        ("c1_hi", 1, 1.0),
        ("last_lo", num_controls - 1, 0.0),
        ("last_hi", num_controls - 1, 10.0),
    ] {
        batch.push(make_stem(
            sid,
            false,
            &[(p_idx, val)],
            vec![stmc_win(p_idx as u16, 0, 0xFF, val)],
        ));
    }
    for &(sid, sw_idx, opt) in &[
        ("sw0_crunch", 2usize, 1u8),
        ("sw0_lead", 2, 2),
        ("sw1_dark", 3, 0),
        ("sw1_bright", 3, 2),
    ] {
        batch.push(make_stem(
            sid,
            false,
            &[(sw_idx, f32::from(opt))],
            vec![stmc_win(sw_idx as u16, 1, opt, f32::from(opt))],
        ));
    }
    for &(sid, u0, u1) in &[
        ("had_pp", 1.0_f32, 1.0_f32),
        ("had_pm", 1.0, -1.0),
        ("had_mp", -1.0, 1.0),
        ("had_mm", -1.0, -1.0),
    ] {
        batch.push(make_stem(
            sid,
            false,
            &[(0, u0), (1, u1)],
            vec![stmc_win(0, 0, 0xFF, u0), stmc_win(1, 0, 0xFF, u1)],
        ));
    }

    let proj_multi = solve_captures_batch_with_options(
        &seed_proj,
        batch,
        Some(&inpc),
        SolverOptions::preserve_backbone(),
    )
    .unwrap();
    let view_multi = DnamProjView::from_bytes(&proj_multi, &crypto).unwrap();
    assert_eq!(view_multi.manifest.base.controls_schema.len(), num_controls);

    let dnam_multi_ultra = export_runtime_dnam(&proj_multi, Some(Tier::Ultra)).unwrap();
    let dnam_multi_pro = export_runtime_dnam(&proj_multi, Some(Tier::Pro)).unwrap();
    let stripped_multi_pro = strip_container_to_tier(&dnam_multi_ultra, Tier::Pro).unwrap();

    // 1. 0.000000 ULP StarCenter invariance on multi-knob model
    let star_multi_audio =
        render_stereo_dnam(&dnam_multi_ultra, Tier::Ultra, &dry[..512], &star_pairs);
    assert_0_ulp_stereo(&star_ref_audio, &star_multi_audio, "StarCenter 0 ULP");

    // 2. Multi-knob, non-zero-default switch, P > 8 knob, 2D Hadamard FM + 0 ULP prefix strip
    for test_overrides in [
        vec![(0usize, 1.0_f32), (1, 1.0)],
        vec![(0, 1.0), (1, -1.0)],
        vec![(2, 2.0), (3, 0.0)],
        vec![(num_controls - 1, 10.0)],
    ] {
        let mut full_c = star_pairs.clone();
        for &(i, v) in &test_overrides {
            full_c[i] = (i, v);
        }
        let y_teacher = render_stereo_dnam(&teacher_dnam, Tier::Ultra, &dry[..2_048], &full_c);
        let y_student = render_stereo_dnam(&dnam_multi_ultra, Tier::Ultra, &dry[..2_048], &full_c);
        let corner_esr = esr_db_stereo(&y_teacher, &y_student);
        assert!(
            corner_esr < -28.0,
            "P={num_controls} corner {test_overrides:?} ESR must beat -28 dB, got {corner_esr:.2} dB"
        );

        let y_pro_strip = render_stereo_dnam(&stripped_multi_pro, Tier::Pro, &dry[..256], &full_c);
        let y_pro_direct = render_stereo_dnam(&dnam_multi_pro, Tier::Pro, &dry[..256], &full_c);
        assert_0_ulp_stereo(&y_pro_strip, &y_pro_direct, "Oracle-02 Pro strip");
    }
}

#[test]
fn test_oracle_02_cp1_arbitrary_p6_and_p16_asymmetric_switches_and_hadamard_fm() {
    // verify[train.pipeline.stage-d-control]
    // verify[train.pipeline.dispatch-solve]
    run_oracle_02_for_p(6);
    run_oracle_02_for_p(16);
}

#[test]
fn test_oracle_03_zero_wav_dnam_proj_vault_fastpath_and_clamping() {
    // verify[train.pipeline.stage-e-pack]
    // verify[train.resume.kronecker-reassembly]
    // verify[train.cache.proj-vault]
    let crypto = DefaultCryptoProvider::default();
    let mut manifest = DnamProjManifest::default_reference();
    manifest.base.io_config.output_mode = "stereo".to_string();
    manifest.base.controls_schema = vec![
        make_range_ctrl("gain", 0.0, 10.0, 5.0),
        make_switch_ctrl("channel", &["Clean", "Crunch", "Lead"], 0),
    ];
    let empty_proj = build_dnam_proj_container(&manifest, &[]).unwrap();

    let dry = synthesize_rich_coreset_dry(CORESET_FP16_FRAMES_PER_STEM);
    let wet_star: Vec<f32> = dry
        .iter()
        .flat_map(|&x| [(1.4 * x).tanh(), (1.2 * x).tanh()])
        .collect();
    let mut spcm_raw = vec![0u8; 64 + 96_000 * 4];
    spcm_raw[0..4].copy_from_slice(b"SPCM");
    spcm_raw[4..6].copy_from_slice(&2u16.to_le_bytes());

    let dry_fp16 = f32_slice_to_fp16_bytes(&dry);
    let inpc = build_proj_inpc_capsule(1, &[0x44; 256], &[0x55; 320], &dry_fp16).unwrap();
    let star_spec = make_vault_spec(
        "star_0",
        true,
        ctrl_meta(5.0, "Clean"),
        &dry,
        &wet_star,
        vec![stmc_win(0, 0, 0xFF, 5.0)],
        Some(spcm_raw),
    );
    let proj_with_spcm = add_capture(&empty_proj, star_spec, Some(&inpc)).unwrap();
    assert!(proj_with_spcm.len() > 65_536);

    let range_hdr = parse_proj_manifest_range(&proj_with_spcm[..65_536]).unwrap();
    assert_ne!(
        range_hdr.preamble.lifecycle_flags & PROJ_FLAG_HAS_RAW_STEMS,
        0
    );

    let compacted = compact_to_capsules(&proj_with_spcm).unwrap();
    let comp_view = DnamProjView::from_bytes(&compacted, &crypto).unwrap();
    let comp_flags = comp_view.preamble.lifecycle_flags;
    assert_eq!(comp_flags & PROJ_FLAG_HAS_RAW_STEMS, 0);
    assert_ne!(comp_flags & PROJ_FLAG_COMPACTED_CAPSULES, 0);
    assert!(comp_view.chunk_slice(ProjChunkKind::Spcm, 0).is_none());
    let stat_star_orig = comp_view
        .chunk_slice(ProjChunkKind::Stat, 0)
        .unwrap()
        .to_vec();
    let dnam_star_orig = export_runtime_dnam(&compacted, Some(Tier::Ultra)).unwrap();
    let star_ctrls = [(0, 5.0), (1, 0.0)];
    let audio_star_orig =
        render_stereo_dnam(&dnam_star_orig, Tier::Ultra, &dry[..512], &star_ctrls);

    let wet_off: Vec<f32> = dry
        .iter()
        .flat_map(|&x| [(3.2 * x).tanh(), (2.9 * x).tanh()])
        .collect();
    let off_spec = make_vault_spec(
        "off_lead_9",
        false,
        ctrl_meta(9.0, "Lead"),
        &dry,
        &wet_off,
        vec![stmc_win(0, 0, 0xFF, 9.0)],
        None,
    );
    let proj_added = add_capture(&compacted, off_spec, None).unwrap();
    let added_view = DnamProjView::from_bytes(&proj_added, &crypto).unwrap();
    let added_schema = &added_view.manifest.base.controls_schema;
    assert_eq!(added_schema[0].safe_clamped_range, Some([5.0, 9.0]));
    assert_eq!(
        added_schema[1].active_options,
        Some(vec!["Clean".to_string(), "Lead".to_string()])
    );
    assert_eq!(
        added_view.chunk_slice(ProjChunkKind::Stat, 0).unwrap(),
        stat_star_orig.as_slice()
    );
    let dnam_added = export_runtime_dnam(&proj_added, Some(Tier::Ultra)).unwrap();
    let audio_star_after_add =
        render_stereo_dnam(&dnam_added, Tier::Ultra, &dry[..512], &star_ctrls);
    assert_0_ulp_stereo(&audio_star_orig, &audio_star_after_add, "Path 1 StarCenter");

    let proj_muted = set_capture_enabled(&proj_added, "off_lead_9", false).unwrap();
    let muted_view = DnamProjView::from_bytes(&proj_muted, &crypto).unwrap();
    let muted_schema = &muted_view.manifest.base.controls_schema;
    assert_eq!(muted_schema[0].safe_clamped_range, Some([5.0, 5.0]));
    assert_eq!(
        muted_schema[1].active_options,
        Some(vec!["Clean".to_string()])
    );
    let proj_removed = remove_capture(&proj_muted, "off_lead_9").unwrap();

    let wet_star_2: Vec<f32> = dry.iter().flat_map(|&x| [0.55 * x, -0.45 * x]).collect();
    let rep_spec = make_vault_spec(
        "star_0",
        true,
        ctrl_meta(3.0, "Crunch"),
        &dry,
        &wet_star_2,
        vec![stmc_win(0, 0, 0xFF, 3.0)],
        None,
    );
    let proj_rep = replace_capture(&proj_removed, "star_0", rep_spec).unwrap();
    let rep_view = DnamProjView::from_bytes(&proj_rep, &crypto).unwrap();
    assert_ne!(
        rep_view.chunk_slice(ProjChunkKind::Stat, 0).unwrap(),
        stat_star_orig.as_slice()
    );
    for tier in [Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra] {
        let rt = export_runtime_dnam(&proj_rep, Some(tier)).unwrap();
        let out = render_stereo_dnam(&rt, tier, &dry[..128], &[(0, 3.0), (1, 1.0)]);
        assert!(out.iter().all(|s| s[0].is_finite() && s[1].is_finite()));
    }
}
