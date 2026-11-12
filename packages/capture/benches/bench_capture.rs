#![forbid(unsafe_code)]
//! Criterion benchmark suite for `dnam-capture` `uplo='U'` Cholesky solver,
//! `< 1.5 ms` `AdaptiveStimulusController::next_frame`, `.dnam-proj` `compact_to_capsules`,
//! and `< 0.2 ms` `export_runtime_dnam` emission.

use std::collections::BTreeMap;

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use dnam_capture::proj::{
    DnamProjManifest, ProjChunkInput, ProjChunkKind, StemProvenance, StmcTrajectoryWindow,
    VaultCaptureSpec, add_capture, build_dnam_proj_container, build_dnam_proj_container_with_cp1,
    build_proj_inpc_capsule, compact_to_capsules, export_runtime_dnam,
};
use dnam_capture::solver::{
    accumulate_uplo_u_gram_and_cross, solve_uplo_u_cholesky_2ch, uplo_u_packed_len,
};
use dnam_capture::stimulus::{AdaptiveStimulusController, StimulusConfig};
use dnam_capture::targets::StemResponse;
use dnam_capture::{CORESET_FP16_FRAMES_PER_STEM, f32_to_fp16_bits};
use dnam_engine::container::ARENA_OFFSET_ULTRA_END;

fn bench_uplo_u_cholesky_lite(c: &mut Criterion) {
    let n = 64usize;
    let mut g = vec![0.0_f64; uplo_u_packed_len(n)];
    let mut cross = vec![[0.0_f64; 2]; n];
    let mut row = vec![0.0_f64; n];
    for (i, item) in row.iter_mut().enumerate() {
        *item = ((i as f64) * 0.13).sin();
    }
    accumulate_uplo_u_gram_and_cross(&mut g, &mut cross, &row, [0.5, -0.25], 1.0);

    c.bench_function("capture_uplo_u_cholesky_64d", |b| {
        b.iter(|| {
            solve_uplo_u_cholesky_2ch(black_box(&g), black_box(&cross), black_box(1.0e-6))
                .expect("benchmark Cholesky solve must succeed")
        });
    });

    let n_ultra = 256usize;
    let mut g_ultra = vec![0.0_f64; uplo_u_packed_len(n_ultra)];
    let mut cross_ultra = vec![[0.0_f64; 2]; n_ultra];
    let row_ultra: Vec<f64> = (0..n_ultra).map(|i| ((i as f64) * 0.09).cos()).collect();
    accumulate_uplo_u_gram_and_cross(&mut g_ultra, &mut cross_ultra, &row_ultra, [0.4, -0.3], 1.0);
    c.bench_function("capture_uplo_u_cholesky_256d_autopatch", |b| {
        b.iter(|| {
            solve_uplo_u_cholesky_2ch(
                black_box(&g_ultra),
                black_box(&cross_ultra),
                black_box(1.0e-4),
            )
            .expect("256-D auto-patch Cholesky solve must succeed")
        });
    });
}

fn bench_adaptive_stimulus_next_frame(c: &mut Criterion) {
    let cfg = StimulusConfig {
        fixed_duration: true,
        max_frames: 1_000_000,
        ..StimulusConfig::default()
    };
    let mut seed_ctrl = AdaptiveStimulusController::new(cfg.clone());
    let first = seed_ctrl.next_frame(None).expect("initial stimulus frame");
    let wet_samples: Vec<f32> = first.in_1.iter().map(|&x| (1.5 * x).tanh()).collect();
    let wet_resp = StemResponse {
        stem_id: "bench_stem".to_string(),
        sample_rate_hz: 48_000,
        out_1: wet_samples,
        out_2: None,
        out_3: None,
        out_4: None,
        internal_nodes: std::collections::BTreeMap::new(),
    };

    c.bench_function("capture_adaptive_stimulus_next_frame_96ms", |b| {
        let mut ctrl = AdaptiveStimulusController::new(cfg.clone());
        let _ = ctrl.next_frame(None);
        b.iter(|| {
            let frame = match ctrl.next_frame(Some(black_box(&wet_resp))) {
                Some(f) => f,
                None => {
                    ctrl = AdaptiveStimulusController::new(cfg.clone());
                    let _ = ctrl.next_frame(None);
                    ctrl.next_frame(Some(black_box(&wet_resp)))
                        .expect("fresh controller must emit frame")
                }
            };
            black_box(frame)
        });
    });
}

fn bench_dnam_proj_compact_and_export(c: &mut Criterion) {
    let mut manifest = DnamProjManifest::default_reference();
    let arena = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
    let mut spcm = vec![0u8; 48_000 * 4];
    spcm[0..4].copy_from_slice(b"SPCM");
    let proj_with_spcm = build_dnam_proj_container_with_cp1(
        &mut manifest,
        &[ProjChunkInput {
            kind: ProjChunkKind::Spcm,
            chunk_id: 0,
            payload: spcm,
        }],
        &arena,
        &[],
    )
    .expect("benchmark .dnam-proj build must succeed");

    c.bench_function("capture_proj_compact_to_capsules", |b| {
        b.iter(|| {
            compact_to_capsules(black_box(&proj_with_spcm))
                .expect("benchmark compact_to_capsules must succeed")
        });
    });

    let compacted_proj =
        compact_to_capsules(&proj_with_spcm).expect("benchmark compact_to_capsules must succeed");
    c.bench_function("capture_proj_export_runtime_dnam", |b| {
        b.iter(|| {
            export_runtime_dnam(black_box(&compacted_proj), None)
                .expect("benchmark export_runtime_dnam must succeed")
        });
    });
}

fn bench_5stage_proj_vault_solver(c: &mut Criterion) {
    let mut manifest = DnamProjManifest::default_reference();
    manifest.base.controls_schema = vec![
        serde_json::from_value(serde_json::json!({
            "id": "gain", "name": "Gain", "type": "range", "lobatto_degree": 2,
            "declared_range": [0.0, 10.0], "safe_clamped_range": [0.0, 10.0], "default": 5.0
        }))
        .expect("range control"),
    ];
    let empty_proj = build_dnam_proj_container(&manifest, &[]).expect("empty proj");
    let dry: Vec<f32> = (0..CORESET_FP16_FRAMES_PER_STEM)
        .map(|i| 0.45 * ((i as f32) * 0.058).sin() + 0.25 * ((i as f32) * 0.19).cos())
        .collect();
    let fp16_bytes = |s: &[f32]| -> Vec<u8> {
        let mut out = Vec::with_capacity(s.len() * 2);
        for &v in s {
            out.extend_from_slice(&f32_to_fp16_bits(v).to_le_bytes());
        }
        out
    };
    let inpc = build_proj_inpc_capsule(1, &[0x11; 256], &[0x22; 320], &fp16_bytes(&dry))
        .expect("inpc capsule");
    let make_spec = |id: &str, is_star: bool, g: f32| VaultCaptureSpec {
        stem_id: id.to_string(),
        is_star_center: is_star,
        enabled: true,
        pad_gain_f32: 1.0,
        fractional_lag_samples_f32: 0.0,
        controls_meta: BTreeMap::from([("gain".to_string(), serde_json::json!(f64::from(g)))]),
        provenance: StemProvenance {
            input_pcm_sha256: "0".repeat(64),
            target_pcm_sha256: "1".repeat(64),
            num_frames: CORESET_FP16_FRAMES_PER_STEM as u64,
        },
        user_metadata: BTreeMap::new(),
        extracted_physics: None,
        windows: vec![StmcTrajectoryWindow {
            control_id: 0,
            ctrl_kind: 0,
            active_opt: 0xFF,
            u_phys: g,
            window_weight: 1.0,
            emr_velocity: 0.0,
        }],
        band_psd_summary_88: [0u8; 88],
        n_comb_bins: 32,
        spectral_summary_4096: [0u8; 4_096],
        wet_coreset_fp16: fp16_bytes(
            &dry.iter()
                .map(|&x| ((0.5 + 0.25 * g) * x).tanh())
                .collect::<Vec<_>>(),
        ),
        optional_spcm: None,
    };
    let star_spec = make_spec("star_0", true, 5.0);
    let seeded_proj = add_capture(&empty_proj, star_spec.clone(), Some(&inpc)).expect("seed solve");
    let off_spec = make_spec("off_9", false, 9.0);

    c.bench_function("capture_solver_path2_full_5stage", |b| {
        b.iter(|| {
            add_capture(
                black_box(&empty_proj),
                black_box(star_spec.clone()),
                Some(black_box(&inpc)),
            )
            .expect("Path 2 5-stage solve must succeed")
        });
    });

    c.bench_function("capture_solver_path1_fastpath_stage_de", |b| {
        b.iter(|| {
            add_capture(black_box(&seeded_proj), black_box(off_spec.clone()), None)
                .expect("Path 1 fast-path solve must succeed")
        });
    });
}

criterion_group!(
    benches,
    bench_uplo_u_cholesky_lite,
    bench_adaptive_stimulus_next_frame,
    bench_dnam_proj_compact_and_export,
    bench_5stage_proj_vault_solver
);
criterion_main!(benches);
