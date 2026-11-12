#![forbid(unsafe_code)]
//! Fail-fast validation, corruption rejection, and boundary-condition tests for
//! in-memory 5-bit-depth WAV / sub-sample alignment (`align`), `DnamProjManifest` (`proj::schema`),
//! and binary payload capsules `CALB`, `INPC`, `STMC`, and `STAT` (`proj::capsules`).

use std::collections::BTreeMap;

use dnam_capture::CaptureBitDepth;
use dnam_capture::align::{
    AlignError, compute_pcm_sha256, decode_wav_bytes, encode_wav_12bit, encode_wav_16bit,
    encode_wav_24bit, encode_wav_32bit_float, encode_wav_32bit_int, encode_wav_with_bit_depth,
    estimate_subsample_lag_and_gain,
};
use dnam_capture::proj::{
    AuditReport, CalbChunk64B, CalibrationSummary, DnamProjManifest, INPC_MONO_BYTES,
    PROJ_STAT_BYTES_FULL, PROJ_STAT_BYTES_LITE, PROJ_STAT_BYTES_PRO, PROJ_STAT_BYTES_ULTRA,
    STMC_FLAG_ENABLED, SolverTelemetrySummary, StemProvenance, StemRegistryEntry,
    StmcTrajectoryWindow, build_proj_inpc_capsule, build_proj_stmc_capsule,
    verify_proj_inpc_capsule, verify_proj_stat_uplo_u, verify_proj_stmc_capsule,
};
use dnam_engine::container::crc32c_compute;

#[test]
fn test_align_wav_codec_and_subsample_error_and_boundary_paths() {
    let e_wav = AlignError::InvalidWav("bad header".to_string());
    let e_aln = AlignError::InvalidAlignmentInput("bad len".to_string());
    assert!(e_wav.to_string().contains("invalid WAV buffer: bad header"));
    assert!(
        e_aln
            .to_string()
            .contains("invalid alignment input: bad len")
    );

    // Zero channels, zero sample rate, and incomplete multi-channel slices rejected across encoders
    assert!(encode_wav_12bit(&[0.0, 0.0], 0, 1).is_err());
    assert!(encode_wav_12bit(&[0.0, 0.0], 48_000, 0).is_err());
    assert!(encode_wav_12bit(&[0.1, 0.2, 0.3], 48_000, 2).is_err());
    assert!(encode_wav_16bit(&[0.0, 0.0], 48_000, 0).is_err());
    assert!(encode_wav_16bit(&[0.1, 0.2, 0.3], 48_000, 2).is_err());
    assert!(encode_wav_24bit(&[0.1], 0, 1).is_err());
    assert!(encode_wav_32bit_int(&[0.1, 0.2, 0.3], 48_000, 2).is_err());
    assert!(encode_wav_32bit_float(&[0.1, 0.2, 0.3], 48_000, 2).is_err());

    // Non-finite inputs (NaN / +/-Inf) sanitize to 0.0; odd byte length (1x 24-bit = 3 B) pads cleanly
    let non_finite = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.25];
    for depth in [
        CaptureBitDepth::Int12,
        CaptureBitDepth::Int16,
        CaptureBitDepth::Int24,
        CaptureBitDepth::Int32,
        CaptureBitDepth::Float32,
    ] {
        let enc = encode_wav_with_bit_depth(&non_finite, 48_000, 1, depth).unwrap();
        let dec = decode_wav_bytes(&enc).unwrap();
        assert_eq!(dec.samples[0], 0.0);
        assert_eq!(dec.samples[1], 0.0);
        assert_eq!(dec.samples[2], 0.0);
        assert!((dec.samples[3] - 0.25).abs() < 1.0e-3);
    }
    let odd_24 = encode_wav_24bit(&[0.125], 48_000, 1).unwrap();
    assert_eq!(odd_24.len() % 2, 0);
    assert_eq!(decode_wav_bytes(&odd_24).unwrap().samples.len(), 1);

    // 12-bit stereo (mask 0x03) and 4-channel (mask 0x00) round-trip
    let w12_st = encode_wav_12bit(&[0.25, -0.25, 0.5, -0.5], 48_000, 2).unwrap();
    assert_eq!(decode_wav_bytes(&w12_st).unwrap().channels, 2);
    let w12_4c = encode_wav_12bit(&[0.1, -0.1, 0.2, -0.2], 48_000, 4).unwrap();
    assert_eq!(decode_wav_bytes(&w12_4c).unwrap().channels, 4);

    // Canonical PCM SHA-256 normalizes NaN and -0.0 to +0.0
    assert_eq!(
        compute_pcm_sha256(&[f32::NAN, -0.0, 0.5]),
        compute_pcm_sha256(&[0.0, 0.0, 0.5])
    );

    // RIFF / fmt / data structural corruption rejections
    assert!(decode_wav_bytes(b"short").is_err());
    let valid_16 = encode_wav_16bit(&[0.1, -0.1, 0.2, -0.2], 48_000, 2).unwrap();
    let mut bad_riff = valid_16.clone();
    bad_riff[0..4].copy_from_slice(b"NOPE");
    assert!(decode_wav_bytes(&bad_riff).is_err());

    let mut bad_size = valid_16.clone();
    bad_size[4..8].copy_from_slice(&2_u32.to_le_bytes());
    assert!(decode_wav_bytes(&bad_size).is_err());

    let mut bad_ch = valid_16.clone();
    bad_ch[22..24].copy_from_slice(&0_u16.to_le_bytes());
    assert!(decode_wav_bytes(&bad_ch).is_err());

    let mut bad_bits = valid_16.clone();
    bad_bits[34..36].copy_from_slice(&20_u16.to_le_bytes());
    assert!(decode_wav_bytes(&bad_bits).is_err());

    let mut bad_tag = valid_16.clone();
    bad_tag[20..22].copy_from_slice(&7_u16.to_le_bytes());
    assert!(decode_wav_bytes(&bad_tag).is_err());

    let mut bad_align = valid_16.clone();
    bad_align[32..34].copy_from_slice(&6_u16.to_le_bytes());
    assert!(decode_wav_bytes(&bad_align).is_err());

    let mut bad_rate = valid_16.clone();
    bad_rate[28..32].copy_from_slice(&123_u32.to_le_bytes());
    assert!(decode_wav_bytes(&bad_rate).is_err());

    // Incomplete stereo frame in data chunk (2 bytes instead of multiple of 4)
    let mut odd_stereo_data = valid_16.clone();
    odd_stereo_data[40..44].copy_from_slice(&2_u32.to_le_bytes());
    odd_stereo_data[4..8].copy_from_slice(&38_u32.to_le_bytes());
    odd_stereo_data.truncate(46);
    assert!(decode_wav_bytes(&odd_stereo_data).is_err());

    // 32-bit float NaN sample & non-multiple-of-4 data length rejected
    let mut nan_f32_wav = encode_wav_32bit_float(&[0.1, 0.2], 48_000, 1).unwrap();
    nan_f32_wav[44..48].copy_from_slice(&f32::NAN.to_le_bytes());
    assert!(decode_wav_bytes(&nan_f32_wav).is_err());

    let mut trunc_f32_wav = encode_wav_32bit_float(&[0.1, 0.2], 48_000, 1).unwrap();
    trunc_f32_wav[40..44].copy_from_slice(&6_u32.to_le_bytes());
    trunc_f32_wav[4..8].copy_from_slice(&42_u32.to_le_bytes());
    trunc_f32_wav.truncate(50);
    assert!(decode_wav_bytes(&trunc_f32_wav).is_err());

    // WAVE_FORMAT_EXTENSIBLE edge cases: valid_bits=0, valid_bits > bits, bad SubFormat, bad GUID
    let mut ext_zero_valid = w12_st.clone();
    ext_zero_valid[38..40].copy_from_slice(&0_u16.to_le_bytes());
    assert!(decode_wav_bytes(&ext_zero_valid).is_ok());

    let mut ext_too_many_valid = w12_st.clone();
    ext_too_many_valid[38..40].copy_from_slice(&24_u16.to_le_bytes());
    assert!(decode_wav_bytes(&ext_too_many_valid).is_err());

    let mut ext_bad_sub = w12_st.clone();
    ext_bad_sub[44..46].copy_from_slice(&99_u16.to_le_bytes());
    assert!(decode_wav_bytes(&ext_bad_sub).is_err());

    let mut ext_bad_guid = w12_st;
    ext_bad_guid[50] ^= 0xFF;
    assert!(decode_wav_bytes(&ext_bad_guid).is_err());

    // Missing data / data-before-fmt / truncated fmt
    let mut no_data = Vec::new();
    no_data.extend_from_slice(b"RIFF");
    no_data.extend_from_slice(&28_u32.to_le_bytes());
    no_data.extend_from_slice(b"WAVEfmt ");
    no_data.extend_from_slice(&16_u32.to_le_bytes());
    no_data.extend_from_slice(&valid_16[20..36]);
    assert!(decode_wav_bytes(&no_data).is_err());

    let mut data_first = Vec::new();
    data_first.extend_from_slice(b"RIFF");
    data_first.extend_from_slice(&16_u32.to_le_bytes());
    data_first.extend_from_slice(b"WAVEdata");
    data_first.extend_from_slice(&4_u32.to_le_bytes());
    data_first.extend_from_slice(&[0; 4]);
    assert!(decode_wav_bytes(&data_first).is_err());

    let mut trunc_fmt = Vec::new();
    trunc_fmt.extend_from_slice(b"RIFF");
    trunc_fmt.extend_from_slice(&22_u32.to_le_bytes());
    trunc_fmt.extend_from_slice(b"WAVEfmt ");
    trunc_fmt.extend_from_slice(&10_u32.to_le_bytes());
    trunc_fmt.extend_from_slice(&[0; 10]);
    assert!(decode_wav_bytes(&trunc_fmt).is_err());

    // Sub-sample alignment error and boundary-lag paths
    assert!(estimate_subsample_lag_and_gain(&[1.0; 4], &[1.0; 4], 2).is_err());
    assert!(estimate_subsample_lag_and_gain(&[1.0; 16], &[1.0; 15], 2).is_err());
    let mut nan_sig = vec![0.1_f32; 16];
    nan_sig[3] = f32::NAN;
    assert!(estimate_subsample_lag_and_gain(&nan_sig, &[0.1; 16], 2).is_err());
    assert!(estimate_subsample_lag_and_gain(&[0.0; 16], &[0.1; 16], 4).is_err());

    // Impulse at boundary lag (lag == max_lag -> fractional refinement skipped cleanly) & inverted polarity
    let mut dry_imp = vec![0.0_f32; 32];
    let mut wet_imp = vec![0.0_f32; 32];
    dry_imp[8] = 1.0;
    wet_imp[11] = -0.5;
    let aln = estimate_subsample_lag_and_gain(&dry_imp, &wet_imp, 3).unwrap();
    assert_eq!(aln.integer_lag_samples, 3);
    assert_eq!(aln.fractional_lag_samples, 0.0);
    assert!(aln.polarity_inverted);
}

#[test]
fn test_proj_manifest_schema_validation_and_jcs_edges() {
    let valid_sha = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    let mut m = DnamProjManifest::default_reference();
    m.version = "1.0.0".to_string();
    assert!(m.validate().is_err());

    let mut m = DnamProjManifest::default_reference();
    m.container_kind = "dnam".to_string();
    assert!(m.validate().is_err());

    let mut m = DnamProjManifest::default_reference();
    m.base.container_kind = "dnam-proj".to_string();
    assert!(m.validate().is_err());

    let mut m = DnamProjManifest::default_reference();
    m.base.version = "1.0.0".to_string();
    assert!(m.validate().is_err());

    let mut m = DnamProjManifest::default_reference();
    m.rig_routing.sample_rate_device_hz = 4_000;
    assert!(m.validate().is_err());

    let mut m = DnamProjManifest::default_reference();
    m.rig_routing.bit_depth = "int8".to_string();
    assert!(m.validate().is_err());

    let mut m = DnamProjManifest::default_reference();
    m.rig_routing.reamp_send_channel = 0;
    assert!(m.validate().is_err());

    let mut m = DnamProjManifest::default_reference();
    m.calibration_summary = Some(CalibrationSummary {
        loopback_lag_samples: f64::NAN,
        signed_gain: 1.0,
        polarity_inverted: false,
        input_reference_dbfs: -20.5,
        noise_floor_dbfs: -96.0,
    });
    assert!(m.validate().is_err());

    let mk_stem =
        |id: &str, cid: u32, en: bool, star: bool, sha: &str, frames: u64| StemRegistryEntry {
            stem_id: id.to_string(),
            chunk_id: cid,
            enabled: en,
            is_star_center: star,
            controls_meta: BTreeMap::new(),
            provenance: StemProvenance {
                input_pcm_sha256: sha.to_string(),
                target_pcm_sha256: sha.to_string(),
                num_frames: frames,
            },
            user_metadata: BTreeMap::new(),
            extracted_physics: None,
        };

    let mut m = DnamProjManifest::default_reference();
    m.stems_registry
        .push(mk_stem("  ", 0, true, false, valid_sha, 100));
    assert!(m.validate().is_err());

    let mut m = DnamProjManifest::default_reference();
    m.stems_registry
        .push(mk_stem("s0", 0, true, false, valid_sha, 100));
    m.stems_registry
        .push(mk_stem("s0", 1, true, false, valid_sha, 100));
    assert!(m.validate().is_err());

    let mut m = DnamProjManifest::default_reference();
    m.stems_registry
        .push(mk_stem("s0", 0, true, false, valid_sha, 100));
    m.stems_registry
        .push(mk_stem("s1", 0, true, false, valid_sha, 100));
    assert!(m.validate().is_err());

    let mut m = DnamProjManifest::default_reference();
    m.stems_registry
        .push(mk_stem("s0", 0, true, false, "UPPERCASE_BAD", 100));
    assert!(m.validate().is_err());

    let mut m = DnamProjManifest::default_reference();
    m.stems_registry
        .push(mk_stem("s0", 0, true, false, valid_sha, 0));
    assert!(m.validate().is_err());

    let mut m = DnamProjManifest::default_reference();
    m.stems_registry
        .push(mk_stem("s0", 0, true, true, valid_sha, 100));
    m.stems_registry
        .push(mk_stem("s1", 1, true, true, valid_sha, 100));
    assert!(m.validate().is_err());

    m.stems_registry[0].enabled = false;
    assert!(m.validate().is_ok());

    let mut m = DnamProjManifest::default_reference();
    m.audit_report = Some(AuditReport::default());
    m.solver_telemetry = Some(SolverTelemetrySummary {
        active_stem_count: 1,
        total_solve_ms: -0.5,
        tier_esr_db: [-40.0; 4],
    });
    assert!(m.validate().is_err());

    let good = DnamProjManifest::default_reference();
    let json = serde_json::to_string_pretty(&good).unwrap();
    let norm = DnamProjManifest::validate_and_normalize_json_str(&json).unwrap();
    assert!(!norm.contains('\n'));
    assert!(DnamProjManifest::validate_and_normalize_json_str("{not json").is_err());
    assert!(DnamProjManifest::validate_and_normalize_json_str(r#"{"unknown_top":1}"#).is_err());
}

#[test]
fn test_proj_calb_inpc_stmc_stat_capsule_validation_edges() {
    assert!(CalbChunk64B::from_bytes(&[0; 16]).is_err());
    let valid_calb = (CalbChunk64B {
        sample_rate_device_hz: 48_000,
        bit_depth_code: 2,
        polarity_inverted: true,
        loopback_lag_samples: 12.5,
        signed_gain: -0.5,
        input_reference_dbfs: -20.5,
        noise_floor_dbfs: -100.0,
    })
    .to_bytes();
    assert!(
        CalbChunk64B::from_bytes(&valid_calb)
            .unwrap()
            .polarity_inverted
    );

    let mut bad = valid_calb;
    bad[0..4].copy_from_slice(b"XXXX");
    assert!(CalbChunk64B::from_bytes(&bad).is_err());

    let mut bad = valid_calb;
    bad[4..6].copy_from_slice(&1_u16.to_le_bytes());
    assert!(CalbChunk64B::from_bytes(&bad).is_err());

    let mut bad = valid_calb;
    bad[6] = 9;
    assert!(CalbChunk64B::from_bytes(&bad).is_err());

    let mut bad = valid_calb;
    bad[60] ^= 0x55;
    assert!(CalbChunk64B::from_bytes(&bad).is_err());

    let mut bad_sr = valid_calb;
    bad_sr[8..12].copy_from_slice(&1_000_u32.to_le_bytes());
    let crc = crc32c_compute(&bad_sr[0..60]);
    bad_sr[60..64].copy_from_slice(&crc.to_le_bytes());
    assert!(CalbChunk64B::from_bytes(&bad_sr).is_err());

    let mut bad_nan = valid_calb;
    bad_nan[16..24].copy_from_slice(&f64::NAN.to_le_bytes());
    let crc = crc32c_compute(&bad_nan[0..60]);
    bad_nan[60..64].copy_from_slice(&crc.to_le_bytes());
    assert!(CalbChunk64B::from_bytes(&bad_nan).is_err());

    let mono_fp16 = vec![0u8; 24_000];
    let stereo_fp16 = vec![0u8; 48_000];
    assert!(build_proj_inpc_capsule(3, &[0; 256], &[0; 320], &mono_fp16).is_err());
    assert!(build_proj_inpc_capsule(1, &[0; 256], &[0; 320], &stereo_fp16).is_err());

    let mut nan_fp16 = mono_fp16.clone();
    nan_fp16[10..12].copy_from_slice(&0x7E00_u16.to_le_bytes());
    assert!(build_proj_inpc_capsule(1, &[0; 256], &[0; 320], &nan_fp16).is_err());

    let valid_inpc = build_proj_inpc_capsule(1, &[1; 256], &[2; 320], &mono_fp16).unwrap();
    assert_eq!(valid_inpc.len(), INPC_MONO_BYTES);
    let valid_inpc_st = build_proj_inpc_capsule(2, &[1; 256], &[2; 320], &stereo_fp16).unwrap();
    assert_eq!(
        verify_proj_inpc_capsule(&valid_inpc_st).unwrap().channels,
        2
    );

    assert!(verify_proj_inpc_capsule(&valid_inpc[..32]).is_err());
    let mut bad = valid_inpc.clone();
    bad[0..4].copy_from_slice(b"NOPE");
    assert!(verify_proj_inpc_capsule(&bad).is_err());

    let mut bad = valid_inpc.clone();
    bad[4..6].copy_from_slice(&1_u16.to_le_bytes());
    assert!(verify_proj_inpc_capsule(&bad).is_err());

    let mut bad = valid_inpc.clone();
    bad[20] = 1;
    assert!(verify_proj_inpc_capsule(&bad).is_err());

    let mut bad = valid_inpc.clone();
    bad[6..8].copy_from_slice(&0_u16.to_le_bytes());
    assert!(verify_proj_inpc_capsule(&bad).is_err());

    let mut bad = valid_inpc.clone();
    bad[8..12].copy_from_slice(&44_100_u32.to_le_bytes());
    assert!(verify_proj_inpc_capsule(&bad).is_err());

    let mut bad = valid_inpc.clone();
    bad.push(0);
    assert!(verify_proj_inpc_capsule(&bad).is_err());

    let mut bad = valid_inpc.clone();
    bad[60] ^= 0xFF;
    assert!(verify_proj_inpc_capsule(&bad).is_err());

    let mut bad = valid_inpc;
    bad[640..642].copy_from_slice(&0x7C00_u16.to_le_bytes());
    let crc = crc32c_compute(&bad[64..]);
    bad[60..64].copy_from_slice(&crc.to_le_bytes());
    assert!(verify_proj_inpc_capsule(&bad).is_err());

    let valid_win = StmcTrajectoryWindow {
        control_id: 0,
        ctrl_kind: 0,
        active_opt: 0,
        u_phys: 0.5,
        window_weight: 1.0,
        emr_velocity: 0.0,
    };
    let mut bad_win = valid_win;
    bad_win.ctrl_kind = 2;
    assert!(StmcTrajectoryWindow::from_bytes(&bad_win.to_bytes()).is_err());
    let mut bad_win2 = valid_win;
    bad_win2.window_weight = -1.0;
    assert!(StmcTrajectoryWindow::from_bytes(&bad_win2.to_bytes()).is_err());

    assert!(
        build_proj_stmc_capsule(
            0,
            0x8000,
            1.0,
            0.0,
            &[0; 88],
            0,
            &[valid_win],
            &[0; 4096],
            &mono_fp16
        )
        .is_err()
    );
    assert!(
        build_proj_stmc_capsule(
            0,
            STMC_FLAG_ENABLED,
            1.0,
            0.0,
            &[0; 88],
            0,
            &[],
            &[0; 4096],
            &mono_fp16
        )
        .is_err()
    );
    assert!(
        build_proj_stmc_capsule(
            0,
            STMC_FLAG_ENABLED,
            f32::NAN,
            0.0,
            &[0; 88],
            0,
            &[valid_win],
            &[0; 4096],
            &mono_fp16
        )
        .is_err()
    );
    assert!(
        build_proj_stmc_capsule(
            0,
            STMC_FLAG_ENABLED,
            1.0,
            0.0,
            &[0; 88],
            0,
            &[valid_win],
            &[0; 4096],
            &stereo_fp16
        )
        .is_err()
    );

    let valid_stmc = build_proj_stmc_capsule(
        0,
        STMC_FLAG_ENABLED,
        1.0,
        0.0,
        &[0; 88],
        0,
        &[valid_win],
        &[0; 4096],
        &mono_fp16,
    )
    .unwrap();
    assert!(verify_proj_stmc_capsule(&valid_stmc[..64]).is_err());

    let mut bad = valid_stmc.clone();
    bad[0..4].copy_from_slice(b"NOPE");
    assert!(verify_proj_stmc_capsule(&bad).is_err());

    let mut bad = valid_stmc.clone();
    bad[4..6].copy_from_slice(&1_u16.to_le_bytes());
    assert!(verify_proj_stmc_capsule(&bad).is_err());

    let mut bad = valid_stmc.clone();
    bad[8..10].copy_from_slice(&0x4000_u16.to_le_bytes());
    assert!(verify_proj_stmc_capsule(&bad).is_err());

    let mut bad = valid_stmc.clone();
    bad[10..12].copy_from_slice(&0_u16.to_le_bytes());
    assert!(verify_proj_stmc_capsule(&bad).is_err());

    let mut bad = valid_stmc.clone();
    bad[12..16].copy_from_slice(&f32::INFINITY.to_le_bytes());
    assert!(verify_proj_stmc_capsule(&bad).is_err());

    let mut bad = valid_stmc.clone();
    bad[124] = 1;
    assert!(verify_proj_stmc_capsule(&bad).is_err());

    let mut bad = valid_stmc.clone();
    bad[108..112].copy_from_slice(&999_u32.to_le_bytes());
    assert!(verify_proj_stmc_capsule(&bad).is_err());

    let mut bad = valid_stmc.clone();
    bad.push(0);
    assert!(verify_proj_stmc_capsule(&bad).is_err());

    let mut bad = valid_stmc;
    bad[160] = 0xAA;
    let crc = crc32c_compute(&bad[128..]);
    bad[120..124].copy_from_slice(&crc.to_le_bytes());
    assert!(verify_proj_stmc_capsule(&bad).is_err());

    assert_eq!(
        verify_proj_stat_uplo_u(&vec![0u8; PROJ_STAT_BYTES_LITE]).unwrap(),
        64
    );
    assert_eq!(
        verify_proj_stat_uplo_u(&vec![0u8; PROJ_STAT_BYTES_FULL]).unwrap(),
        128
    );
    assert_eq!(
        verify_proj_stat_uplo_u(&vec![0u8; PROJ_STAT_BYTES_PRO]).unwrap(),
        192
    );
    assert_eq!(
        verify_proj_stat_uplo_u(&vec![0u8; PROJ_STAT_BYTES_ULTRA]).unwrap(),
        256
    );
    assert!(verify_proj_stat_uplo_u(&[0u8; 128]).is_err());

    let mut nan_stat = vec![0u8; PROJ_STAT_BYTES_LITE];
    nan_stat[16..24].copy_from_slice(&f64::NAN.to_le_bytes());
    assert!(verify_proj_stat_uplo_u(&nan_stat).is_err());
}

#[test]
fn test_align_regimes_validation_and_bitdepth_edges() {
    use dnam_capture::align::{
        DecayTailDecision, calibrate_loopback_calb, estimate_multiregime_alignment,
        evaluate_acoustic_null_homing, evaluate_live_decay_tail, measure_differential_pad_ratio,
        realign_fractional_kaiser64, rescale_padded_stem,
    };
    use std::f64::consts::PI;

    assert!(realign_fractional_kaiser64(&[], 1.5, 0.0, false).is_empty());
    assert_eq!(
        realign_fractional_kaiser64(&[0.5, -0.25], 0.0, 0.0, true),
        vec![-0.5, 0.25]
    );
    let nan_re = realign_fractional_kaiser64(&[0.25, f32::NAN, -0.25], f64::NAN, f64::NAN, false);
    assert!(nan_re.iter().all(|v| v.is_finite()));

    let sig: Vec<f32> = (0..512)
        .map(|i| 0.4 * (2.0 * PI * 330.0 * (i as f64) / 48_000.0).sin() as f32)
        .collect();
    assert!(estimate_multiregime_alignment(&sig, &sig, 0, 16).is_err());
    assert!(estimate_multiregime_alignment(&sig[..4], &sig[..4], 48_000, 2).is_err());
    let mut bad = sig.clone();
    bad[10] = f32::NAN;
    assert!(estimate_multiregime_alignment(&bad, &sig, 48_000, 16).is_err());

    assert!(calibrate_loopback_calb(&sig, &sig, 0, CaptureBitDepth::Float32, -20.0, 16).is_err());
    assert!(
        calibrate_loopback_calb(&sig, &sig, 48_000, CaptureBitDepth::Float32, f32::NAN, 16)
            .is_err()
    );
    assert!(
        calibrate_loopback_calb(
            &sig[..64],
            &sig[..64],
            48_000,
            CaptureBitDepth::Float32,
            -20.0,
            8
        )
        .is_err()
    );
    for (bd, code) in [
        (CaptureBitDepth::Int12, 0_u8),
        (CaptureBitDepth::Int16, 1),
        (CaptureBitDepth::Int24, 2),
        (CaptureBitDepth::Int32, 3),
        (CaptureBitDepth::Float32, 4),
    ] {
        let rep = calibrate_loopback_calb(&sig, &sig, 48_000, bd, -20.0, 16).expect("calb");
        assert_eq!(rep.calb.bit_depth_code, code);
    }

    let zeros = vec![0.0_f32; 512];
    let loud: Vec<f32> = sig.iter().map(|&x| 5.0 * x).collect();
    assert!(measure_differential_pad_ratio(&sig, &loud, 16).is_err());
    assert!(rescale_padded_stem(&sig, 0.0).is_err());
    let rescaled = rescale_padded_stem(&[0.1, f32::NAN], 0.5).expect("nan clamped to 0");
    assert!((rescaled[0] - 0.2).abs() < 1.0e-5 && rescaled[1] == 0.0);
    assert!(evaluate_acoustic_null_homing(&zeros, &sig, 1.0e-5).is_err());
    assert!(matches!(
        evaluate_live_decay_tail(&[], f32::NAN, 0, 1_000),
        DecayTailDecision::Complete { .. }
    ));
}

#[derive(Default)]
struct CoupledTiltTarget {
    lp_state: f32,
}

impl dnam_capture::DistillationTarget for CoupledTiltTarget {
    fn reset_state(&mut self) {
        self.lp_state = 0.0;
    }

    fn render_stem(
        &mut self,
        stim: &dnam_capture::StemStimulus,
    ) -> Result<dnam_capture::StemResponse, dnam_engine::schema::DnamError> {
        let mut out_1 = Vec::with_capacity(stim.in_1.len());
        let mut out_2 = stim.in_2.as_ref().map(|r| Vec::with_capacity(r.len()));
        for (i, &x) in stim.in_1.iter().enumerate() {
            let c0 = stim
                .control_trajectories
                .as_ref()
                .and_then(|t| t.first()?.get(i).copied())
                .unwrap_or_else(|| stim.control_values.first().copied().unwrap_or(0.5));
            let c1 = stim
                .control_trajectories
                .as_ref()
                .and_then(|t| t.get(1)?.get(i).copied())
                .unwrap_or_else(|| stim.control_values.get(1).copied().unwrap_or(0.5));
            self.lp_state = 0.94 * self.lp_state + 0.06 * x;
            let hp = x - self.lp_state;
            let y = (3.0 * (1.0 - c0) + 0.05) * self.lp_state + (3.0 * c1 + 0.05) * hp;
            out_1.push(y);
            if let Some(ref mut r) = out_2 {
                r.push(0.8 * y);
            }
        }
        Ok(dnam_capture::StemResponse {
            stem_id: stim.stem_id.clone(),
            sample_rate_hz: stim.sample_rate_hz,
            out_1,
            out_2,
            out_3: None,
            out_4: None,
            internal_nodes: BTreeMap::new(),
        })
    }
}

#[test]
fn test_coupled_hadamard_diagonals_stereo_pad_and_batch_guards() {
    use dnam_capture::DistillationTarget;
    use dnam_capture::align::generate_mic_pad_probe_400ms;
    use dnam_capture::stimulus::{
        AdaptiveStimulusController, AdaptiveTrainingOptions, BatchStimulusOptions, StimulusConfig,
        TopologyProbeMode, generate_refinement_stimulus_batch, generate_stimulus_batch,
        recapture_clipped_stem, solve_from_responses, train_adaptive_target,
    };
    use dnam_engine::schema::ControlSchemaDef;

    let mk_ctrl = |id: &str| -> ControlSchemaDef {
        serde_json::from_value(serde_json::json!({
            "id": id,
            "name": id,
            "type": "range",
            "lobatto_degree": 2,
            "declared_range": [0.0, 1.0],
            "safe_clamped_range": [0.0, 1.0],
            "default": 0.5
        }))
        .unwrap()
    };

    let mut manifest = DnamProjManifest::default_reference();
    manifest.base.io_config.inputs = 2;
    manifest.base.io_config.output_mode = "stereo".to_string();
    manifest.rig_routing.input_channels = 2;
    manifest.rig_routing.output_channels = 2;
    manifest.base.controls_schema = vec![mk_ctrl("bass_tilt"), mk_ctrl("treble_tilt")];

    let mut target = CoupledTiltTarget::default();
    let rep = train_adaptive_target(
        manifest.clone(),
        &mut target,
        &AdaptiveTrainingOptions {
            topology_mode: TopologyProbeMode::TrueStereo2To2,
            prefer_mode_b: true,
            max_frames_per_stem: 6,
            run_live_null_test: false,
            ..AdaptiveTrainingOptions::default()
        },
    )
    .expect("coupled Hadamard Mode 1");
    assert_eq!(rep.hadamard_diagonals_run, 1);
    assert!(rep.null_test.is_none());

    let probe = generate_mic_pad_probe_400ms(48_000);
    let padded_probe: Vec<f32> = probe.iter().map(|&x| 0.316_227_76 * x).collect();
    let mut ctrl_st = AdaptiveStimulusController::new(StimulusConfig {
        stem_id: "stop_ctrl_0_1".to_string(),
        topology_mode: TopologyProbeMode::TrueStereo2To2,
        max_frames: 6,
        max_tail_frames: 0,
        ..StimulusConfig::default()
    });
    let mut last_r = None;
    while let Some(fr) = ctrl_st.next_frame(last_r.as_ref()) {
        last_r = Some(target.render_frame(&fr).unwrap());
    }
    let mut meta = BTreeMap::new();
    meta.insert("bass_tilt".to_string(), serde_json::json!(1.0));
    meta.insert("treble_tilt".to_string(), serde_json::json!(0.5));
    let (_, st_spec) = ctrl_st.seal_inpc_and_vault_spec(false, meta).unwrap();
    assert_eq!(st_spec.wet_coreset_fp16.len(), 48_000);
    let (_, st_cal) = recapture_clipped_stem(
        &rep.proj_bytes,
        "stop_ctrl_0_1",
        &probe,
        &padded_probe,
        st_spec,
    )
    .expect("stereo pad recapture");
    assert!((st_cal.pad_attenuation_db - (-10.0)).abs() <= 0.1);

    assert!(
        generate_stimulus_batch(
            &manifest,
            &BatchStimulusOptions {
                sample_rate_hz: 0,
                ..BatchStimulusOptions::default()
            },
        )
        .is_err()
    );
    let st_batch = generate_stimulus_batch(
        &manifest,
        &BatchStimulusOptions {
            topology_mode: TopologyProbeMode::TrueStereo2To2,
            ..BatchStimulusOptions::default()
        },
    )
    .expect("stereo batch");
    assert!(st_batch.stems[0].in_2.is_some());
    assert!(
        generate_refinement_stimulus_batch(&manifest, &st_batch, &[])
            .unwrap()
            .is_none()
    );
    assert!(solve_from_responses(manifest, &[(&st_batch, &[])], None).is_err());
}
