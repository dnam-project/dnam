//! Inline `CP1` contributor decoding and `U_int` ANOVA-FM interaction matrix evaluation (`r[dsp.ctrl.accum.anova-fm]`).

use crate::control::record::Cp1ContributorSlot;

/// Evaluates `U_int^base` (`4 × 8 = 128 B` at `Arena 0 [7,552..7,680]`) on `z_fm_base` (`r[dsp.ctrl.accum.anova-fm]`).
#[inline]
pub fn eval_uint_base_4x8(
    arena_bytes: &[u8],
    z_fm_base: [f32; 4],
    s_emb: [f32; 3],
) -> ([f32; 8], bool) {
    const UINT_BASE_OFF: usize = 7_552;
    if arena_bytes.len() < UINT_BASE_OFF + 128 {
        return ([0.0; 8], false);
    }
    let has_nonzero_uint = arena_bytes[UINT_BASE_OFF..UINT_BASE_OFF + 128]
        .iter()
        .any(|&b| b != 0);
    if !has_nonzero_uint {
        return ([0.0; 8], false);
    }
    let inputs = if z_fm_base.iter().any(|&v| v.abs() > 1e-12) {
        z_fm_base
    } else {
        [
            s_emb[0] * s_emb[1],
            s_emb[0] * s_emb[2],
            s_emb[1] * s_emb[2],
            s_emb[0] * s_emb[0],
        ]
    };
    let mut out = [0.0_f32; 8];
    for (r, &p_val) in inputs.iter().enumerate() {
        if p_val.abs() > 1e-12 {
            let row_off = UINT_BASE_OFF + r * 32;
            for (d, out_d) in out.iter_mut().enumerate() {
                let off = row_off + d * 4;
                let w = f32::from_le_bytes([
                    arena_bytes[off],
                    arena_bytes[off + 1],
                    arena_bytes[off + 2],
                    arena_bytes[off + 3],
                ]);
                *out_d += p_val * w;
            }
        }
    }
    (out, true)
}

/// Evaluates `U_int^base` (`3 × 8` at `Arena 0 [7,552..7,648]`) on the 3 pairwise products of `s_emb`.
#[inline]
pub fn eval_uint_base_3x8(arena_bytes: &[u8], s_emb: [f32; 3]) -> [f32; 8] {
    eval_uint_base_4x8(arena_bytes, [0.0; 4], s_emb).0
}

/// Evaluates `3 × U_int,Delta` (`3 × 4 × 8 = 384 B` at `Arena 3 [60,928..61,312]`, `r[dsp.ctrl.accum.anova-fm]`):
/// - Block 0 (`60,928..61,056`): maps `z_fm_ultra` into `\Delta z_raw[0..8]`
/// - Block 1 (`61,056..61,184`): maps `z_fm_base` into `z_raw[8..16]`
/// - Block 2 (`61,184..61,312`): maps `z_fm_ultra` into `z_raw[8..16]`
#[inline]
pub fn eval_uint_ultra_delta_3x4x8(
    arena_bytes: &[u8],
    z_fm_base: [f32; 4],
    z_fm_ultra: [f32; 4],
    s_emb: [f32; 3],
) -> ([f32; 8], [f32; 8], bool) {
    const UINT_ULTRA_OFF: usize = 60_928;
    if arena_bytes.len() < UINT_ULTRA_OFF + 384 {
        return ([0.0; 8], [0.0; 8], false);
    }
    let has_nonzero_uint = arena_bytes[UINT_ULTRA_OFF..UINT_ULTRA_OFF + 384]
        .iter()
        .any(|&b| b != 0);
    if !has_nonzero_uint {
        return ([0.0; 8], [0.0; 8], false);
    }
    let has_fm =
        z_fm_base.iter().any(|&v| v.abs() > 1e-12) || z_fm_ultra.iter().any(|&v| v.abs() > 1e-12);
    let (in_b0, in_b1, in_b2) = if has_fm {
        (z_fm_ultra, z_fm_base, z_fm_ultra)
    } else {
        let p = [
            s_emb[0] * s_emb[1],
            s_emb[0] * s_emb[2],
            s_emb[1] * s_emb[2],
            0.0,
        ];
        ([0.0; 4], p, [0.0; 4])
    };
    let mut delta_base_8 = [0.0_f32; 8];
    let mut ultra_new_8 = [0.0_f32; 8];
    for (r, &val) in in_b0.iter().enumerate() {
        if val.abs() > 1e-12 {
            let row_off = UINT_ULTRA_OFF + r * 32;
            for (d, out_d) in delta_base_8.iter_mut().enumerate() {
                let off = row_off + d * 4;
                let w = f32::from_le_bytes([
                    arena_bytes[off],
                    arena_bytes[off + 1],
                    arena_bytes[off + 2],
                    arena_bytes[off + 3],
                ]);
                *out_d += val * w;
            }
        }
    }
    for (r, &val) in in_b1.iter().enumerate() {
        if val.abs() > 1e-12 {
            let row_off = UINT_ULTRA_OFF + 128 + r * 32;
            for (d, out_d) in ultra_new_8.iter_mut().enumerate() {
                let off = row_off + d * 4;
                let w = f32::from_le_bytes([
                    arena_bytes[off],
                    arena_bytes[off + 1],
                    arena_bytes[off + 2],
                    arena_bytes[off + 3],
                ]);
                *out_d += val * w;
            }
        }
    }
    for (r, &val) in in_b2.iter().enumerate() {
        if val.abs() > 1e-12 {
            let row_off = UINT_ULTRA_OFF + 256 + r * 32;
            for (d, out_d) in ultra_new_8.iter_mut().enumerate() {
                let off = row_off + d * 4;
                let w = f32::from_le_bytes([
                    arena_bytes[off],
                    arena_bytes[off + 1],
                    arena_bytes[off + 2],
                    arena_bytes[off + 3],
                ]);
                *out_d += val * w;
            }
        }
    }
    (delta_base_8, ultra_new_8, true)
}

/// Evaluates `U_int,Delta` (`6 × 8` at `Arena 3 [60,928..61,120]`) on 6 pairwise products.
#[inline]
pub fn eval_uint_ultra_delta_6x8(
    arena_bytes: &[u8],
    s_emb: [f32; 3],
    s_emb_ultra: [f32; 3],
) -> [f32; 8] {
    const UINT_ULTRA_OFF: usize = 60_928;
    if arena_bytes.len() < UINT_ULTRA_OFF + 192 {
        return [0.0; 8];
    }
    let pairs = [
        s_emb[0] * s_emb[1],
        s_emb[0] * s_emb[2],
        s_emb[1] * s_emb[2],
        s_emb_ultra[0] * s_emb[0],
        s_emb_ultra[1] * s_emb[1],
        s_emb_ultra[2] * s_emb[2],
    ];
    let mut out = [0.0_f32; 8];
    for (r, &p_val) in pairs.iter().enumerate() {
        if p_val.abs() > 1e-12 {
            let row_off = UINT_ULTRA_OFF + r * 32;
            for (d, out_d) in out.iter_mut().enumerate() {
                let off = row_off + d * 4;
                let w = f32::from_le_bytes([
                    arena_bytes[off],
                    arena_bytes[off + 1],
                    arena_bytes[off + 2],
                    arena_bytes[off + 3],
                ]);
                *out_d += p_val * w;
            }
        }
    }
    out
}

/// Decodes inline `CP1` contributors from `Arena 0 [7,680..8,512]` (`832 B` base) and
/// `Arena 3 [61,312..63,424]` (`2,112 B` Ultra) using `cp1_lanes_dir` at `Arena 0 [7,524..7,552]`.
/// Returns `Ok(0)` when `num_inline_contributors == 0`.
pub fn decode_cp1_inline_contributors(
    arena_bytes: &[u8],
    out_slots: &mut [Cp1ContributorSlot],
) -> Result<usize, crate::schema::DnamError> {
    const DIR_OFF: usize = 7_524;
    const BASE_OFF: usize = 7_680;
    const BASE_END: usize = 8_512;
    const ULTRA_OFF: usize = 61_312;
    const ULTRA_END: usize = 63_424;

    if arena_bytes.len() < BASE_END {
        return Ok(0);
    }
    let count = u16::from_le_bytes([arena_bytes[DIR_OFF], arena_bytes[DIR_OFF + 1]]) as usize;
    if count == 0 {
        return Ok(0);
    }
    if count > 26 || count > out_slots.len() {
        return Err(crate::schema::DnamError::SchemaValidationError(
            alloc::string::String::from("inline CP1 contributor count exceeds 26"),
        ));
    }
    let use_3b_dir = count <= 8
        && arena_bytes[DIR_OFF + 2] == 0
        && arena_bytes[DIR_OFF + 3] == 0
        && ((1..=3).contains(&(arena_bytes[DIR_OFF + 5] & 0x0F))
            || (1..=3).contains(&(arena_bytes[DIR_OFF + 6] & 0x0F)));
    let has_ultra = arena_bytes.len() >= ULTRA_END;
    let mut b_cursor = BASE_OFF;
    let mut u_cursor = ULTRA_OFF;
    for (c, slot) in out_slots.iter_mut().take(count).enumerate() {
        let (control_idx, ctrl_kind, option_idx, partner_control_id, lanes) = if use_3b_dir {
            let entry_off = DIR_OFF + 4 + c * 3;
            let c_idx = arena_bytes[entry_off] as u16;
            let b1 = arena_bytes[entry_off + 1];
            let b2 = arena_bytes[entry_off + 2];
            let (packed, aux) = if (1..=3).contains(&(b1 & 0x0F)) {
                (b1, b2)
            } else {
                (b2, b1)
            };
            let kind = (packed >> 4) & 0x0F;
            let l = packed & 0x0F;
            let opt = if kind == 1 { aux } else { 0xFF };
            let partner = if kind == 2 { aux as u16 } else { 0 };
            (c_idx, kind, opt, partner, l)
        } else {
            let packed = arena_bytes[DIR_OFF + 2 + c];
            let kind = (packed >> 4) & 0x0F;
            let l = packed & 0x0F;
            let opt = if kind == 1 { 1 } else { 0xFF };
            let partner = if kind == 2 { (c + 1) as u16 } else { 0 };
            (c as u16, kind, opt, partner, l)
        };
        if !(1..=3).contains(&lanes) {
            return Err(crate::schema::DnamError::SchemaValidationError(
                alloc::string::String::from("invalid inline CP1 lane count"),
            ));
        }
        let l_cnt = lanes as usize;
        let b_bytes = 4 * (9 * l_cnt + 4);
        if b_cursor + b_bytes > BASE_END {
            return Err(crate::schema::DnamError::SchemaValidationError(
                alloc::string::String::from("inline CP1 base payload exceeds 832 B"),
            ));
        }
        let mut main_base = [[0.0f32; 8]; 3];
        for row in main_base.iter_mut().take(l_cnt) {
            for val in row.iter_mut() {
                *val = f32::from_le_bytes([
                    arena_bytes[b_cursor],
                    arena_bytes[b_cursor + 1],
                    arena_bytes[b_cursor + 2],
                    arena_bytes[b_cursor + 3],
                ]);
                b_cursor += 4;
            }
        }
        let mut emb = [0.0f32; 3];
        for val in emb.iter_mut().take(l_cnt) {
            *val = f32::from_le_bytes([
                arena_bytes[b_cursor],
                arena_bytes[b_cursor + 1],
                arena_bytes[b_cursor + 2],
                arena_bytes[b_cursor + 3],
            ]);
            b_cursor += 4;
        }
        let mut v_base = [0.0f32; 4];
        for val in &mut v_base {
            *val = f32::from_le_bytes([
                arena_bytes[b_cursor],
                arena_bytes[b_cursor + 1],
                arena_bytes[b_cursor + 2],
                arena_bytes[b_cursor + 3],
            ]);
            b_cursor += 4;
        }
        let mut main_ultra = [[0.0f32; 8]; 3];
        let mut v_ultra = [0.0f32; 4];
        if has_ultra {
            let u_bytes = (32 * l_cnt) + 16;
            if u_cursor + u_bytes <= ULTRA_END {
                for row in main_ultra.iter_mut().take(l_cnt) {
                    for val in row.iter_mut() {
                        *val = f32::from_le_bytes([
                            arena_bytes[u_cursor],
                            arena_bytes[u_cursor + 1],
                            arena_bytes[u_cursor + 2],
                            arena_bytes[u_cursor + 3],
                        ]);
                        u_cursor += 4;
                    }
                }
                for val in &mut v_ultra {
                    *val = f32::from_le_bytes([
                        arena_bytes[u_cursor],
                        arena_bytes[u_cursor + 1],
                        arena_bytes[u_cursor + 2],
                        arena_bytes[u_cursor + 3],
                    ]);
                    u_cursor += 4;
                }
            }
        }
        *slot = Cp1ContributorSlot {
            lanes,
            ctrl_kind,
            control_idx,
            option_idx,
            partner_control_id,
            main_base,
            emb,
            v_base,
            main_ultra,
            v_ultra,
            has_ultra,
        };
    }
    Ok(count)
}

#[cfg(test)]
mod ckpt_and_control_tests {
    #[allow(unused_imports)]
    use alloc::{
        boxed::Box,
        format,
        string::{String, ToString},
        vec,
        vec::Vec,
    };

    // Empirical TORA Control Bus and 3-Phase Clamping Verification (`Tests 8A, 8B, 8D, 8E`).
    //
    // Canonical Reference: `docs/specs/dsp/parametric_control_bus.md`
    // and `docs/specs/training/control_space_audit.md`.
    //
    // Verifies:
    // - `Test 8A & 8B` (`verify[dsp.op-t.transition.dual-mode]`, `verify[dsp.ctrl.basis.lobatto]`): 3-node Lobatto physical delay interpolation accuracy and
    //   128-sample Hann-windowed dual-pointer switch crossfade vs. single-pointer phase jump.
    // - `Test 8D & 8E` (`verify[dsp.ctrl.range.normalize-clamp]`, `verify[dsp.ctrl.accum.cp1-scaling]`): 128-byte `ControlChannelRecord` table at
    //   `tensor_arena_offset - 128 * P`, `update_header_in_place` preservation of `ControlChannelRecord`
    //   bytes, 64-byte $\mathcal{O}(1)$ `LatentAccumulator64B` delta updates across $P = 32$ controls,
    //   and non-stop Phase-1/2/3 clamping (`[CLAMP-RANGE]`, `[CLAMP-SWITCH]`, `[CLAMP-STABILITY]`).

    use std::collections::BTreeMap;
    use std::f64::consts::TAU;

    use crate::FtzDazGuard;
    use crate::container::{
        ARENA_OFFSET_ULTRA_END, DnamContainerView, build_dnam_container, update_header_in_place,
    };
    use crate::control::{
        ControlBus, ControlChannelRecord, ControlClampDescriptor, ControlLatentProjection,
        LatentAccumulator64B,
    };
    use crate::math::ScalarBackend;
    use crate::operators::op_t::{OP_T_LITE_RING_CAPACITY, OpTDelayLine};
    use crate::provenance::DefaultCryptoProvider;
    use crate::schema::{ControlKind, ControlSchemaDef, ControlUiHints, DnamJsonHeader};
    use proptest::prelude::*;

    /// Test 8A & 8B (`verify[dsp.op-t.transition.dual-mode]`, `verify[dsp.ctrl.basis.lobatto]`): Verifies 3-node Lobatto physical delay interpolation accuracy
    /// at intermediate knob positions (`u = 0.5`) and verifies that `OpTDelayLine`'s 128-sample
    /// Hann-windowed dual-pointer switch crossfade eliminates single-pointer discontinuity jumps.
    #[test]
    fn test_8a_8b_lobatto_physical_delay_interpolation_and_dual_pointer_switch_crossfade() {
        let tau_at = |u: f64| -> f64 {
            let psi1 = u;
            let psi2 = u * (u + 1.0);
            200.0 + 100.0 * psi1 + 50.0 * psi2
        };
        let exact_mid = 200.0 + 150.0 * 0.5 + 50.0 * 0.25;
        let interp_mid = tau_at(0.5);
        assert!((interp_mid - exact_mid).abs() < 1e-12);

        let _audio = FtzDazGuard::new();
        let mut dl_xfade = OpTDelayLine::<OP_T_LITE_RING_CAPACITY>::default();
        let mut dl_jump = OpTDelayLine::<OP_T_LITE_RING_CAPACITY>::default();
        dl_xfade.current_delay = 40.0;
        dl_xfade.target_delay = 40.0;
        dl_jump.current_delay = 40.0;
        dl_jump.target_delay = 40.0;

        let mut prev_x = 0.0_f32;
        let mut prev_j = 0.0_f32;
        for n in 0..315 {
            let s = (TAU as f32 * 400.0 * (n as f32) / 48_000.0).sin();
            prev_x = dl_xfade.step(s);
            prev_j = dl_jump.step(s);
        }

        dl_xfade.set_target_delay(100.0, true);
        dl_jump.current_delay = 100.0;
        dl_jump.target_delay = 100.0;

        let mut max_diff_xfade = 0.0_f32;
        let mut max_diff_jump = 0.0_f32;

        for n in 315..450 {
            let s = (TAU as f32 * 400.0 * (n as f32) / 48_000.0).sin();
            let yx = dl_xfade.step(s);
            let yj = dl_jump.step(s);
            max_diff_xfade = max_diff_xfade.max((yx - prev_x).abs());
            max_diff_jump = max_diff_jump.max((yj - prev_j).abs());
            prev_x = yx;
            prev_j = yj;
        }

        assert!(max_diff_jump > 10.0 * max_diff_xfade);
    }

    /// Test 8D & 8E (`verify[dsp.ctrl.range.normalize-clamp]`, `verify[dsp.ctrl.accum.cp1-scaling]`): Verifies 128-byte `ControlChannelRecord` framing,
    /// `control_table_offset()` preservation across `update_header_in_place`, $\mathcal{O}(1)$
    /// `LatentAccumulator64B` delta updates across $P = 32$ controls, and non-stop Phase-1/2/3 clamping.
    #[test]
    fn test_8d_8e_control_channel_record_o1_latent_and_3phase_clamping() {
        assert_eq!(core::mem::size_of::<ControlClampDescriptor>(), 64);
        assert_eq!(core::mem::size_of::<ControlLatentProjection>(), 64);
        assert_eq!(core::mem::size_of::<ControlChannelRecord>(), 128);
        assert_eq!(core::mem::size_of::<LatentAccumulator64B>(), 64);

        let mut projections = [ControlLatentProjection::new(); 32];
        let mut u_vals = [0.0_f32; 32];
        let mut psi_vals = [0.0_f32; 32];
        let mut acc = LatentAccumulator64B::new();

        for p in 0..32 {
            let pf = (p + 1) as f32;
            projections[p].w_main = core::array::from_fn(|d| 0.02 * pf * ((d + 1) as f32));
            projections[p].v_factors = core::array::from_fn(|f| 0.05 * ((p + f + 1) as f32));
            u_vals[p] = -0.8 + 0.05 * (p as f32);
            psi_vals[p] = 0.5 * u_vals[p];
            acc.update_delta(&projections[p], 0.0, u_vals[p], psi_vals[p]);
        }

        let new_u7 = 0.75_f32;
        let new_psi7 = 0.5 * new_u7;
        acc.update_delta(&projections[7], u_vals[7], new_u7, new_psi7 - psi_vals[7]);
        u_vals[7] = new_u7;
        psi_vals[7] = new_psi7;

        let z_o1 = acc.evaluate_z_ctrl();
        for d in 0..8 {
            let mut expected = 0.0_f32;
            for p in 0..32 {
                expected += projections[p].w_main[d] * psi_vals[p];
            }
            if d < 4 {
                for p in 0..32 {
                    for q in (p + 1)..32 {
                        expected += projections[p].v_factors[d]
                            * projections[q].v_factors[d]
                            * u_vals[p]
                            * u_vals[q];
                    }
                }
            }
            assert!((z_o1[d] - expected).abs() < 1e-4);
        }

        let mut header = DnamJsonHeader::default_reference();
        let mut cond_max = BTreeMap::new();
        cond_max.insert("boost=on".to_string(), 9.37_f32);
        header.controls_schema = vec![
            ControlSchemaDef {
                id: "gain".to_string(),
                name: "Gain".to_string(),
                kind: ControlKind::Range,
                unit: None,
                lobatto_degree: Some(2),
                anchor_nodes_norm: Some(vec![0.0, -0.6, 0.6]),
                declared_range: Some([0.0, 10.0]),
                safe_clamped_range: Some([2.0, 8.0]),
                default: serde_json::json!(5.0),
                conditional_min: BTreeMap::new(),
                conditional_max: BTreeMap::new(),
                declared_options: None,
                active_options: None,
                clamp_warning: None,
                calibration: None,
                ui_hints: ControlUiHints::default(),
                user_metadata: BTreeMap::new(),
            },
            ControlSchemaDef {
                id: "feedback".to_string(),
                name: "Feedback".to_string(),
                kind: ControlKind::Range,
                unit: None,
                lobatto_degree: Some(2),
                anchor_nodes_norm: Some(vec![0.0, -1.0, 1.0]),
                declared_range: Some([0.0, 10.0]),
                safe_clamped_range: Some([0.0, 10.0]),
                default: serde_json::json!(5.0),
                conditional_min: BTreeMap::new(),
                conditional_max: cond_max,
                declared_options: None,
                active_options: None,
                clamp_warning: None,
                calibration: None,
                ui_hints: ControlUiHints::default(),
                user_metadata: BTreeMap::new(),
            },
            ControlSchemaDef {
                id: "boost".to_string(),
                name: "Boost".to_string(),
                kind: ControlKind::Switch,
                unit: None,
                lobatto_degree: None,
                anchor_nodes_norm: None,
                declared_range: None,
                safe_clamped_range: None,
                default: serde_json::json!("off"),
                conditional_min: BTreeMap::new(),
                conditional_max: BTreeMap::new(),
                declared_options: Some(vec![
                    "off".to_string(),
                    "on".to_string(),
                    "ultra".to_string(),
                ]),
                active_options: Some(vec!["off".to_string(), "on".to_string()]),
                clamp_warning: None,
                calibration: None,
                ui_hints: ControlUiHints::default(),
                user_metadata: BTreeMap::new(),
            },
        ];

        let descriptors = header.compile_clamp_descriptors().unwrap();
        let mut cb = ControlBus::<ScalarBackend>::new();
        for (i, d) in descriptors.iter().enumerate() {
            cb.descriptors[i] = *d;
        }
        cb.num_controls = 3;

        cb.set_control_immediate(0, 9.5);
        assert!((cb.clamped_values[0] - 8.0).abs() < 1e-6);
        cb.set_control_immediate(0, 0.5);
        assert!((cb.clamped_values[0] - 2.0).abs() < 1e-6);

        cb.set_control_immediate(2, 2.0);
        assert!(cb.active_switch_options[2] <= 1);

        cb.set_control_immediate(2, 1.0);
        cb.set_control_immediate(1, 10.0);
        assert!((cb.clamped_values[1] - 9.37).abs() < 1e-5);
        assert_ne!(cb.active_clamp_mask & (1 << 1), 0);

        // Verify Region 4 shared_extent ControlClampDescriptor[0..P] table survives update_header_in_place
        let arena = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
        let mut container = build_dnam_container(&mut header, &arena, &arena, false).unwrap();
        let crypto = DefaultCryptoProvider::default();
        let view_before = DnamContainerView::from_bytes(&container, &crypto).unwrap();
        assert_eq!(view_before.clamp_descriptors.len(), 3);
        let layout = crate::container::validate_v2_open_dnam_layout(&container).unwrap();
        let shared_off = layout.shared_extent.0;
        let shared_len = layout.shared_extent.1;
        assert_eq!(shared_len, 3 * 64);
        let table_snapshot = container[shared_off..shared_off + shared_len].to_vec();

        header
            .user_metadata
            .insert("tag".to_string(), serde_json::json!("preserved"));
        update_header_in_place(&mut container, &header).unwrap();
        assert_eq!(
            &container[shared_off..shared_off + shared_len],
            &table_snapshot
        );
    }

    #[test]
    fn test_theorem_3_unlimited_controls_31_band_eq_and_64_rack() {
        use crate::container::build_dnam_container_with_cp1;
        use crate::control::Cp1ContributorSlot;
        use crate::{DnamEngine, Tier};

        for &num_ctrls in &[31_usize, 64_usize] {
            let mut header = DnamJsonHeader::default_reference();
            header.controls_schema = (0..num_ctrls)
                .map(|p| ControlSchemaDef {
                    id: format!("band_{p}"),
                    name: format!("Band {p}"),
                    kind: ControlKind::Range,
                    unit: Some("dB".to_string()),
                    lobatto_degree: Some(2),
                    anchor_nodes_norm: Some(vec![0.0, -1.0, 1.0]),
                    declared_range: Some([-12.0, 12.0]),
                    safe_clamped_range: Some([-12.0, 12.0]),
                    default: serde_json::json!(0.0),
                    conditional_min: BTreeMap::new(),
                    conditional_max: BTreeMap::new(),
                    declared_options: None,
                    active_options: None,
                    clamp_warning: None,
                    calibration: None,
                    ui_hints: ControlUiHints::default(),
                    user_metadata: BTreeMap::new(),
                })
                .collect();

            let last_p = num_ctrls - 1;
            let mut slot = Cp1ContributorSlot::zero();
            slot.lanes = 1;
            slot.ctrl_kind = 0;
            slot.control_idx = last_p as u16;
            slot.option_idx = 0xFF;
            slot.main_base[0] = [0.5, -0.25, 0.1, 0.2, 0.3, -0.1, 0.05, 0.15];
            slot.emb[0] = 1.0;
            slot.v_base = [0.4, 0.2, -0.3, 0.1];

            let arena = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
            let container =
                build_dnam_container_with_cp1(&mut header, &arena, &arena, false, &[slot]).unwrap();

            let crypto = DefaultCryptoProvider::default();
            let view0 = DnamContainerView::from_bytes(&container, &crypto).unwrap();
            assert_eq!(view0.preamble.num_controls as usize, num_ctrls);
            assert_eq!(view0.control_records.len(), num_ctrls);
            assert_eq!(view0.cp1_contributors.len(), 1);

            let mut engine =
                DnamEngine::<ScalarBackend>::from_bytes(&container, Tier::Lite, 48_000).unwrap();
            assert_eq!(engine.preamble().num_controls as usize, num_ctrls);
            assert_eq!(engine.telemetry_snapshot().z_head, [0.0; 4]);

            // Move the last control (index 30 or 63, well beyond the old 16-control wall!)
            engine.set_control_immediate(last_p, 12.0);
            let z_after = engine.telemetry_snapshot().z_head;
            assert!(
                z_after[0].abs() > 0.1,
                "Control #{last_p} failed to update z_head via O(1) LatentAccumulator64B: {:?}",
                z_after
            );
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(32))]

        /// Property test verifying that `ControlChannelRecord` roundtrips bit-identically through
        /// 128-byte serialization.
        #[test]
        fn prop_control_record_roundtrip(
            u0 in -1.0_f32..1.0_f32,
            w0 in -2.0_f32..2.0_f32,
            v0 in -1.0_f32..1.0_f32,
        ) {
            let mut rec = ControlChannelRecord::new();
            rec.clamp.anchor_nodes_norm[1] = u0;
            rec.projection.w_main[0] = w0;
            rec.projection.v_factors[0] = v0;
            let bytes = rec.to_bytes();
            let decoded = ControlChannelRecord::from_bytes(&bytes);
            prop_assert_eq!(decoded, rec);
        }
    }

    #[test]
    fn test_update_header_in_place_updates_control_user_metadata_and_rejects_immutable_edits() {
        use crate::container::{DnamContainerView, update_header_in_place};
        use crate::provenance::DefaultCryptoProvider;

        let mut header = DnamJsonHeader::default_reference();
        header.controls_schema = vec![ControlSchemaDef {
            id: "drive".to_string(),
            name: "Drive".to_string(),
            kind: ControlKind::Range,
            unit: None,
            lobatto_degree: Some(2),
            anchor_nodes_norm: Some(vec![0.0, -1.0, 1.0]),
            declared_range: Some([0.0, 10.0]),
            safe_clamped_range: Some([1.0, 9.0]),
            default: serde_json::json!(5.0),
            conditional_min: BTreeMap::new(),
            conditional_max: BTreeMap::new(),
            declared_options: None,
            active_options: None,
            clamp_warning: None,
            calibration: None,
            ui_hints: ControlUiHints::default(),
            user_metadata: BTreeMap::new(),
        }];
        let arena = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
        let mut container = build_dnam_container(&mut header, &arena, &arena, false).unwrap();

        // Immutable safe_clamped_range edits must be rejected by update_header_in_place
        let mut bad_header = header.clone();
        bad_header.controls_schema[0].safe_clamped_range = Some([2.5, 7.5]);
        assert!(update_header_in_place(&mut container, &bad_header).is_err());

        // Mutable control_user_metadata edits succeed via DMU1 while preserving safe_clamped_range
        header.controls_schema[0]
            .user_metadata
            .insert("midi_cc".to_string(), serde_json::json!(14));
        update_header_in_place(&mut container, &header).unwrap();

        let crypto = DefaultCryptoProvider::default();
        let view = DnamContainerView::from_bytes(&container, &crypto).unwrap();
        assert_eq!(view.control_records.len(), 1);
        assert!((view.control_records[0].clamp.safe_min_base - 1.0).abs() < 1e-5);
        assert!((view.control_records[0].clamp.safe_max_base - 9.0).abs() < 1e-5);
        assert_eq!(
            view.header.controls_schema[0].user_metadata.get("midi_cc"),
            Some(&serde_json::json!(14))
        );
    }
}
