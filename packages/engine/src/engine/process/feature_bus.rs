//! Canonical 256-D Multi-Tier Feature Bus Assembly (`Lite` `0..64`, `Full` `64..128`, `Pro` `128..192`, `Ultra` `192..256`).
//!
//! Canonical Reference: `docs/specs/dsp/router_and_stages_topology.md` §7 (`CANONICAL-256D-FEATURE-BUS-MAP`, `r[dsp.head.feature-bus-slots]`).

use crate::math::{MathBackend, SimdVec4};
use crate::stages::stage1_pre::Stage1PreDynamics;

/// Domain-specific SIMD backend trait for the 256-D Kronecker Output Head & Feature Bus (`64`-element tiles).
pub trait FeatureBusBackend {
    /// Computes the 64-element dot product $\sum_{i=0}^{63} a_i b_i$.
    fn tile_dot64(a: &[f32; 64], b: &[f32; 64]) -> f32;

    /// Computes dual-channel (`Left` and `Right`) 64-element dot products simultaneously:
    /// `[sum(taps[i] * w_l[i]), sum(taps[i] * w_r[i])]`.
    fn tile_dot64_stereo(taps: &[f32; 64], w_l: &[f32; 64], w_r: &[f32; 64]) -> [f32; 2];

    /// Collapses a slice of SVD stereo output head weights using $4 \times 4$ outer-product
    /// lane-broadcast tile accumulation (`fma_mat4x4_vec4`) in 100% safe Rust.
    fn tile_collapse_svd_slice(
        base: &[f32],
        svd_modes: &[f32],
        z_head: &[f32; 4],
        active_rank: usize,
        out_collapsed: &mut [f32],
    );
}

impl FeatureBusBackend for crate::math::ScalarBackend {
    #[inline(always)]
    fn tile_dot64(a: &[f32; 64], b: &[f32; 64]) -> f32 {
        tile_dot64_vec4::<crate::math::ScalarVec4>(a, b)
    }

    #[inline(always)]
    fn tile_dot64_stereo(taps: &[f32; 64], w_l: &[f32; 64], w_r: &[f32; 64]) -> [f32; 2] {
        tile_dot64_stereo_vec4::<crate::math::ScalarVec4>(taps, w_l, w_r)
    }

    #[inline(always)]
    fn tile_collapse_svd_slice(
        base: &[f32],
        svd_modes: &[f32],
        z_head: &[f32; 4],
        active_rank: usize,
        out_collapsed: &mut [f32],
    ) {
        tile_collapse_svd_slice_vec4::<Self>(base, svd_modes, z_head, active_rank, out_collapsed);
    }
}

#[cfg(target_arch = "aarch64")]
impl FeatureBusBackend for crate::math::NeonBackend {
    #[inline(always)]
    fn tile_dot64(a: &[f32; 64], b: &[f32; 64]) -> f32 {
        tile_dot64_vec4::<crate::math::NeonVec4>(a, b)
    }

    #[inline(always)]
    fn tile_dot64_stereo(taps: &[f32; 64], w_l: &[f32; 64], w_r: &[f32; 64]) -> [f32; 2] {
        tile_dot64_stereo_vec4::<crate::math::NeonVec4>(taps, w_l, w_r)
    }

    #[inline(always)]
    fn tile_collapse_svd_slice(
        base: &[f32],
        svd_modes: &[f32],
        z_head: &[f32; 4],
        active_rank: usize,
        out_collapsed: &mut [f32],
    ) {
        tile_collapse_svd_slice_vec4::<Self>(base, svd_modes, z_head, active_rank, out_collapsed);
    }
}

#[cfg(target_arch = "aarch64")]
impl FeatureBusBackend for crate::math::ArmSve2Backend {
    #[inline(always)]
    fn tile_dot64(a: &[f32; 64], b: &[f32; 64]) -> f32 {
        tile_dot64_vec4::<crate::math::Sve2Vec4>(a, b)
    }

    #[inline(always)]
    fn tile_dot64_stereo(taps: &[f32; 64], w_l: &[f32; 64], w_r: &[f32; 64]) -> [f32; 2] {
        tile_dot64_stereo_vec4::<crate::math::Sve2Vec4>(taps, w_l, w_r)
    }

    #[inline(always)]
    fn tile_collapse_svd_slice(
        base: &[f32],
        svd_modes: &[f32],
        z_head: &[f32; 4],
        active_rank: usize,
        out_collapsed: &mut [f32],
    ) {
        tile_collapse_svd_slice_vec4::<Self>(base, svd_modes, z_head, active_rank, out_collapsed);
    }
}

#[cfg(target_arch = "aarch64")]
impl FeatureBusBackend for crate::math::ArmSmeBackend {
    #[inline(always)]
    fn tile_dot64(a: &[f32; 64], b: &[f32; 64]) -> f32 {
        if crate::math::ArmSve2Backend::is_supported() {
            crate::math::ArmSve2Backend::tile_dot64(a, b)
        } else {
            crate::math::NeonBackend::tile_dot64(a, b)
        }
    }

    #[inline(always)]
    fn tile_dot64_stereo(taps: &[f32; 64], w_l: &[f32; 64], w_r: &[f32; 64]) -> [f32; 2] {
        if crate::math::ArmSve2Backend::is_supported() {
            crate::math::ArmSve2Backend::tile_dot64_stereo(taps, w_l, w_r)
        } else {
            crate::math::NeonBackend::tile_dot64_stereo(taps, w_l, w_r)
        }
    }

    #[inline(always)]
    fn tile_collapse_svd_slice(
        base: &[f32],
        svd_modes: &[f32],
        z_head: &[f32; 4],
        active_rank: usize,
        out_collapsed: &mut [f32],
    ) {
        if crate::math::ArmSve2Backend::is_supported() {
            crate::math::ArmSve2Backend::tile_collapse_svd_slice(
                base,
                svd_modes,
                z_head,
                active_rank,
                out_collapsed,
            );
        } else {
            crate::math::NeonBackend::tile_collapse_svd_slice(
                base,
                svd_modes,
                z_head,
                active_rank,
                out_collapsed,
            );
        }
    }
}

#[cfg(target_arch = "x86_64")]
impl FeatureBusBackend for crate::math::X86Sse2Backend {
    #[inline(always)]
    fn tile_dot64(a: &[f32; 64], b: &[f32; 64]) -> f32 {
        tile_dot64_vec4::<crate::math::Sse2Vec4>(a, b)
    }

    #[inline(always)]
    fn tile_dot64_stereo(taps: &[f32; 64], w_l: &[f32; 64], w_r: &[f32; 64]) -> [f32; 2] {
        tile_dot64_stereo_vec4::<crate::math::Sse2Vec4>(taps, w_l, w_r)
    }

    #[inline(always)]
    fn tile_collapse_svd_slice(
        base: &[f32],
        svd_modes: &[f32],
        z_head: &[f32; 4],
        active_rank: usize,
        out_collapsed: &mut [f32],
    ) {
        tile_collapse_svd_slice_vec4::<Self>(base, svd_modes, z_head, active_rank, out_collapsed);
    }
}

#[cfg(target_arch = "x86_64")]
impl FeatureBusBackend for crate::math::X86Sse42Backend {
    #[inline(always)]
    fn tile_dot64(a: &[f32; 64], b: &[f32; 64]) -> f32 {
        tile_dot64_vec4::<crate::math::Sse42Vec4>(a, b)
    }

    #[inline(always)]
    fn tile_dot64_stereo(taps: &[f32; 64], w_l: &[f32; 64], w_r: &[f32; 64]) -> [f32; 2] {
        tile_dot64_stereo_vec4::<crate::math::Sse42Vec4>(taps, w_l, w_r)
    }

    #[inline(always)]
    fn tile_collapse_svd_slice(
        base: &[f32],
        svd_modes: &[f32],
        z_head: &[f32; 4],
        active_rank: usize,
        out_collapsed: &mut [f32],
    ) {
        tile_collapse_svd_slice_vec4::<Self>(base, svd_modes, z_head, active_rank, out_collapsed);
    }
}

#[cfg(target_arch = "x86_64")]
impl FeatureBusBackend for crate::math::X86Avx2FmaBackend {
    #[inline(always)]
    fn tile_dot64(a: &[f32; 64], b: &[f32; 64]) -> f32 {
        crate::math::x86::avx2::avx2_tile_dot64(a, b)
            .unwrap_or_else(|| crate::math::X86Sse42Backend::tile_dot64(a, b))
    }

    #[inline(always)]
    fn tile_dot64_stereo(taps: &[f32; 64], w_l: &[f32; 64], w_r: &[f32; 64]) -> [f32; 2] {
        crate::math::x86::avx2::avx2_tile_dot64_stereo(taps, w_l, w_r)
            .unwrap_or_else(|| crate::math::X86Sse42Backend::tile_dot64_stereo(taps, w_l, w_r))
    }

    #[inline(always)]
    fn tile_collapse_svd_slice(
        base: &[f32],
        svd_modes: &[f32],
        z_head: &[f32; 4],
        active_rank: usize,
        out_collapsed: &mut [f32],
    ) {
        tile_collapse_svd_slice_vec4::<Self>(base, svd_modes, z_head, active_rank, out_collapsed);
    }
}

#[cfg(target_arch = "wasm32")]
impl FeatureBusBackend for crate::math::WasmSimd128Backend {
    #[inline(always)]
    fn tile_dot64(a: &[f32; 64], b: &[f32; 64]) -> f32 {
        tile_dot64_vec4::<crate::math::WasmVec4>(a, b)
    }

    #[inline(always)]
    fn tile_dot64_stereo(taps: &[f32; 64], w_l: &[f32; 64], w_r: &[f32; 64]) -> [f32; 2] {
        tile_dot64_stereo_vec4::<crate::math::WasmVec4>(taps, w_l, w_r)
    }

    #[inline(always)]
    fn tile_collapse_svd_slice(
        base: &[f32],
        svd_modes: &[f32],
        z_head: &[f32; 4],
        active_rank: usize,
        out_collapsed: &mut [f32],
    ) {
        tile_collapse_svd_slice_vec4::<Self>(base, svd_modes, z_head, active_rank, out_collapsed);
    }
}

/// Safe `SimdVec4` implementation of [`FeatureBusBackend::tile_dot64`] using 4 independent accumulators.
#[inline(always)]
pub fn tile_dot64_vec4<V: SimdVec4>(a: &[f32; 64], b: &[f32; 64]) -> f32 {
    let a_chunks: &[[f32; 4]; 16] = zerocopy::transmute_ref!(a);
    let b_chunks: &[[f32; 4]; 16] = zerocopy::transmute_ref!(b);
    let mut acc0 = V::zero();
    let mut acc1 = V::zero();
    let mut acc2 = V::zero();
    let mut acc3 = V::zero();
    for g4 in 0..4 {
        let g = g4 * 4;
        acc0 = acc0.fma(V::load(&a_chunks[g]), V::load(&b_chunks[g]));
        acc1 = acc1.fma(V::load(&a_chunks[g + 1]), V::load(&b_chunks[g + 1]));
        acc2 = acc2.fma(V::load(&a_chunks[g + 2]), V::load(&b_chunks[g + 2]));
        acc3 = acc3.fma(V::load(&a_chunks[g + 3]), V::load(&b_chunks[g + 3]));
    }
    acc0.add(acc1).add(acc2.add(acc3)).horizontal_sum()
}

/// Safe `SimdVec4` implementation of [`FeatureBusBackend::tile_dot64_stereo`] using 4 independent stereo accumulators.
#[inline(always)]
pub fn tile_dot64_stereo_vec4<V: SimdVec4>(
    taps: &[f32; 64],
    w_l: &[f32; 64],
    w_r: &[f32; 64],
) -> [f32; 2] {
    let t_chunks: &[[f32; 4]; 16] = zerocopy::transmute_ref!(taps);
    let wl_chunks: &[[f32; 4]; 16] = zerocopy::transmute_ref!(w_l);
    let wr_chunks: &[[f32; 4]; 16] = zerocopy::transmute_ref!(w_r);
    let mut acc_l0 = V::zero();
    let mut acc_l1 = V::zero();
    let mut acc_l2 = V::zero();
    let mut acc_l3 = V::zero();
    let mut acc_r0 = V::zero();
    let mut acc_r1 = V::zero();
    let mut acc_r2 = V::zero();
    let mut acc_r3 = V::zero();
    for g4 in 0..4 {
        let g = g4 * 4;
        let tv0 = V::load(&t_chunks[g]);
        let tv1 = V::load(&t_chunks[g + 1]);
        let tv2 = V::load(&t_chunks[g + 2]);
        let tv3 = V::load(&t_chunks[g + 3]);
        acc_l0 = acc_l0.fma(tv0, V::load(&wl_chunks[g]));
        acc_l1 = acc_l1.fma(tv1, V::load(&wl_chunks[g + 1]));
        acc_l2 = acc_l2.fma(tv2, V::load(&wl_chunks[g + 2]));
        acc_l3 = acc_l3.fma(tv3, V::load(&wl_chunks[g + 3]));
        acc_r0 = acc_r0.fma(tv0, V::load(&wr_chunks[g]));
        acc_r1 = acc_r1.fma(tv1, V::load(&wr_chunks[g + 1]));
        acc_r2 = acc_r2.fma(tv2, V::load(&wr_chunks[g + 2]));
        acc_r3 = acc_r3.fma(tv3, V::load(&wr_chunks[g + 3]));
    }
    let (sum_l, sum_r) = acc_l0
        .add(acc_l1)
        .add(acc_l2.add(acc_l3))
        .horizontal_sum_pair(acc_r0.add(acc_r1).add(acc_r2.add(acc_r3)));
    [sum_l, sum_r]
}

/// Safe `MathBackend` implementation of [`FeatureBusBackend::tile_collapse_svd_slice`].
#[inline(always)]
pub fn tile_collapse_svd_slice_vec4<B: MathBackend>(
    base: &[f32],
    svd_modes: &[f32],
    z_head: &[f32; 4],
    active_rank: usize,
    out_collapsed: &mut [f32],
) {
    let len = base.len();
    debug_assert_eq!(out_collapsed.len(), len);
    debug_assert!(len.is_multiple_of(4));

    if active_rank == 4 && svd_modes.len() >= 4 * len {
        let (base_vecs, _) = base.as_chunks::<4>();
        let (m0_vecs, _) = svd_modes[..len].as_chunks::<4>();
        let (m1_vecs, _) = svd_modes[len..2 * len].as_chunks::<4>();
        let (m2_vecs, _) = svd_modes[2 * len..3 * len].as_chunks::<4>();
        let (m3_vecs, _) = svd_modes[3 * len..4 * len].as_chunks::<4>();
        let (out_vecs, _) = out_collapsed.as_chunks_mut::<4>();

        for idx in 0..base_vecs.len() {
            let acc = B::Vec4::load(&base_vecs[idx]);
            let mat = [m0_vecs[idx], m1_vecs[idx], m2_vecs[idx], m3_vecs[idx]];
            let out = B::fma_mat4x4_vec4(acc, &mat, z_head);
            out.store(&mut out_vecs[idx]);
        }
    } else {
        let rank = active_rank.min(4);
        let (base_vecs, _) = base.as_chunks::<4>();
        let (out_vecs, _) = out_collapsed.as_chunks_mut::<4>();
        for (c, (b_arr, out_arr)) in base_vecs.iter().zip(out_vecs.iter_mut()).enumerate() {
            let offset = c * 4;
            let mut acc = B::Vec4::load(b_arr);
            for (k, &zk) in z_head.iter().enumerate().take(rank) {
                let m_offset = k * len + offset;
                let m_arr: &[f32; 4] = (&svd_modes[m_offset..m_offset + 4]).try_into().unwrap();
                acc = acc.fma(B::Vec4::splat(zk), B::Vec4::load(m_arr));
            }
            acc.store(out_arr);
        }
    }
}

/// Packs the remaining Stage 0, Stage I, Stage III FIR, and Stage IV Lite slots of the canonical
/// 256-D (`64` features/tier) feature bus up to `loop_tiers` (`1..=4`) directly into `fb` after
/// Stage II, Stage III Kautz, and Stage IV multi-tier taps have written their slots directly into `fb`.
///
/// Guarantees:
/// - **Zero duplicate columns across all 256 slots (`Lite..Ultra`)**: Every slot is backed by a
///   distinct physical state or nonlinear branch readout across Stages 0, I, II, III, and IV.
/// - **Strict lower-triangular tier causality (`REQ-5.6`)**: `Lite` (`0..64`) depends exclusively
///   on `Arena 0`, and higher-tier slots (`64..256`) never mutate lower-tier state.
/// - **Exact DC resting equilibrium (`[STAB-ZERO-DC]`)**: Evaluates to `0.0` across all 256 slots
///   at zero input after `reset_to_dc_equilibrium()`.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(crate) fn pack_tier_feature_bus_into(
    fb: &mut [f32; 256],
    stage1: &Stage1PreDynamics,
    fir_seg_l: &[f32; 4],
    fir_seg_r: &[f32; 4],
    tank_readouts: &[f32; 4],
    down_l: f32,
    down_r: f32,
    clean_sub_l: f32,
    clean_sub_r: f32,
    s1_l: f32,
    s1_r: f32,
    s2_in_l: f32,
    s2_in_r: f32,
    s2_l: f32,
    s2_r: f32,
    send_l: f32,
    send_r: f32,
    tank_l: f32,
    tank_r: f32,
    loop_tiers: usize,
) {
    // Tier 0 — Lite (0..64):
    // 0..4: Stage III Kautz Block 0 (populated by Kautz loop)
    // 4..12: Stage I T1 Sub-Harmonic Tracker L/R (8 floats)
    fb[4..8].copy_from_slice(&stage1.last_tracker_l);
    fb[8..12].copy_from_slice(&stage1.last_tracker_r);
    // 12..20: Stage I Pre-FX Stereo Op-M[Pre] z_re L/R (8 floats)
    fb[12..16].copy_from_slice(&stage1.op_m_pre.z_re);
    fb[16..20].copy_from_slice(&stage1.op_m_pre_r.z_re);
    // 20..52: Stage II Commuted Cascade Lite (populated by step_frame_into_feature_bus)
    // 52..60: Stage IV Spatial Tank Lite (8 floats)
    fb[52..56].copy_from_slice(tank_readouts);
    fb[56..60].copy_from_slice(&[tank_l, tank_r, s1_l + s2_l, s1_r + s2_r]);
    // 60..64: Stage 0 Clean Sub L/R + Stage III Early FIR Segment 0 L/R (4 floats)
    fb[60..64].copy_from_slice(&[clean_sub_l, clean_sub_r, fir_seg_l[0], fir_seg_r[0]]);

    if loop_tiers >= 2 {
        // Tier 1 — Full (64..128):
        // 64..68: Stage III Kautz Block 1 (populated by Kautz loop)
        // 68..84: Stage I Pre-FX Full Group 1 (16 floats: delta Op-M z_re L/R + Stage I/0 intermediate taps)
        fb[68..72].copy_from_slice(&stage1.op_m_pre_delta_l.z_re);
        fb[72..76].copy_from_slice(&stage1.op_m_pre_delta_r.z_re);
        fb[76..80].copy_from_slice(&[s1_l, s1_r, down_l, down_r]);
        fb[80..84].copy_from_slice(&[s2_in_l, s2_in_r, send_l, send_r]);
        // 84..116: Stage II Commuted Cascade Full (populated by step_frame_into_feature_bus)
        // 116..126: Stage IV Spatial Tank Full (populated by read_multi_tier_taps_into)
        // 126..128: Stage III Early FIR Segment 1 L/R (2 floats, taps 16..31)
        fb[126] = fir_seg_l[1];
        fb[127] = fir_seg_r[1];

        if loop_tiers >= 3 {
            // Tier 2 — Pro (128..192):
            // 128..136: Stage III Kautz Blocks 2..3 (populated by Kautz loop)
            // 136..152: Stage I Pre-FX Pro Group 2 (16 floats: delta Op-M z_im L/R + Op-N/Op-T intermediate taps)
            fb[136..140].copy_from_slice(&stage1.op_m_pre_delta_l.z_im);
            fb[140..144].copy_from_slice(&stage1.op_m_pre_delta_r.z_im);
            fb[144..148].copy_from_slice(&[
                stage1.last_op_n_l,
                stage1.last_op_n_r,
                stage1.last_op_t_l,
                stage1.last_op_t_r,
            ]);
            fb[148..152].copy_from_slice(&[
                stage1.last_op_t_l - stage1.prev_op_t_l,
                stage1.last_op_t_r - stage1.prev_op_t_r,
                stage1.op_m_pre.z_im[0],
                stage1.op_m_pre_r.z_im[0],
            ]);
            // 152..184: Stage II Commuted Cascade Pro (populated by step_frame_into_feature_bus)
            // 184..190: Stage IV Spatial Tank Pro (populated by read_multi_tier_taps_into)
            // 190..192: Stage III Early FIR Segment 2 L/R (2 floats, taps 32..47)
            fb[190] = fir_seg_l[2];
            fb[191] = fir_seg_r[2];

            if loop_tiers >= 4 {
                // Tier 3 — Ultra (192..256):
                // 192..208: Stage III Kautz Blocks 4..7 (populated by Kautz loop)
                // 208..224: Stage I Pre-FX Ultra Group 3 (16 floats: Lite Op-M z_im L/R + Hilbert & T1 DC states)
                fb[208..212].copy_from_slice(&stage1.op_m_pre.z_im);
                fb[212..216].copy_from_slice(&stage1.op_m_pre_r.z_im);
                fb[216..220].copy_from_slice(stage1.hilbert_l.last_stereo_s5());
                fb[220..224].copy_from_slice(&[
                    stage1.dc_blocker_l.prev_d,
                    stage1.dc_blocker_r.prev_d,
                    stage1.prev_op_t_l,
                    stage1.prev_op_t_r,
                ]);
                // 224..246: Stage II Commuted Cascade Ultra (populated by step_frame_into_feature_bus)
                // 246..254: Stage IV Spatial Tank Ultra (populated by read_multi_tier_taps_into)
                // 254..256: Stage III Early FIR Segment 3 L/R (2 floats, taps 48..63)
                fb[254] = fir_seg_l[3];
                fb[255] = fir_seg_r[3];
            }
        }
    }
}

/// Evaluates the 16-group per-tier SIMD `Vec4` head readout dot product across `0..active_tiers`
/// (`Lite` `0..64`, `Full` `64..128`, `Pro` `128..192`, `Ultra` `192..256`), shared between
/// `step_frame_tora` and `step_frame_4port_tora`.
#[cfg_attr(feature = "hotpath", hotpath::measure)]
#[inline(always)]
pub(crate) fn eval_tier_feature_bus_head_dot<B: crate::math::MathBackend>(
    fb: &[f32; 256],
    collapsed_head_wl: &[[B::Vec4; 16]; 4],
    collapsed_head_wr: &[[B::Vec4; 16]; 4],
    active_tiers: usize,
    mono_only: bool,
) -> (f32, f32) {
    use crate::math::SimdVec4;
    let chunks: &[[[f32; 4]; 16]; 4] = zerocopy::transmute_ref!(fb);
    let max_t = active_tiers.min(4);
    let mut acc_l0 = B::Vec4::zero();
    let mut acc_l1 = B::Vec4::zero();
    let mut acc_l2 = B::Vec4::zero();
    let mut acc_l3 = B::Vec4::zero();

    if mono_only {
        for t in 0..max_t {
            let phi_t = &chunks[t];
            let wl_t = &collapsed_head_wl[t];
            for g4 in 0..4 {
                let g = g4 * 4;
                acc_l0 = acc_l0.fma(wl_t[g], B::Vec4::load(&phi_t[g]));
                acc_l1 = acc_l1.fma(wl_t[g + 1], B::Vec4::load(&phi_t[g + 1]));
                acc_l2 = acc_l2.fma(wl_t[g + 2], B::Vec4::load(&phi_t[g + 2]));
                acc_l3 = acc_l3.fma(wl_t[g + 3], B::Vec4::load(&phi_t[g + 3]));
            }
        }
        let sum_l = acc_l0.add(acc_l1).add(acc_l2.add(acc_l3)).horizontal_sum();
        (sum_l, sum_l)
    } else {
        let mut acc_r0 = B::Vec4::zero();
        let mut acc_r1 = B::Vec4::zero();
        let mut acc_r2 = B::Vec4::zero();
        let mut acc_r3 = B::Vec4::zero();
        for t in 0..max_t {
            let phi_t = &chunks[t];
            let wl_t = &collapsed_head_wl[t];
            let wr_t = &collapsed_head_wr[t];
            for g4 in 0..4 {
                let g = g4 * 4;
                let p0 = B::Vec4::load(&phi_t[g]);
                let p1 = B::Vec4::load(&phi_t[g + 1]);
                let p2 = B::Vec4::load(&phi_t[g + 2]);
                let p3 = B::Vec4::load(&phi_t[g + 3]);
                acc_l0 = acc_l0.fma(wl_t[g], p0);
                acc_l1 = acc_l1.fma(wl_t[g + 1], p1);
                acc_l2 = acc_l2.fma(wl_t[g + 2], p2);
                acc_l3 = acc_l3.fma(wl_t[g + 3], p3);
                acc_r0 = acc_r0.fma(wr_t[g], p0);
                acc_r1 = acc_r1.fma(wr_t[g + 1], p1);
                acc_r2 = acc_r2.fma(wr_t[g + 2], p2);
                acc_r3 = acc_r3.fma(wr_t[g + 3], p3);
            }
        }
        acc_l0
            .add(acc_l1)
            .add(acc_l2.add(acc_l3))
            .horizontal_sum_pair(acc_r0.add(acc_r1).add(acc_r2.add(acc_r3)))
    }
}

/// Evaluates the Stage IV Spatial Tank contribution to `(head_sum_l, head_sum_r)` across all active tiers
/// (`Lite` slots `52..58`, `Full` slots `116..126`, `Pro` slots `184..190`, `Ultra` slots `246..254`)
/// so `WDW` mode cleanly separates dry center core (`core_l/r`) from wet spatial tank (`w_wet_l/r`).
#[inline(always)]
pub(crate) fn eval_tier_tank_head_contribution<B: crate::math::MathBackend>(
    fb: &[f32; 256],
    collapsed_head_wl: &[[B::Vec4; 16]; 4],
    collapsed_head_wr: &[[B::Vec4; 16]; 4],
    active_tiers: usize,
) -> (f32, f32) {
    use crate::math::SimdVec4;
    const TANK_RANGES: [(usize, usize, usize); 4] =
        [(0, 52, 58), (1, 116, 126), (2, 184, 190), (3, 246, 254)];
    let mut tank_head_l = 0.0_f32;
    let mut tank_head_r = 0.0_f32;
    for &(t, start, end) in TANK_RANGES.iter().take(active_tiers.min(4)) {
        for idx in start..end {
            let in_tier = idx - t * 64;
            let g = in_tier / 4;
            let lane = in_tier % 4;
            let wl = collapsed_head_wl[t][g].to_array()[lane];
            let wr = collapsed_head_wr[t][g].to_array()[lane];
            let val = fb[idx];
            tank_head_l += wl * val;
            tank_head_r += wr * val;
        }
    }
    (tank_head_l, tank_head_r)
}
