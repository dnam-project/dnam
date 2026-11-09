//! Stage III: Early SIMD FIR + Orthonormal Kautz Modal Bank (96 B Lite RAM).
//!
//! Canonical Reference: `docs/dsp/router_and_stages_topology.md` §2.3 & `docs/dsp/op_m_modal_kautz.md` §4.

use crate::operators::op_m::KautzModalBank;

/// State RAM footprint for Stage III Lite: strictly 96 B.
pub const STAGE3_CABINET_RAM_BYTES: usize = 96;

/// Circular buffer capacity for Stage III early reflection FIR & ZOH onset delay (`D_onset in [14, 63]`).
pub const STAGE3_FIR_HISTORY_CAPACITY: usize = 64;
const STAGE3_FIR_HISTORY_MASK: usize = STAGE3_FIR_HISTORY_CAPACITY - 1;

/// Stage III: Early FIR + Orthonormal Kautz Modal Bank (96 B Lite nominal RAM budget).
#[repr(C, align(64))]
#[derive(Clone, Debug)]
pub struct Stage3CabinetModals {
    /// Early reflection FIR circular delay history (`64` samples to support ZOH-scaled onset delay up to `192 kHz`).
    pub fir_history: [f32; STAGE3_FIR_HISTORY_CAPACITY],
    pub fir_write_ptr: usize,
    /// ZOH-rescaled Kautz modal bank onset delay in samples (`D_onset = round(16 / rho_sr)`, default `16`).
    pub onset_delay_samples: usize,
    /// 4-pole Orthonormal Kautz Modal Bank for Lite tier tail decay.
    pub kautz_bank: KautzModalBank<4>,
}

impl Default for Stage3CabinetModals {
    fn default() -> Self {
        Self::new()
    }
}

impl Stage3CabinetModals {
    pub const fn new() -> Self {
        Self {
            fir_history: [0.0; STAGE3_FIR_HISTORY_CAPACITY],
            fir_write_ptr: 0,
            onset_delay_samples: 16,
            kautz_bank: KautzModalBank::new(),
        }
    }

    pub fn reset(&mut self) {
        self.fir_history.fill(0.0);
        self.fir_write_ptr = 0;
        self.kautz_bank.reset();
    }

    /// Reconfigures the Kautz onset tap delay $D_{\text{onset}} = \operatorname{round}(16 / \rho_{\text{sr}}) \in [14, 63]$ (`ref[dsp.zoh.stage3-fir-onset]`).
    pub fn reconfigure_sample_rate(&mut self, zoh: &crate::container::ZohRescaledParams) {
        let d_f64 = zoh.rescale_alignment_delay(16.0) as f64;
        self.onset_delay_samples = (d_f64.round() as usize).clamp(14, STAGE3_FIR_HISTORY_MASK);
    }

    /// Evaluates the 16-tap Early FIR + parallel `clean_sub` feed (without stepping the internal
    /// Lite Kautz bank when external tier-local Kautz banks are used), returning `(fir_plus_sub, x_delayed_onset)`.
    /// Direct callers must hold a [`crate::FtzDazGuard`] for the enclosing batch.
    #[cfg_attr(feature = "hotpath", hotpath::measure)]
    #[inline(always)]
    pub fn step_fir_only_with_delayed_tap(
        &mut self,
        x: f32,
        clean_sub: f32,
        fir_taps: &[f32; 16],
    ) -> (f32, f32) {
        use crate::math::{DefaultMathBackend, MathBackend, SimdVec4};
        type V = <DefaultMathBackend as MathBackend>::Vec4;

        let ptr = self.fir_write_ptr & STAGE3_FIR_HISTORY_MASK;
        let d_onset = self.onset_delay_samples.clamp(1, STAGE3_FIR_HISTORY_MASK);
        let x_delayed = self.fir_history
            [(ptr + STAGE3_FIR_HISTORY_CAPACITY - d_onset) & STAGE3_FIR_HISTORY_MASK];
        self.fir_history[ptr] = x;
        self.fir_write_ptr = (ptr + 1) & STAGE3_FIR_HISTORY_MASK;

        let mut acc = V::zero();
        for blk in 0..4 {
            let i = blk * 4;
            let h = V::from_array([
                self.fir_history[(ptr + STAGE3_FIR_HISTORY_CAPACITY - i) & STAGE3_FIR_HISTORY_MASK],
                self.fir_history
                    [(ptr + STAGE3_FIR_HISTORY_CAPACITY - 1 - i) & STAGE3_FIR_HISTORY_MASK],
                self.fir_history
                    [(ptr + STAGE3_FIR_HISTORY_CAPACITY - 2 - i) & STAGE3_FIR_HISTORY_MASK],
                self.fir_history
                    [(ptr + STAGE3_FIR_HISTORY_CAPACITY - 3 - i) & STAGE3_FIR_HISTORY_MASK],
            ]);
            let t = V::from_array([
                fir_taps[i],
                fir_taps[i + 1],
                fir_taps[i + 2],
                fir_taps[i + 3],
            ]);
            acc = acc.fma(h, t);
        }
        let early_fir = acc.horizontal_sum();
        (early_fir + clean_sub, x_delayed)
    }

    /// Evaluates a stereo pair (`self` = Left, `right` = Right) of 16-tap Early FIRs sharing
    /// `fir_taps`, loading each 4-tap coefficient vector once (`0 FDIV`, `0 FSQRT`).
    #[inline(always)]
    pub fn step_fir_stereo_with_delayed_tap<B: crate::math::MathBackend>(
        &mut self,
        right: &mut Self,
        x_l: f32,
        x_r: f32,
        clean_sub_l: f32,
        clean_sub_r: f32,
        fir_taps: &[f32; 16],
    ) -> (f32, f32, f32, f32) {
        self.step_fir_stereo_lr_taps_with_delayed_tap::<B>(
            right,
            x_l,
            x_r,
            clean_sub_l,
            clean_sub_r,
            fir_taps,
            fir_taps,
        )
    }

    /// Evaluates a stereo pair (`self` = Left, `right` = Right) of 16-tap Early FIRs with
    /// distinct Left/Right tap arrays `fir_taps_l` and `fir_taps_r` (`0 FDIV`, `0 FSQRT`).
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    pub fn step_fir_stereo_lr_taps_with_delayed_tap<B: crate::math::MathBackend>(
        &mut self,
        right: &mut Self,
        x_l: f32,
        x_r: f32,
        clean_sub_l: f32,
        clean_sub_r: f32,
        fir_taps_l: &[f32; 16],
        fir_taps_r: &[f32; 16],
    ) -> (f32, f32, f32, f32) {
        use crate::math::SimdVec4;
        let ptr_l = self.fir_write_ptr & STAGE3_FIR_HISTORY_MASK;
        let ptr_r = right.fir_write_ptr & STAGE3_FIR_HISTORY_MASK;
        let d_onset_l = self.onset_delay_samples.clamp(1, STAGE3_FIR_HISTORY_MASK);
        let d_onset_r = right.onset_delay_samples.clamp(1, STAGE3_FIR_HISTORY_MASK);
        let x_del_l = self.fir_history
            [(ptr_l + STAGE3_FIR_HISTORY_CAPACITY - d_onset_l) & STAGE3_FIR_HISTORY_MASK];
        let x_del_r = right.fir_history
            [(ptr_r + STAGE3_FIR_HISTORY_CAPACITY - d_onset_r) & STAGE3_FIR_HISTORY_MASK];
        self.fir_history[ptr_l] = x_l;
        right.fir_history[ptr_r] = x_r;
        self.fir_write_ptr = (ptr_l + 1) & STAGE3_FIR_HISTORY_MASK;
        right.fir_write_ptr = (ptr_r + 1) & STAGE3_FIR_HISTORY_MASK;

        let tap_chunks_l: &[[f32; 4]; 4] = zerocopy::transmute_ref!(fir_taps_l);
        let tap_chunks_r: &[[f32; 4]; 4] = zerocopy::transmute_ref!(fir_taps_r);
        let mut acc_l = B::Vec4::zero();
        let mut acc_r = B::Vec4::zero();
        for blk in 0..4 {
            let i = blk * 4;
            let idx0_l = (ptr_l + STAGE3_FIR_HISTORY_CAPACITY - i) & STAGE3_FIR_HISTORY_MASK;
            let idx1_l = (ptr_l + STAGE3_FIR_HISTORY_CAPACITY - 1 - i) & STAGE3_FIR_HISTORY_MASK;
            let idx2_l = (ptr_l + STAGE3_FIR_HISTORY_CAPACITY - 2 - i) & STAGE3_FIR_HISTORY_MASK;
            let idx3_l = (ptr_l + STAGE3_FIR_HISTORY_CAPACITY - 3 - i) & STAGE3_FIR_HISTORY_MASK;
            let hl = B::Vec4::from_array([
                self.fir_history[idx0_l],
                self.fir_history[idx1_l],
                self.fir_history[idx2_l],
                self.fir_history[idx3_l],
            ]);
            let hr = B::Vec4::from_array([
                right.fir_history
                    [(ptr_r + STAGE3_FIR_HISTORY_CAPACITY - i) & STAGE3_FIR_HISTORY_MASK],
                right.fir_history
                    [(ptr_r + STAGE3_FIR_HISTORY_CAPACITY - 1 - i) & STAGE3_FIR_HISTORY_MASK],
                right.fir_history
                    [(ptr_r + STAGE3_FIR_HISTORY_CAPACITY - 2 - i) & STAGE3_FIR_HISTORY_MASK],
                right.fir_history
                    [(ptr_r + STAGE3_FIR_HISTORY_CAPACITY - 3 - i) & STAGE3_FIR_HISTORY_MASK],
            ]);
            let tl = B::Vec4::load(&tap_chunks_l[blk]);
            let tr = B::Vec4::load(&tap_chunks_r[blk]);
            acc_l = acc_l.fma(hl, tl);
            acc_r = acc_r.fma(hr, tr);
        }
        (
            acc_l.horizontal_sum() + clean_sub_l,
            acc_r.horizontal_sum() + clean_sub_r,
            x_del_l,
            x_del_r,
        )
    }

    /// Evaluates higher-tier (`t = 1..fir_loop_tiers`) 16-tap segments of the contiguous 64-tap
    /// Stage III Early FIR against delay offsets `16..31` (`Full`), `32..47` (`Pro`), and `48..63` (`Ultra`),
    /// scaled by `g_fir` (`0 FDIV`, `0 FSQRT`). Must be called immediately after `step_fir_stereo_lr_taps_with_delayed_tap`.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    pub fn eval_higher_tier_fir_segments<B: crate::math::MathBackend>(
        &self,
        right: &Self,
        s3_fir_sub_l: f32,
        s3_fir_sub_r: f32,
        clean_sub_l: f32,
        clean_sub_r: f32,
        fir_taps_l: &[[f32; 16]; 4],
        fir_taps_r: &[[f32; 16]; 4],
        fir_loop_tiers: usize,
        g_fir: f32,
    ) -> ([f32; 4], [f32; 4]) {
        use crate::math::SimdVec4;
        let mut out_l = [0.0f32; 4];
        let mut out_r = [0.0f32; 4];
        if (g_fir - 1.0).abs() <= 1e-9 {
            out_l[0] = s3_fir_sub_l;
            out_r[0] = s3_fir_sub_r;
        } else {
            out_l[0] = (s3_fir_sub_l - clean_sub_l) * g_fir + clean_sub_l;
            out_r[0] = (s3_fir_sub_r - clean_sub_r) * g_fir + clean_sub_r;
        }
        let latest_l =
            (self.fir_write_ptr + STAGE3_FIR_HISTORY_CAPACITY - 1) & STAGE3_FIR_HISTORY_MASK;
        let latest_r =
            (right.fir_write_ptr + STAGE3_FIR_HISTORY_CAPACITY - 1) & STAGE3_FIR_HISTORY_MASK;
        let max_t = fir_loop_tiers.min(4);
        for t in 1..max_t {
            let base_delay = t * 16;
            let tap_chunks_l: &[[f32; 4]; 4] = zerocopy::transmute_ref!(&fir_taps_l[t]);
            let tap_chunks_r: &[[f32; 4]; 4] = zerocopy::transmute_ref!(&fir_taps_r[t]);
            let mut acc_l = B::Vec4::zero();
            let mut acc_r = B::Vec4::zero();
            for blk in 0..4 {
                let d = base_delay + blk * 4;
                let hl = B::Vec4::from_array([
                    self.fir_history
                        [(latest_l + STAGE3_FIR_HISTORY_CAPACITY - d) & STAGE3_FIR_HISTORY_MASK],
                    self.fir_history[(latest_l + STAGE3_FIR_HISTORY_CAPACITY - 1 - d)
                        & STAGE3_FIR_HISTORY_MASK],
                    self.fir_history[(latest_l + STAGE3_FIR_HISTORY_CAPACITY - 2 - d)
                        & STAGE3_FIR_HISTORY_MASK],
                    self.fir_history[(latest_l + STAGE3_FIR_HISTORY_CAPACITY - 3 - d)
                        & STAGE3_FIR_HISTORY_MASK],
                ]);
                let hr = B::Vec4::from_array([
                    right.fir_history
                        [(latest_r + STAGE3_FIR_HISTORY_CAPACITY - d) & STAGE3_FIR_HISTORY_MASK],
                    right.fir_history[(latest_r + STAGE3_FIR_HISTORY_CAPACITY - 1 - d)
                        & STAGE3_FIR_HISTORY_MASK],
                    right.fir_history[(latest_r + STAGE3_FIR_HISTORY_CAPACITY - 2 - d)
                        & STAGE3_FIR_HISTORY_MASK],
                    right.fir_history[(latest_r + STAGE3_FIR_HISTORY_CAPACITY - 3 - d)
                        & STAGE3_FIR_HISTORY_MASK],
                ]);
                let tl = B::Vec4::load(&tap_chunks_l[blk]);
                let tr = B::Vec4::load(&tap_chunks_r[blk]);
                acc_l = acc_l.fma(hl, tl);
                acc_r = acc_r.fma(hr, tr);
            }
            out_l[t] = acc_l.horizontal_sum() * g_fir;
            out_r[t] = acc_r.horizontal_sum() * g_fir;
        }
        (out_l, out_r)
    }

    /// Evaluates Stage III for input `x` and parallel `clean_sub` feed, returning `(stage3_out, x_delayed_16)`
    /// so external Kautz modal tiers can be driven by the 16-sample-delayed onset tap $x[n - 16]$ (`ref[dsp.stage3.cabinet-bank]`).
    /// Direct callers must hold a [`crate::FtzDazGuard`] for the enclosing batch.
    #[inline(always)]
    pub fn step_with_delayed_tap(
        &mut self,
        x: f32,
        clean_sub: f32,
        fir_taps: &[f32; 16],
    ) -> (f32, f32) {
        let (fir_sub, x_delayed_16) = self.step_fir_only_with_delayed_tap(x, clean_sub, fir_taps);
        let tail = self.kautz_bank.step(x_delayed_16);
        (fir_sub + tail, x_delayed_16)
    }

    /// Evaluates Stage III for input `x` and parallel `clean_sub` feed.
    /// Direct callers must hold a [`crate::FtzDazGuard`] for the enclosing batch.
    #[inline(always)]
    pub fn step(&mut self, x: f32, clean_sub: f32, fir_taps: &[f32; 16]) -> f32 {
        self.step_with_delayed_tap(x, clean_sub, fir_taps).0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_stage3_cabinet_modals() {
        let _audio = crate::FtzDazGuard::new();
        let mut cab = Stage3CabinetModals::default();
        cab.reset();
        let taps = [0.1f32; 16];
        let out = cab.step(1.0, 0.2, &taps);
        assert!(out.is_finite());
        assert_eq!(cab.fir_write_ptr, 1);
        cab.reset();
        assert_eq!(cab.fir_write_ptr, 0);
        assert_eq!(cab.fir_history, [0.0; STAGE3_FIR_HISTORY_CAPACITY]);
    }

    proptest! {
        #[test]
        fn prop_stage3_cabinet_linearity_and_dc_gain(
            x in -2.0_f32..2.0_f32,
            clean_sub in -1.0_f32..1.0_f32,
            tap_val in -0.2_f32..0.2_f32,
            scale in -3.0_f32..3.0_f32,
        ) {
            let _audio = crate::FtzDazGuard::new();
            let taps = [tap_val; 16];
            let mut cab1 = Stage3CabinetModals::default();
            let mut cab2 = Stage3CabinetModals::default();

            let y1 = cab1.step(x, clean_sub, &taps);
            let y2 = cab2.step(x * scale, clean_sub * scale, &taps);
            prop_assert!((y1 * scale - y2).abs() < 1e-4);

            // After 16 constant steps with zero Kautz residues, output equals 16 * tap_val * x + clean_sub
            let mut cab_dc = Stage3CabinetModals::default();
            let mut last = 0.0;
            for _ in 0..16 {
                last = cab_dc.step(x, clean_sub, &taps);
            }
            let expected = 16.0 * tap_val * x + clean_sub;
            prop_assert!((last - expected).abs() < 1e-4);
        }
    }
}
