//! Stage III: Early SIMD FIR + Orthonormal Kautz Modal Bank (96 B Lite RAM).
//!
//! Canonical Reference: `docs/dsp/router_and_stages_topology.md` §2.3 & `docs/dsp/op_m_modal_kautz.md` §4.

use crate::operators::op_m::KautzModalBank;

/// State RAM footprint for Stage III Lite: strictly 96 B.
pub const STAGE3_CABINET_RAM_BYTES: usize = 96;

/// Stage III: Early FIR + Orthonormal Kautz Modal Bank (96 B Lite RAM).
#[repr(C, align(64))]
#[derive(Clone, Debug)]
pub struct Stage3CabinetModals {
    /// Early reflection FIR circular delay history (16 taps in Lite).
    pub fir_history: [f32; 16],
    pub fir_write_ptr: usize,
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
            fir_history: [0.0; 16],
            fir_write_ptr: 0,
            kautz_bank: KautzModalBank::new(),
        }
    }

    pub fn reset(&mut self) {
        self.fir_history.fill(0.0);
        self.fir_write_ptr = 0;
        self.kautz_bank.reset();
    }

    /// Evaluates the 16-tap Early FIR + parallel `clean_sub` feed (without stepping the internal
    /// Lite Kautz bank when external tier-local Kautz banks are used), returning `(fir_plus_sub, x_delayed_16)`.
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

        let ptr = self.fir_write_ptr & 0x0F;
        let x_delayed_16 = self.fir_history[ptr];
        self.fir_history[ptr] = x;
        self.fir_write_ptr = (ptr + 1) & 0x0F;

        let mut acc = V::zero();
        for blk in 0..4 {
            let i = blk * 4;
            let h = V::from_array([
                self.fir_history[(ptr + 16 - i) & 0x0F],
                self.fir_history[(ptr + 15 - i) & 0x0F],
                self.fir_history[(ptr + 14 - i) & 0x0F],
                self.fir_history[(ptr + 13 - i) & 0x0F],
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
        (early_fir + clean_sub, x_delayed_16)
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
        use crate::math::SimdVec4;
        let ptr_l = self.fir_write_ptr & 0x0F;
        let ptr_r = right.fir_write_ptr & 0x0F;
        let x_del_l = self.fir_history[ptr_l];
        let x_del_r = right.fir_history[ptr_r];
        self.fir_history[ptr_l] = x_l;
        right.fir_history[ptr_r] = x_r;
        self.fir_write_ptr = (ptr_l + 1) & 0x0F;
        right.fir_write_ptr = (ptr_r + 1) & 0x0F;

        let tap_chunks: &[[f32; 4]; 4] = zerocopy::transmute_ref!(fir_taps);
        let mut acc_l = B::Vec4::zero();
        let mut acc_r = B::Vec4::zero();
        for (blk, tap_chunk) in tap_chunks.iter().enumerate() {
            let i = blk * 4;
            let idx0_l = (ptr_l + 16 - i) & 0x0F;
            let idx1_l = (ptr_l + 15 - i) & 0x0F;
            let idx2_l = (ptr_l + 14 - i) & 0x0F;
            let idx3_l = (ptr_l + 13 - i) & 0x0F;
            let hl = B::Vec4::from_array([
                self.fir_history[idx0_l],
                self.fir_history[idx1_l],
                self.fir_history[idx2_l],
                self.fir_history[idx3_l],
            ]);
            let hr = B::Vec4::from_array([
                right.fir_history[(ptr_r + 16 - i) & 0x0F],
                right.fir_history[(ptr_r + 15 - i) & 0x0F],
                right.fir_history[(ptr_r + 14 - i) & 0x0F],
                right.fir_history[(ptr_r + 13 - i) & 0x0F],
            ]);
            let t = B::Vec4::load(tap_chunk);
            acc_l = acc_l.fma(hl, t);
            acc_r = acc_r.fma(hr, t);
        }
        (
            acc_l.horizontal_sum() + clean_sub_l,
            acc_r.horizontal_sum() + clean_sub_r,
            x_del_l,
            x_del_r,
        )
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
        assert_eq!(cab.fir_history, [0.0; 16]);
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
