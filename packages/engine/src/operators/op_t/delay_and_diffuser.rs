//! Op-T Catmull-Rom Delay Line, Granular Windowing, Schroeder Diffuser & Ballistic Ducking VCA.
//!
//! Canonical Reference: `docs/specs/dsp/op_t_transport.md` (`ref[dsp.op-t.reader.catmull-rom]`, `ref[dsp.op-t.transition.dual-mode]`, `ref[dsp.op-t.diffuser.recurrence]`, `ref[dsp.op-t.ducking.vca]`).

use crate::math::{
    SimdVec4, flush_recursive_state_f32,
    fp16::{f32_to_fp16, fp16_to_f32},
};

/// Domain-specific SIMD backend trait for `Op-T` (Transport, Catmull-Rom & Cubic Lagrange Reader).
pub trait OpTBackend<V: SimdVec4> {
    /// Evaluates 4-tap Cubic Lagrange interpolation weights for 4 delay channels simultaneously.
    fn cubic_lagrange_weights_4ch(vmu: V) -> (V, V, V, V);
}

/// Safe `SimdVec4` implementation of [`OpTBackend::cubic_lagrange_weights_4ch`].
#[inline(always)]
pub fn cubic_lagrange_weights_4ch_vec4<V: SimdVec4>(vmu: V) -> (V, V, V, V) {
    let one = V::splat(1.0);
    let two = V::splat(2.0);
    let inv_six = V::splat(1.0 / 6.0);
    let neg_inv_six = V::splat(-1.0 / 6.0);
    let half = V::splat(0.5);
    let neg_half = V::splat(-0.5);

    let mu_p1 = vmu.add(one);
    let mu_m1 = vmu.sub(one);
    let mu_m2 = vmu.sub(two);

    let c_m1 = vmu.mul(mu_m1).mul(mu_m2).mul(neg_inv_six);
    let c_0 = mu_p1.mul(mu_m1).mul(mu_m2).mul(half);
    let c_1 = mu_p1.mul(vmu).mul(mu_m2).mul(neg_half);
    let c_2 = mu_p1.mul(vmu).mul(mu_m1).mul(inv_six);
    (c_m1, c_0, c_1, c_2)
}

impl OpTBackend<crate::math::ScalarVec4> for crate::math::ScalarBackend {
    #[inline(always)]
    fn cubic_lagrange_weights_4ch(
        vmu: crate::math::ScalarVec4,
    ) -> (
        crate::math::ScalarVec4,
        crate::math::ScalarVec4,
        crate::math::ScalarVec4,
        crate::math::ScalarVec4,
    ) {
        cubic_lagrange_weights_4ch_vec4(vmu)
    }
}

#[cfg(target_arch = "aarch64")]
impl OpTBackend<crate::math::NeonVec4> for crate::math::NeonBackend {
    #[inline(always)]
    fn cubic_lagrange_weights_4ch(
        vmu: crate::math::NeonVec4,
    ) -> (
        crate::math::NeonVec4,
        crate::math::NeonVec4,
        crate::math::NeonVec4,
        crate::math::NeonVec4,
    ) {
        cubic_lagrange_weights_4ch_vec4(vmu)
    }
}

#[cfg(target_arch = "aarch64")]
impl OpTBackend<crate::math::Sve2Vec4> for crate::math::ArmSve2Backend {
    #[inline(always)]
    fn cubic_lagrange_weights_4ch(
        vmu: crate::math::Sve2Vec4,
    ) -> (
        crate::math::Sve2Vec4,
        crate::math::Sve2Vec4,
        crate::math::Sve2Vec4,
        crate::math::Sve2Vec4,
    ) {
        cubic_lagrange_weights_4ch_vec4(vmu)
    }
}

#[cfg(target_arch = "aarch64")]
impl OpTBackend<crate::math::NeonVec4> for crate::math::ArmSmeBackend {
    #[inline(always)]
    fn cubic_lagrange_weights_4ch(
        vmu: crate::math::NeonVec4,
    ) -> (
        crate::math::NeonVec4,
        crate::math::NeonVec4,
        crate::math::NeonVec4,
        crate::math::NeonVec4,
    ) {
        cubic_lagrange_weights_4ch_vec4(vmu)
    }
}

#[cfg(target_arch = "x86_64")]
impl OpTBackend<crate::math::Sse2Vec4> for crate::math::X86Sse2Backend {
    #[inline(always)]
    fn cubic_lagrange_weights_4ch(
        vmu: crate::math::Sse2Vec4,
    ) -> (
        crate::math::Sse2Vec4,
        crate::math::Sse2Vec4,
        crate::math::Sse2Vec4,
        crate::math::Sse2Vec4,
    ) {
        cubic_lagrange_weights_4ch_vec4(vmu)
    }
}

#[cfg(target_arch = "x86_64")]
impl OpTBackend<crate::math::Sse42Vec4> for crate::math::X86Sse42Backend {
    #[inline(always)]
    fn cubic_lagrange_weights_4ch(
        vmu: crate::math::Sse42Vec4,
    ) -> (
        crate::math::Sse42Vec4,
        crate::math::Sse42Vec4,
        crate::math::Sse42Vec4,
        crate::math::Sse42Vec4,
    ) {
        cubic_lagrange_weights_4ch_vec4(vmu)
    }
}

#[cfg(target_arch = "x86_64")]
impl OpTBackend<crate::math::Avx2FmaVec4> for crate::math::X86Avx2FmaBackend {
    #[inline(always)]
    fn cubic_lagrange_weights_4ch(
        vmu: crate::math::Avx2FmaVec4,
    ) -> (
        crate::math::Avx2FmaVec4,
        crate::math::Avx2FmaVec4,
        crate::math::Avx2FmaVec4,
        crate::math::Avx2FmaVec4,
    ) {
        cubic_lagrange_weights_4ch_vec4(vmu)
    }
}

#[cfg(target_arch = "wasm32")]
impl OpTBackend<crate::math::WasmVec4> for crate::math::WasmSimd128Backend {
    #[inline(always)]
    fn cubic_lagrange_weights_4ch(
        vmu: crate::math::WasmVec4,
    ) -> (
        crate::math::WasmVec4,
        crate::math::WasmVec4,
        crate::math::WasmVec4,
        crate::math::WasmVec4,
    ) {
        cubic_lagrange_weights_4ch_vec4(vmu)
    }
}

/// Maximum delay line capacity for Op-T Lite buffer (`2,048` samples = `4,096` bytes fp16).
pub const OP_T_LITE_RING_CAPACITY: usize = 2048;
pub const OP_T_FULL_RING_CAPACITY: usize = 4096;
pub const OP_T_ULTRA_RING_CAPACITY: usize = 8192;

/// Co-prime Schroeder all-pass diffuser delays (`ref[dsp.op-t.diffuser.recurrence]`).
pub const SCHROEDER_COPRIME_DELAYS: [usize; 8] = [19, 31, 47, 73, 103, 139, 179, 223];

/// Stateless 4-tap Catmull-Rom cubic Hermite interpolation in Horner form (`ref[dsp.op-t.reader.catmull-rom]`).
///
/// Evaluates:
/// `c3 = -0.5*s[-1] + 1.5*s[0] - 1.5*s[1] + 0.5*s[2]`
/// `c2 = s[-1] - 2.5*s[0] + 2.0*s[1] - 0.5*s[2]`
/// `c1 = -0.5*s[-1] + 0.5*s[1]`
/// `c0 = s[0]`
/// `d = ((c3 * mu + c2) * mu + c1) * mu + c0`
#[inline(always)]
pub fn catmull_rom_hermite_cubic(s_m1: f32, s_0: f32, s_1: f32, s_2: f32, mu: f32) -> f32 {
    let c3 = -0.5 * s_m1 + 1.5 * s_0 - 1.5 * s_1 + 0.5 * s_2;
    let c2 = s_m1 - 2.5 * s_0 + 2.0 * s_1 - 0.5 * s_2;
    let c1 = -0.5 * s_m1 + 0.5 * s_1;
    let c0 = s_0;
    ((c3 * mu + c2) * mu + c1) * mu + c0
}

/// Computes partition-of-unity grain weights for Hermite-Hann grains (Mode M = 2, Family 3).
///
/// Exact identity: `H(phi_A) + H(phi_B) == 1.000000000000`.
#[inline(always)]
pub fn hermite_hann_partition_of_unity(phi_a: f32) -> (f32, f32) {
    let u_a = 1.0 - (2.0 * phi_a - 1.0).abs();
    let u_a_clamped = u_a.clamp(0.0, 1.0);
    let h_a = u_a_clamped * u_a_clamped * (3.0 - 2.0 * u_a_clamped);
    let h_b = 1.0 - h_a;
    (h_a, h_b)
}

/// Computes partition-of-unity 3-phase raised-cosine weights (Mode M = 3, Family 6).
///
/// Exact identity: `W0 + W1 + W2 == 1.000000000000` with non-negative weights.
#[inline(always)]
pub fn raised_cosine_barberpole_3phase(phi: f32) -> [f32; 3] {
    const TWO_PI: f32 = core::f32::consts::TAU;
    const ONE_THIRD: f32 = 1.0 / 3.0;

    let phi0 = phi.fract();
    let phi1 = (phi + ONE_THIRD).fract();
    let phi2 = (phi + 2.0 * ONE_THIRD).fract();

    let c0 = fast_cos_unit(phi0 * TWO_PI);
    let c1 = fast_cos_unit(phi1 * TWO_PI);
    let _c2 = fast_cos_unit(phi2 * TWO_PI);

    let w0 = ONE_THIRD * (1.0 + c0).max(0.0);
    let w1 = ONE_THIRD * (1.0 + c1).max(0.0);
    let w2 = (1.0 - w0 - w1).max(0.0);
    [w0, w1, w2]
}

/// Approximate cosine for phase in `[0, 2*pi]` via minimax polynomial (`0 FDIV`, `0 libm`, bounded `O(1)` range reduction).
#[allow(clippy::excessive_precision)]
#[inline(always)]
fn fast_cos_unit(mut x: f32) -> f32 {
    const PI: f32 = core::f32::consts::PI;
    const TWO_PI: f32 = core::f32::consts::TAU;
    const INV_TWO_PI: f32 = 1.0 / core::f32::consts::TAU;
    if !(0.0..TWO_PI).contains(&x) {
        let q = crate::math::floor_f32(x * INV_TWO_PI);
        x = (x - q * TWO_PI).clamp(0.0, TWO_PI);
    }
    if x > PI {
        x = TWO_PI - x;
    }
    let z = x - PI * 0.5;
    let z2 = z * z;
    -z * (0.99999995 - z2 * (0.16666650 - z2 * 0.00833290))
}

/// Single-delay line with power-of-2 ring buffer mask, Catmull-Rom reader,
/// partitioned Schroeder diffuser, and 128-sample switch crossfade.
#[derive(Clone, Debug)]
pub struct OpTDelayLine<const CAPACITY: usize> {
    pub ring: [u16; CAPACITY],
    pub write_ptr: usize,
    pub current_delay: f32,
    pub target_delay: f32,
    pub xfade_active: bool,
    pub xfade_idx: usize,
    pub xfade_len: usize,
    pub cached_xfade_len: usize,
    pub inv_xfade_denom: f32,
    pub old_delay: f32,
    pub diff_state: [f32; 256],
    pub diff_idx: usize,
}

impl<const CAPACITY: usize> Default for OpTDelayLine<CAPACITY> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const CAPACITY: usize> OpTDelayLine<CAPACITY> {
    pub const MASK: usize = CAPACITY - 1;
    pub const HALF_CAPACITY: usize = CAPACITY / 2;
    pub const HALF_MASK: usize = (CAPACITY / 2) - 1;
    pub const DEFAULT_INV_XFADE_DENOM: f32 = (1.0_f64 / 127.0_f64) as f32;

    pub const fn new() -> Self {
        Self {
            ring: [0; CAPACITY],
            write_ptr: 0,
            current_delay: 64.0,
            target_delay: 64.0,
            xfade_active: false,
            xfade_idx: 0,
            xfade_len: 128,
            cached_xfade_len: 128,
            inv_xfade_denom: Self::DEFAULT_INV_XFADE_DENOM,
            old_delay: 64.0,
            diff_state: [0.0; 256],
            diff_idx: 0,
        }
    }

    /// Updates `xfade_len` (clamped to `>= 16`) and precomputes its reciprocal `1 / (len - 1)` (`0 FDIV`).
    #[inline]
    pub fn set_xfade_len(&mut self, len: usize) {
        let clamped = len.max(16);
        self.xfade_len = clamped;
        self.cached_xfade_len = clamped;
        self.inv_xfade_denom = crate::math::newton6_reciprocal_f64((clamped - 1) as f64) as f32;
    }

    #[inline(always)]
    fn xfade_inv_denom(&mut self, len: usize) -> f32 {
        if len == self.cached_xfade_len {
            self.inv_xfade_denom
        } else {
            let inv = crate::math::newton6_reciprocal_f64((len - 1) as f64) as f32;
            self.cached_xfade_len = len;
            self.inv_xfade_denom = inv;
            inv
        }
    }

    pub fn reset(&mut self) {
        self.ring.fill(0);
        self.write_ptr = 0;
        self.current_delay = self.target_delay;
        self.xfade_active = false;
        self.xfade_idx = 0;
        self.old_delay = self.target_delay;
        self.diff_state.fill(0.0);
        self.diff_idx = 0;
    }

    /// Sets delay target with dual-mode transition (`ref[dsp.op-t.transition.dual-mode]`).
    pub fn set_target_delay(&mut self, target: f32, is_switch_jump: bool) {
        let max_delay = Self::HALF_CAPACITY.saturating_sub(4).max(4) as f32;
        let clamped = target.clamp(4.0, max_delay);
        if is_switch_jump && (clamped - self.current_delay).abs() > 16.0 {
            self.old_delay = self.current_delay;
            self.target_delay = clamped;
            self.xfade_active = true;
            self.xfade_idx = 0;
        } else {
            self.target_delay = clamped;
        }
    }

    /// Reads fractional delay via 4-tap Catmull-Rom interpolation (`ref[dsp.op-t.reader.catmull-rom]`).
    /// Direct callers must hold a [`crate::FtzDazGuard`] for the enclosing batch.
    #[inline(always)]
    pub fn read_delay(&self, delay: f32) -> f32 {
        let max_delay = CAPACITY.saturating_sub(4).max(2) as f32;
        let d_clamped = delay.clamp(2.0, max_delay);
        let d_int = d_clamped as usize;
        let mu = d_clamped - (d_int as f32);

        let p_0 = (self.write_ptr + CAPACITY - d_int) & Self::MASK;
        let s_0 = fp16_to_f32(self.ring[p_0]);
        if mu == 0.0 {
            return s_0;
        }
        let p_m1 = (p_0 + 1) & Self::MASK;
        let p_1 = (p_0 + CAPACITY - 1) & Self::MASK;
        let p_2 = (p_0 + CAPACITY - 2) & Self::MASK;

        let s_m1 = fp16_to_f32(self.ring[p_m1]);
        let s_1 = fp16_to_f32(self.ring[p_1]);
        let s_2 = fp16_to_f32(self.ring[p_2]);

        catmull_rom_hermite_cubic(s_m1, s_0, s_1, s_2, mu)
    }

    /// Advances the delay line by 1 audio sample: writes `input` and returns delayed output.
    /// Direct callers must hold a [`crate::FtzDazGuard`] for the enclosing batch.
    #[inline(always)]
    pub fn step(&mut self, input: f32) -> f32 {
        if !self.xfade_active {
            let diff = self.target_delay - self.current_delay;
            if diff.abs() > 0.0001 {
                self.current_delay += diff.clamp(-0.25, 0.25);
            } else {
                self.current_delay = self.target_delay;
            }
        }

        let out = if self.xfade_active {
            let len = self.xfade_len.max(16);
            let inv_denom = self.xfade_inv_denom(len);
            let d_old = self.read_delay(self.old_delay);
            let d_new = self.read_delay(self.target_delay);
            let w = (self.xfade_idx as f32) * inv_denom;
            let alpha_xf = 0.5 * (1.0 - fast_cos_unit(w * core::f32::consts::PI));
            let mixed = (1.0 - alpha_xf) * d_old + alpha_xf * d_new;
            self.xfade_idx += 1;
            if self.xfade_idx >= len {
                self.xfade_active = false;
                self.current_delay = self.target_delay;
            }
            mixed
        } else {
            self.read_delay(self.current_delay)
        };

        self.ring[self.write_ptr] = f32_to_fp16(input);
        self.write_ptr = (self.write_ptr + 1) & Self::MASK;
        out
    }

    /// Reads fractional delay for a partitioned stereo pair (`ring[0..CAPACITY/2]` Left,
    /// `ring[CAPACITY/2..CAPACITY]` Right) via 4-tap Catmull-Rom interpolation (`ref[dsp.op-t.reader.catmull-rom]`).
    /// Direct callers must hold a [`crate::FtzDazGuard`] for the enclosing batch.
    #[inline(always)]
    pub fn read_delay_stereo(&self, delay: f32) -> (f32, f32) {
        let max_delay = Self::HALF_CAPACITY.saturating_sub(4).max(2) as f32;
        let d_clamped = delay.clamp(2.0, max_delay);
        let d_int = d_clamped as usize;
        let mu = d_clamped - (d_int as f32);
        let w_ptr = self.write_ptr & Self::HALF_MASK;

        let p_0 = (w_ptr + Self::HALF_CAPACITY - d_int) & Self::HALF_MASK;
        let r_off = Self::HALF_CAPACITY;
        if mu == 0.0 {
            return (
                fp16_to_f32(self.ring[p_0]),
                fp16_to_f32(self.ring[r_off + p_0]),
            );
        }

        let (l_2, l_1, l_0, l_m1, r_2, r_1, r_0, r_m1) = if p_0 >= 2
            && p_0 + 1 < Self::HALF_CAPACITY
        {
            let base_l = p_0 - 2;
            let base_r = r_off + base_l;
            let [l2, l1, l0, lm1] =
                <crate::math::DefaultMathBackend as crate::math::MathBackend>::fp16x4_to_f32x4(
                    self.ring[base_l..base_l + 4].try_into().unwrap(),
                );
            let [r2, r1, r0, rm1] =
                <crate::math::DefaultMathBackend as crate::math::MathBackend>::fp16x4_to_f32x4(
                    self.ring[base_r..base_r + 4].try_into().unwrap(),
                );
            (l2, l1, l0, lm1, r2, r1, r0, rm1)
        } else {
            let p_m1 = (p_0 + 1) & Self::HALF_MASK;
            let p_1 = (p_0 + Self::HALF_CAPACITY - 1) & Self::HALF_MASK;
            let p_2 = (p_0 + Self::HALF_CAPACITY - 2) & Self::HALF_MASK;
            let [l2, l1, l0, lm1] =
                <crate::math::DefaultMathBackend as crate::math::MathBackend>::fp16x4_to_f32x4(&[
                    self.ring[p_2],
                    self.ring[p_1],
                    self.ring[p_0],
                    self.ring[p_m1],
                ]);
            let [r2, r1, r0, rm1] =
                <crate::math::DefaultMathBackend as crate::math::MathBackend>::fp16x4_to_f32x4(&[
                    self.ring[r_off + p_2],
                    self.ring[r_off + p_1],
                    self.ring[r_off + p_0],
                    self.ring[r_off + p_m1],
                ]);
            (l2, l1, l0, lm1, r2, r1, r0, rm1)
        };

        (
            catmull_rom_hermite_cubic(l_m1, l_0, l_1, l_2, mu),
            catmull_rom_hermite_cubic(r_m1, r_0, r_1, r_2, mu),
        )
    }

    /// Advances the partitioned stereo delay line by 1 stereo audio frame (`in_l`, `in_r`)
    /// with zero L/R crosstalk and single-frame slew/crossfade progression (`ref[dsp.op-t.transition.dual-mode]`).
    /// Direct callers must hold a [`crate::FtzDazGuard`] for the enclosing batch.
    #[inline(always)]
    pub fn step_stereo(&mut self, in_l: f32, in_r: f32) -> (f32, f32) {
        if !self.xfade_active {
            let diff = self.target_delay - self.current_delay;
            if diff.abs() > 0.0001 {
                self.current_delay += diff.clamp(-0.25, 0.25);
            } else {
                self.current_delay = self.target_delay;
            }
        }

        let (out_l, out_r) = if self.xfade_active {
            let len = self.xfade_len.max(16);
            let inv_denom = self.xfade_inv_denom(len);
            let (l_old, r_old) = self.read_delay_stereo(self.old_delay);
            let (l_new, r_new) = self.read_delay_stereo(self.target_delay);
            let w = (self.xfade_idx as f32) * inv_denom;
            let alpha_xf = 0.5 * (1.0 - fast_cos_unit(w * core::f32::consts::PI));
            let one_minus_a = 1.0 - alpha_xf;
            let mix_l = one_minus_a * l_old + alpha_xf * l_new;
            let mix_r = one_minus_a * r_old + alpha_xf * r_new;
            self.xfade_idx += 1;
            if self.xfade_idx >= len {
                self.xfade_active = false;
                self.current_delay = self.target_delay;
            }
            (mix_l, mix_r)
        } else {
            self.read_delay_stereo(self.current_delay)
        };

        let w_ptr = self.write_ptr & Self::HALF_MASK;
        let [h_l, h_r, _, _] =
            <crate::math::DefaultMathBackend as crate::math::MathBackend>::f32x4_to_fp16x4(&[
                in_l, in_r, 0.0, 0.0,
            ]);
        self.ring[w_ptr] = h_l;
        self.ring[Self::HALF_CAPACITY + w_ptr] = h_r;
        self.write_ptr = (w_ptr + 1) & Self::HALF_MASK;
        (out_l, out_r)
    }

    /// Steps the partitioned in-buffer Schroeder all-pass diffuser (`ref[dsp.op-t.diffuser.recurrence]`).
    /// Direct callers must hold a [`crate::FtzDazGuard`] for the enclosing batch.
    #[inline(always)]
    pub fn step_diffuser(&mut self, input: f32, delay_samples: usize, a_diff: f32) -> f32 {
        let d = delay_samples.clamp(1, 255);
        let a = a_diff.clamp(-0.75, 0.75);

        let read_idx = (self.diff_idx + 256 - d) & 0xFF;
        let u_delayed = self.diff_state[read_idx];
        let v = input - a * u_delayed;
        let y = a * v + u_delayed;
        self.diff_state[self.diff_idx] = flush_recursive_state_f32(v);
        self.diff_idx = (self.diff_idx + 1) & 0xFF;
        y
    }
}

/// 1-pole ballistic envelope ducking VCA (Family 4).
#[derive(Clone, Copy, Debug)]
pub struct BallisticDuckingVca {
    pub env: f32,
    pub alpha_att: f32,
    pub alpha_rel: f32,
    pub threshold: f32,
    pub kappa: f32,
    pub min_gain: f32,
}

impl Default for BallisticDuckingVca {
    fn default() -> Self {
        Self::new()
    }
}

impl BallisticDuckingVca {
    pub const fn new() -> Self {
        Self {
            env: 0.0,
            alpha_att: 0.99,
            alpha_rel: 0.9995,
            threshold: 0.1,
            kappa: 2.0,
            min_gain: 0.1,
        }
    }

    #[inline(always)]
    pub fn reset(&mut self) {
        self.env = 0.0;
    }

    /// Applies ducking to one wet sample. Direct callers must hold a
    /// [`crate::FtzDazGuard`] for the enclosing batch.
    #[inline(always)]
    pub fn step(&mut self, sidechain: f32, wet: f32) -> f32 {
        let abs_sc = sidechain.abs();
        let alpha = if abs_sc > self.env {
            self.alpha_att
        } else {
            self.alpha_rel
        };
        let next_env = alpha * self.env + (1.0 - alpha) * abs_sc;
        self.env = flush_recursive_state_f32(next_env);
        let excess = (next_env - self.threshold).max(0.0);
        let g = (1.0 - self.kappa * excess).clamp(self.min_gain, 1.0);
        g * wet
    }

    /// Evaluates a stereo-linked ballistic ducking step once per stereo frame (`1x fs`),
    /// preserving the stereo image `(wet_l, wet_r)` with identical L/R ducking gain.
    /// Direct callers must hold a [`crate::FtzDazGuard`] for the enclosing batch.
    #[inline(always)]
    pub fn step_stereo(
        &mut self,
        sidechain_l: f32,
        sidechain_r: f32,
        wet_l: f32,
        wet_r: f32,
    ) -> (f32, f32) {
        let abs_sc = 0.5 * (sidechain_l.abs() + sidechain_r.abs());
        let alpha = if abs_sc > self.env {
            self.alpha_att
        } else {
            self.alpha_rel
        };
        let next_env = alpha * self.env + (1.0 - alpha) * abs_sc;
        self.env = flush_recursive_state_f32(next_env);
        let excess = (next_env - self.threshold).max(0.0);
        let g = (1.0 - self.kappa * excess).clamp(self.min_gain, 1.0);
        (g * wet_l, g * wet_r)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FtzDazGuard;
    use proptest::prelude::*;

    #[test]
    fn test_interpolation_and_windowing() {
        let v = catmull_rom_hermite_cubic(1.0, 2.0, 3.0, 4.0, 0.5);
        assert!((v - 2.5).abs() < 1e-4);

        for phi in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let (ha, hb) = hermite_hann_partition_of_unity(phi);
            assert!((ha + hb - 1.0).abs() < 1e-6);
        }

        for phi in [0.1, 0.33, 0.5, 0.8] {
            let [w0, w1, w2] = raised_cosine_barberpole_3phase(phi);
            assert!((w0 + w1 + w2 - 1.0).abs() < 1e-5);
        }

        for x in [
            -core::f32::consts::PI,
            0.0,
            1.5,
            core::f32::consts::PI,
            core::f32::consts::TAU,
            10.0,
        ] {
            let c = fast_cos_unit(x);
            assert!(c.is_finite());
        }
    }

    #[test]
    fn test_op_t_delay_line_and_diffuser() {
        let _audio = FtzDazGuard::new();
        let mut dl = OpTDelayLine::<OP_T_LITE_RING_CAPACITY>::default();
        dl.reset();

        for i in 0..100 {
            let _ = dl.step(i as f32);
        }

        dl.set_target_delay(80.0, false);
        for _ in 0..200 {
            let _ = dl.step(0.5);
        }
        assert!((dl.current_delay - 80.0).abs() < 0.1);

        dl.set_xfade_len(32);
        assert_eq!(dl.xfade_len, 32);
        dl.set_target_delay(150.0, true);
        assert!(dl.xfade_active);
        for _ in 0..32 {
            let _ = dl.step(0.5);
        }
        assert!(!dl.xfade_active);
        assert_eq!(dl.current_delay, 150.0);

        dl.xfade_len = 24;
        dl.set_target_delay(64.0, true);
        assert!(dl.xfade_active);
        for _ in 0..24 {
            let (ol, or) = dl.step_stereo(0.5, -0.25);
            assert!(ol.is_finite() && or.is_finite());
        }
        assert!(!dl.xfade_active);
        assert_eq!(dl.current_delay, 64.0);

        let d_out = dl.step_diffuser(0.8, 31, 0.5);
        assert!(d_out.is_finite());

        dl.reset();
        assert_eq!(dl.write_ptr, 0);
    }

    #[test]
    fn test_ballistic_ducking_vca() {
        let _audio = FtzDazGuard::new();
        let mut vca = BallisticDuckingVca::default();
        let out_low = vca.step(0.01, 1.0);
        assert!((out_low - 1.0).abs() < 0.05);

        for _ in 0..50 {
            let _ = vca.step(0.9, 1.0);
        }
        let out_ducked = vca.step(0.9, 1.0);
        assert!(out_ducked < 0.9);

        let mut vca_s = BallisticDuckingVca::default();
        let (dl, dr) = vca_s.step_stereo(1.0, 0.0, 0.8, 0.4);
        assert!((dl - 2.0 * dr).abs() < 1e-6);
    }

    #[test]
    fn test_catmull_rom_exact_dc_gain_and_diffuser_energy() {
        let _audio = FtzDazGuard::new();
        for step in 0..=100 {
            let mu = (step as f32) / 100.0;
            let val = catmull_rom_hermite_cubic(1.0, 1.0, 1.0, 1.0, mu);
            assert!(
                (val - 1.0).abs() < 1e-6,
                "Catmull-Rom DC gain diverged at mu={}: val={}",
                mu,
                val
            );
        }

        let mut dl = OpTDelayLine::<OP_T_LITE_RING_CAPACITY>::default();
        let a_diff = 0.65_f32;
        let d = 19_usize;
        let mut energy_in = 0.0_f64;
        let mut energy_out = 0.0_f64;

        for n in 0..1000 {
            let x = if n == 0 { 1.0 } else { 0.0 };
            let y = dl.step_diffuser(x, d, a_diff);
            energy_in += (x as f64) * (x as f64);
            energy_out += (y as f64) * (y as f64);
        }
        assert!(
            (energy_in - energy_out).abs() < 1e-4,
            "Diffuser energy unconserved: in={}, out={}",
            energy_in,
            energy_out
        );
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn prop_catmull_rom_linear_reproduction_and_windows(
            a in -5.0f32..5.0f32,
            b in -5.0f32..5.0f32,
            mu in 0.0f32..1.0f32,
        ) {
            let ym1 = a - b;
            let y0 = a;
            let y1 = a + b;
            let y2 = a + 2.0 * b;
            let interp = catmull_rom_hermite_cubic(ym1, y0, y1, y2, mu);
            let exact = a + b * mu;
            prop_assert!((interp - exact).abs() < 1e-4);

            let (ha, hb) = hermite_hann_partition_of_unity(mu);
            prop_assert!((ha + hb - 1.0).abs() < 1e-6);
            prop_assert!(ha >= -1e-6 && hb >= -1e-6);

            let [w0, w1, w2] = raised_cosine_barberpole_3phase(mu);
            prop_assert!((w0 + w1 + w2 - 1.0).abs() < 1e-5);
            prop_assert!(w0 >= -1e-5 && w1 >= -1e-5 && w2 >= -1e-5);
        }

        #[test]
        fn prop_schroeder_diffuser_parseval_energy(
            a_diff in -0.7f32..0.7f32,
            d in 4usize..32usize,
        ) {
            let _audio = FtzDazGuard::new();
            let mut dl = OpTDelayLine::<OP_T_LITE_RING_CAPACITY>::default();
            let mut e_in = 0.0_f64;
            let mut e_out = 0.0_f64;
            for n in 0..512 {
                let x = if n < 8 { ((n as f32) * 0.7 + 0.2).sin() } else { 0.0 };
                let y = dl.step_diffuser(x, d, a_diff);
                e_in += (x as f64) * (x as f64);
                e_out += (y as f64) * (y as f64);
            }
            prop_assert!((e_in - e_out).abs() < 5e-3);
        }
    }
}
