//! Stage II 6-Tap Dilated `4×4` Matrix Convolution Domain Backend (`Stage2Backend`).
//!
//! Canonical Reference: `docs/specs/dsp/op_n_commuted_cascade.md` §4 & `docs/specs/dsp/router_and_stages_topology.md` §2.2.

use crate::math::SimdVec4;

/// Domain-specific SIMD backend trait for Stage II 6-tap dilated `4×4` matrix convolutions.
pub trait Stage2Backend<V: SimdVec4> {
    /// 6-tap dilated convolution over 1 input group (`6 × 4×4` tiles).
    fn conv6_taps_1group(bias: V, w_tiles: &[[[f32; 4]; 4]; 6], h_taps: &[&[[f32; 4]; 4]; 6]) -> V;

    /// 6-tap dilated convolution over 2 input groups (`12 × 4×4` tiles).
    fn conv6_taps_2groups(
        bias: V,
        w_tiles: &[[[f32; 4]; 4]; 12],
        h_taps: &[&[[f32; 4]; 4]; 6],
    ) -> V;

    /// 6-tap dilated convolution over 3 input groups (`18 × 4×4` tiles).
    fn conv6_taps_3groups(
        bias: V,
        w_tiles: &[[[f32; 4]; 4]; 18],
        h_taps: &[&[[f32; 4]; 4]; 6],
    ) -> V;

    /// 6-tap dilated convolution over 4 input groups (`24 × 4×4` tiles).
    fn conv6_taps_4groups(
        bias: V,
        w_tiles: &[[[f32; 4]; 4]; 24],
        h_taps: &[&[[f32; 4]; 4]; 6],
    ) -> V;
}

/// Safe `SimdVec4` implementation of [`Stage2Backend::conv6_taps_1group`] using 4 independent
/// lane accumulators (`za0..za3`) and indexed lane broadcast FMA (`V::fma_lane::<0..3>`).
#[inline(always)]
pub fn conv6_taps_1group_vec4<V: SimdVec4>(
    bias: V,
    w_tiles: &[[[f32; 4]; 4]; 6],
    h_taps: &[&[[f32; 4]; 4]; 6],
) -> V {
    let mut za0 = bias;
    let mut za1 = V::zero();
    let mut za2 = V::zero();
    let mut za3 = V::zero();
    for t in 0..6 {
        let hv = V::load(&h_taps[t][0]);
        let w = &w_tiles[t];
        za0 = za0.fma_lane::<0>(V::load(&w[0]), hv);
        za1 = za1.fma_lane::<1>(V::load(&w[1]), hv);
        za2 = za2.fma_lane::<2>(V::load(&w[2]), hv);
        za3 = za3.fma_lane::<3>(V::load(&w[3]), hv);
    }
    za0.add(za1).add(za2.add(za3))
}

/// Safe `SimdVec4` implementation of [`Stage2Backend::conv6_taps_2groups`].
#[inline(always)]
pub fn conv6_taps_2groups_vec4<V: SimdVec4>(
    bias: V,
    w_tiles: &[[[f32; 4]; 4]; 12],
    h_taps: &[&[[f32; 4]; 4]; 6],
) -> V {
    let mut za0 = bias;
    let mut za1 = V::zero();
    let mut za2 = V::zero();
    let mut za3 = V::zero();
    for t in 0..6 {
        let h = h_taps[t];
        for g in 0..2 {
            let hv = V::load(&h[g]);
            let w = &w_tiles[t * 2 + g];
            za0 = za0.fma_lane::<0>(V::load(&w[0]), hv);
            za1 = za1.fma_lane::<1>(V::load(&w[1]), hv);
            za2 = za2.fma_lane::<2>(V::load(&w[2]), hv);
            za3 = za3.fma_lane::<3>(V::load(&w[3]), hv);
        }
    }
    za0.add(za1).add(za2.add(za3))
}

/// Safe `SimdVec4` implementation of [`Stage2Backend::conv6_taps_3groups`].
#[inline(always)]
pub fn conv6_taps_3groups_vec4<V: SimdVec4>(
    bias: V,
    w_tiles: &[[[f32; 4]; 4]; 18],
    h_taps: &[&[[f32; 4]; 4]; 6],
) -> V {
    let mut za0 = bias;
    let mut za1 = V::zero();
    let mut za2 = V::zero();
    let mut za3 = V::zero();
    for t in 0..6 {
        let h = h_taps[t];
        for g in 0..3 {
            let hv = V::load(&h[g]);
            let w = &w_tiles[t * 3 + g];
            za0 = za0.fma_lane::<0>(V::load(&w[0]), hv);
            za1 = za1.fma_lane::<1>(V::load(&w[1]), hv);
            za2 = za2.fma_lane::<2>(V::load(&w[2]), hv);
            za3 = za3.fma_lane::<3>(V::load(&w[3]), hv);
        }
    }
    za0.add(za1).add(za2.add(za3))
}

/// Safe `SimdVec4` implementation of [`Stage2Backend::conv6_taps_4groups`].
#[inline(always)]
pub fn conv6_taps_4groups_vec4<V: SimdVec4>(
    bias: V,
    w_tiles: &[[[f32; 4]; 4]; 24],
    h_taps: &[&[[f32; 4]; 4]; 6],
) -> V {
    let mut za0 = bias;
    let mut za1 = V::zero();
    let mut za2 = V::zero();
    let mut za3 = V::zero();
    for t in 0..6 {
        let h = h_taps[t];
        for g in 0..4 {
            let hv = V::load(&h[g]);
            let w = &w_tiles[t * 4 + g];
            za0 = za0.fma_lane::<0>(V::load(&w[0]), hv);
            za1 = za1.fma_lane::<1>(V::load(&w[1]), hv);
            za2 = za2.fma_lane::<2>(V::load(&w[2]), hv);
            za3 = za3.fma_lane::<3>(V::load(&w[3]), hv);
        }
    }
    za0.add(za1).add(za2.add(za3))
}

impl Stage2Backend<crate::math::ScalarVec4> for crate::math::ScalarBackend {
    #[inline(always)]
    fn conv6_taps_1group(
        bias: crate::math::ScalarVec4,
        w_tiles: &[[[f32; 4]; 4]; 6],
        h_taps: &[&[[f32; 4]; 4]; 6],
    ) -> crate::math::ScalarVec4 {
        conv6_taps_1group_vec4(bias, w_tiles, h_taps)
    }

    #[inline(always)]
    fn conv6_taps_2groups(
        bias: crate::math::ScalarVec4,
        w_tiles: &[[[f32; 4]; 4]; 12],
        h_taps: &[&[[f32; 4]; 4]; 6],
    ) -> crate::math::ScalarVec4 {
        conv6_taps_2groups_vec4(bias, w_tiles, h_taps)
    }

    #[inline(always)]
    fn conv6_taps_3groups(
        bias: crate::math::ScalarVec4,
        w_tiles: &[[[f32; 4]; 4]; 18],
        h_taps: &[&[[f32; 4]; 4]; 6],
    ) -> crate::math::ScalarVec4 {
        conv6_taps_3groups_vec4(bias, w_tiles, h_taps)
    }

    #[inline(always)]
    fn conv6_taps_4groups(
        bias: crate::math::ScalarVec4,
        w_tiles: &[[[f32; 4]; 4]; 24],
        h_taps: &[&[[f32; 4]; 4]; 6],
    ) -> crate::math::ScalarVec4 {
        conv6_taps_4groups_vec4(bias, w_tiles, h_taps)
    }
}

#[cfg(target_arch = "aarch64")]
impl Stage2Backend<crate::math::NeonVec4> for crate::math::NeonBackend {
    #[inline(always)]
    fn conv6_taps_1group(
        bias: crate::math::NeonVec4,
        w_tiles: &[[[f32; 4]; 4]; 6],
        h_taps: &[&[[f32; 4]; 4]; 6],
    ) -> crate::math::NeonVec4 {
        conv6_taps_1group_vec4(bias, w_tiles, h_taps)
    }

    #[inline(always)]
    fn conv6_taps_2groups(
        bias: crate::math::NeonVec4,
        w_tiles: &[[[f32; 4]; 4]; 12],
        h_taps: &[&[[f32; 4]; 4]; 6],
    ) -> crate::math::NeonVec4 {
        conv6_taps_2groups_vec4(bias, w_tiles, h_taps)
    }

    #[inline(always)]
    fn conv6_taps_3groups(
        bias: crate::math::NeonVec4,
        w_tiles: &[[[f32; 4]; 4]; 18],
        h_taps: &[&[[f32; 4]; 4]; 6],
    ) -> crate::math::NeonVec4 {
        conv6_taps_3groups_vec4(bias, w_tiles, h_taps)
    }

    #[inline(always)]
    fn conv6_taps_4groups(
        bias: crate::math::NeonVec4,
        w_tiles: &[[[f32; 4]; 4]; 24],
        h_taps: &[&[[f32; 4]; 4]; 6],
    ) -> crate::math::NeonVec4 {
        conv6_taps_4groups_vec4(bias, w_tiles, h_taps)
    }
}

#[cfg(target_arch = "aarch64")]
impl Stage2Backend<crate::math::Sve2Vec4> for crate::math::ArmSve2Backend {
    #[inline(always)]
    fn conv6_taps_1group(
        bias: crate::math::Sve2Vec4,
        w_tiles: &[[[f32; 4]; 4]; 6],
        h_taps: &[&[[f32; 4]; 4]; 6],
    ) -> crate::math::Sve2Vec4 {
        conv6_taps_1group_vec4(bias, w_tiles, h_taps)
    }

    #[inline(always)]
    fn conv6_taps_2groups(
        bias: crate::math::Sve2Vec4,
        w_tiles: &[[[f32; 4]; 4]; 12],
        h_taps: &[&[[f32; 4]; 4]; 6],
    ) -> crate::math::Sve2Vec4 {
        conv6_taps_2groups_vec4(bias, w_tiles, h_taps)
    }

    #[inline(always)]
    fn conv6_taps_3groups(
        bias: crate::math::Sve2Vec4,
        w_tiles: &[[[f32; 4]; 4]; 18],
        h_taps: &[&[[f32; 4]; 4]; 6],
    ) -> crate::math::Sve2Vec4 {
        conv6_taps_3groups_vec4(bias, w_tiles, h_taps)
    }

    #[inline(always)]
    fn conv6_taps_4groups(
        bias: crate::math::Sve2Vec4,
        w_tiles: &[[[f32; 4]; 4]; 24],
        h_taps: &[&[[f32; 4]; 4]; 6],
    ) -> crate::math::Sve2Vec4 {
        conv6_taps_4groups_vec4(bias, w_tiles, h_taps)
    }
}

#[cfg(target_arch = "aarch64")]
impl Stage2Backend<crate::math::NeonVec4> for crate::math::ArmSmeBackend {
    #[inline(always)]
    fn conv6_taps_1group(
        bias: crate::math::NeonVec4,
        w_tiles: &[[[f32; 4]; 4]; 6],
        h_taps: &[&[[f32; 4]; 4]; 6],
    ) -> crate::math::NeonVec4 {
        conv6_taps_1group_vec4(bias, w_tiles, h_taps)
    }

    #[inline(always)]
    fn conv6_taps_2groups(
        bias: crate::math::NeonVec4,
        w_tiles: &[[[f32; 4]; 4]; 12],
        h_taps: &[&[[f32; 4]; 4]; 6],
    ) -> crate::math::NeonVec4 {
        conv6_taps_2groups_vec4(bias, w_tiles, h_taps)
    }

    #[inline(always)]
    fn conv6_taps_3groups(
        bias: crate::math::NeonVec4,
        w_tiles: &[[[f32; 4]; 4]; 18],
        h_taps: &[&[[f32; 4]; 4]; 6],
    ) -> crate::math::NeonVec4 {
        conv6_taps_3groups_vec4(bias, w_tiles, h_taps)
    }

    #[inline(always)]
    fn conv6_taps_4groups(
        bias: crate::math::NeonVec4,
        w_tiles: &[[[f32; 4]; 4]; 24],
        h_taps: &[&[[f32; 4]; 4]; 6],
    ) -> crate::math::NeonVec4 {
        conv6_taps_4groups_vec4(bias, w_tiles, h_taps)
    }
}

#[cfg(target_arch = "x86_64")]
impl Stage2Backend<crate::math::Sse2Vec4> for crate::math::X86Sse2Backend {
    #[inline(always)]
    fn conv6_taps_1group(
        bias: crate::math::Sse2Vec4,
        w_tiles: &[[[f32; 4]; 4]; 6],
        h_taps: &[&[[f32; 4]; 4]; 6],
    ) -> crate::math::Sse2Vec4 {
        conv6_taps_1group_vec4(bias, w_tiles, h_taps)
    }

    #[inline(always)]
    fn conv6_taps_2groups(
        bias: crate::math::Sse2Vec4,
        w_tiles: &[[[f32; 4]; 4]; 12],
        h_taps: &[&[[f32; 4]; 4]; 6],
    ) -> crate::math::Sse2Vec4 {
        conv6_taps_2groups_vec4(bias, w_tiles, h_taps)
    }

    #[inline(always)]
    fn conv6_taps_3groups(
        bias: crate::math::Sse2Vec4,
        w_tiles: &[[[f32; 4]; 4]; 18],
        h_taps: &[&[[f32; 4]; 4]; 6],
    ) -> crate::math::Sse2Vec4 {
        conv6_taps_3groups_vec4(bias, w_tiles, h_taps)
    }

    #[inline(always)]
    fn conv6_taps_4groups(
        bias: crate::math::Sse2Vec4,
        w_tiles: &[[[f32; 4]; 4]; 24],
        h_taps: &[&[[f32; 4]; 4]; 6],
    ) -> crate::math::Sse2Vec4 {
        conv6_taps_4groups_vec4(bias, w_tiles, h_taps)
    }
}

#[cfg(target_arch = "x86_64")]
impl Stage2Backend<crate::math::Sse42Vec4> for crate::math::X86Sse42Backend {
    #[inline(always)]
    fn conv6_taps_1group(
        bias: crate::math::Sse42Vec4,
        w_tiles: &[[[f32; 4]; 4]; 6],
        h_taps: &[&[[f32; 4]; 4]; 6],
    ) -> crate::math::Sse42Vec4 {
        conv6_taps_1group_vec4(bias, w_tiles, h_taps)
    }

    #[inline(always)]
    fn conv6_taps_2groups(
        bias: crate::math::Sse42Vec4,
        w_tiles: &[[[f32; 4]; 4]; 12],
        h_taps: &[&[[f32; 4]; 4]; 6],
    ) -> crate::math::Sse42Vec4 {
        conv6_taps_2groups_vec4(bias, w_tiles, h_taps)
    }

    #[inline(always)]
    fn conv6_taps_3groups(
        bias: crate::math::Sse42Vec4,
        w_tiles: &[[[f32; 4]; 4]; 18],
        h_taps: &[&[[f32; 4]; 4]; 6],
    ) -> crate::math::Sse42Vec4 {
        conv6_taps_3groups_vec4(bias, w_tiles, h_taps)
    }

    #[inline(always)]
    fn conv6_taps_4groups(
        bias: crate::math::Sse42Vec4,
        w_tiles: &[[[f32; 4]; 4]; 24],
        h_taps: &[&[[f32; 4]; 4]; 6],
    ) -> crate::math::Sse42Vec4 {
        conv6_taps_4groups_vec4(bias, w_tiles, h_taps)
    }
}

#[cfg(target_arch = "x86_64")]
impl Stage2Backend<crate::math::Avx2FmaVec4> for crate::math::X86Avx2FmaBackend {
    #[inline(always)]
    fn conv6_taps_1group(
        bias: crate::math::Avx2FmaVec4,
        w_tiles: &[[[f32; 4]; 4]; 6],
        h_taps: &[&[[f32; 4]; 4]; 6],
    ) -> crate::math::Avx2FmaVec4 {
        conv6_taps_1group_vec4(bias, w_tiles, h_taps)
    }

    #[inline(always)]
    fn conv6_taps_2groups(
        bias: crate::math::Avx2FmaVec4,
        w_tiles: &[[[f32; 4]; 4]; 12],
        h_taps: &[&[[f32; 4]; 4]; 6],
    ) -> crate::math::Avx2FmaVec4 {
        conv6_taps_2groups_vec4(bias, w_tiles, h_taps)
    }

    #[inline(always)]
    fn conv6_taps_3groups(
        bias: crate::math::Avx2FmaVec4,
        w_tiles: &[[[f32; 4]; 4]; 18],
        h_taps: &[&[[f32; 4]; 4]; 6],
    ) -> crate::math::Avx2FmaVec4 {
        conv6_taps_3groups_vec4(bias, w_tiles, h_taps)
    }

    #[inline(always)]
    fn conv6_taps_4groups(
        bias: crate::math::Avx2FmaVec4,
        w_tiles: &[[[f32; 4]; 4]; 24],
        h_taps: &[&[[f32; 4]; 4]; 6],
    ) -> crate::math::Avx2FmaVec4 {
        conv6_taps_4groups_vec4(bias, w_tiles, h_taps)
    }
}

#[cfg(target_arch = "wasm32")]
impl Stage2Backend<crate::math::WasmVec4> for crate::math::WasmSimd128Backend {
    #[inline(always)]
    fn conv6_taps_1group(
        bias: crate::math::WasmVec4,
        w_tiles: &[[[f32; 4]; 4]; 6],
        h_taps: &[&[[f32; 4]; 4]; 6],
    ) -> crate::math::WasmVec4 {
        conv6_taps_1group_vec4(bias, w_tiles, h_taps)
    }

    #[inline(always)]
    fn conv6_taps_2groups(
        bias: crate::math::WasmVec4,
        w_tiles: &[[[f32; 4]; 4]; 12],
        h_taps: &[&[[f32; 4]; 4]; 6],
    ) -> crate::math::WasmVec4 {
        conv6_taps_2groups_vec4(bias, w_tiles, h_taps)
    }

    #[inline(always)]
    fn conv6_taps_3groups(
        bias: crate::math::WasmVec4,
        w_tiles: &[[[f32; 4]; 4]; 18],
        h_taps: &[&[[f32; 4]; 4]; 6],
    ) -> crate::math::WasmVec4 {
        conv6_taps_3groups_vec4(bias, w_tiles, h_taps)
    }

    #[inline(always)]
    fn conv6_taps_4groups(
        bias: crate::math::WasmVec4,
        w_tiles: &[[[f32; 4]; 4]; 24],
        h_taps: &[&[[f32; 4]; 4]; 6],
    ) -> crate::math::WasmVec4 {
        conv6_taps_4groups_vec4(bias, w_tiles, h_taps)
    }
}
