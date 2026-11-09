//! Block-level runtime-dispatched DNAM engine (`DynamicDnamEngine<'a>`).

pub mod dispatch;

use crate::engine::DnamEngine;
use crate::math::{BackendKind, ScalarBackend};
use crate::provenance::{CryptoProvider, DefaultCryptoProvider};
use crate::schema::DnamError;
use crate::types::Tier;

/// Runtime-dispatched wrapper around [`DnamEngine`] supporting block-level backend selection
/// ([`BackendKind`]) with **zero per-sample branching**.
#[derive(Debug)]
pub enum DynamicDnamEngine<'a> {
    /// Portable `#![no_std]` IEEE-754 scalar backend.
    Scalar(DnamEngine<'a, ScalarBackend>),
    /// ARM64 NEON `float32x4_t` backend.
    #[cfg(target_arch = "aarch64")]
    Neon(DnamEngine<'a, crate::math::neon::NeonBackend>),
    /// ARMv9.2-A Scalable Matrix Extension (`ARM SME` / `SME2`) + NEON hybrid backend.
    #[cfg(target_arch = "aarch64")]
    ArmSme(DnamEngine<'a, crate::math::arm_sme::ArmSmeBackend>),
    /// x86_64 AVX2+FMA / SSE2 backend.
    #[cfg(target_arch = "x86_64")]
    X86Simd(DnamEngine<'a, crate::math::x86_simd::X86SimdBackend>),
    /// WebAssembly SIMD128 `v128` backend.
    #[cfg(target_arch = "wasm32")]
    WasmSimd128(DnamEngine<'a, crate::math::wasm_simd128::WasmSimd128Backend>),
}

impl<'a> DynamicDnamEngine<'a> {
    pub(crate) fn new_empty(max_supported_tier: Tier, backend: BackendKind) -> Self {
        match backend {
            #[cfg(target_arch = "aarch64")]
            BackendKind::ArmSme => Self::ArmSme(DnamEngine::new_empty(max_supported_tier)),
            #[cfg(target_arch = "aarch64")]
            BackendKind::Neon | BackendKind::ArmNeon => {
                Self::Neon(DnamEngine::new_empty(max_supported_tier))
            }
            #[cfg(target_arch = "x86_64")]
            BackendKind::X86Simd | BackendKind::X86Avx2Fma | BackendKind::X86Sse42 => {
                Self::X86Simd(DnamEngine::new_empty(max_supported_tier))
            }
            #[cfg(target_arch = "wasm32")]
            BackendKind::WasmSimd128 => {
                Self::WasmSimd128(DnamEngine::new_empty(max_supported_tier))
            }
            _ => Self::Scalar(DnamEngine::new_empty(max_supported_tier)),
        }
    }

    pub(crate) fn init_from_view(
        &mut self,
        view: crate::container::DnamContainerView<'a>,
        tier: Tier,
        max_supported_tier: Tier,
        sample_rate_hz: u32,
    ) -> Result<(), DnamError> {
        match self {
            Self::Scalar(e) => e.init_from_view(view, tier, max_supported_tier, sample_rate_hz),
            #[cfg(target_arch = "aarch64")]
            Self::Neon(e) => e.init_from_view(view, tier, max_supported_tier, sample_rate_hz),
            #[cfg(target_arch = "aarch64")]
            Self::ArmSme(e) => e.init_from_view(view, tier, max_supported_tier, sample_rate_hz),
            #[cfg(target_arch = "x86_64")]
            Self::X86Simd(e) => e.init_from_view(view, tier, max_supported_tier, sample_rate_hz),
            #[cfg(target_arch = "wasm32")]
            Self::WasmSimd128(e) => {
                e.init_from_view(view, tier, max_supported_tier, sample_rate_hz)
            }
        }
    }

    /// Creates a [`DynamicDnamEngine`] using runtime CPU feature detection
    /// ([`crate::math::detect_best_backend`]: `ArmSme` if `FEAT_SME` is supported on `aarch64`,
    /// otherwise `Neon`, `X86Simd`, `WasmSimd128`, or `Scalar`).
    pub fn from_bytes(bytes: &'a [u8], tier: Tier, sample_rate_hz: u32) -> Result<Self, DnamError> {
        Self::from_bytes_with_backend(
            bytes,
            tier,
            sample_rate_hz,
            crate::math::detect_best_backend(),
        )
    }

    /// Creates a [`DynamicDnamEngine`] capped at `max_supported_tier` using runtime CPU feature detection.
    pub fn from_bytes_with_max_tier(
        bytes: &'a [u8],
        tier: Tier,
        max_supported_tier: Tier,
        sample_rate_hz: u32,
    ) -> Result<Self, DnamError> {
        let crypto = DefaultCryptoProvider::default();
        let view = crate::container::DnamContainerView::from_bytes_with_max_tier(
            bytes,
            &crypto,
            max_supported_tier,
        )?;
        let mut engine = Self::new_empty(max_supported_tier, crate::math::detect_best_backend());
        engine.init_from_view(view, tier, max_supported_tier, sample_rate_hz)?;
        Ok(engine)
    }

    /// Creates an owned `'static` [`DynamicDnamEngine`] using runtime CPU feature detection.
    pub fn from_bytes_owned(
        bytes: &[u8],
        tier: Tier,
        sample_rate_hz: u32,
    ) -> Result<DynamicDnamEngine<'static>, DnamError> {
        let crypto = DefaultCryptoProvider::default();
        Self::from_bytes_owned_with_crypto(bytes, tier, sample_rate_hz, &crypto)
    }

    /// Creates an owned `'static` [`DynamicDnamEngine`] capped at `max_supported_tier` using runtime CPU feature detection.
    pub fn from_bytes_owned_with_max_tier(
        bytes: &[u8],
        tier: Tier,
        max_supported_tier: Tier,
        sample_rate_hz: u32,
    ) -> Result<DynamicDnamEngine<'static>, DnamError> {
        let crypto = DefaultCryptoProvider::default();
        let view = crate::container::DnamContainerView::from_bytes_with_max_tier(
            bytes,
            &crypto,
            max_supported_tier,
        )?;
        let arena_bytes = view.arena_storage.as_bytes();
        let mut owned_arena =
            crate::memory::AlignedBuffer::<crate::memory::CacheLine64>::new_zeroed(
                arena_bytes.len() / 64,
            );
        owned_arena.as_bytes_mut().copy_from_slice(arena_bytes);
        let owned_view = crate::container::DnamContainerView {
            preamble: view.preamble,
            header: view.header,
            verification_status: view.verification_status,
            protection_profile: view.protection_profile,
            control_records: view.control_records,
            clamp_descriptors: view.clamp_descriptors,
            cp1_contributors: view.cp1_contributors,
            arena_storage: crate::memory::ArenaStorage::from_owned_buffer(owned_arena),
        };
        let mut engine = DynamicDnamEngine::<'static>::new_empty(
            max_supported_tier,
            crate::math::detect_best_backend(),
        );
        engine.init_from_view(owned_view, tier, max_supported_tier, sample_rate_hz)?;
        Ok(engine)
    }

    /// Creates an owned `'static` [`DynamicDnamEngine`] using a custom [`CryptoProvider`] and runtime CPU feature detection.
    pub fn from_bytes_owned_with_crypto<C: CryptoProvider>(
        bytes: &[u8],
        tier: Tier,
        sample_rate_hz: u32,
        crypto: &C,
    ) -> Result<DynamicDnamEngine<'static>, DnamError> {
        Self::from_bytes_owned_with_backend_and_crypto(
            bytes,
            tier,
            sample_rate_hz,
            crate::math::detect_best_backend(),
            crypto,
        )
    }

    /// Creates an owned `'static` [`DynamicDnamEngine`] with an explicit [`BackendKind`].
    pub fn from_bytes_owned_with_backend(
        bytes: &[u8],
        tier: Tier,
        sample_rate_hz: u32,
        backend: BackendKind,
    ) -> Result<DynamicDnamEngine<'static>, DnamError> {
        let crypto = DefaultCryptoProvider::default();
        Self::from_bytes_owned_with_backend_and_crypto(
            bytes,
            tier,
            sample_rate_hz,
            backend,
            &crypto,
        )
    }

    /// Creates an owned `'static` [`DynamicDnamEngine`] with an explicit [`BackendKind`] and [`CryptoProvider`].
    pub fn from_bytes_owned_with_backend_and_crypto<C: CryptoProvider>(
        bytes: &[u8],
        tier: Tier,
        sample_rate_hz: u32,
        backend: BackendKind,
        crypto: &C,
    ) -> Result<DynamicDnamEngine<'static>, DnamError> {
        let view = crate::container::DnamContainerView::from_bytes(bytes, crypto)?;
        let arena_bytes = view.arena_storage.as_bytes();
        let mut owned_arena =
            crate::memory::AlignedBuffer::<crate::memory::CacheLine64>::new_zeroed(
                arena_bytes.len() / 64,
            );
        owned_arena.as_bytes_mut().copy_from_slice(arena_bytes);
        let owned_view = crate::container::DnamContainerView {
            preamble: view.preamble,
            header: view.header,
            verification_status: view.verification_status,
            protection_profile: view.protection_profile,
            control_records: view.control_records,
            clamp_descriptors: view.clamp_descriptors,
            cp1_contributors: view.cp1_contributors,
            arena_storage: crate::memory::ArenaStorage::from_owned_buffer(owned_arena),
        };
        let mut engine = DynamicDnamEngine::<'static>::new_empty(Tier::Ultra, backend);
        engine.init_from_view(owned_view, tier, Tier::Ultra, sample_rate_hz)?;
        Ok(engine)
    }

    /// Creates a [`DynamicDnamEngine`] with an explicit [`BackendKind`] (falling back to `Scalar`
    /// if the requested ISA is not compiled for the current target).
    pub fn from_bytes_with_backend(
        bytes: &'a [u8],
        tier: Tier,
        sample_rate_hz: u32,
        backend: BackendKind,
    ) -> Result<Self, DnamError> {
        let crypto = DefaultCryptoProvider::default();
        let view = crate::container::DnamContainerView::from_bytes(bytes, &crypto)?;
        let mut engine = Self::new_empty(Tier::Ultra, backend);
        engine.init_from_view(view, tier, Tier::Ultra, sample_rate_hz)?;
        Ok(engine)
    }

    /// Returns the active [`BackendKind`].
    #[inline]
    pub const fn backend_kind(&self) -> BackendKind {
        match self {
            Self::Scalar(_) => BackendKind::Scalar,
            #[cfg(target_arch = "aarch64")]
            Self::Neon(_) => BackendKind::Neon,
            #[cfg(target_arch = "aarch64")]
            Self::ArmSme(_) => BackendKind::ArmSme,
            #[cfg(target_arch = "x86_64")]
            Self::X86Simd(_) => BackendKind::X86Simd,
            #[cfg(target_arch = "wasm32")]
            Self::WasmSimd128(_) => BackendKind::WasmSimd128,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::{ARENA_OFFSET_ULTRA_END, build_dnam_container};
    use crate::schema::DnamJsonHeader;
    extern crate alloc;
    use alloc::vec;

    #[test]
    fn test_dynamic_engine_constructors() {
        let mut header = DnamJsonHeader::default_reference();
        let arena = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
        let container = build_dnam_container(&mut header, &arena, &arena, false).unwrap();

        let e1 = DynamicDnamEngine::from_bytes(&container, Tier::Lite, 48000).unwrap();
        let expected_backend = match crate::math::detect_best_backend() {
            BackendKind::ArmNeon => BackendKind::Neon,
            BackendKind::X86Sse42 | BackendKind::X86Avx2Fma => BackendKind::X86Simd,
            backend => backend,
        };
        assert_eq!(e1.backend_kind(), expected_backend);

        let e2 =
            DynamicDnamEngine::from_bytes_with_max_tier(&container, Tier::Lite, Tier::Full, 48000)
                .unwrap();
        assert!(e2.effective_tier() <= Tier::Full);

        let e3 = DynamicDnamEngine::from_bytes_owned(&container, Tier::Lite, 48000).unwrap();
        assert_eq!(e3.effective_tier(), Tier::Lite);

        let e4 = DynamicDnamEngine::from_bytes_owned_with_max_tier(
            &container,
            Tier::Lite,
            Tier::Pro,
            48000,
        )
        .unwrap();
        assert_eq!(e4.effective_tier(), Tier::Lite);

        let crypto = DefaultCryptoProvider::default();
        let e5 =
            DynamicDnamEngine::from_bytes_owned_with_crypto(&container, Tier::Lite, 48000, &crypto)
                .unwrap();
        assert_eq!(e5.effective_tier(), Tier::Lite);

        let e6 = DynamicDnamEngine::from_bytes_owned_with_backend(
            &container,
            Tier::Lite,
            48000,
            BackendKind::Scalar,
        )
        .unwrap();
        assert_eq!(e6.backend_kind(), BackendKind::Scalar);

        let e7 = DynamicDnamEngine::from_bytes_with_backend(
            &container,
            Tier::Lite,
            48000,
            BackendKind::Scalar,
        )
        .unwrap();
        assert_eq!(e7.backend_kind(), BackendKind::Scalar);
    }
}
