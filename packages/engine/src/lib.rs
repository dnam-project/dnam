//! `dnam-engine` — Pure Rust `#![no_std]` + `alloc` Real-Time Stereo Neural Audio Modeling & Differentiable DSP Core.
//!
//! Unifies the three subsystems of **Dynamic Neural Audio Modeler (DNAM / "Tri-Physics")**:
//! 1. **Core 1 ([`core1`]):** 12-layer Hybrid Dilated-Conv (`1x B-Spline PolyADAA`) + Radial-Polar
//!    `SelectiveRgLru` stack with 3-state DC resting equilibrium (`0.000000` Sample-0 transient).
//! 2. **Core 2 ([`core2`]):** Unified Stereo In-Loop Diffusive Modulated Unitary Feedback Delay Network
//!    (`M-UFDN`) with dynamic 64B-aligned `fp16` delay buffers (`0 KB` when bypassed), 64-sample TZF
//!    reference tap, 4-tap Cubic Lagrange interpolation, 128-sample switch crossfades, Givens+Householder
//!    unitary mixing, and Coercive-Saturating Lyapunov loop (`S'(0) == 1.0000`).
//! 3. **Control Bus ([`control`]):** $3\text{ kHz}$ Envelopes/LFOs + Node-Anchored Lobatto HDMR & Rank-4
//!    Factorization Machine with 64-byte [`ControlClampDescriptor`](control::ControlClampDescriptor)
//!    enforcement (`[CLAMP-RANGE]`, `[CLAMP-SWITCH]`, `[CLAMP-STABILITY]`, `[CLAMP-DYNAMIC]`),
//!    1-pole anti-zipper smoothing, and static `0 MACs/sample` collapsed head caching.
//!
//! ## Core Real-Time Invariants
//! - **100% `#![no_std]` + `alloc` Foundation:** Zero OS calls, zero locks, and **zero heap allocations**
//!   inside [`DnamEngine::process_block`], [`DnamEngine::process_block_in_place`],
//!   [`DnamEngine::process_mono_to_stereo`], [`DnamEngine::process_mono`], and [`DnamEngine::reload_model_in_place`].
//! - **Zero `FSQRT`, `FDIV`, or `libm` Calls** at $48\text{ kHz}$ and $3\text{ kHz}$.
//! - **Automatic Hardware Denormal Suppression ([`FtzDazGuard`]):** Sets `FPCR.FZ` (`aarch64`) or
//!   `MXCSR` `FTZ | DAZ` (`x86_64`) on block entry and restores on exit.

#![no_std]

extern crate alloc;

pub mod container;
pub mod control;
pub mod core1;
pub mod core2;
pub mod ffi;
pub mod math;
pub mod memory;
pub mod provenance;
pub mod schema;

use crate::container::{ARENA_OFFSET_LITE_END, DnamContainerView, DnamPreamble, ZohRescaledParams};
use crate::control::{ControlBus, MAX_CONTROLS};
use crate::core1::Core1Stack;
use crate::core2::ModulatedUfdn;
use crate::math::{BackendKind, DefaultMathBackend, MathBackend, ScalarBackend};
use crate::memory::{AlignedBuffer, AlignedFp16Block, ArenaStorage, CacheLine64};
use crate::provenance::{CryptoProvider, DefaultCryptoProvider, VerificationStatus};
use crate::schema::{DnamError, DnamJsonHeader};

/// Slimmable execution tier (`Lite`, `Full`, or `Ultra`).
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Tier {
    /// **`Lite` ($C=4$ Core 1 channels, $M=4$ FDN lines, `10.5 KB` `Arena 0`):**
    /// Ultra-low-power tier (`~920`–`1,850 MACs/sample`).
    Lite = 0,
    /// **`Full` ($C=8$ Core 1 channels, $M=8$ FDN lines, `21.8 KB` `Arena 0..1`):**
    /// Standard real-time studio/pedalboard tier (`~3,150`–`4,625 MACs/sample`).
    Full = 1,
    /// **`Ultra` ($C=12$ Core 1 channels, $M=8$ FDN lines, `35.4 KB` `Arena 0..2`):**
    /// Mastering-grade tier (`~6,300`–`7,775 MACs/sample`, `8 + 4` SIMD loop).
    Ultra = 2,
}

impl Tier {
    /// Converts a raw `u8` (`0`, `1`, `2`) into a [`Tier`], clamping values $\ge 2$ to [`Tier::Ultra`].
    #[inline]
    pub const fn from_u8(val: u8) -> Self {
        match val {
            0 => Self::Lite,
            1 => Self::Full,
            _ => Self::Ultra,
        }
    }

    /// Returns the number of active Core 1 channels ($C \in \{4, 8, 12\}$).
    #[inline]
    pub const fn channels(self) -> usize {
        match self {
            Self::Lite => 4,
            Self::Full => 8,
            Self::Ultra => 12,
        }
    }

    /// Returns the number of 4-channel SIMD tier groups in Core 1 (`1`, `2`, or `3`).
    #[inline]
    pub const fn core1_groups(self) -> usize {
        match self {
            Self::Lite => 1,
            Self::Full => 2,
            Self::Ultra => 3,
        }
    }

    /// Returns the number of 4-line SIMD groups in Core 2 `M-UFDN` (`1` for `Lite`, `2` for `Full`/`Ultra`).
    #[inline]
    pub const fn fdn_groups(self) -> usize {
        match self {
            Self::Lite => 1,
            Self::Full | Self::Ultra => 2,
        }
    }
}

// ============================================================================
// Hardware Denormal Suppression Guard (`FtzDazGuard`: `FPCR.FZ` / `MXCSR`)
// ============================================================================

/// Scoped RAII guard that enables hardware **Flush-to-Zero (`FTZ`)** and **Denormals-Are-Zero (`DAZ`)**
/// on `aarch64` (`FPCR` bit 24) and `x86_64` (`MXCSR` bits 6 & 15 = `0x8040`) for the duration of an
/// audio block, restoring the caller's previous floating-point control register on drop.
#[derive(Debug)]
pub struct FtzDazGuard {
    #[cfg(target_arch = "aarch64")]
    prev_fpcr: u64,
    #[cfg(target_arch = "x86_64")]
    prev_mxcsr: u32,
}

impl FtzDazGuard {
    /// Enables hardware `FTZ`/`DAZ` for the current scope.
    #[inline(always)]
    pub fn new() -> Self {
        #[cfg(target_arch = "aarch64")]
        {
            let prev_fpcr: u64;
            // SAFETY: Reading and setting bit 24 (FZ) of the ARM64 user-space FPCR register is safe
            // and prevents subnormal traps inside recursive IIR/RG-LRU/FDN filters.
            unsafe {
                core::arch::asm!("mrs {}, fpcr", out(reg) prev_fpcr, options(nomem, nostack, preserves_flags));
                let new_fpcr = prev_fpcr | (1_u64 << 24);
                if new_fpcr != prev_fpcr {
                    core::arch::asm!("msr fpcr, {}", in(reg) new_fpcr, options(nomem, nostack, preserves_flags));
                }
            }
            Self { prev_fpcr }
        }
        #[cfg(target_arch = "x86_64")]
        {
            let mut prev_mxcsr: u32 = 0;
            // SAFETY: `stmxcsr` and `ldmxcsr` read/write the user-space SSE MXCSR control word.
            unsafe {
                core::arch::asm!("stmxcsr [{}]", in(reg) &mut prev_mxcsr, options(nostack, preserves_flags));
                let new_mxcsr = prev_mxcsr | 0x8040;
                if new_mxcsr != prev_mxcsr {
                    core::arch::asm!("ldmxcsr [{}]", in(reg) &new_mxcsr, options(nostack, preserves_flags));
                }
            }
            Self { prev_mxcsr }
        }
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            Self {}
        }
    }
}

impl Default for FtzDazGuard {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for FtzDazGuard {
    #[inline(always)]
    fn drop(&mut self) {
        #[cfg(target_arch = "aarch64")]
        {
            // SAFETY: Restores the caller's original FPCR value saved in `FtzDazGuard::new`.
            unsafe {
                core::arch::asm!("msr fpcr, {}", in(reg) self.prev_fpcr, options(nomem, nostack, preserves_flags));
            }
        }
        #[cfg(target_arch = "x86_64")]
        {
            // SAFETY: Restores the caller's original MXCSR value saved in `FtzDazGuard::new`.
            unsafe {
                core::arch::asm!("ldmxcsr [{}]", in(reg) &self.prev_mxcsr, options(nostack, preserves_flags));
            }
        }
    }
}

// ============================================================================
// 60 fps Real-Time Telemetry Snapshot (`#[repr(C)]`)
// ============================================================================

/// Lock-free, copyable `#[repr(C)]` telemetry snapshot for 60 fps UI visualizers
/// (`TriPhysicsTelemetry.tsx`, `packages/plugin`, and `packages/nodejs`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DnamTelemetrySnapshot {
    /// Active slimmable tier (`0 = Lite`, `1 = Full`, `2 = Ultra`).
    pub active_tier: u8,
    /// `1` if Core 2 (`M-UFDN`) is active, `0` if bypassed (`0 KB` delay RAM).
    pub core2_active: u8,
    /// `1` if $3\text{ kHz}$ LFOs and envelope followers are active, `0` otherwise.
    pub lfo_env_active: u8,
    /// `1` if stereo output head is active, `0` if mono output mode.
    pub stereo_out: u8,
    /// Bitmask of controls currently clamped by `[CLAMP-RANGE]`, `[CLAMP-SWITCH]`, or `[CLAMP-STABILITY]`.
    pub active_clamp_mask: u16,
    /// Current host control-bus decimation stride $K_{\text{ctrl}}(f_{s,\text{host}})$.
    pub control_decimation_host: u16,
    /// Peak `PolyADAA` normalized drive $|u_c| \in [0, 1]$ in the latest block.
    pub max_adaa_drive: f32,
    /// Peak `SelectiveRgLru` pole radius $r_c[n] \le 0.9995$ in the latest block.
    pub max_pole_radius: f32,
    /// Peak `M-UFDN` feedback loop gain $g_{\text{eff},m} \le 0.9990$ in the latest block.
    pub max_effective_loop_gain: f32,
    /// Reported $T_{60}$ decay tail time in milliseconds.
    pub tail_time_ms: f32,
    /// Current collapsed SVD head coordinates $\mathbf{z}_{\text{head}} \in \mathbb{R}^4$.
    pub z_head: [f32; 4],
    /// Current 3-band envelope follower states (`[Low, Mid, High, 0.0]`).
    pub env_states: [f32; 4],
    /// Current 4-harmonic Chebyshev LFO waves ($T_1..T_4$ for LFO 0 and LFO 1).
    pub lfo_harmonics: [f32; 8],
    /// Current clamped physical control values across all 16 slots.
    pub clamped_controls: [f32; MAX_CONTROLS],
}

// ============================================================================
// Primary Generic Real-Time Engine (`DnamEngine<'a, B>`)
// ============================================================================

/// Zero-allocation real-time DNAM ("Tri-Physics") inference engine parameterized over a
/// [`MathBackend`] (`DefaultMathBackend` by default).
#[derive(Debug)]
pub struct DnamEngine<'a, B: MathBackend = DefaultMathBackend> {
    preamble: DnamPreamble,
    header: Option<DnamJsonHeader>,
    verification_status: VerificationStatus,
    arena_storage: ArenaStorage<'a>,
    tier: Tier,
    max_available_tier: Tier,
    zoh: ZohRescaledParams,
    ctrl_sub_counter: usize,
    ctrl_energy_accum: f32,
    pub control_bus: ControlBus<B>,
    pub core1: Core1Stack<B>,
    pub core2: ModulatedUfdn<'a, B>,
}

impl<'a, B: MathBackend> DnamEngine<'a, B> {
    /// Loads a `.dnam` or `.dnam-ckpt` container from `bytes` using the default [`CryptoProvider`],
    /// borrowing `Arena 0..2` with **zero copies** when `bytes` is 64-byte aligned (`mmap`) and unencrypted.
    pub fn from_bytes(bytes: &'a [u8], tier: Tier, sample_rate_hz: u32) -> Result<Self, DnamError> {
        let crypto = DefaultCryptoProvider::default();
        Self::from_bytes_with_crypto(bytes, tier, sample_rate_hz, &crypto)
    }

    /// Loads a `.dnam` or `.dnam-ckpt` container into a `'static` owned engine by copying the
    /// 64-byte aligned tensor arena into an owned [`AlignedBuffer`].
    pub fn from_bytes_owned(
        bytes: &[u8],
        tier: Tier,
        sample_rate_hz: u32,
    ) -> Result<DnamEngine<'static, B>, DnamError> {
        let crypto = DefaultCryptoProvider::default();
        let view = DnamContainerView::from_bytes(bytes, &crypto)?;
        let arena_bytes = view.arena_storage.as_bytes();
        let mut owned_arena = AlignedBuffer::<CacheLine64>::new_zeroed(arena_bytes.len() / 64);
        owned_arena.as_bytes_mut().copy_from_slice(arena_bytes);

        let owned_view = DnamContainerView {
            preamble: view.preamble,
            header: view.header,
            verification_status: view.verification_status,
            arena_storage: ArenaStorage::from_owned_buffer(owned_arena),
        };
        DnamEngine::<'static, B>::from_view(owned_view, tier, sample_rate_hz)
    }

    /// Loads a `.dnam` or `.dnam-ckpt` container using a custom [`CryptoProvider`] (e.g. hardware
    /// `OP-TEE`, Apple Secure Enclave `SEP`, Windows/Linux `TPM 2.0`, or keyed [`RustCryptoProvider`](provenance::RustCryptoProvider)).
    pub fn from_bytes_with_crypto<C: CryptoProvider>(
        bytes: &'a [u8],
        tier: Tier,
        sample_rate_hz: u32,
        crypto: &C,
    ) -> Result<Self, DnamError> {
        let view = DnamContainerView::from_bytes(bytes, crypto)?;
        Self::from_view(view, tier, sample_rate_hz)
    }

    /// Zero-Copy Scratch Loader:
    /// Parses the `.dnam` container via `#![no_std]` `serde` + `alloc` while borrowing both the
    /// 64-byte aligned tensor arena and caller-supplied `scratch` for Core 2 `fp16` delay lines.
    pub fn from_bytes_with_scratch(
        bytes: &'a [u8],
        scratch: &'a mut [AlignedFp16Block],
        tier: Tier,
        sample_rate_hz: u32,
    ) -> Result<Self, DnamError> {
        let crypto = DefaultCryptoProvider::default();
        let view = DnamContainerView::from_bytes(bytes, &crypto)?;
        let ring_log2 = view.preamble.ring_buffer_pow2_log2;
        let core2 = ModulatedUfdn::with_borrowed_scratch(scratch, ring_log2);
        Self::from_view_with_core2(view, core2, tier, sample_rate_hz)
    }

    /// Pre-allocates a [`DnamEngine`] with maximum `Arena 0..2` (`36,224 B`) and Core 2 `fp16`
    /// delay capacity (`max_ring_log2`, e.g. `15` for 32,768 samples) so subsequent calls to
    /// [`reload_model_in_place`](Self::reload_model_in_place) execute with **0 heap allocations**.
    pub fn with_max_capacity(max_ring_log2: u8) -> Self {
        let owned_arena = AlignedBuffer::<CacheLine64>::new_zeroed(36_224 / 64);
        let zoh = ZohRescaledParams::compute(48_000, 48_000, 16, 0.9993, 0.995, 64.0, 128)
            .unwrap_or(ZohRescaledParams {
                sample_rate_host_hz: 48_000,
                rho_sr: 1.0,
                control_decimation_host: 16,
                rho_ctrl: 1.0,
                r_ac_host: 0.9993,
                r_dc_host: 0.995,
                tau_ref_samples_host: 64.0,
                switch_xfade_samples_host: 128,
            });
        let preamble = DnamPreamble {
            magic: crate::container::DNAM_MAGIC,
            version_major: 1,
            version_minor: 0,
            flags: crate::container::FLAG_HAS_LITE
                | crate::container::FLAG_EXEC_ORDER_SOA
                | crate::container::FLAG_DEFAULT_ANCHORED
                | crate::container::FLAG_STEREO_OUT,
            num_controls: 0,
            ring_buffer_pow2_log2: 0,
            sample_rate_train_hz: 48_000,
            json_header_byte_len: 0,
            tensor_arena_offset: 4096,
            tensor_arena_byte_len: ARENA_OFFSET_LITE_END,
            arena_offset_lite_end: ARENA_OFFSET_LITE_END,
            arena_offset_full_end: ARENA_OFFSET_LITE_END,
            arena_offset_ultra_end: ARENA_OFFSET_LITE_END,
            arena_crc32c_lite: 0,
            arena_crc32c_full: 0,
            arena_crc32c_ultra: 0,
            num_ckpt_stems: 0,
            control_decimation: 16,
            head_svd_rank: 4,
            tail_time_ms: 0.0,
            header_crc32c: 0,
        };
        Self {
            preamble,
            header: None,
            verification_status: VerificationStatus::Unsigned,
            arena_storage: ArenaStorage::from_owned_buffer(owned_arena),
            tier: Tier::Lite,
            max_available_tier: Tier::Lite,
            zoh,
            ctrl_sub_counter: 0,
            ctrl_energy_accum: 0.0,
            control_bus: ControlBus::new(),
            core1: Core1Stack::new(),
            core2: ModulatedUfdn::with_max_capacity(max_ring_log2),
        }
    }

    /// Hot-swaps a new `.dnam` model in-place using the Zero-JSON hardware fast-path with
    /// **zero heap allocations** (provided `self` was created with sufficient `with_max_capacity`).
    pub fn reload_model_in_place(
        &mut self,
        bytes: &'a [u8],
        tier: Tier,
        sample_rate_hz: u32,
    ) -> Result<(), DnamError> {
        let preamble = DnamPreamble::from_bytes(bytes)?;
        if preamble.is_encrypted_arena() {
            return Err(DnamError::EncryptedArenaRequiresDeviceKey);
        }
        let arena_off = preamble.tensor_arena_offset as usize;
        let arena_len = preamble.tensor_arena_byte_len as usize;
        if bytes.len() < arena_off + arena_len {
            return Err(DnamError::BufferTooSmall {
                expected: arena_off + arena_len,
                actual: bytes.len(),
            });
        }
        let computed_hdr_crc = crate::container::compute_header_crc32c(bytes, arena_off);
        if computed_hdr_crc != preamble.header_crc32c {
            return Err(DnamError::Crc32cMismatch {
                region: "header",
                expected: preamble.header_crc32c,
                computed: computed_hdr_crc,
            });
        }
        let raw_arena = &bytes[arena_off..arena_off + arena_len];
        DnamContainerView::verify_arena_prefix_crcs(&preamble, raw_arena)?;

        // Copy into pre-allocated Owned buffer if already Owned and large enough, or borrow if 64B-aligned
        let copy_len = raw_arena.len().min(36_224);
        if !self.arena_storage.reload_in_place(&raw_arena[..copy_len]) {
            self.arena_storage = ArenaStorage::from_slice_or_copy(raw_arena);
        }

        self.preamble = preamble;
        self.header = None;
        self.verification_status = VerificationStatus::Unsigned;
        self.max_available_tier = Self::detect_max_tier(&preamble, copy_len);
        self.tier = tier.min(self.max_available_tier);

        self.reconfigure_for_sample_rate(sample_rate_hz)?;
        Ok(())
    }

    fn from_view(
        view: DnamContainerView<'a>,
        tier: Tier,
        sample_rate_hz: u32,
    ) -> Result<Self, DnamError> {
        let ring_log2 = if view.preamble.is_core2_active() {
            view.preamble.ring_buffer_pow2_log2
        } else {
            0
        };
        let core2 = ModulatedUfdn::with_max_capacity(ring_log2);
        Self::from_view_with_core2(view, core2, tier, sample_rate_hz)
    }

    fn from_view_with_core2(
        view: DnamContainerView<'a>,
        core2: ModulatedUfdn<'a, B>,
        tier: Tier,
        sample_rate_hz: u32,
    ) -> Result<Self, DnamError> {
        let arena_len = view.arena_storage.as_bytes().len();
        let max_available_tier = Self::detect_max_tier(&view.preamble, arena_len);
        let active_tier = tier.min(max_available_tier);

        let zoh = ZohRescaledParams::compute(
            view.preamble.sample_rate_train_hz,
            sample_rate_hz,
            view.preamble.control_decimation.max(8),
            0.9993,
            0.995,
            64.0,
            128,
        )?;

        let mut engine = Self {
            preamble: view.preamble,
            header: Some(view.header),
            verification_status: view.verification_status,
            arena_storage: view.arena_storage,
            tier: active_tier,
            max_available_tier,
            zoh,
            ctrl_sub_counter: 0,
            ctrl_energy_accum: 0.0,
            control_bus: ControlBus::new(),
            core1: Core1Stack::new(),
            core2,
        };

        engine.reconfigure_for_sample_rate(sample_rate_hz)?;
        Ok(engine)
    }

    fn detect_max_tier(preamble: &DnamPreamble, arena_len: usize) -> Tier {
        if (preamble.flags & crate::container::FLAG_HAS_ULTRA_DELTA) != 0
            && arena_len >= (preamble.arena_offset_ultra_end as usize)
        {
            Tier::Ultra
        } else if (preamble.flags & crate::container::FLAG_HAS_FULL_DELTA) != 0
            && arena_len >= (preamble.arena_offset_full_end as usize)
        {
            Tier::Full
        } else {
            Tier::Lite
        }
    }

    fn reconfigure_for_sample_rate(&mut self, sample_rate_hz: u32) -> Result<(), DnamError> {
        let arena = self.arena_storage.as_bytes();
        let r_ac_tr = if arena.len() >= 128 {
            let v = f32::from_le_bytes([arena[124], arena[125], arena[126], arena[127]]);
            if v > 0.9 && v < 1.0 { v } else { 0.9993 }
        } else {
            0.9993
        };

        self.zoh = ZohRescaledParams::compute(
            self.preamble.sample_rate_train_hz,
            sample_rate_hz,
            self.preamble.control_decimation.max(8),
            r_ac_tr,
            0.995,
            64.0,
            128,
        )?;

        self.control_bus.load_from_arena0(
            arena,
            self.preamble.num_controls as usize,
            self.zoh.rho_ctrl,
        );
        // If JSON header was parsed and binary clamp descriptors in Arena 0 were not pre-populated,
        // compile descriptors from header.controls_schema
        if let Some(ref hdr) = self.header
            && !hdr.controls_schema.is_empty()
            && self.control_bus.descriptors[0].ctrl_kind == 0xFF
            && let Ok(compiled) = hdr.compile_clamp_descriptors()
        {
            self.control_bus.descriptors = compiled;
            self.control_bus.num_controls = hdr.controls_schema.len();
            self.control_bus.reset_to_defaults();
        }

        self.core1.load_from_arena(arena, &self.zoh);
        self.core2.load_from_arena(
            arena,
            self.preamble.is_core2_active(),
            self.preamble.ring_buffer_pow2_log2,
            &self.zoh,
        );
        self.ctrl_sub_counter = 0;
        self.ctrl_energy_accum = 0.0;
        Ok(())
    }

    /// Returns the active [`Tier`].
    #[inline]
    pub const fn tier(&self) -> Tier {
        self.tier
    }

    /// Switches the active [`Tier`] (`Lite`, `Full`, `Ultra`) at runtime with `0` allocations
    /// and `0.00 dB` LUFS continuity via `tier_gain_trim_lin`.
    #[inline]
    pub fn set_tier(&mut self, tier: Tier) {
        let clamped = tier.min(self.max_available_tier);
        if clamped != self.tier {
            self.tier = clamped;
            let arena = self.arena_storage.as_bytes();
            self.core1.collapse_stereo_head_3khz(
                arena,
                &self.control_bus.z_head,
                self.tier.core1_groups(),
            );
        }
    }

    /// Rescales all discrete-time poles, delays, and control-bus strides for `sample_rate_hz`
    /// ($44,100\text{--}96,000\text{ Hz}$).
    #[inline]
    pub fn set_sample_rate(&mut self, sample_rate_hz: u32) -> Result<(), DnamError> {
        self.reconfigure_for_sample_rate(sample_rate_hz)
    }

    /// Returns the current host sample rate in Hz.
    #[inline]
    pub const fn sample_rate_hz(&self) -> u32 {
        self.zoh.sample_rate_host_hz
    }

    /// Returns the container's cryptographic [`VerificationStatus`].
    #[inline]
    pub const fn verification_status(&self) -> &VerificationStatus {
        &self.verification_status
    }

    /// Returns the 64-byte [`DnamPreamble`].
    #[inline]
    pub const fn preamble(&self) -> &DnamPreamble {
        &self.preamble
    }

    /// Returns the parsed [`DnamJsonHeader`] (or `None` before a model is loaded into `with_max_capacity`).
    #[inline]
    pub const fn header(&self) -> Option<&DnamJsonHeader> {
        self.header.as_ref()
    }

    /// Sets a user control value by index (`0..num_controls`) with smooth $3\text{ kHz}$ anti-zipper
    /// interpolation for continuous `range` controls.
    #[inline]
    pub fn set_control_value(&mut self, ctrl_idx: usize, physical_value: f32) {
        self.control_bus.set_control_value(ctrl_idx, physical_value);
    }

    /// Sets a user control value and snaps immediately without anti-zipper lag, refreshing the
    /// collapsed stereo output head immediately.
    #[inline]
    pub fn set_control_immediate(&mut self, ctrl_idx: usize, physical_value: f32) {
        self.control_bus
            .set_control_immediate(ctrl_idx, physical_value);
        let arena = self.arena_storage.as_bytes();
        if self
            .control_bus
            .tick_3khz(0.0, self.preamble.is_lfo_env_active(), arena)
        {
            self.core1.collapse_stereo_head_3khz(
                arena,
                &self.control_bus.z_head,
                self.tier.core1_groups(),
            );
        }
    }

    /// Applies a named built-in preset from `header.presets` and snaps all controls to the preset's
    /// settings without transient pops.
    pub fn apply_preset(&mut self, preset_id: &str) -> Result<(), DnamError> {
        let Some(ref hdr) = self.header else {
            return Err(DnamError::SchemaValidationError(
                "cannot apply named preset without a loaded JSON header".into(),
            ));
        };
        let preset = hdr
            .presets
            .iter()
            .find(|p| p.id == preset_id)
            .ok_or_else(|| {
                DnamError::SchemaValidationError(alloc::format!("unknown preset id: {preset_id}"))
            })?;

        for (ctrl_idx, ctrl_def) in hdr.controls_schema.iter().enumerate() {
            if let Some(val) = preset.controls.get(&ctrl_def.id) {
                match ctrl_def.kind {
                    crate::schema::ControlKind::Range => {
                        if let Some(num) = val.as_f64() {
                            self.control_bus.set_control_immediate(ctrl_idx, num as f32);
                        }
                    }
                    crate::schema::ControlKind::Switch => {
                        if let Some(s) = val.as_str()
                            && let Some(ref opts) = ctrl_def.declared_options
                            && let Some(opt_idx) = opts.iter().position(|o| o == s)
                        {
                            self.control_bus
                                .set_control_immediate(ctrl_idx, opt_idx as f32);
                        }
                    }
                }
            }
        }
        let arena = self.arena_storage.as_bytes();
        let _ = self
            .control_bus
            .tick_3khz(0.0, self.preamble.is_lfo_env_active(), arena);
        self.core1.collapse_stereo_head_3khz(
            arena,
            &self.control_bus.z_head,
            self.tier.core1_groups(),
        );
        Ok(())
    }

    /// Resets Core 1 and Core 2 to their analytical 3-state DC resting equilibrium (`0.000000` Sample-0 transient).
    #[inline]
    pub fn reset_to_dc_equilibrium(&mut self) {
        self.core1.reset_to_dc_equilibrium();
        self.core2.reset_state();
        self.ctrl_sub_counter = 0;
        self.ctrl_energy_accum = 0.0;
    }

    /// Captures a lock-free [`DnamTelemetrySnapshot`] of the current engine state.
    pub fn telemetry_snapshot(&self) -> DnamTelemetrySnapshot {
        DnamTelemetrySnapshot {
            active_tier: self.tier as u8,
            core2_active: u8::from(self.core2.active),
            lfo_env_active: u8::from(self.preamble.is_lfo_env_active()),
            stereo_out: u8::from(self.preamble.is_stereo_out()),
            active_clamp_mask: self.control_bus.active_clamp_mask,
            control_decimation_host: self.zoh.control_decimation_host as u16,
            max_adaa_drive: self.core1.max_adaa_drive,
            max_pole_radius: self.core1.max_pole_radius,
            max_effective_loop_gain: self.core2.max_effective_loop_gain,
            tail_time_ms: self.preamble.tail_time_ms,
            z_head: self.control_bus.z_head,
            env_states: self.control_bus.env_states,
            lfo_harmonics: self.control_bus.lfo_harmonics,
            clamped_controls: self.control_bus.clamped_values,
        }
    }

    /// Internal single-sample execution step shared across all 4 I/O topologies.
    #[inline(always)]
    fn step_frame(&mut self, in_l: f32, in_r: f32, force_mono_head: bool) -> [f32; 2] {
        let arena = self.arena_storage.as_bytes();
        let k_ctrl = self.zoh.control_decimation_host;
        let core1_groups = self.tier.core1_groups();
        let fdn_groups = self.tier.fdn_groups();
        let mono_only = force_mono_head || !self.preamble.is_stereo_out();

        // Accumulate block energy for 3 kHz envelope follower
        self.ctrl_energy_accum += 0.5 * (in_l.abs() + in_r.abs());
        self.ctrl_sub_counter += 1;
        if self.ctrl_sub_counter >= k_ctrl {
            let avg_energy = self.ctrl_energy_accum * (1.0 / 16.0);
            self.ctrl_sub_counter = 0;
            self.ctrl_energy_accum = 0.0;
            if self
                .control_bus
                .tick_3khz(avg_energy, self.preamble.is_lfo_env_active(), arena)
            {
                self.core1
                    .collapse_stereo_head_3khz(arena, &self.control_bus.z_head, core1_groups);
            }
        }

        let core1_out = self
            .core1
            .process_sample(in_l, in_r, arena, core1_groups, mono_only);

        let (wet_l, wet_r, dry_l, dry_r) = if self.core2.active {
            let c2 = self.core2.process_sample(
                in_l,
                in_r,
                &self.core1.latest_core1_tap,
                &self.control_bus.core2_delay_mod_samples,
                fdn_groups,
            );
            (core1_out[0] + c2[0], core1_out[1] + c2[1], c2[2], c2[3])
        } else {
            (core1_out[0], core1_out[1], in_l, in_r)
        };

        let trim = self.core1.tier_gain_trim_lin[self.tier as usize];
        let dry_g = self.core1.mix_dry_gain;
        let wet_g = self.core1.mix_wet_gain * trim;

        let out_l = dry_g * dry_l + wet_g * wet_l;
        let out_r = if mono_only {
            out_l
        } else {
            dry_g * dry_r + wet_g * wet_r
        };
        [out_l, out_r]
    }

    /// **Topology 1 (`Stereo -> Stereo` Separate Buffers):**
    /// Processes `left_in` and `right_in` into `left_out` and `right_out` with zero heap allocations
    /// and automatic [`FtzDazGuard`] denormal suppression.
    pub fn process_block(
        &mut self,
        left_in: &[f32],
        right_in: &[f32],
        left_out: &mut [f32],
        right_out: &mut [f32],
    ) {
        let _ftz = FtzDazGuard::new();
        let n = left_in
            .len()
            .min(right_in.len())
            .min(left_out.len())
            .min(right_out.len());
        self.core1.max_adaa_drive_vec = [0.0; 4];
        self.core1.max_adaa_drive = 0.0;
        self.core1.max_pole_radius = 0.0;
        self.core2.max_effective_loop_gain = 0.0;

        for i in 0..n {
            let [l, r] = self.step_frame(left_in[i], right_in[i], false);
            left_out[i] = l;
            right_out[i] = r;
        }
    }

    /// **Topology 2 (`Stereo -> Stereo` In-Place):**
    /// Processes `left_io` and `right_io` in-place (VST3/CLAP standard buffer topology) with zero heap allocations.
    pub fn process_block_in_place(&mut self, left_io: &mut [f32], right_io: &mut [f32]) {
        let _ftz = FtzDazGuard::new();
        let n = left_io.len().min(right_io.len());
        self.core1.max_adaa_drive_vec = [0.0; 4];
        self.core1.max_adaa_drive = 0.0;
        self.core1.max_pole_radius = 0.0;
        self.core2.max_effective_loop_gain = 0.0;

        for i in 0..n {
            let [l, r] = self.step_frame(left_io[i], right_io[i], false);
            left_io[i] = l;
            right_io[i] = r;
        }
    }

    /// **Topology 3 (`Mono -> Stereo`):**
    /// Feeds a single mono DI input (`mono_in`) into true-stereo Core 1 + Core 2 (`left_out`, `right_out`).
    pub fn process_mono_to_stereo(
        &mut self,
        mono_in: &[f32],
        left_out: &mut [f32],
        right_out: &mut [f32],
    ) {
        let _ftz = FtzDazGuard::new();
        let n = mono_in.len().min(left_out.len()).min(right_out.len());
        self.core1.max_adaa_drive_vec = [0.0; 4];
        self.core1.max_adaa_drive = 0.0;
        self.core1.max_pole_radius = 0.0;
        self.core2.max_effective_loop_gain = 0.0;

        for i in 0..n {
            let x = mono_in[i];
            let [l, r] = self.step_frame(x, x, false);
            left_out[i] = l;
            right_out[i] = r;
        }
    }

    /// **Topology 4 (`Mono -> Mono`, 50% Output Head MAC Savings):**
    /// Evaluates only Channel 0 of the collapsed 16-tap output head (`force_mono_head = true`),
    /// saving 50% of output head MACs on mono pedalboard blocks.
    pub fn process_mono(&mut self, mono_in: &[f32], mono_out: &mut [f32]) {
        let _ftz = FtzDazGuard::new();
        let n = mono_in.len().min(mono_out.len());
        self.core1.max_adaa_drive_vec = [0.0; 4];
        self.core1.max_adaa_drive = 0.0;
        self.core1.max_pole_radius = 0.0;
        self.core2.max_effective_loop_gain = 0.0;

        for i in 0..n {
            let x = mono_in[i];
            let [l, _] = self.step_frame(x, x, true);
            mono_out[i] = l;
        }
    }
}

// ============================================================================
// Block-Level Runtime-Dispatched Engine (`DynamicDnamEngine<'a>`)
// ============================================================================

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

    /// Creates an owned `'static` [`DynamicDnamEngine`] using runtime CPU feature detection.
    pub fn from_bytes_owned(
        bytes: &[u8],
        tier: Tier,
        sample_rate_hz: u32,
    ) -> Result<DynamicDnamEngine<'static>, DnamError> {
        #[cfg(target_arch = "aarch64")]
        {
            if crate::math::arm_sme::ArmSmeBackend::is_supported() {
                return Ok(DynamicDnamEngine::ArmSme(DnamEngine::from_bytes_owned(
                    bytes,
                    tier,
                    sample_rate_hz,
                )?));
            }
            Ok(DynamicDnamEngine::Neon(DnamEngine::from_bytes_owned(
                bytes,
                tier,
                sample_rate_hz,
            )?))
        }
        #[cfg(target_arch = "x86_64")]
        {
            return Ok(DynamicDnamEngine::X86Simd(DnamEngine::from_bytes_owned(
                bytes,
                tier,
                sample_rate_hz,
            )?));
        }
        #[cfg(target_arch = "wasm32")]
        {
            return Ok(DynamicDnamEngine::WasmSimd128(
                DnamEngine::from_bytes_owned(bytes, tier, sample_rate_hz)?,
            ));
        }
        #[cfg(not(any(
            target_arch = "aarch64",
            target_arch = "x86_64",
            target_arch = "wasm32"
        )))]
        {
            Ok(DynamicDnamEngine::Scalar(DnamEngine::from_bytes_owned(
                bytes,
                tier,
                sample_rate_hz,
            )?))
        }
    }

    /// Creates a [`DynamicDnamEngine`] with an explicit [`BackendKind`] (falling back to `Scalar`
    /// if the requested ISA is not compiled for the current target).
    pub fn from_bytes_with_backend(
        bytes: &'a [u8],
        tier: Tier,
        sample_rate_hz: u32,
        backend: BackendKind,
    ) -> Result<Self, DnamError> {
        match backend {
            #[cfg(target_arch = "aarch64")]
            BackendKind::ArmSme => Ok(Self::ArmSme(DnamEngine::from_bytes(
                bytes,
                tier,
                sample_rate_hz,
            )?)),
            #[cfg(target_arch = "aarch64")]
            BackendKind::Neon | BackendKind::ArmNeon => Ok(Self::Neon(DnamEngine::from_bytes(
                bytes,
                tier,
                sample_rate_hz,
            )?)),
            #[cfg(target_arch = "x86_64")]
            BackendKind::X86Simd | BackendKind::X86Avx2Fma | BackendKind::X86Sse42 => Ok(
                Self::X86Simd(DnamEngine::from_bytes(bytes, tier, sample_rate_hz)?),
            ),
            #[cfg(target_arch = "wasm32")]
            BackendKind::WasmSimd128 => Ok(Self::WasmSimd128(DnamEngine::from_bytes(
                bytes,
                tier,
                sample_rate_hz,
            )?)),
            _ => Ok(Self::Scalar(DnamEngine::from_bytes(
                bytes,
                tier,
                sample_rate_hz,
            )?)),
        }
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

    /// Switches the active [`Tier`].
    #[inline]
    pub fn set_tier(&mut self, tier: Tier) {
        match self {
            Self::Scalar(e) => e.set_tier(tier),
            #[cfg(target_arch = "aarch64")]
            Self::Neon(e) => e.set_tier(tier),
            #[cfg(target_arch = "aarch64")]
            Self::ArmSme(e) => e.set_tier(tier),
            #[cfg(target_arch = "x86_64")]
            Self::X86Simd(e) => e.set_tier(tier),
            #[cfg(target_arch = "wasm32")]
            Self::WasmSimd128(e) => e.set_tier(tier),
        }
    }

    /// Rescales discrete-time parameters for a new host sample rate.
    #[inline]
    pub fn set_sample_rate(&mut self, sample_rate_hz: u32) -> Result<(), DnamError> {
        match self {
            Self::Scalar(e) => e.set_sample_rate(sample_rate_hz),
            #[cfg(target_arch = "aarch64")]
            Self::Neon(e) => e.set_sample_rate(sample_rate_hz),
            #[cfg(target_arch = "aarch64")]
            Self::ArmSme(e) => e.set_sample_rate(sample_rate_hz),
            #[cfg(target_arch = "x86_64")]
            Self::X86Simd(e) => e.set_sample_rate(sample_rate_hz),
            #[cfg(target_arch = "wasm32")]
            Self::WasmSimd128(e) => e.set_sample_rate(sample_rate_hz),
        }
    }

    /// Sets a user control value with smooth $3\text{ kHz}$ anti-zipper interpolation.
    #[inline]
    pub fn set_control_value(&mut self, ctrl_idx: usize, value: f32) {
        match self {
            Self::Scalar(e) => e.set_control_value(ctrl_idx, value),
            #[cfg(target_arch = "aarch64")]
            Self::Neon(e) => e.set_control_value(ctrl_idx, value),
            #[cfg(target_arch = "aarch64")]
            Self::ArmSme(e) => e.set_control_value(ctrl_idx, value),
            #[cfg(target_arch = "x86_64")]
            Self::X86Simd(e) => e.set_control_value(ctrl_idx, value),
            #[cfg(target_arch = "wasm32")]
            Self::WasmSimd128(e) => e.set_control_value(ctrl_idx, value),
        }
    }

    /// Sets a user control value and snaps immediately.
    #[inline]
    pub fn set_control_immediate(&mut self, ctrl_idx: usize, value: f32) {
        match self {
            Self::Scalar(e) => e.set_control_immediate(ctrl_idx, value),
            #[cfg(target_arch = "aarch64")]
            Self::Neon(e) => e.set_control_immediate(ctrl_idx, value),
            #[cfg(target_arch = "aarch64")]
            Self::ArmSme(e) => e.set_control_immediate(ctrl_idx, value),
            #[cfg(target_arch = "x86_64")]
            Self::X86Simd(e) => e.set_control_immediate(ctrl_idx, value),
            #[cfg(target_arch = "wasm32")]
            Self::WasmSimd128(e) => e.set_control_immediate(ctrl_idx, value),
        }
    }

    /// Resets all states to their analytical DC resting equilibrium.
    #[inline]
    pub fn reset_to_dc_equilibrium(&mut self) {
        match self {
            Self::Scalar(e) => e.reset_to_dc_equilibrium(),
            #[cfg(target_arch = "aarch64")]
            Self::Neon(e) => e.reset_to_dc_equilibrium(),
            #[cfg(target_arch = "aarch64")]
            Self::ArmSme(e) => e.reset_to_dc_equilibrium(),
            #[cfg(target_arch = "x86_64")]
            Self::X86Simd(e) => e.reset_to_dc_equilibrium(),
            #[cfg(target_arch = "wasm32")]
            Self::WasmSimd128(e) => e.reset_to_dc_equilibrium(),
        }
    }

    /// Processes a stereo audio block (`Stereo -> Stereo`).
    #[inline]
    pub fn process_block(
        &mut self,
        left_in: &[f32],
        right_in: &[f32],
        left_out: &mut [f32],
        right_out: &mut [f32],
    ) {
        match self {
            Self::Scalar(e) => e.process_block(left_in, right_in, left_out, right_out),
            #[cfg(target_arch = "aarch64")]
            Self::Neon(e) => e.process_block(left_in, right_in, left_out, right_out),
            #[cfg(target_arch = "aarch64")]
            Self::ArmSme(e) => e.process_block(left_in, right_in, left_out, right_out),
            #[cfg(target_arch = "x86_64")]
            Self::X86Simd(e) => e.process_block(left_in, right_in, left_out, right_out),
            #[cfg(target_arch = "wasm32")]
            Self::WasmSimd128(e) => e.process_block(left_in, right_in, left_out, right_out),
        }
    }

    /// Processes a mono audio block (`Mono -> Mono`).
    #[inline]
    pub fn process_mono(&mut self, mono_in: &[f32], mono_out: &mut [f32]) {
        match self {
            Self::Scalar(e) => e.process_mono(mono_in, mono_out),
            #[cfg(target_arch = "aarch64")]
            Self::Neon(e) => e.process_mono(mono_in, mono_out),
            #[cfg(target_arch = "aarch64")]
            Self::ArmSme(e) => e.process_mono(mono_in, mono_out),
            #[cfg(target_arch = "x86_64")]
            Self::X86Simd(e) => e.process_mono(mono_in, mono_out),
            #[cfg(target_arch = "wasm32")]
            Self::WasmSimd128(e) => e.process_mono(mono_in, mono_out),
        }
    }

    /// Returns a lock-free [`DnamTelemetrySnapshot`].
    #[inline]
    pub fn telemetry_snapshot(&self) -> DnamTelemetrySnapshot {
        match self {
            Self::Scalar(e) => e.telemetry_snapshot(),
            #[cfg(target_arch = "aarch64")]
            Self::Neon(e) => e.telemetry_snapshot(),
            #[cfg(target_arch = "aarch64")]
            Self::ArmSme(e) => e.telemetry_snapshot(),
            #[cfg(target_arch = "x86_64")]
            Self::X86Simd(e) => e.telemetry_snapshot(),
            #[cfg(target_arch = "wasm32")]
            Self::WasmSimd128(e) => e.telemetry_snapshot(),
        }
    }
}
