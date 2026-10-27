//! Primary generic real-time DNAM ("Tri-Physics") inference engine (`DnamEngine<'a, B>`).

use crate::container::{ARENA_OFFSET_LITE_END, DnamContainerView, DnamPreamble, ZohRescaledParams};
use crate::control::ControlBus;
use crate::core1::Core1Stack;
use crate::core2::ModulatedUfdn;
use crate::math::{DefaultMathBackend, MathBackend};
use crate::memory::{AlignedBuffer, AlignedFp16Block, ArenaStorage, CacheLine64};
use crate::provenance::{CryptoProvider, DefaultCryptoProvider, VerificationStatus};
use crate::schema::{
    ControlCalibrationRole, DnamError, DnamJsonHeader, DnamProtectionProfile, TempoSubdivision,
    cold_pow_f64,
};
use crate::types::{DnamTelemetrySnapshot, FtzDazGuard, Tier};

/// Zero-allocation real-time DNAM ("Tri-Physics") inference engine parameterized over a
/// [`MathBackend`] (`DefaultMathBackend` by default).
#[derive(Debug)]
pub struct DnamEngine<'a, B: MathBackend = DefaultMathBackend> {
    preamble: DnamPreamble,
    header: Option<DnamJsonHeader>,
    verification_status: VerificationStatus,
    protection_profile: DnamProtectionProfile,
    arena_storage: ArenaStorage<'a>,
    tier: Tier,
    max_available_tier: Tier,
    zoh: ZohRescaledParams,
    ctrl_sub_counter: usize,
    ctrl_energy_accum: f32,
    bypassed: bool,
    trails_enabled: bool,
    freeze_mutes_dry: bool,
    stereo_width: f32,
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

    /// Loads a `.dnam`, `.dnam-ckpt`, or `.dnam-seal` (`b"DNSL"`) container into a `'static` owned engine
    /// by copying the 64-byte aligned tensor arena into an owned [`AlignedBuffer`].
    pub fn from_bytes_owned(
        bytes: &[u8],
        tier: Tier,
        sample_rate_hz: u32,
    ) -> Result<DnamEngine<'static, B>, DnamError> {
        let crypto = DefaultCryptoProvider::default();
        Self::from_bytes_owned_with_crypto(bytes, tier, sample_rate_hz, &crypto)
    }

    /// Loads a `.dnam`, `.dnam-ckpt`, or `.dnam-seal` (`b"DNSL"`) container into a `'static` owned engine
    /// using a custom [`CryptoProvider`].
    pub fn from_bytes_owned_with_crypto<C: CryptoProvider>(
        bytes: &[u8],
        tier: Tier,
        sample_rate_hz: u32,
        crypto: &C,
    ) -> Result<DnamEngine<'static, B>, DnamError> {
        let view = DnamContainerView::from_bytes(bytes, crypto)?;
        let arena_bytes = view.arena_storage.as_bytes();
        let mut owned_arena = AlignedBuffer::<CacheLine64>::new_zeroed(arena_bytes.len() / 64);
        owned_arena.as_bytes_mut().copy_from_slice(arena_bytes);

        let owned_view = DnamContainerView {
            preamble: view.preamble,
            header: view.header,
            verification_status: view.verification_status,
            protection_profile: view.protection_profile,
            arena_storage: ArenaStorage::from_owned_buffer(owned_arena),
        };
        DnamEngine::<'static, B>::from_view(owned_view, tier, sample_rate_hz)
    }

    /// Loads a `.dnam` or `.dnam-ckpt` container using a custom [`CryptoProvider`] (e.g. hardware
    /// `OP-TEE`, Apple Secure Enclave `SEP`, Windows/Linux `TPM 2.0`, or keyed [`RustCryptoProvider`](crate::provenance::RustCryptoProvider)).
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
            protection_profile: DnamProtectionProfile::Open,
            arena_storage: ArenaStorage::from_owned_buffer(owned_arena),
            tier: Tier::Lite,
            max_available_tier: Tier::Lite,
            zoh,
            ctrl_sub_counter: 0,
            ctrl_energy_accum: 0.0,
            bypassed: false,
            trails_enabled: false,
            freeze_mutes_dry: false,
            stereo_width: 1.0,
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
        self.freeze_mutes_dry = false;
        self.verification_status = VerificationStatus::Unsigned;
        self.protection_profile = DnamProtectionProfile::Open;
        self.max_available_tier = Self::detect_max_tier(&preamble, copy_len);
        self.tier = tier.min(self.max_available_tier);

        self.reconfigure_for_sample_rate(sample_rate_hz)?;
        Ok(())
    }

    /// Hot-swaps a new `.dnam` model in-place into an owned engine without requiring `bytes`
    /// to outlive `'a` (updating `ArenaStorage::Owned` in-place with zero allocations when capacity suffices).
    pub fn reload_model_in_place_owned(
        &mut self,
        bytes: &[u8],
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

        let copy_len = raw_arena.len().min(36_224);
        if !self.arena_storage.reload_in_place(&raw_arena[..copy_len]) {
            self.arena_storage = ArenaStorage::from_slice_owned(&raw_arena[..copy_len]);
        }

        self.preamble = preamble;
        self.header = None;
        self.freeze_mutes_dry = false;
        self.verification_status = VerificationStatus::Unsigned;
        self.protection_profile = DnamProtectionProfile::Open;
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

        let freeze_mutes_dry = view.header.io_config.capabilities.freeze_mutes_dry;
        let mut engine = Self {
            preamble: view.preamble,
            header: Some(view.header),
            verification_status: view.verification_status,
            protection_profile: view.protection_profile,
            arena_storage: view.arena_storage,
            tier: active_tier,
            max_available_tier,
            zoh,
            ctrl_sub_counter: 0,
            ctrl_energy_accum: 0.0,
            bypassed: false,
            trails_enabled: false,
            freeze_mutes_dry,
            stereo_width: 1.0,
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
            && (self.control_bus.descriptors[0].ctrl_kind == 0xFF
                || self.control_bus.descriptors[0].scale_norm == 0.0)
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

    /// Returns the cached [`ZohRescaledParams`] for the current host sample rate.
    #[inline]
    pub const fn zoh(&self) -> &ZohRescaledParams {
        &self.zoh
    }

    /// Returns the container's cryptographic [`VerificationStatus`].
    #[inline]
    pub const fn verification_status(&self) -> &VerificationStatus {
        &self.verification_status
    }

    /// Returns the container's resolved [`DnamProtectionProfile`].
    #[inline]
    pub const fn protection_profile(&self) -> DnamProtectionProfile {
        self.protection_profile
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
        self.refresh_calibrated_control_effects(ctrl_idx, physical_value);
    }

    /// Sets a user control value and snaps immediately without anti-zipper lag, refreshing the
    /// collapsed stereo output head and Core 2 $3\text{ kHz}$ modulation immediately.
    #[inline]
    pub fn set_control_immediate(&mut self, ctrl_idx: usize, physical_value: f32) {
        self.control_bus
            .set_control_immediate(ctrl_idx, physical_value);
        self.refresh_calibrated_control_effects(ctrl_idx, physical_value);
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
        if self.control_bus.core2_mod_dirty && self.core2.active {
            self.core2.apply_control_bus_3khz(
                &self.control_bus.core2_gain_targets,
                &self.control_bus.core2_warp_mod,
                &self.control_bus.core2_grain_len_scale,
                &self.control_bus.core2_state_mod,
                self.control_bus.env_states[3],
            );
            self.control_bus.core2_mod_dirty = false;
        }
    }

    /// Updates engine-level calibrated effects (`StereoMode`, `StereoWidth`, `FreezeLatch`,
    /// `GrainSizeMs`, `GrainPitchSemitones`, `GrainSpray`) when `ctrl_idx` changes.
    fn refresh_calibrated_control_effects(&mut self, ctrl_idx: usize, physical_value: f32) {
        let Some(ref hdr) = self.header else {
            return;
        };
        let Some(ctrl_def) = hdr.controls_schema.get(ctrl_idx) else {
            return;
        };
        let Some(ref cal) = ctrl_def.calibration else {
            return;
        };
        let decl = ctrl_def.declared_range.unwrap_or([0.0, 10.0]);
        let span = (decl[1] - decl[0]).max(1e-6);
        let norm_01 = ((physical_value - decl[0]) / span).clamp(0.0, 1.0);
        let norm_u = 2.0 * norm_01 - 1.0;

        match cal.role {
            ControlCalibrationRole::StereoMode => {
                if let Some(ref opts) = ctrl_def.declared_options {
                    let idx = (physical_value.round() as isize).max(0) as usize;
                    if let Some(opt_label) = opts.get(idx)
                        && let Some(w) = cal.stereo_width_for_option(opt_label)
                    {
                        self.stereo_width = w;
                    }
                }
            }
            ControlCalibrationRole::StereoWidth => {
                self.stereo_width = norm_01;
            }
            ControlCalibrationRole::FreezeLatch => {
                if ctrl_def.kind == crate::schema::ControlKind::Switch {
                    if let Some(ref opts) = ctrl_def.declared_options {
                        let idx = (physical_value.round() as isize).max(0) as usize;
                        if let Some(opt_label) = opts.get(idx)
                            && let Some(alpha) = cal.freeze_amount_for_option(opt_label)
                        {
                            self.core2.set_freeze_amount(alpha);
                        }
                    }
                } else {
                    self.core2.set_freeze_amount(norm_01);
                }
            }
            ControlCalibrationRole::GrainSizeMs => {
                let val_ms = cal.evaluate_lobatto_anchors(norm_u).unwrap_or_else(|| {
                    if decl[1] <= 10.0 {
                        10.0 + norm_01 * 240.0
                    } else {
                        physical_value.clamp(5.0, 500.0)
                    }
                });
                self.core2.set_grain_params(
                    Some(val_ms),
                    None,
                    None,
                    self.zoh.sample_rate_host_hz as f32,
                );
            }
            ControlCalibrationRole::GrainPitchSemitones => {
                let val_st = cal.evaluate_lobatto_anchors(norm_u).unwrap_or_else(|| {
                    if decl[0] >= 0.0 && decl[1] <= 10.0 {
                        norm_u * 12.0
                    } else {
                        physical_value.clamp(-24.0, 24.0)
                    }
                });
                self.core2.set_grain_params(
                    None,
                    Some(val_st),
                    None,
                    self.zoh.sample_rate_host_hz as f32,
                );
            }
            ControlCalibrationRole::GrainSpray => {
                let val_spray = cal
                    .evaluate_lobatto_anchors(norm_u)
                    .unwrap_or(norm_01)
                    .clamp(0.0, 1.0);
                self.core2.set_grain_params(
                    None,
                    None,
                    Some(val_spray),
                    self.zoh.sample_rate_host_hz as f32,
                );
            }
            _ => {}
        }
    }

    /// Engages or disengages Core 2 Unitary Spectral Freeze (`alpha = 1.0` vs `0.0`).
    #[inline]
    pub fn set_freeze(&mut self, freeze: bool) {
        self.core2.set_freeze(freeze);
    }

    /// Sets the continuous Core 2 Unitary Spectral Freeze amount $\alpha_{\text{freeze}} \in [0.0, 1.0]$.
    #[inline]
    pub fn set_freeze_amount(&mut self, alpha: f32) {
        self.core2.set_freeze_amount(alpha);
    }

    /// Returns `true` if Core 2 Unitary Spectral Freeze is currently engaged ($\alpha_{\text{freeze}} > 0.5$).
    #[inline]
    pub fn is_frozen(&self) -> bool {
        self.core2.is_frozen()
    }

    /// Configures bypass and spillover **Trails** (`bypassed = true, trails = true`).
    ///
    /// - When `bypassed && !trails`: true hard bypass (output equals dry input, `0` wet signal).
    /// - When `bypassed && trails`: dry input passes through at unity while new audio injection into
    ///   Core 1 / Core 2 is muted (`0.0`), allowing Core 2's `ModulatedUfdn` delay/reverb tank to ring out naturally.
    #[inline]
    pub fn set_bypass(&mut self, bypassed: bool, trails: bool) {
        self.bypassed = bypassed;
        self.trails_enabled = trails;
    }

    /// Returns `(bypassed, trails_enabled)`.
    #[inline]
    pub const fn bypass_state(&self) -> (bool, bool) {
        (self.bypassed, self.trails_enabled)
    }

    /// Sets the Mid/Side stereo width $\alpha_{\text{width}} \in [0.0, 1.0]$ (`0.0` = Mono, `1.0` = Full Stereo).
    #[inline]
    pub fn set_stereo_width(&mut self, width: f32) {
        self.stereo_width = width.clamp(0.0, 1.0);
    }

    /// Returns the active Mid/Side stereo width $\alpha_{\text{width}} \in [0.0, 1.0]$.
    #[inline]
    pub const fn stereo_width(&self) -> f32 {
        self.stereo_width
    }

    /// Configures host parallel Dry/Wet `Blend` ($b \in [0.0, 1.0]$), output gain trim in dB,
    /// and auto-unity leveling from `header.io_config.{capabilities, leveling}`.
    ///
    /// - Uses linear voltage mixing ($\text{dry} = 1 - b$, $\text{wet} = b$) so correlated bass fundamentals
    ///   sum to `0.0 dB` at `50%` blend.
    /// - When `io_config.capabilities.wet_polarity_inverted` is `true`, inverts `wet_g` (`wet_g *= -1.0`)
    ///   so `0%` blend matches bypass polarity and `50%` blend sums constructively without phase cancellation.
    /// - When `apply_unity_leveling` is `true`, adds `io_config.leveling.calibrated_trim_db` to `output_gain_db`
    ///   on the wet path so at default control settings ($\mathbf{u} = \mathbf{0}$) and `0.0 dB` output trim,
    ///   the engine outputs exact unity (`+ target_gain_db`).
    pub fn configure_host_mix_and_leveling(
        &mut self,
        blend_wet: f32,
        output_gain_db: f32,
        apply_unity_leveling: bool,
    ) {
        let b = blend_wet.clamp(0.0, 1.0);
        let (pol_sign, cal_trim_db) = if let Some(ref hdr) = self.header {
            let sign = if hdr.io_config.capabilities.wet_polarity_inverted {
                -1.0_f32
            } else {
                1.0_f32
            };
            let trim = if apply_unity_leveling && hdr.io_config.leveling.auto_unity_enabled {
                hdr.io_config.leveling.calibrated_trim_db
            } else {
                0.0_f32
            };
            (sign, trim)
        } else {
            (1.0_f32, 0.0_f32)
        };

        let out_scale = cold_pow_f64(10.0, (output_gain_db as f64) / 20.0) as f32;
        let wet_trim_scale =
            cold_pow_f64(10.0, ((output_gain_db + cal_trim_db) as f64) / 20.0) as f32;

        self.core1.mix_dry_gain = (1.0 - b) * out_scale;
        self.core1.mix_wet_gain = pol_sign * b * wet_trim_scale;
    }

    /// Synchronizes all calibrated delay-time (`DelayTimeMs`) and LFO-rate (`LfoRateHz`) controls
    /// (and any associated `Range` switch in `conditional_physical_anchors`) to `(bpm, subdivision_override)`.
    ///
    /// Returns the number of calibrated controls updated.
    pub fn sync_tempo(
        &mut self,
        bpm: f32,
        subdivision_override: Option<TempoSubdivision>,
        fold_octave: bool,
    ) -> usize {
        let Some(ref hdr) = self.header else {
            return 0;
        };
        let mut updates: alloc::vec::Vec<(usize, f32)> = alloc::vec::Vec::new();
        for (ctrl_idx, ctrl_def) in hdr.controls_schema.iter().enumerate() {
            let Some(ref cal) = ctrl_def.calibration else {
                continue;
            };
            let Some(res) = cal.resolve_tempo_sync(bpm, subdivision_override, fold_octave) else {
                continue;
            };
            if let Some((sw_id, opt_label)) = res.switch_selection
                && let Some((sw_idx, sw_def)) = hdr
                    .controls_schema
                    .iter()
                    .enumerate()
                    .find(|(_, c)| c.id == sw_id)
                && let Some(ref opts) = sw_def.declared_options
                && let Some(opt_idx) = opts.iter().position(|o| o == &opt_label)
            {
                updates.push((sw_idx, opt_idx as f32));
            }
            let decl = ctrl_def.declared_range.unwrap_or([0.0, 10.0]);
            let phys = decl[0] + 0.5 * (res.normalized_u + 1.0) * (decl[1] - decl[0]);
            updates.push((ctrl_idx, phys));
        }
        let count = updates.len();
        for (idx, val) in updates {
            self.set_control_immediate(idx, val);
        }
        count
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

        let mut pending: alloc::vec::Vec<(usize, f32)> = alloc::vec::Vec::new();
        for (ctrl_idx, ctrl_def) in hdr.controls_schema.iter().enumerate() {
            if let Some(val) = preset.controls.get(&ctrl_def.id) {
                match ctrl_def.kind {
                    crate::schema::ControlKind::Range => {
                        if let Some(num) = val.as_f64() {
                            pending.push((ctrl_idx, num as f32));
                        }
                    }
                    crate::schema::ControlKind::Switch => {
                        if let Some(s) = val.as_str()
                            && let Some(ref opts) = ctrl_def.declared_options
                            && let Some(opt_idx) = opts.iter().position(|o| o == s)
                        {
                            pending.push((ctrl_idx, opt_idx as f32));
                        }
                    }
                }
            }
        }
        for (ctrl_idx, phys_val) in pending {
            self.control_bus.set_control_immediate(ctrl_idx, phys_val);
            self.refresh_calibrated_control_effects(ctrl_idx, phys_val);
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
        if self.control_bus.core2_mod_dirty && self.core2.active {
            self.core2.apply_control_bus_3khz(
                &self.control_bus.core2_gain_targets,
                &self.control_bus.core2_warp_mod,
                &self.control_bus.core2_grain_len_scale,
                &self.control_bus.core2_state_mod,
                self.control_bus.env_states[3],
            );
            self.control_bus.core2_mod_dirty = false;
        }
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

    /// Internal single-sample execution step parameterized by compile-time tier group counts.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn step_frame_const<const GROUPS: usize, const FDN_GROUPS: usize>(
        core1: &mut Core1Stack<B>,
        core2: &mut ModulatedUfdn<'a, B>,
        control_bus: &mut ControlBus<B>,
        ctrl_sub_counter: &mut usize,
        ctrl_energy_accum: &mut f32,
        in_l: f32,
        in_r: f32,
        arena: &[u8],
        tiles: &[[[f32; 4]; 4]],
        k_ctrl: usize,
        lfo_env_active: bool,
        mono_only: bool,
        dry_g: f32,
        wet_g: f32,
        bypassed: bool,
        trails_enabled: bool,
        freeze_mutes_dry: bool,
        stereo_width: f32,
    ) -> [f32; 2] {
        if bypassed && !trails_enabled {
            return if mono_only {
                [in_l, in_l]
            } else {
                [in_l, in_r]
            };
        }

        let live_scale = if !bypassed && freeze_mutes_dry && core2.active {
            (1.0 - core2.freeze_alpha_smoothed).clamp(0.0, 1.0)
        } else {
            1.0
        };
        let (eff_in_l, eff_in_r) = if bypassed {
            (0.0, 0.0)
        } else {
            (in_l * live_scale, in_r * live_scale)
        };

        // Accumulate block energy for 3 kHz envelope follower
        *ctrl_energy_accum += 0.5 * (eff_in_l.abs() + eff_in_r.abs());
        *ctrl_sub_counter += 1;
        if *ctrl_sub_counter >= k_ctrl {
            let avg_energy = *ctrl_energy_accum * (1.0 / 16.0);
            *ctrl_sub_counter = 0;
            *ctrl_energy_accum = 0.0;
            if control_bus.tick_3khz(avg_energy, lfo_env_active, arena) {
                core1.collapse_stereo_head_3khz(arena, &control_bus.z_head, GROUPS);
            }
            if control_bus.core2_mod_dirty && core2.active {
                core2.apply_control_bus_3khz(
                    &control_bus.core2_gain_targets,
                    &control_bus.core2_warp_mod,
                    &control_bus.core2_grain_len_scale,
                    &control_bus.core2_state_mod,
                    control_bus.env_states[3],
                );
                control_bus.core2_mod_dirty = false;
            }
        }

        let h_vecs = core1.process_backbone_tiles::<GROUPS>(eff_in_l, eff_in_r, tiles);

        let (mut wet_l, mut wet_r, dry_l, dry_r) = if core2.active {
            let zero_tap = [[0.0_f32; 4]; 2];
            let tap_ref = if bypassed {
                &zero_tap
            } else {
                &core1.latest_core1_tap
            };
            let c2 = core2.process_sample(
                eff_in_l,
                eff_in_r,
                tap_ref,
                &control_bus.core2_delay_mod_samples,
                FDN_GROUPS,
            );
            let core1_out = core1.process_head_from_taps::<GROUPS>(
                h_vecs,
                if bypassed {
                    None
                } else {
                    Some((&core2.latest_ap_out, core2.head_tap_inject_gain))
                },
                mono_only,
            );
            if bypassed {
                (c2[0], c2[1], in_l, in_r)
            } else {
                (
                    core1_out[0] + c2[0],
                    core1_out[1] + c2[1],
                    c2[2] * live_scale,
                    c2[3] * live_scale,
                )
            }
        } else if bypassed {
            let _ = core1.process_head_from_taps::<GROUPS>(h_vecs, None, mono_only);
            (0.0, 0.0, in_l, in_r)
        } else {
            let core1_out = core1.process_head_from_taps::<GROUPS>(h_vecs, None, mono_only);
            (core1_out[0], core1_out[1], in_l, in_r)
        };

        if !mono_only && stereo_width < 0.999_999 {
            let mid = 0.5 * (wet_l + wet_r);
            let side = 0.5 * (wet_l - wet_r);
            wet_l = mid + stereo_width * side;
            wet_r = mid - stereo_width * side;
        }

        let eff_dry_g = if bypassed { 1.0 } else { dry_g };
        let out_l = eff_dry_g * dry_l + wet_g * wet_l;
        let out_r = if mono_only {
            out_l
        } else {
            eff_dry_g * dry_r + wet_g * wet_r
        };
        [out_l, out_r]
    }

    #[inline(always)]
    fn reset_block_telemetry(&mut self) {
        self.core1.max_adaa_drive_vec = [0.0; 4];
        self.core1.max_adaa_drive = 0.0;
        self.core1.max_pole_radius = 0.0;
        self.core2.max_effective_loop_gain = 0.0;
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
        self.reset_block_telemetry();
        let arena = self.arena_storage.as_bytes();
        let tiles = Core1Stack::<B>::arena_tiles(arena);
        let k_ctrl = self.zoh.control_decimation_host;
        let lfo_env_active = self.preamble.is_lfo_env_active();
        let mono_only = !self.preamble.is_stereo_out();
        let dry_g = self.core1.mix_dry_gain;
        let wet_g = self.core1.mix_wet_gain * self.core1.tier_gain_trim_lin[self.tier as usize];
        let bypassed = self.bypassed;
        let trails = self.trails_enabled;
        let freeze_mutes_dry = self.freeze_mutes_dry;
        let width = self.stereo_width;

        match self.tier {
            Tier::Lite => {
                for i in 0..n {
                    let [l, r] = Self::step_frame_const::<1, 1>(
                        &mut self.core1,
                        &mut self.core2,
                        &mut self.control_bus,
                        &mut self.ctrl_sub_counter,
                        &mut self.ctrl_energy_accum,
                        left_in[i],
                        right_in[i],
                        arena,
                        tiles,
                        k_ctrl,
                        lfo_env_active,
                        mono_only,
                        dry_g,
                        wet_g,
                        bypassed,
                        trails,
                        freeze_mutes_dry,
                        width,
                    );
                    left_out[i] = l;
                    right_out[i] = r;
                }
            }
            Tier::Full => {
                for i in 0..n {
                    let [l, r] = Self::step_frame_const::<2, 2>(
                        &mut self.core1,
                        &mut self.core2,
                        &mut self.control_bus,
                        &mut self.ctrl_sub_counter,
                        &mut self.ctrl_energy_accum,
                        left_in[i],
                        right_in[i],
                        arena,
                        tiles,
                        k_ctrl,
                        lfo_env_active,
                        mono_only,
                        dry_g,
                        wet_g,
                        bypassed,
                        trails,
                        freeze_mutes_dry,
                        width,
                    );
                    left_out[i] = l;
                    right_out[i] = r;
                }
            }
            Tier::Ultra => {
                for i in 0..n {
                    let [l, r] = Self::step_frame_const::<3, 2>(
                        &mut self.core1,
                        &mut self.core2,
                        &mut self.control_bus,
                        &mut self.ctrl_sub_counter,
                        &mut self.ctrl_energy_accum,
                        left_in[i],
                        right_in[i],
                        arena,
                        tiles,
                        k_ctrl,
                        lfo_env_active,
                        mono_only,
                        dry_g,
                        wet_g,
                        bypassed,
                        trails,
                        freeze_mutes_dry,
                        width,
                    );
                    left_out[i] = l;
                    right_out[i] = r;
                }
            }
        }
        if n > 0 {
            self.core1
                .finalize_block_telemetry(self.tier.core1_groups());
        }
    }

    /// **Topology 2 (`Stereo -> Stereo` In-Place):**
    /// Processes `left_io` and `right_io` in-place (VST3/CLAP standard buffer topology) with zero heap allocations.
    pub fn process_block_in_place(&mut self, left_io: &mut [f32], right_io: &mut [f32]) {
        let _ftz = FtzDazGuard::new();
        let n = left_io.len().min(right_io.len());
        self.reset_block_telemetry();
        let arena = self.arena_storage.as_bytes();
        let tiles = Core1Stack::<B>::arena_tiles(arena);
        let k_ctrl = self.zoh.control_decimation_host;
        let lfo_env_active = self.preamble.is_lfo_env_active();
        let mono_only = !self.preamble.is_stereo_out();
        let dry_g = self.core1.mix_dry_gain;
        let wet_g = self.core1.mix_wet_gain * self.core1.tier_gain_trim_lin[self.tier as usize];
        let bypassed = self.bypassed;
        let trails = self.trails_enabled;
        let freeze_mutes_dry = self.freeze_mutes_dry;
        let width = self.stereo_width;

        match self.tier {
            Tier::Lite => {
                for i in 0..n {
                    let [l, r] = Self::step_frame_const::<1, 1>(
                        &mut self.core1,
                        &mut self.core2,
                        &mut self.control_bus,
                        &mut self.ctrl_sub_counter,
                        &mut self.ctrl_energy_accum,
                        left_io[i],
                        right_io[i],
                        arena,
                        tiles,
                        k_ctrl,
                        lfo_env_active,
                        mono_only,
                        dry_g,
                        wet_g,
                        bypassed,
                        trails,
                        freeze_mutes_dry,
                        width,
                    );
                    left_io[i] = l;
                    right_io[i] = r;
                }
            }
            Tier::Full => {
                for i in 0..n {
                    let [l, r] = Self::step_frame_const::<2, 2>(
                        &mut self.core1,
                        &mut self.core2,
                        &mut self.control_bus,
                        &mut self.ctrl_sub_counter,
                        &mut self.ctrl_energy_accum,
                        left_io[i],
                        right_io[i],
                        arena,
                        tiles,
                        k_ctrl,
                        lfo_env_active,
                        mono_only,
                        dry_g,
                        wet_g,
                        bypassed,
                        trails,
                        freeze_mutes_dry,
                        width,
                    );
                    left_io[i] = l;
                    right_io[i] = r;
                }
            }
            Tier::Ultra => {
                for i in 0..n {
                    let [l, r] = Self::step_frame_const::<3, 2>(
                        &mut self.core1,
                        &mut self.core2,
                        &mut self.control_bus,
                        &mut self.ctrl_sub_counter,
                        &mut self.ctrl_energy_accum,
                        left_io[i],
                        right_io[i],
                        arena,
                        tiles,
                        k_ctrl,
                        lfo_env_active,
                        mono_only,
                        dry_g,
                        wet_g,
                        bypassed,
                        trails,
                        freeze_mutes_dry,
                        width,
                    );
                    left_io[i] = l;
                    right_io[i] = r;
                }
            }
        }
        if n > 0 {
            self.core1
                .finalize_block_telemetry(self.tier.core1_groups());
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
        self.reset_block_telemetry();
        let arena = self.arena_storage.as_bytes();
        let tiles = Core1Stack::<B>::arena_tiles(arena);
        let k_ctrl = self.zoh.control_decimation_host;
        let lfo_env_active = self.preamble.is_lfo_env_active();
        let mono_only = !self.preamble.is_stereo_out();
        let dry_g = self.core1.mix_dry_gain;
        let wet_g = self.core1.mix_wet_gain * self.core1.tier_gain_trim_lin[self.tier as usize];
        let bypassed = self.bypassed;
        let trails = self.trails_enabled;
        let freeze_mutes_dry = self.freeze_mutes_dry;
        let width = self.stereo_width;

        match self.tier {
            Tier::Lite => {
                for i in 0..n {
                    let x = mono_in[i];
                    let [l, r] = Self::step_frame_const::<1, 1>(
                        &mut self.core1,
                        &mut self.core2,
                        &mut self.control_bus,
                        &mut self.ctrl_sub_counter,
                        &mut self.ctrl_energy_accum,
                        x,
                        x,
                        arena,
                        tiles,
                        k_ctrl,
                        lfo_env_active,
                        mono_only,
                        dry_g,
                        wet_g,
                        bypassed,
                        trails,
                        freeze_mutes_dry,
                        width,
                    );
                    left_out[i] = l;
                    right_out[i] = r;
                }
            }
            Tier::Full => {
                for i in 0..n {
                    let x = mono_in[i];
                    let [l, r] = Self::step_frame_const::<2, 2>(
                        &mut self.core1,
                        &mut self.core2,
                        &mut self.control_bus,
                        &mut self.ctrl_sub_counter,
                        &mut self.ctrl_energy_accum,
                        x,
                        x,
                        arena,
                        tiles,
                        k_ctrl,
                        lfo_env_active,
                        mono_only,
                        dry_g,
                        wet_g,
                        bypassed,
                        trails,
                        freeze_mutes_dry,
                        width,
                    );
                    left_out[i] = l;
                    right_out[i] = r;
                }
            }
            Tier::Ultra => {
                for i in 0..n {
                    let x = mono_in[i];
                    let [l, r] = Self::step_frame_const::<3, 2>(
                        &mut self.core1,
                        &mut self.core2,
                        &mut self.control_bus,
                        &mut self.ctrl_sub_counter,
                        &mut self.ctrl_energy_accum,
                        x,
                        x,
                        arena,
                        tiles,
                        k_ctrl,
                        lfo_env_active,
                        mono_only,
                        dry_g,
                        wet_g,
                        bypassed,
                        trails,
                        freeze_mutes_dry,
                        width,
                    );
                    left_out[i] = l;
                    right_out[i] = r;
                }
            }
        }
        if n > 0 {
            self.core1
                .finalize_block_telemetry(self.tier.core1_groups());
        }
    }

    /// **Topology 4 (`Mono -> Mono`, 50% Output Head MAC Savings):**
    /// Evaluates only Channel 0 of the collapsed 16-tap output head (`force_mono_head = true`),
    /// saving 50% of output head MACs on mono pedalboard blocks.
    pub fn process_mono(&mut self, mono_in: &[f32], mono_out: &mut [f32]) {
        let _ftz = FtzDazGuard::new();
        let n = mono_in.len().min(mono_out.len());
        self.reset_block_telemetry();
        let arena = self.arena_storage.as_bytes();
        let tiles = Core1Stack::<B>::arena_tiles(arena);
        let k_ctrl = self.zoh.control_decimation_host;
        let lfo_env_active = self.preamble.is_lfo_env_active();
        let dry_g = self.core1.mix_dry_gain;
        let wet_g = self.core1.mix_wet_gain * self.core1.tier_gain_trim_lin[self.tier as usize];
        let bypassed = self.bypassed;
        let trails = self.trails_enabled;
        let freeze_mutes_dry = self.freeze_mutes_dry;
        let width = self.stereo_width;

        match self.tier {
            Tier::Lite => {
                for i in 0..n {
                    let x = mono_in[i];
                    let [l, _] = Self::step_frame_const::<1, 1>(
                        &mut self.core1,
                        &mut self.core2,
                        &mut self.control_bus,
                        &mut self.ctrl_sub_counter,
                        &mut self.ctrl_energy_accum,
                        x,
                        x,
                        arena,
                        tiles,
                        k_ctrl,
                        lfo_env_active,
                        true,
                        dry_g,
                        wet_g,
                        bypassed,
                        trails,
                        freeze_mutes_dry,
                        width,
                    );
                    mono_out[i] = l;
                }
            }
            Tier::Full => {
                for i in 0..n {
                    let x = mono_in[i];
                    let [l, _] = Self::step_frame_const::<2, 2>(
                        &mut self.core1,
                        &mut self.core2,
                        &mut self.control_bus,
                        &mut self.ctrl_sub_counter,
                        &mut self.ctrl_energy_accum,
                        x,
                        x,
                        arena,
                        tiles,
                        k_ctrl,
                        lfo_env_active,
                        true,
                        dry_g,
                        wet_g,
                        bypassed,
                        trails,
                        freeze_mutes_dry,
                        width,
                    );
                    mono_out[i] = l;
                }
            }
            Tier::Ultra => {
                for i in 0..n {
                    let x = mono_in[i];
                    let [l, _] = Self::step_frame_const::<3, 2>(
                        &mut self.core1,
                        &mut self.core2,
                        &mut self.control_bus,
                        &mut self.ctrl_sub_counter,
                        &mut self.ctrl_energy_accum,
                        x,
                        x,
                        arena,
                        tiles,
                        k_ctrl,
                        lfo_env_active,
                        true,
                        dry_g,
                        wet_g,
                        bypassed,
                        trails,
                        freeze_mutes_dry,
                        width,
                    );
                    mono_out[i] = l;
                }
            }
        }
        if n > 0 {
            self.core1
                .finalize_block_telemetry(self.tier.core1_groups());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::{ARENA_OFFSET_ULTRA_END, build_dnam_container};
    use crate::math::ScalarBackend;
    use crate::schema::{
        BlendMode, ControlCalibration, ControlKind, ControlSchemaDef, ControlUiHints, PresetDef,
    };
    use alloc::collections::BTreeMap;
    use alloc::string::String;

    #[test]
    fn test_dnam_engine_presets_and_reload_errors() {
        let mut header = DnamJsonHeader::default_reference(false);
        header.controls_schema = alloc::vec![
            ControlSchemaDef {
                id: String::from("gain"),
                name: String::from("Gain"),
                kind: ControlKind::Range,
                unit: None,
                lobatto_degree: Some(2),
                anchor_nodes_norm: None,
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
            },
            ControlSchemaDef {
                id: String::from("channel"),
                name: String::from("Channel"),
                kind: ControlKind::Switch,
                unit: None,
                lobatto_degree: None,
                anchor_nodes_norm: None,
                declared_range: None,
                safe_clamped_range: None,
                default: serde_json::json!("Clean"),
                conditional_min: BTreeMap::new(),
                conditional_max: BTreeMap::new(),
                declared_options: Some(alloc::vec![String::from("Clean"), String::from("Lead")]),
                active_options: Some(alloc::vec![String::from("Clean"), String::from("Lead")]),
                clamp_warning: None,
                calibration: None,
                ui_hints: ControlUiHints::default(),
                user_metadata: BTreeMap::new(),
            },
        ];
        header.presets = alloc::vec![PresetDef {
            id: String::from("lead_boost"),
            name: String::from("Lead Boost"),
            description: None,
            controls: BTreeMap::from([
                (String::from("gain"), serde_json::json!(8.0)),
                (String::from("channel"), serde_json::json!("Lead")),
            ]),
            user_metadata: BTreeMap::new(),
        }];

        let arena = alloc::vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
        let container = build_dnam_container(&mut header, &arena, &arena, false).unwrap();

        let mut engine =
            DnamEngine::<ScalarBackend>::from_bytes(&container, Tier::Ultra, 48_000).unwrap();
        assert_eq!(engine.tier(), Tier::Ultra);
        assert_eq!(engine.sample_rate_hz(), 48_000);
        assert_eq!(engine.zoh().sample_rate_host_hz, 48_000);
        assert!(engine.header().is_some());
        assert_eq!(engine.preamble().num_controls, 2);
        assert_eq!(engine.verification_status(), &VerificationStatus::Unsigned);

        engine.apply_preset("lead_boost").unwrap();
        assert!(engine.apply_preset("missing").is_err());

        // reload_model_in_place clears header -> apply_preset returns error
        let mut cap_engine = DnamEngine::<ScalarBackend>::with_max_capacity(8);
        assert!(cap_engine.apply_preset("lead_boost").is_err());
        cap_engine
            .reload_model_in_place(&container, Tier::Full, 48_000)
            .unwrap();
        assert_eq!(cap_engine.tier(), Tier::Full);

        // Reload error branches: short buffer, bad header CRC, encrypted flag
        assert!(
            cap_engine
                .reload_model_in_place(&container[..4096], Tier::Lite, 48_000)
                .is_err()
        );
        let mut bad_crc = container.clone();
        bad_crc[60] ^= 0xFF;
        assert!(
            cap_engine
                .reload_model_in_place(&bad_crc, Tier::Lite, 48_000)
                .is_err()
        );
        let mut enc_flag = container.clone();
        enc_flag[6] |= (crate::container::FLAG_ENCRYPTED_ARENA & 0xFF) as u8;
        enc_flag[7] |= ((crate::container::FLAG_ENCRYPTED_ARENA >> 8) & 0xFF) as u8;
        assert!(
            cap_engine
                .reload_model_in_place(&enc_flag, Tier::Lite, 48_000)
                .is_err()
        );

        let owned_eng =
            DnamEngine::<ScalarBackend>::from_bytes_owned(&container, Tier::Lite, 48_000).unwrap();
        assert_eq!(owned_eng.tier(), Tier::Lite);
    }

    #[test]
    fn test_dnam_engine_leveling_blend_trails_tempo_and_stereo_mode() {
        let mut header = DnamJsonHeader::default_reference(true);
        header.io_config.capabilities.blend_mode = BlendMode::HostParallel;
        header.io_config.capabilities.wet_polarity_inverted = true;
        header.io_config.capabilities.supports_trails = true;
        header.io_config.capabilities.tempo_sync_capable = true;
        header.io_config.leveling.auto_unity_enabled = true;
        header.io_config.leveling.calibrated_trim_db = 6.0206; // ~2.0x linear gain

        let mut cond_anchors = BTreeMap::new();
        cond_anchors.insert(String::from("range=0.01-0.1s"), [10.0, 55.0, 100.0]);
        cond_anchors.insert(String::from("range=0.1-1s"), [100.0, 550.0, 1000.0]);
        cond_anchors.insert(String::from("range=1-2s"), [1000.0, 1500.0, 2000.0]);

        header.controls_schema = alloc::vec![
            ControlSchemaDef {
                id: String::from("time"),
                name: String::from("Time"),
                kind: ControlKind::Range,
                unit: Some(String::from("ms")),
                lobatto_degree: Some(2),
                anchor_nodes_norm: None,
                declared_range: Some([0.0, 10.0]),
                safe_clamped_range: Some([0.0, 10.0]),
                default: serde_json::json!(5.0),
                conditional_min: BTreeMap::new(),
                conditional_max: BTreeMap::new(),
                declared_options: None,
                active_options: None,
                clamp_warning: None,
                calibration: Some(ControlCalibration {
                    role: ControlCalibrationRole::DelayTimeMs,
                    lobatto_physical_anchors: None,
                    conditional_physical_anchors: cond_anchors,
                    tempo_sync_default_subdivision: Some(TempoSubdivision::EighthDotted),
                }),
                ui_hints: ControlUiHints::default(),
                user_metadata: BTreeMap::new(),
            },
            ControlSchemaDef {
                id: String::from("range"),
                name: String::from("Range"),
                kind: ControlKind::Switch,
                unit: None,
                lobatto_degree: None,
                anchor_nodes_norm: None,
                declared_range: None,
                safe_clamped_range: None,
                default: serde_json::json!("0.1-1s"),
                conditional_min: BTreeMap::new(),
                conditional_max: BTreeMap::new(),
                declared_options: Some(alloc::vec![
                    String::from("0.01-0.1s"),
                    String::from("0.1-1s"),
                    String::from("1-2s"),
                ]),
                active_options: Some(alloc::vec![
                    String::from("0.01-0.1s"),
                    String::from("0.1-1s"),
                    String::from("1-2s"),
                ]),
                clamp_warning: None,
                calibration: None,
                ui_hints: ControlUiHints::default(),
                user_metadata: BTreeMap::new(),
            },
            ControlSchemaDef {
                id: String::from("mode"),
                name: String::from("Mode"),
                kind: ControlKind::Switch,
                unit: None,
                lobatto_degree: None,
                anchor_nodes_norm: None,
                declared_range: None,
                safe_clamped_range: None,
                default: serde_json::json!("Stereo"),
                conditional_min: BTreeMap::new(),
                conditional_max: BTreeMap::new(),
                declared_options: Some(alloc::vec![String::from("Mono"), String::from("Stereo")]),
                active_options: Some(alloc::vec![String::from("Mono"), String::from("Stereo")]),
                clamp_warning: None,
                calibration: Some(ControlCalibration {
                    role: ControlCalibrationRole::StereoMode,
                    lobatto_physical_anchors: None,
                    conditional_physical_anchors: BTreeMap::new(),
                    tempo_sync_default_subdivision: None,
                }),
                ui_hints: ControlUiHints::default(),
                user_metadata: BTreeMap::new(),
            },
        ];

        let arena = alloc::vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
        let container = build_dnam_container(&mut header, &arena, &arena, false).unwrap();
        let mut engine =
            DnamEngine::<ScalarBackend>::from_bytes(&container, Tier::Lite, 48_000).unwrap();

        // 1. Test configure_host_mix_and_leveling at 50% blend with wet_polarity_inverted=true and +6.0206 dB trim
        engine.configure_host_mix_and_leveling(0.5, 0.0, true);
        assert!((engine.core1.mix_dry_gain - 0.5).abs() < 1e-4);
        assert!((engine.core1.mix_wet_gain - (-1.0)).abs() < 1e-3);

        // 2. Test StereoMode switch ("Mono" = index 0 -> stereo_width = 0.0; "Stereo" = index 1 -> stereo_width = 1.0)
        assert!((engine.stereo_width() - 1.0).abs() < 1e-6);
        engine.set_control_immediate(2, 0.0);
        assert!(engine.stereo_width().abs() < 1e-6);
        engine.set_control_immediate(2, 1.0);
        assert!((engine.stereo_width() - 1.0).abs() < 1e-6);

        // 3. Test sync_tempo: 120 BPM + 1/2. = 1500 ms -> range="1-2s" (index 2) and time=5.0
        let updated = engine.sync_tempo(120.0, Some(TempoSubdivision::HalfDotted), false);
        assert_eq!(updated, 2);
        let snap = engine.telemetry_snapshot();
        assert!((snap.clamped_controls[0] - 5.0).abs() < 1e-3);
        assert!((snap.clamped_controls[1] - 2.0).abs() < 1e-3);

        // 4. Test hard bypass and trails bypass
        engine.set_bypass(true, false);
        assert_eq!(engine.bypass_state(), (true, false));
        let mut l_out = [0.0_f32; 4];
        let mut r_out = [0.0_f32; 4];
        engine.process_block(
            &[0.25, -0.5, 0.75, -0.25],
            &[-0.25, 0.5, -0.75, 0.25],
            &mut l_out,
            &mut r_out,
        );
        assert_eq!(l_out, [0.25, -0.5, 0.75, -0.25]);
        assert_eq!(r_out, [-0.25, 0.5, -0.75, 0.25]);

        engine.set_bypass(true, true);
        assert_eq!(engine.bypass_state(), (true, true));
        engine.process_block(
            &[0.25, -0.5, 0.75, -0.25],
            &[-0.25, 0.5, -0.75, 0.25],
            &mut l_out,
            &mut r_out,
        );
        assert_eq!(l_out, [0.25, -0.5, 0.75, -0.25]);
    }

    #[test]
    fn test_dnam_engine_wt_ufdn_freeze_grain_calibration_and_3tier_pre_head_routing() {
        let mut header = DnamJsonHeader::default_reference(true);
        header.io_config.capabilities.supports_freeze = true;
        header.io_config.capabilities.freeze_mutes_dry = true;
        header.controls_schema = alloc::vec![
            ControlSchemaDef {
                id: String::from("freeze"),
                name: String::from("Freeze"),
                kind: ControlKind::Switch,
                unit: None,
                lobatto_degree: None,
                anchor_nodes_norm: None,
                declared_range: None,
                safe_clamped_range: None,
                default: serde_json::json!("Off"),
                conditional_min: BTreeMap::new(),
                conditional_max: BTreeMap::new(),
                declared_options: Some(alloc::vec![String::from("Off"), String::from("Freeze")]),
                active_options: Some(alloc::vec![String::from("Off"), String::from("Freeze")]),
                clamp_warning: None,
                calibration: Some(ControlCalibration {
                    role: ControlCalibrationRole::FreezeLatch,
                    lobatto_physical_anchors: None,
                    conditional_physical_anchors: BTreeMap::new(),
                    tempo_sync_default_subdivision: None,
                }),
                ui_hints: ControlUiHints::default(),
                user_metadata: BTreeMap::new(),
            },
            ControlSchemaDef {
                id: String::from("grain_size"),
                name: String::from("Grain Size"),
                kind: ControlKind::Range,
                unit: Some(String::from("ms")),
                lobatto_degree: Some(2),
                anchor_nodes_norm: None,
                declared_range: Some([0.0, 10.0]),
                safe_clamped_range: Some([0.0, 10.0]),
                default: serde_json::json!(5.0),
                conditional_min: BTreeMap::new(),
                conditional_max: BTreeMap::new(),
                declared_options: None,
                active_options: None,
                clamp_warning: None,
                calibration: Some(ControlCalibration {
                    role: ControlCalibrationRole::GrainSizeMs,
                    lobatto_physical_anchors: Some([20.0, 80.0, 200.0]),
                    conditional_physical_anchors: BTreeMap::new(),
                    tempo_sync_default_subdivision: Some(TempoSubdivision::Sixteenth),
                }),
                ui_hints: ControlUiHints::default(),
                user_metadata: BTreeMap::new(),
            },
            ControlSchemaDef {
                id: String::from("pitch"),
                name: String::from("Pitch"),
                kind: ControlKind::Range,
                unit: Some(String::from("st")),
                lobatto_degree: Some(2),
                anchor_nodes_norm: None,
                declared_range: Some([0.0, 10.0]),
                safe_clamped_range: Some([0.0, 10.0]),
                default: serde_json::json!(5.0),
                conditional_min: BTreeMap::new(),
                conditional_max: BTreeMap::new(),
                declared_options: None,
                active_options: None,
                clamp_warning: None,
                calibration: Some(ControlCalibration {
                    role: ControlCalibrationRole::GrainPitchSemitones,
                    lobatto_physical_anchors: Some([-12.0, 0.0, 12.0]),
                    conditional_physical_anchors: BTreeMap::new(),
                    tempo_sync_default_subdivision: None,
                }),
                ui_hints: ControlUiHints::default(),
                user_metadata: BTreeMap::new(),
            },
            ControlSchemaDef {
                id: String::from("spray"),
                name: String::from("Spray"),
                kind: ControlKind::Range,
                unit: None,
                lobatto_degree: Some(2),
                anchor_nodes_norm: None,
                declared_range: Some([0.0, 10.0]),
                safe_clamped_range: Some([0.0, 10.0]),
                default: serde_json::json!(0.0),
                conditional_min: BTreeMap::new(),
                conditional_max: BTreeMap::new(),
                declared_options: None,
                active_options: None,
                clamp_warning: None,
                calibration: Some(ControlCalibration {
                    role: ControlCalibrationRole::GrainSpray,
                    lobatto_physical_anchors: Some([0.0, 0.4, 1.0]),
                    conditional_physical_anchors: BTreeMap::new(),
                    tempo_sync_default_subdivision: None,
                }),
                ui_hints: ControlUiHints::default(),
                user_metadata: BTreeMap::new(),
            },
        ];

        let mut arena = alloc::vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
        // Populate head_tap_inject_gain = 0.75 at 6784 + 252 and non-zero W_ctrl_core2 at 9216
        arena[6784 + 252..6784 + 256].copy_from_slice(&0.75_f32.to_le_bytes());
        arena[9216..9220].copy_from_slice(&0.05_f32.to_le_bytes());

        let container = build_dnam_container(&mut header, &arena, &arena, false).unwrap();
        let mut engine =
            DnamEngine::<ScalarBackend>::from_bytes(&container, Tier::Ultra, 48_000).unwrap();

        assert!(!engine.is_frozen());
        engine.set_control_immediate(1, 8.0); // GrainSizeMs
        engine.set_control_immediate(2, 10.0); // GrainPitchSemitones (+12 st)
        engine.set_control_immediate(3, 5.0); // GrainSpray (0.4)
        engine.set_control_immediate(0, 1.0); // FreezeLatch -> "Freeze"
        assert!(engine.is_frozen());

        // Tempo sync on GrainSizeMs (120 BPM Sixteenth = 125 ms)
        let synced = engine.sync_tempo(120.0, None, false);
        assert_eq!(synced, 1);

        // Process across all 3 tiers (Ultra, Full, Lite) with 3-tier Pre-Head Kronecker routing
        let l_in = [0.2_f32; 32];
        let r_in = [-0.2_f32; 32];
        let mut l_out = [0.0_f32; 32];
        let mut r_out = [0.0_f32; 32];
        for tier in [Tier::Ultra, Tier::Full, Tier::Lite] {
            engine.set_tier(tier);
            engine.process_block(&l_in, &r_in, &mut l_out, &mut r_out);
            assert!(l_out.iter().all(|x| x.is_finite()));
            assert!(r_out.iter().all(|x| x.is_finite()));
        }

        engine.set_freeze_amount(0.25);
        assert!(!engine.is_frozen());
        engine.set_freeze(true);
        assert!(engine.is_frozen());
    }
}
