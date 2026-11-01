//! Primary generic real-time DNAM ("Tri-Physics") inference engine (`DnamEngine<'a, B>`).

use crate::container::{
    ARENA_LITE_END_CL, ARENA_OFFSET_LITE_END, DnamContainerView, DnamPreamble, ZohRescaledParams,
};
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

/// Exact linear-phase Plugin Delay Compensation (`PDC`) in host samples when `offline_bounce`
/// ($4\times$ 2-stage symmetric half-band FIR polyphase oversampling) is active.
pub const OFFLINE_BOUNCE_PDC_SAMPLES: usize = 12;

/// Stage 1 ($1\times \leftrightarrow 2\times$) 19-tap symmetric half-band polyphase odd-branch taps ($K_1 = 5$).
/// Exact dyadic rationals with denominator $2^{16} = 65536$, summing to $0.50000000$.
const HB_STAGE1_TAPS: [f32; 5] = [
    35.0 / 65536.0,
    -405.0 / 65536.0,
    2268.0 / 65536.0,
    -8820.0 / 65536.0,
    39690.0 / 65536.0,
];

/// Stage 2 ($2\times \leftrightarrow 4\times$) 11-tap symmetric half-band polyphase odd-branch taps ($K_2 = 3$).
/// Exact dyadic rationals with denominator $2^8 = 256$, summing to $0.50000000$.
const HB_STAGE2_TAPS: [f32; 3] = [3.0 / 256.0, -25.0 / 256.0, 150.0 / 256.0];

/// Zero-allocation 2-stage $4\times$ symmetric half-band linear-phase polyphase oversampler state.
///
/// Round-trip group delay is `(2 * K1 - 1) + K2 = 9 + 3 = 12` host samples ([`OFFLINE_BOUNCE_PDC_SAMPLES`]).
#[derive(Clone, Debug)]
pub struct Oversampler4xState {
    up1_hist: [[f32; 2]; 10],
    up2_hist: [[f32; 2]; 6],
    dn2_even_hist: [[f32; 2]; 6],
    dn2_odd_delay: [[f32; 2]; 4],
    stage2_align_z1: [f32; 2],
    dn1_even_hist: [[f32; 2]; 10],
    dn1_odd_delay: [[f32; 2]; 6],
}

impl Default for Oversampler4xState {
    fn default() -> Self {
        Self::new()
    }
}

impl Oversampler4xState {
    /// Creates a zeroed [`Oversampler4xState`].
    #[inline]
    pub const fn new() -> Self {
        Self {
            up1_hist: [[0.0; 2]; 10],
            up2_hist: [[0.0; 2]; 6],
            dn2_even_hist: [[0.0; 2]; 6],
            dn2_odd_delay: [[0.0; 2]; 4],
            stage2_align_z1: [0.0; 2],
            dn1_even_hist: [[0.0; 2]; 10],
            dn1_odd_delay: [[0.0; 2]; 6],
        }
    }

    /// Resets all internal FIR delay lines to zero (or DC `[dc_l, dc_r]`).
    #[inline]
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// Upsamples one $1\times$ stereo frame `[in_l, in_r]` into four chronological $4\times$ stereo frames.
    #[inline(always)]
    pub fn upsample_4x(&mut self, in_l: f32, in_r: f32) -> [[f32; 2]; 4] {
        for k in (1..10).rev() {
            self.up1_hist[k] = self.up1_hist[k - 1];
        }
        self.up1_hist[0] = [in_l, in_r];

        let mut s2x_0 = [0.0_f32; 2];
        for (k, &h) in HB_STAGE1_TAPS.iter().enumerate() {
            let a = self.up1_hist[k];
            let b = self.up1_hist[9 - k];
            s2x_0[0] += h * (a[0] + b[0]);
            s2x_0[1] += h * (a[1] + b[1]);
        }
        let s2x_1 = self.up1_hist[4];
        let s2x = [s2x_0, s2x_1];

        let mut out_4x = [[0.0_f32; 2]; 4];
        for (m, &u2) in s2x.iter().enumerate() {
            for k in (1..6).rev() {
                self.up2_hist[k] = self.up2_hist[k - 1];
            }
            self.up2_hist[0] = u2;

            let mut s4x_0 = [0.0_f32; 2];
            for (k, &h) in HB_STAGE2_TAPS.iter().enumerate() {
                let a = self.up2_hist[k];
                let b = self.up2_hist[5 - k];
                s4x_0[0] += h * (a[0] + b[0]);
                s4x_0[1] += h * (a[1] + b[1]);
            }
            let s4x_1 = self.up2_hist[2];
            out_4x[m * 2] = s4x_0;
            out_4x[m * 2 + 1] = s4x_1;
        }
        out_4x
    }

    /// Downsamples four chronological $4\times$ stereo frames `w4x` into one $1\times$ stereo frame.
    #[inline(always)]
    pub fn downsample_4x(&mut self, w4x: [[f32; 2]; 4]) -> [f32; 2] {
        let mut v2x = [[0.0_f32; 2]; 2];
        for m in 0..2 {
            let w0 = w4x[m * 2];
            let w1 = w4x[m * 2 + 1];

            for k in (1..6).rev() {
                self.dn2_even_hist[k] = self.dn2_even_hist[k - 1];
            }
            self.dn2_even_hist[0] = w0;

            for k in (1..4).rev() {
                self.dn2_odd_delay[k] = self.dn2_odd_delay[k - 1];
            }
            self.dn2_odd_delay[0] = w1;

            let mut filt2 = [0.0_f32; 2];
            for (k, &h) in HB_STAGE2_TAPS.iter().enumerate() {
                let a = self.dn2_even_hist[k];
                let b = self.dn2_even_hist[5 - k];
                filt2[0] += h * (a[0] + b[0]);
                filt2[1] += h * (a[1] + b[1]);
            }
            let odd2 = self.dn2_odd_delay[3];
            let raw2 = [0.5 * (filt2[0] + odd2[0]), 0.5 * (filt2[1] + odd2[1])];
            v2x[m] = self.stage2_align_z1;
            self.stage2_align_z1 = raw2;
        }

        for k in (1..10).rev() {
            self.dn1_even_hist[k] = self.dn1_even_hist[k - 1];
        }
        self.dn1_even_hist[0] = v2x[0];

        for k in (1..6).rev() {
            self.dn1_odd_delay[k] = self.dn1_odd_delay[k - 1];
        }
        self.dn1_odd_delay[0] = v2x[1];

        let mut filt1 = [0.0_f32; 2];
        for (k, &h) in HB_STAGE1_TAPS.iter().enumerate() {
            let a = self.dn1_even_hist[k];
            let b = self.dn1_even_hist[9 - k];
            filt1[0] += h * (a[0] + b[0]);
            filt1[1] += h * (a[1] + b[1]);
        }
        let odd1 = self.dn1_odd_delay[5];
        [0.5 * (filt1[0] + odd1[0]), 0.5 * (filt1[1] + odd1[1])]
    }
}

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
    max_supported_tier: Tier,
    offline_bounce: bool,
    maximize_tier_on_bounce: bool,
    oversampler: Oversampler4xState,
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
    /// Creates an unconfigured zero-allocation [`DnamEngine`] shell (`sret` in-place friendly).
    pub(crate) fn new_empty(max_supported_tier: Tier) -> Self {
        let zoh = ZohRescaledParams {
            sample_rate_host_hz: 48_000,
            rho_sr: 1.0,
            control_decimation_host: 16,
            rho_ctrl: 1.0,
            r_ac_host: 0.9993,
            r_dc_host: 0.995,
            tau_ref_samples_host: 64.0,
            switch_xfade_samples_host: 128,
        };
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
            arena_lite_end_cl: ARENA_LITE_END_CL,
            arena_full_end_cl: ARENA_LITE_END_CL,
            arena_pro_end_cl: ARENA_LITE_END_CL,
            arena_ultra_end_cl: ARENA_LITE_END_CL,
            arena_crc32c_lite: 0,
            arena_crc32c_full: 0,
            arena_crc32c_pro: 0,
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
            arena_storage: ArenaStorage::from_slice_or_copy(&[]),
            tier: Tier::Lite,
            max_available_tier: Tier::Lite,
            max_supported_tier,
            offline_bounce: false,
            maximize_tier_on_bounce: true,
            oversampler: Oversampler4xState::new(),
            zoh,
            ctrl_sub_counter: 0,
            ctrl_energy_accum: 0.0,
            bypassed: false,
            trails_enabled: false,
            freeze_mutes_dry: false,
            stereo_width: 1.0,
            control_bus: ControlBus::new(),
            core1: Core1Stack::new(),
            core2: ModulatedUfdn::new_inactive(),
        }
    }

    /// Loads a `.dnam` or `.dnam-ckpt` container from `bytes` using the default [`CryptoProvider`],
    /// borrowing `Arena 0..3` with **zero copies** when `bytes` is 64-byte aligned (`mmap`) and unencrypted.
    pub fn from_bytes(bytes: &'a [u8], tier: Tier, sample_rate_hz: u32) -> Result<Self, DnamError> {
        let crypto = DefaultCryptoProvider::default();
        Self::from_bytes_with_crypto(bytes, tier, sample_rate_hz, &crypto)
    }

    /// Loads a `.dnam` or `.dnam-ckpt` container capped at `max_supported_tier` (loading, verifying,
    /// and allocating only the prefix up to `max_supported_tier`).
    pub fn from_bytes_with_max_tier(
        bytes: &'a [u8],
        tier: Tier,
        max_supported_tier: Tier,
        sample_rate_hz: u32,
    ) -> Result<Self, DnamError> {
        let crypto = DefaultCryptoProvider::default();
        let view = DnamContainerView::from_bytes_with_max_tier(bytes, &crypto, max_supported_tier)?;
        let mut engine = Self::new_empty(max_supported_tier);
        engine.init_from_view(view, tier, max_supported_tier, sample_rate_hz)?;
        Ok(engine)
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
    /// capped at `max_supported_tier`.
    pub fn from_bytes_owned_with_max_tier(
        bytes: &[u8],
        tier: Tier,
        max_supported_tier: Tier,
        sample_rate_hz: u32,
    ) -> Result<DnamEngine<'static, B>, DnamError> {
        let crypto = DefaultCryptoProvider::default();
        let view = DnamContainerView::from_bytes_with_max_tier(bytes, &crypto, max_supported_tier)?;
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
        let mut engine = DnamEngine::<'static, B>::new_empty(max_supported_tier);
        engine.init_from_view(owned_view, tier, max_supported_tier, sample_rate_hz)?;
        Ok(engine)
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
        let mut engine = DnamEngine::<'static, B>::new_empty(Tier::Ultra);
        engine.init_from_view(owned_view, tier, Tier::Ultra, sample_rate_hz)?;
        Ok(engine)
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
        let mut engine = Self::new_empty(Tier::Ultra);
        engine.init_from_view(view, tier, Tier::Ultra, sample_rate_hz)?;
        Ok(engine)
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
        let mut engine = Self::new_empty(Tier::Ultra);
        engine.init_from_view_with_core2(view, core2, tier, Tier::Ultra, sample_rate_hz)?;
        Ok(engine)
    }

    /// Pre-allocates a [`DnamEngine`] with maximum `Arena 0..3` (`64,000 B`) and Core 2 `fp16`
    /// delay capacity (`max_ring_log2`, e.g. `15` for 32,768 samples) so subsequent calls to
    /// [`reload_model_in_place`](Self::reload_model_in_place) execute with **0 heap allocations**.
    pub fn with_max_capacity(max_ring_log2: u8) -> Self {
        Self::with_max_tier(max_ring_log2, Tier::Ultra)
    }

    /// Pre-allocates a [`DnamEngine`] capped at `max_supported_tier` (`10,752 B` for `Lite`,
    /// `22,336 B` for `Full`, `36,224 B` for `Pro`, or `64,000 B` for `Ultra`) and corresponding
    /// Core 2 `fp16` delay line groups (`1`, `2`, or `4` groups).
    pub fn with_max_tier(max_ring_log2: u8, max_supported_tier: Tier) -> Self {
        let mut engine = Self::new_empty(max_supported_tier);
        engine.arena_storage = ArenaStorage::with_max_tier(max_supported_tier);
        engine.core2 = ModulatedUfdn::with_max_capacity_for_groups(
            max_ring_log2,
            max_supported_tier.fdn_groups(),
        );
        engine
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
        let copy_len = raw_arena
            .len()
            .min(self.max_supported_tier.expected_arena_bytes());
        DnamContainerView::verify_arena_prefix_crcs(&preamble, &raw_arena[..copy_len])?;

        if !self.arena_storage.reload_in_place(&raw_arena[..copy_len]) {
            self.arena_storage = ArenaStorage::from_slice_or_copy(&raw_arena[..copy_len]);
        }

        self.preamble = preamble;
        self.header = None;
        self.freeze_mutes_dry = false;
        self.verification_status = VerificationStatus::Unsigned;
        self.protection_profile = DnamProtectionProfile::Open;
        self.max_available_tier = Self::detect_max_tier(&preamble, copy_len);
        self.tier = tier
            .min(self.max_available_tier)
            .min(self.max_supported_tier);

        self.reconfigure_for_sample_rate_with_mode(sample_rate_hz, true)?;
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
        let copy_len = raw_arena
            .len()
            .min(self.max_supported_tier.expected_arena_bytes());
        DnamContainerView::verify_arena_prefix_crcs(&preamble, &raw_arena[..copy_len])?;

        if !self.arena_storage.reload_in_place(&raw_arena[..copy_len]) {
            self.arena_storage = ArenaStorage::from_slice_owned(&raw_arena[..copy_len]);
        }

        self.preamble = preamble;
        self.header = None;
        self.freeze_mutes_dry = false;
        self.verification_status = VerificationStatus::Unsigned;
        self.protection_profile = DnamProtectionProfile::Open;
        self.max_available_tier = Self::detect_max_tier(&preamble, copy_len);
        self.tier = tier
            .min(self.max_available_tier)
            .min(self.max_supported_tier);

        self.reconfigure_for_sample_rate_with_mode(sample_rate_hz, true)?;
        Ok(())
    }

    pub(crate) fn init_from_view(
        &mut self,
        view: DnamContainerView<'a>,
        tier: Tier,
        max_supported_tier: Tier,
        sample_rate_hz: u32,
    ) -> Result<(), DnamError> {
        let ring_log2 =
            if view.preamble.is_core2_active() && view.preamble.ring_buffer_pow2_log2 >= 8 {
                (view.preamble.ring_buffer_pow2_log2 + 2).min(18)
            } else {
                view.preamble.ring_buffer_pow2_log2
            };
        let core2 =
            ModulatedUfdn::with_max_capacity_for_groups(ring_log2, max_supported_tier.fdn_groups());
        self.init_from_view_with_core2(view, core2, tier, max_supported_tier, sample_rate_hz)
    }

    fn init_from_view_with_core2(
        &mut self,
        view: DnamContainerView<'a>,
        core2: ModulatedUfdn<'a, B>,
        tier: Tier,
        max_supported_tier: Tier,
        sample_rate_hz: u32,
    ) -> Result<(), DnamError> {
        let arena_len = view.arena_storage.as_bytes().len();
        let max_available_tier = Self::detect_max_tier(&view.preamble, arena_len);
        let active_tier = tier.min(max_available_tier).min(max_supported_tier);

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
        self.preamble = view.preamble;
        self.header = Some(view.header);
        self.verification_status = view.verification_status;
        self.protection_profile = view.protection_profile;
        self.arena_storage = view.arena_storage;
        self.tier = active_tier;
        self.max_available_tier = max_available_tier;
        self.max_supported_tier = max_supported_tier;
        self.offline_bounce = false;
        self.maximize_tier_on_bounce = true;
        self.oversampler.reset();
        self.zoh = zoh;
        self.ctrl_sub_counter = 0;
        self.ctrl_energy_accum = 0.0;
        self.bypassed = false;
        self.trails_enabled = false;
        self.freeze_mutes_dry = freeze_mutes_dry;
        self.stereo_width = 1.0;
        self.core2 = core2;

        self.reconfigure_for_sample_rate_with_mode(sample_rate_hz, true)?;
        Ok(())
    }

    fn detect_max_tier(preamble: &DnamPreamble, arena_len: usize) -> Tier {
        if (preamble.flags & crate::container::FLAG_HAS_ULTRA_DELTA) != 0
            && preamble.arena_ultra_end_cl > 0
            && arena_len >= (preamble.arena_offset_ultra_end() as usize)
        {
            Tier::Ultra
        } else if (preamble.flags & crate::container::FLAG_HAS_PRO_DELTA) != 0
            && preamble.arena_pro_end_cl > 0
            && arena_len >= (preamble.arena_offset_pro_end() as usize)
        {
            Tier::Pro
        } else if (preamble.flags & crate::container::FLAG_HAS_FULL_DELTA) != 0
            && preamble.arena_full_end_cl > 0
            && arena_len >= (preamble.arena_offset_full_end() as usize)
        {
            Tier::Full
        } else {
            Tier::Lite
        }
    }

    fn reconfigure_for_sample_rate(&mut self, sample_rate_hz: u32) -> Result<(), DnamError> {
        self.reconfigure_for_sample_rate_with_mode(sample_rate_hz, false)
    }

    fn reconfigure_for_sample_rate_with_mode(
        &mut self,
        sample_rate_hz: u32,
        reset_controls: bool,
    ) -> Result<(), DnamError> {
        let arena_len = self.arena_storage.as_bytes().len();
        let r_ac_tr = if arena_len >= 128 {
            let arena = self.arena_storage.as_bytes();
            let v = f32::from_le_bytes([arena[124], arena[125], arena[126], arena[127]]);
            if v > 0.9 && v < 1.0 { v } else { 0.9993 }
        } else {
            0.9993
        };

        let oversample_factor = if self.offline_bounce { 4 } else { 1 };
        self.zoh = ZohRescaledParams::compute_oversampled(
            self.preamble.sample_rate_train_hz,
            sample_rate_hz,
            oversample_factor,
            self.preamble.control_decimation.max(8),
            r_ac_tr,
            0.995,
            64.0,
            128,
        )?;

        let prev_mix_dry = self.core1.mix_dry_gain;
        let prev_mix_wet = self.core1.mix_wet_gain;

        if reset_controls {
            let arena = self.arena_storage.as_bytes();
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
            self.control_bus.refresh_static_state(arena);
        } else {
            let arena = self.arena_storage.as_bytes();
            self.control_bus
                .update_sample_rate_params(arena, self.zoh.rho_ctrl);
            self.control_bus.reset_dynamic_state();
            self.control_bus.refresh_static_state(arena);
        }

        let eff_tier = self.effective_tier();
        let eff_ring_log2 = if self.offline_bounce && self.preamble.ring_buffer_pow2_log2 >= 8 {
            (self.preamble.ring_buffer_pow2_log2 + 2).min(18)
        } else {
            self.preamble.ring_buffer_pow2_log2
        };
        {
            let arena = self.arena_storage.as_bytes();
            self.core1.load_from_arena(arena, &self.zoh);
            if !reset_controls {
                self.core1.mix_dry_gain = prev_mix_dry;
                self.core1.mix_wet_gain = prev_mix_wet;
            }
            self.core1.collapse_stereo_head_3khz_with_ultra(
                arena,
                &self.control_bus.z_head,
                &self.control_bus.z_head_ultra,
                eff_tier.core1_groups(),
            );
            self.core2.load_from_arena(
                arena,
                self.preamble.is_core2_active(),
                eff_ring_log2,
                &self.zoh,
            );
        }
        let num_ctrls = self.control_bus.num_controls;
        for ctrl_idx in 0..num_ctrls {
            let raw_val = self.control_bus.raw_values[ctrl_idx];
            self.refresh_calibrated_control_effects(ctrl_idx, raw_val);
        }
        if self.core2.active {
            self.core2.apply_control_bus_3khz(
                &self.control_bus.core2_gain_targets,
                &self.control_bus.core2_warp_mod,
                &self.control_bus.core2_grain_len_scale,
                &self.control_bus.core2_state_mod,
                self.control_bus.env_states[3],
            );
            self.control_bus.core2_mod_dirty = false;
        }
        self.oversampler.reset();
        self.ctrl_sub_counter = 0;
        self.ctrl_energy_accum = 0.0;
        Ok(())
    }

    /// Returns the requested [`Tier`] (clamped to `min(max_available_tier, max_supported_tier)`).
    #[inline]
    pub const fn tier(&self) -> Tier {
        self.tier
    }

    /// Returns the highest tier present in the loaded `.dnam` container.
    #[inline]
    pub const fn max_available_tier(&self) -> Tier {
        self.max_available_tier
    }

    /// Returns the plugin/host ceiling [`Tier`] (`max_supported_tier`).
    #[inline]
    pub const fn max_supported_tier(&self) -> Tier {
        self.max_supported_tier
    }

    /// Sets the plugin/host ceiling [`Tier`] (`max_supported_tier`), clamping the active tier if necessary.
    #[inline]
    pub fn set_max_supported_tier(&mut self, max_tier: Tier) {
        self.max_supported_tier = max_tier;
        if self.tier > max_tier {
            self.tier = max_tier.min(self.max_available_tier);
        }
        let eff_tier = self.effective_tier();
        let arena = self.arena_storage.as_bytes();
        self.core1.collapse_stereo_head_3khz_with_ultra(
            arena,
            &self.control_bus.z_head,
            &self.control_bus.z_head_ultra,
            eff_tier.core1_groups(),
        );
    }

    /// Returns the effective [`Tier`] currently executing on the audio thread:
    /// - When `offline_bounce && maximize_tier_on_bounce`: `min(max_available_tier, max_supported_tier)`.
    /// - Otherwise: `self.tier`.
    #[inline]
    pub fn effective_tier(&self) -> Tier {
        if self.offline_bounce && self.maximize_tier_on_bounce {
            self.max_available_tier.min(self.max_supported_tier)
        } else {
            self.tier
                .min(self.max_available_tier)
                .min(self.max_supported_tier)
        }
    }

    /// Returns `true` if Offline Bounce mode ($4\times$ linear-phase polyphase oversampling) is enabled.
    #[inline]
    pub const fn offline_bounce(&self) -> bool {
        self.offline_bounce
    }

    /// Enables or disables Offline Bounce mode ($4\times$ linear-phase polyphase oversampling, `12`-sample PDC).
    #[inline]
    pub fn set_offline_bounce(&mut self, enabled: bool) {
        if self.offline_bounce != enabled {
            self.offline_bounce = enabled;
            let sr = self.zoh.sample_rate_host_hz;
            let _ = self.reconfigure_for_sample_rate(sr);
        }
    }

    /// Returns `true` if `maximize_tier_on_bounce` is enabled (automatically promoting the effective tier
    /// to `min(max_available_tier, max_supported_tier)` when `offline_bounce` is active).
    #[inline]
    pub const fn maximize_tier_on_bounce(&self) -> bool {
        self.maximize_tier_on_bounce
    }

    /// Enables or disables automatic tier promotion (`min(max_available_tier, max_supported_tier)`)
    /// during Offline Bounce.
    #[inline]
    pub fn set_maximize_tier_on_bounce(&mut self, enabled: bool) {
        if self.maximize_tier_on_bounce != enabled {
            self.maximize_tier_on_bounce = enabled;
            let eff_tier = self.effective_tier();
            let arena = self.arena_storage.as_bytes();
            self.core1.collapse_stereo_head_3khz_with_ultra(
                arena,
                &self.control_bus.z_head,
                &self.control_bus.z_head_ultra,
                eff_tier.core1_groups(),
            );
        }
    }

    /// Returns the current Plugin Delay Compensation (`PDC`) latency in host samples:
    /// - `0` samples in real-time (`!offline_bounce`) mode.
    /// - [`OFFLINE_BOUNCE_PDC_SAMPLES`] (`12`) samples in `offline_bounce` ($4\times$ linear-phase oversampling) mode.
    #[inline]
    pub const fn latency_samples(&self) -> usize {
        if self.offline_bounce {
            OFFLINE_BOUNCE_PDC_SAMPLES
        } else {
            0
        }
    }

    /// Renders a complete stereo buffer (`left_in`, `right_in` -> `left_out`, `right_out`) in
    /// Offline Bounce mode (`offline_bounce = true`), automatically compensating for the
    /// [`OFFLINE_BOUNCE_PDC_SAMPLES`] (`12`-sample) linear-phase filter lead-in so `left_out[0]`
    /// is sample-accurately aligned with `left_in[0]`.
    pub fn render_offline_bounce(
        &mut self,
        left_in: &[f32],
        right_in: &[f32],
        left_out: &mut [f32],
        right_out: &mut [f32],
    ) {
        let prev_bounce = self.offline_bounce;
        if !prev_bounce {
            self.set_offline_bounce(true);
        }
        self.reset_to_dc_equilibrium();

        let n = left_in
            .len()
            .min(right_in.len())
            .min(left_out.len())
            .min(right_out.len());
        let pdc = OFFLINE_BOUNCE_PDC_SAMPLES;
        let mut sink_l = [0.0_f32; OFFLINE_BOUNCE_PDC_SAMPLES];
        let mut sink_r = [0.0_f32; OFFLINE_BOUNCE_PDC_SAMPLES];

        if n <= pdc {
            let zeros = [0.0_f32; OFFLINE_BOUNCE_PDC_SAMPLES];
            self.process_block(
                &left_in[..n],
                &right_in[..n],
                &mut sink_l[..n],
                &mut sink_r[..n],
            );
            let rem = pdc - n;
            if rem > 0 {
                self.process_block(
                    &zeros[..rem],
                    &zeros[..rem],
                    &mut sink_l[..rem],
                    &mut sink_r[..rem],
                );
            }
            self.process_block(
                &zeros[..n],
                &zeros[..n],
                &mut left_out[..n],
                &mut right_out[..n],
            );
        } else {
            self.process_block(&left_in[..pdc], &right_in[..pdc], &mut sink_l, &mut sink_r);
            let body_len = n - pdc;
            self.process_block(
                &left_in[pdc..n],
                &right_in[pdc..n],
                &mut left_out[..body_len],
                &mut right_out[..body_len],
            );
            let zeros = [0.0_f32; OFFLINE_BOUNCE_PDC_SAMPLES];
            self.process_block(
                &zeros,
                &zeros,
                &mut left_out[body_len..n],
                &mut right_out[body_len..n],
            );
        }

        if !prev_bounce {
            self.set_offline_bounce(false);
        }
    }

    /// Switches the active [`Tier`] (`Lite`, `Full`, `Pro`, `Ultra`) at runtime with `0` allocations
    /// and `0.00 dB` LUFS continuity via `tier_gain_trim_lin`.
    #[inline]
    pub fn set_tier(&mut self, tier: Tier) {
        let clamped = tier
            .min(self.max_available_tier)
            .min(self.max_supported_tier);
        if clamped != self.tier {
            self.tier = clamped;
            let eff_tier = self.effective_tier();
            let arena = self.arena_storage.as_bytes();
            self.core1.collapse_stereo_head_3khz_with_ultra(
                arena,
                &self.control_bus.z_head,
                &self.control_bus.z_head_ultra,
                eff_tier.core1_groups(),
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
        let eff_tier = self.effective_tier();
        let arena = self.arena_storage.as_bytes();
        self.control_bus.refresh_static_state(arena);
        self.core1.collapse_stereo_head_3khz_with_ultra(
            arena,
            &self.control_bus.z_head,
            &self.control_bus.z_head_ultra,
            eff_tier.core1_groups(),
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
        let eff_sr_hz = (self
            .zoh
            .sample_rate_host_hz
            .saturating_mul(if self.offline_bounce { 4 } else { 1 }))
            as f32;

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
                self.core2
                    .set_grain_params(Some(val_ms), None, None, eff_sr_hz);
            }
            ControlCalibrationRole::GrainPitchSemitones => {
                let val_st = cal.evaluate_lobatto_anchors(norm_u).unwrap_or_else(|| {
                    if decl[0] >= 0.0 && decl[1] <= 10.0 {
                        norm_u * 12.0
                    } else {
                        physical_value.clamp(-24.0, 24.0)
                    }
                });
                self.core2
                    .set_grain_params(None, Some(val_st), None, eff_sr_hz);
            }
            ControlCalibrationRole::GrainSpray => {
                let val_spray = cal
                    .evaluate_lobatto_anchors(norm_u)
                    .unwrap_or(norm_01)
                    .clamp(0.0, 1.0);
                self.core2
                    .set_grain_params(None, None, Some(val_spray), eff_sr_hz);
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
        let eff_tier = self.effective_tier();
        let arena = self.arena_storage.as_bytes();
        self.control_bus.refresh_static_state(arena);
        self.core1.collapse_stereo_head_3khz_with_ultra(
            arena,
            &self.control_bus.z_head,
            &self.control_bus.z_head_ultra,
            eff_tier.core1_groups(),
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
        self.control_bus.reset_dynamic_state();
        let eff_tier = self.effective_tier();
        {
            let arena = self.arena_storage.as_bytes();
            self.control_bus.refresh_static_state(arena);
            self.core1.reset_to_dc_equilibrium();
            self.core1.collapse_stereo_head_3khz_with_ultra(
                arena,
                &self.control_bus.z_head,
                &self.control_bus.z_head_ultra,
                eff_tier.core1_groups(),
            );
        }
        self.core2.reset_state();
        if self.core2.active {
            self.core2.apply_control_bus_3khz(
                &self.control_bus.core2_gain_targets,
                &self.control_bus.core2_warp_mod,
                &self.control_bus.core2_grain_len_scale,
                &self.control_bus.core2_state_mod,
                self.control_bus.env_states[3],
            );
            self.control_bus.core2_mod_dirty = false;
        }
        self.oversampler.reset();
        self.ctrl_sub_counter = 0;
        self.ctrl_energy_accum = 0.0;
    }

    /// Captures a lock-free [`DnamTelemetrySnapshot`] of the current engine state.
    pub fn telemetry_snapshot(&self) -> DnamTelemetrySnapshot {
        DnamTelemetrySnapshot {
            active_tier: self.effective_tier() as u8,
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
        energy_norm_scale: f32,
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
            let avg_energy = *ctrl_energy_accum * energy_norm_scale;
            *ctrl_sub_counter = 0;
            *ctrl_energy_accum = 0.0;
            if control_bus.tick_3khz(avg_energy, lfo_env_active, arena) {
                core1.collapse_stereo_head_3khz_with_ultra(
                    arena,
                    &control_bus.z_head,
                    &control_bus.z_head_ultra,
                    GROUPS,
                );
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
            let zero_tap = [[0.0_f32; 4]; 4];
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

    /// Internal $4\times$ linear-phase polyphase oversampled host frame step (`12`-sample PDC).
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn step_frame_4x_const<const GROUPS: usize, const FDN_GROUPS: usize>(
        oversampler: &mut Oversampler4xState,
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
        let up4 = oversampler.upsample_4x(in_l, in_r);
        let mut w4 = [[0.0_f32; 2]; 4];
        for m in 0..4 {
            w4[m] = Self::step_frame_const::<GROUPS, FDN_GROUPS>(
                core1,
                core2,
                control_bus,
                ctrl_sub_counter,
                ctrl_energy_accum,
                up4[m][0],
                up4[m][1],
                arena,
                tiles,
                k_ctrl,
                1.0 / 64.0,
                lfo_env_active,
                mono_only,
                dry_g,
                wet_g,
                bypassed,
                trails_enabled,
                freeze_mutes_dry,
                stereo_width,
            );
        }
        let [out_l, out_r] = oversampler.downsample_4x(w4);
        if mono_only {
            [out_l, out_l]
        } else {
            [out_l, out_r]
        }
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
        let eff_tier = self.effective_tier();
        let arena = self.arena_storage.as_bytes();
        let tiles = Core1Stack::<B>::arena_tiles(arena);
        let k_ctrl = self.zoh.control_decimation_host;
        let lfo_env_active = self.preamble.is_lfo_env_active();
        let mono_only = !self.preamble.is_stereo_out();
        let dry_g = self.core1.mix_dry_gain;
        let wet_g = self.core1.mix_wet_gain * self.core1.tier_gain_trim_lin[eff_tier as usize];
        let bypassed = self.bypassed;
        let trails = self.trails_enabled;
        let freeze_mutes_dry = self.freeze_mutes_dry;
        let width = self.stereo_width;

        macro_rules! run_stereo {
            ($G:literal, $FG:literal) => {
                if self.offline_bounce {
                    for i in 0..n {
                        let [l, r] = Self::step_frame_4x_const::<$G, $FG>(
                            &mut self.oversampler,
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
                } else {
                    for i in 0..n {
                        let [l, r] = Self::step_frame_const::<$G, $FG>(
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
                            1.0 / 16.0,
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
            };
        }

        match eff_tier {
            Tier::Lite => run_stereo!(1, 1),
            Tier::Full => run_stereo!(2, 2),
            Tier::Pro => run_stereo!(3, 2),
            Tier::Ultra => run_stereo!(4, 4),
        }
        if n > 0 {
            self.core1.finalize_block_telemetry(eff_tier.core1_groups());
        }
    }

    /// **Topology 2 (`Stereo -> Stereo` In-Place):**
    /// Processes `left_io` and `right_io` in-place (VST3/CLAP standard buffer topology) with zero heap allocations.
    pub fn process_block_in_place(&mut self, left_io: &mut [f32], right_io: &mut [f32]) {
        let _ftz = FtzDazGuard::new();
        let n = left_io.len().min(right_io.len());
        self.reset_block_telemetry();
        let eff_tier = self.effective_tier();
        let arena = self.arena_storage.as_bytes();
        let tiles = Core1Stack::<B>::arena_tiles(arena);
        let k_ctrl = self.zoh.control_decimation_host;
        let lfo_env_active = self.preamble.is_lfo_env_active();
        let mono_only = !self.preamble.is_stereo_out();
        let dry_g = self.core1.mix_dry_gain;
        let wet_g = self.core1.mix_wet_gain * self.core1.tier_gain_trim_lin[eff_tier as usize];
        let bypassed = self.bypassed;
        let trails = self.trails_enabled;
        let freeze_mutes_dry = self.freeze_mutes_dry;
        let width = self.stereo_width;

        macro_rules! run_in_place {
            ($G:literal, $FG:literal) => {
                if self.offline_bounce {
                    for i in 0..n {
                        let [l, r] = Self::step_frame_4x_const::<$G, $FG>(
                            &mut self.oversampler,
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
                } else {
                    for i in 0..n {
                        let [l, r] = Self::step_frame_const::<$G, $FG>(
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
                            1.0 / 16.0,
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
            };
        }

        match eff_tier {
            Tier::Lite => run_in_place!(1, 1),
            Tier::Full => run_in_place!(2, 2),
            Tier::Pro => run_in_place!(3, 2),
            Tier::Ultra => run_in_place!(4, 4),
        }
        if n > 0 {
            self.core1.finalize_block_telemetry(eff_tier.core1_groups());
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
        let eff_tier = self.effective_tier();
        let arena = self.arena_storage.as_bytes();
        let tiles = Core1Stack::<B>::arena_tiles(arena);
        let k_ctrl = self.zoh.control_decimation_host;
        let lfo_env_active = self.preamble.is_lfo_env_active();
        let mono_only = !self.preamble.is_stereo_out();
        let dry_g = self.core1.mix_dry_gain;
        let wet_g = self.core1.mix_wet_gain * self.core1.tier_gain_trim_lin[eff_tier as usize];
        let bypassed = self.bypassed;
        let trails = self.trails_enabled;
        let freeze_mutes_dry = self.freeze_mutes_dry;
        let width = self.stereo_width;

        macro_rules! run_m2s {
            ($G:literal, $FG:literal) => {
                if self.offline_bounce {
                    for i in 0..n {
                        let x = mono_in[i];
                        let [l, r] = Self::step_frame_4x_const::<$G, $FG>(
                            &mut self.oversampler,
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
                } else {
                    for i in 0..n {
                        let x = mono_in[i];
                        let [l, r] = Self::step_frame_const::<$G, $FG>(
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
                            1.0 / 16.0,
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
            };
        }

        match eff_tier {
            Tier::Lite => run_m2s!(1, 1),
            Tier::Full => run_m2s!(2, 2),
            Tier::Pro => run_m2s!(3, 2),
            Tier::Ultra => run_m2s!(4, 4),
        }
        if n > 0 {
            self.core1.finalize_block_telemetry(eff_tier.core1_groups());
        }
    }

    /// **Topology 4 (`Mono -> Mono`, 50% Output Head MAC Savings):**
    /// Evaluates only Channel 0 of the collapsed 16-tap output head (`force_mono_head = true`),
    /// saving 50% of output head MACs on mono pedalboard blocks.
    pub fn process_mono(&mut self, mono_in: &[f32], mono_out: &mut [f32]) {
        let _ftz = FtzDazGuard::new();
        let n = mono_in.len().min(mono_out.len());
        self.reset_block_telemetry();
        let eff_tier = self.effective_tier();
        let arena = self.arena_storage.as_bytes();
        let tiles = Core1Stack::<B>::arena_tiles(arena);
        let k_ctrl = self.zoh.control_decimation_host;
        let lfo_env_active = self.preamble.is_lfo_env_active();
        let dry_g = self.core1.mix_dry_gain;
        let wet_g = self.core1.mix_wet_gain * self.core1.tier_gain_trim_lin[eff_tier as usize];
        let bypassed = self.bypassed;
        let trails = self.trails_enabled;
        let freeze_mutes_dry = self.freeze_mutes_dry;
        let width = self.stereo_width;

        macro_rules! run_mono {
            ($G:literal, $FG:literal) => {
                if self.offline_bounce {
                    for i in 0..n {
                        let x = mono_in[i];
                        let [l, _] = Self::step_frame_4x_const::<$G, $FG>(
                            &mut self.oversampler,
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
                } else {
                    for i in 0..n {
                        let x = mono_in[i];
                        let [l, _] = Self::step_frame_const::<$G, $FG>(
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
                            1.0 / 16.0,
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
            };
        }

        match eff_tier {
            Tier::Lite => run_mono!(1, 1),
            Tier::Full => run_mono!(2, 2),
            Tier::Pro => run_mono!(3, 2),
            Tier::Ultra => run_mono!(4, 4),
        }
        if n > 0 {
            self.core1.finalize_block_telemetry(eff_tier.core1_groups());
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

        // Process across all 4 tiers (Ultra, Pro, Full, Lite) with 4-tier Pre-Head Kronecker routing
        let l_in = [0.2_f32; 32];
        let r_in = [-0.2_f32; 32];
        let mut l_out = [0.0_f32; 32];
        let mut r_out = [0.0_f32; 32];
        for tier in [Tier::Ultra, Tier::Pro, Tier::Full, Tier::Lite] {
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

    #[test]
    fn test_4_tiers_max_supported_tier_and_offline_bounce_4x_oversampling() {
        let mut header = DnamJsonHeader::default_reference(false);
        header.controls_schema.push(ControlSchemaDef {
            id: String::from("gain"),
            name: String::from("Gain"),
            kind: ControlKind::Range,
            unit: None,
            lobatto_degree: Some(5),
            anchor_nodes_norm: None,
            declared_range: Some([0.0, 10.0]),
            safe_clamped_range: Some([0.0, 10.0]),
            default: serde_json::json!(5.0),
            conditional_min: BTreeMap::new(),
            conditional_max: BTreeMap::new(),
            declared_options: None,
            active_options: None,
            clamp_warning: None,
            calibration: None,
            ui_hints: ControlUiHints::default(),
            user_metadata: BTreeMap::new(),
        });
        let mut arena = alloc::vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
        // Configure dry=1.0, wet=0.0 at Arena 0 offset 116..124 so we can verify exact 4x oversampler linear phase!
        arena[116..120].copy_from_slice(&1.0_f32.to_le_bytes());
        arena[120..124].copy_from_slice(&0.0_f32.to_le_bytes());
        let container = build_dnam_container(&mut header, &arena, &arena, false).unwrap();

        // 1. Verify max_supported_tier ceiling on from_bytes_with_max_tier
        let mut capped_eng = DnamEngine::<ScalarBackend>::from_bytes_with_max_tier(
            &container,
            Tier::Ultra,
            Tier::Full,
            48_000,
        )
        .unwrap();
        assert_eq!(capped_eng.max_supported_tier(), Tier::Full);
        assert_eq!(capped_eng.max_available_tier(), Tier::Full);
        assert_eq!(capped_eng.tier(), Tier::Full);
        capped_eng.set_tier(Tier::Pro);
        assert_eq!(capped_eng.tier(), Tier::Full);

        // 2. Full Ultra engine with offline_bounce and maximize_tier_on_bounce
        let mut eng =
            DnamEngine::<ScalarBackend>::from_bytes(&container, Tier::Lite, 48_000).unwrap();
        assert_eq!(eng.tier(), Tier::Lite);
        assert_eq!(eng.effective_tier(), Tier::Lite);
        assert_eq!(eng.latency_samples(), 0);
        assert!(!eng.offline_bounce());
        assert!(eng.maximize_tier_on_bounce());

        eng.set_offline_bounce(true);
        assert!(eng.offline_bounce());
        assert_eq!(eng.latency_samples(), 12);
        assert_eq!(eng.effective_tier(), Tier::Ultra);

        eng.set_maximize_tier_on_bounce(false);
        assert_eq!(eng.effective_tier(), Tier::Lite);
        eng.set_maximize_tier_on_bounce(true);
        assert_eq!(eng.effective_tier(), Tier::Ultra);

        // Respect max_supported_tier during offline bounce promotion
        eng.set_max_supported_tier(Tier::Pro);
        assert_eq!(eng.effective_tier(), Tier::Pro);
        eng.set_max_supported_tier(Tier::Ultra);
        assert_eq!(eng.effective_tier(), Tier::Ultra);

        // 3. Verify 4x linear-phase oversampler has exact 12-sample group delay and symmetric impulse response
        eng.reset_to_dc_equilibrium();
        let mut imp_in = [0.0_f32; 32];
        imp_in[0] = 1.0;
        let zero_in = [0.0_f32; 32];
        let mut imp_out_l = [0.0_f32; 32];
        let mut imp_out_r = [0.0_f32; 32];
        eng.process_block(&imp_in, &zero_in, &mut imp_out_l, &mut imp_out_r);

        // Peak must be at sample 12, and imp_out_l[12 - k] == imp_out_l[12 + k] (exact linear phase!)
        let peak_idx = imp_out_l
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;
        assert_eq!(peak_idx, 12);
        for k in 1..=12 {
            assert!(
                (imp_out_l[12 - k] - imp_out_l[12 + k]).abs() < 1e-6,
                "Asymmetry at k={k}: {} vs {}",
                imp_out_l[12 - k],
                imp_out_l[12 + k]
            );
        }

        // 4. Verify render_offline_bounce strips the 12-sample PDC lead-in so peak lands at sample 0!
        eng.set_offline_bounce(false);
        eng.set_control_immediate(0, 8.75);
        assert!((eng.control_bus.raw_values[0] - 8.75).abs() < 1e-6);
        let mut bounce_out_l = [0.0_f32; 32];
        let mut bounce_out_r = [0.0_f32; 32];
        eng.render_offline_bounce(&imp_in, &zero_in, &mut bounce_out_l, &mut bounce_out_r);
        assert!(
            (bounce_out_l[0] - imp_out_l[12]).abs() < 1e-6,
            "render_offline_bounce must align sample 12 to index 0 (got {} vs {})",
            bounce_out_l[0],
            imp_out_l[12]
        );
        // Control values must be preserved across render_offline_bounce and set_sample_rate
        assert!((eng.control_bus.raw_values[0] - 8.75).abs() < 1e-6);
        eng.set_sample_rate(96_000).unwrap();
        assert!((eng.control_bus.raw_values[0] - 8.75).abs() < 1e-6);

        // 5. Verify set_control_immediate does not step LFO phasors, and render_offline_bounce
        //    is bit-identical whether preceded by real-time LFO playback / slewed set_control_value
        //    or called on a fresh engine.
        let mut lfo_hdr = DnamJsonHeader::default_reference(true);
        lfo_hdr.core2_mufdn.ring_buffer_pow2_log2 = 14;
        lfo_hdr.control_bus.lfo_oscillators = 2;
        lfo_hdr.control_bus.envelopes = 3;
        lfo_hdr.controls_schema.push(ControlSchemaDef {
            id: String::from("grain_ms"),
            name: String::from("Grain Size"),
            kind: ControlKind::Range,
            unit: Some(String::from("ms")),
            lobatto_degree: Some(2),
            anchor_nodes_norm: Some(alloc::vec![0.0, -1.0, 1.0]),
            declared_range: Some([10.0, 250.0]),
            safe_clamped_range: Some([10.0, 250.0]),
            default: serde_json::json!(50.0),
            conditional_min: BTreeMap::new(),
            conditional_max: BTreeMap::new(),
            declared_options: None,
            active_options: None,
            clamp_warning: None,
            calibration: Some(ControlCalibration {
                role: ControlCalibrationRole::GrainSizeMs,
                lobatto_physical_anchors: Some([10.0, 50.0, 250.0]),
                conditional_physical_anchors: BTreeMap::new(),
                tempo_sync_default_subdivision: None,
            }),
            ui_hints: ControlUiHints::default(),
            user_metadata: BTreeMap::new(),
        });
        let mut lfo_arena = alloc::vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
        lfo_arena[0..4].copy_from_slice(&1.0_f32.to_le_bytes());
        lfo_arena[4224..4228].copy_from_slice(&0.5_f32.to_le_bytes());
        lfo_arena[4736..4740].copy_from_slice(&0.25_f32.to_le_bytes());
        lfo_arena[8960..8964].copy_from_slice(&1.0_f32.to_le_bytes());
        // Populate LFO/envelope rows (rows 4..7) in W_ctrl_core2 (9216..10240)
        for col in 0..32 {
            let off = 9216 + (4 * 32 + col) * 4;
            lfo_arena[off..off + 4].copy_from_slice(&0.2_f32.to_le_bytes());
        }
        let lfo_container =
            build_dnam_container(&mut lfo_hdr, &lfo_arena, &lfo_arena, false).unwrap();

        let mut eng_fresh =
            DnamEngine::<ScalarBackend>::from_bytes(&lfo_container, Tier::Ultra, 48_000).unwrap();
        let phasor_before = eng_fresh.control_bus.lfo_phasors;
        eng_fresh.set_control_immediate(0, 200.0);
        assert_eq!(eng_fresh.control_bus.lfo_phasors, phasor_before);
        let grain_1x = eng_fresh.core2.groups[0].grain_len_samples[0];
        eng_fresh.set_offline_bounce(true);
        let grain_4x = eng_fresh.core2.groups[0].grain_len_samples[0];
        assert!(
            (grain_4x - 4.0 * grain_1x).abs() < 1e-2,
            "4x offline bounce must scale calibrated GrainSizeMs by 4x: got {grain_4x} vs 4 * {grain_1x}"
        );
        eng_fresh.set_offline_bounce(false);

        let mut ref_bounce_l = [0.0_f32; 64];
        let mut ref_bounce_r = [0.0_f32; 64];
        let mut test_in = [0.0_f32; 64];
        for (i, s) in test_in.iter_mut().enumerate() {
            *s = ((i as f32) * 0.17).sin() * 0.5;
        }
        eng_fresh.render_offline_bounce(&test_in, &test_in, &mut ref_bounce_l, &mut ref_bounce_r);

        let mut eng_dirty =
            DnamEngine::<ScalarBackend>::from_bytes(&lfo_container, Tier::Ultra, 48_000).unwrap();
        let mut rt_l = [0.0_f32; 64];
        let mut rt_r = [0.0_f32; 64];
        // Advance LFO/envelope states in real-time and set control via slewed set_control_value
        eng_dirty.process_block(&test_in, &test_in, &mut rt_l, &mut rt_r);
        eng_dirty.set_control_value(0, 200.0);
        let mut dirty_bounce_l = [0.0_f32; 64];
        let mut dirty_bounce_r = [0.0_f32; 64];
        eng_dirty.render_offline_bounce(
            &test_in,
            &test_in,
            &mut dirty_bounce_l,
            &mut dirty_bounce_r,
        );
        assert_eq!(dirty_bounce_l, ref_bounce_l);
        assert_eq!(dirty_bounce_r, ref_bounce_r);
    }
}
