//! Primary generic real-time DNAM inference engine (`DnamEngine<'a, B>`).

pub mod oversampler;
pub mod preview_guard;
pub mod process;
pub mod runtime;
pub mod tempo;

use crate::container::{
    ARENA_LITE_END_CL, ARENA_OFFSET_LITE_END, DnamContainerView, DnamPreamble, ZohRescaledParams,
};
use crate::control::ControlBus;
use crate::math::{DefaultMathBackend, MathBackend};
use crate::memory::{AlignedBuffer, AlignedFp16Block, ArenaStorage, CacheLine64};
use crate::provenance::{CryptoProvider, DefaultCryptoProvider, VerificationStatus};
use crate::schema::{
    ControlCalibrationRole, DnamError, DnamJsonHeader, DnamProtectionProfile, TempoSubdivision,
    cold_pow_f64,
};
use crate::stages::StagePipeline;
use crate::types::{DnamTelemetrySnapshot, Tier};

pub use oversampler::{OFFLINE_BOUNCE_PDC_SAMPLES, Oversampler4xState};
pub use preview_guard::PreviewGuard;
pub use tempo::TempoSyncState;

/// Zero-allocation real-time DNAM inference engine parameterized over a
/// [`MathBackend`] (`DefaultMathBackend` by default).
#[derive(Debug)]
pub struct DnamEngine<'a, B: MathBackend = DefaultMathBackend> {
    pub(crate) preamble: DnamPreamble,
    pub(crate) header: Option<DnamJsonHeader>,
    pub(crate) verification_status: VerificationStatus,
    pub(crate) protection_profile: DnamProtectionProfile,
    pub(crate) arena_storage: ArenaStorage<'a>,
    pub(crate) tier: Tier,
    pub(crate) max_available_tier: Tier,
    pub(crate) max_supported_tier: Tier,
    pub(crate) offline_bounce: bool,
    pub(crate) maximize_tier_on_bounce: bool,
    pub(crate) oversampler: Oversampler4xState,
    pub(crate) preview_guard: PreviewGuard,
    pub(crate) tempo_state: TempoSyncState,
    pub(crate) zoh: ZohRescaledParams,
    pub(crate) ctrl_sub_counter: usize,
    pub(crate) ctrl_energy_accum: f32,
    pub(crate) bypassed: bool,
    pub(crate) trails_enabled: bool,
    pub(crate) freeze_mutes_dry: bool,
    pub(crate) stereo_width: f32,
    pub control_bus: ControlBus<B>,
    pub stage_pipeline: StagePipeline,
    pub mix_dry_gain: f32,
    pub mix_wet_gain: f32,
}

impl<'a, B: MathBackend> DnamEngine<'a, B> {
    /// Creates an unconfigured zero-allocation [`DnamEngine`] shell.
    pub(crate) fn new_empty(max_supported_tier: Tier) -> Self {
        let zoh = ZohRescaledParams {
            sample_rate_host_hz: 48_000,
            rho_sr: 1.0,
            control_decimation_host: 16,
            rho_ctrl: 1.0,
            alpha_ramp_host: 0.0625,
            r_ac_host: 0.9993,
            r_dc_host: 0.995,
            tau_ref_samples_host: 64.0,
            switch_xfade_samples_host: 128,
        };
        let preamble = DnamPreamble {
            magic: crate::container::DNAM_MAGIC,
            version_major: 1,
            version_minor: 0,
            flags: crate::container::FLAG_HAS_LITE | crate::container::FLAG_STEREO_OUT,
            num_controls: 0,
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
            _reserved: 0,
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
            preview_guard: PreviewGuard::new(),
            tempo_state: TempoSyncState::new(),
            zoh,
            ctrl_sub_counter: 0,
            ctrl_energy_accum: 0.0,
            bypassed: false,
            trails_enabled: false,
            freeze_mutes_dry: false,
            stereo_width: 1.0,
            control_bus: ControlBus::new(),
            stage_pipeline: StagePipeline::new(),
            mix_dry_gain: 0.0,
            mix_wet_gain: 1.0,
        }
    }

    /// Loads a `.dnam` or `.dnam-ckpt` container from `bytes` using the default [`CryptoProvider`].
    pub fn from_bytes(bytes: &'a [u8], tier: Tier, sample_rate_hz: u32) -> Result<Self, DnamError> {
        let crypto = DefaultCryptoProvider::default();
        Self::from_bytes_with_crypto(bytes, tier, sample_rate_hz, &crypto)
    }

    /// Loads a `.dnam` or `.dnam-ckpt` container capped at `max_supported_tier`.
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

    /// Creates an owned `'static` [`DnamEngine`].
    pub fn from_bytes_owned(
        bytes: &[u8],
        tier: Tier,
        sample_rate_hz: u32,
    ) -> Result<DnamEngine<'static, B>, DnamError> {
        let crypto = DefaultCryptoProvider::default();
        Self::from_bytes_owned_with_crypto(bytes, tier, sample_rate_hz, &crypto)
    }

    /// Creates an owned `'static` [`DnamEngine`] capped at `max_supported_tier`.
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
            control_records: view.control_records,
            arena_storage: ArenaStorage::from_owned_buffer(owned_arena),
        };
        let mut engine = DnamEngine::<'static, B>::new_empty(max_supported_tier);
        engine.init_from_view(owned_view, tier, max_supported_tier, sample_rate_hz)?;
        Ok(engine)
    }

    /// Loads a `.dnam` container with a custom [`CryptoProvider`].
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
            control_records: view.control_records,
            arena_storage: ArenaStorage::from_owned_buffer(owned_arena),
        };
        let mut engine = DnamEngine::<'static, B>::new_empty(Tier::Ultra);
        engine.init_from_view(owned_view, tier, Tier::Ultra, sample_rate_hz)?;
        Ok(engine)
    }

    /// Loads a borrowed `.dnam` container with a custom [`CryptoProvider`].
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

    /// Zero-Copy Scratch Loader: borrows caller-supplied `scratch` and arena.
    pub fn from_bytes_with_scratch(
        bytes: &'a [u8],
        _scratch: &'a mut [AlignedFp16Block],
        tier: Tier,
        sample_rate_hz: u32,
    ) -> Result<Self, DnamError> {
        Self::from_bytes(bytes, tier, sample_rate_hz)
    }

    /// Pre-allocates a [`DnamEngine`] with maximum `Arena 0..3` (`64,000 B`).
    pub fn with_max_capacity() -> Self {
        Self::with_max_tier(Tier::Ultra)
    }

    /// Pre-allocates a [`DnamEngine`] capped at `max_supported_tier`.
    pub fn with_max_tier(max_supported_tier: Tier) -> Self {
        let mut engine = Self::new_empty(max_supported_tier);
        engine.arena_storage = ArenaStorage::with_max_tier(max_supported_tier);
        engine.control_bus.ensure_control_capacity(256);
        engine
    }

    #[inline]
    pub const fn tier(&self) -> Tier {
        self.tier
    }

    #[inline]
    pub const fn max_available_tier(&self) -> Tier {
        self.max_available_tier
    }

    #[inline]
    pub const fn max_supported_tier(&self) -> Tier {
        self.max_supported_tier
    }

    #[inline]
    pub fn set_max_supported_tier(&mut self, max_tier: Tier) {
        self.max_supported_tier = max_tier;
        if self.tier > max_tier {
            self.tier = max_tier;
        }
    }

    #[inline]
    pub fn effective_tier(&self) -> Tier {
        if self.offline_bounce && self.maximize_tier_on_bounce {
            self.max_available_tier.min(self.max_supported_tier)
        } else {
            self.tier
        }
    }

    #[inline]
    pub const fn offline_bounce(&self) -> bool {
        self.offline_bounce
    }

    #[inline]
    pub fn set_offline_bounce(&mut self, enabled: bool) {
        if self.offline_bounce != enabled {
            self.offline_bounce = enabled;
            let _ = self.reconfigure_for_sample_rate(self.zoh.sample_rate_host_hz);
        }
    }

    #[inline]
    pub const fn maximize_tier_on_bounce(&self) -> bool {
        self.maximize_tier_on_bounce
    }

    #[inline]
    pub fn set_maximize_tier_on_bounce(&mut self, enabled: bool) {
        self.maximize_tier_on_bounce = enabled;
    }

    #[inline]
    pub const fn latency_samples(&self) -> usize {
        if self.offline_bounce {
            OFFLINE_BOUNCE_PDC_SAMPLES
        } else {
            0
        }
    }

    #[inline]
    pub fn set_tier(&mut self, tier: Tier) {
        let max_tier = self.max_available_tier.min(self.max_supported_tier);
        self.tier = tier.min(max_tier);
    }

    #[inline]
    pub fn set_sample_rate(&mut self, sample_rate_hz: u32) -> Result<(), DnamError> {
        self.reconfigure_for_sample_rate(sample_rate_hz)
    }

    #[inline]
    pub const fn sample_rate_hz(&self) -> u32 {
        self.zoh.sample_rate_host_hz
    }

    #[inline]
    pub const fn zoh(&self) -> &ZohRescaledParams {
        &self.zoh
    }

    #[inline]
    pub const fn verification_status(&self) -> &VerificationStatus {
        &self.verification_status
    }

    #[inline]
    pub const fn protection_profile(&self) -> DnamProtectionProfile {
        self.protection_profile
    }

    #[inline]
    pub const fn preamble(&self) -> &DnamPreamble {
        &self.preamble
    }

    #[inline]
    pub const fn header(&self) -> Option<&DnamJsonHeader> {
        self.header.as_ref()
    }

    #[inline]
    pub fn set_control_value(&mut self, ctrl_idx: usize, physical_value: f32) {
        self.control_bus.set_control_value(ctrl_idx, physical_value);
        self.refresh_calibrated_control_effects(ctrl_idx, physical_value);
    }

    #[inline]
    pub fn set_control_immediate(&mut self, ctrl_idx: usize, physical_value: f32) {
        self.control_bus
            .set_control_immediate(ctrl_idx, physical_value);
        self.refresh_calibrated_control_effects(ctrl_idx, physical_value);
        let arena = self.arena_storage.as_bytes();
        self.control_bus.refresh_static_state(arena);
        if self.control_bus.tank_mod_dirty {
            self.apply_tank_modulations();
            self.control_bus.tank_mod_dirty = false;
        }
    }

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
                let u = (2.0 * norm_01 - 1.0).clamp(-1.0, 1.0);
                let w = cal.evaluate_lobatto_anchors(u).unwrap_or_else(|| {
                    if decl[1] <= 2.0 {
                        physical_value
                    } else {
                        norm_01
                    }
                });
                self.stereo_width = w.clamp(0.0, 2.0);
            }
            ControlCalibrationRole::FreezeLatch => {
                if ctrl_def.kind == crate::schema::ControlKind::Switch {
                    if let Some(ref opts) = ctrl_def.declared_options {
                        let idx = (physical_value.round() as isize).max(0) as usize;
                        if let Some(opt_label) = opts.get(idx)
                            && let Some(alpha) = cal.freeze_amount_for_option(opt_label)
                        {
                            self.set_freeze_amount(alpha);
                        }
                    }
                } else {
                    self.set_freeze_amount(norm_01);
                }
            }
            _ => {}
        }
    }

    #[inline]
    pub fn set_freeze(&mut self, freeze: bool) {
        self.stage_pipeline.stage4.freeze = freeze;
    }

    #[inline]
    pub fn set_freeze_amount(&mut self, alpha: f32) {
        self.stage_pipeline.stage4.freeze = alpha > 0.5;
    }

    #[inline]
    pub fn is_frozen(&self) -> bool {
        self.stage_pipeline.stage4.freeze
    }

    #[inline]
    pub fn set_bypass(&mut self, bypassed: bool, trails: bool) {
        self.bypassed = bypassed;
        self.trails_enabled = trails;
    }

    #[inline]
    pub const fn bypass_state(&self) -> (bool, bool) {
        (self.bypassed, self.trails_enabled)
    }

    #[inline]
    pub fn set_stereo_width(&mut self, width: f32) {
        self.stereo_width = width.clamp(0.0, 2.0);
    }

    #[inline]
    pub const fn stereo_width(&self) -> f32 {
        self.stereo_width
    }

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

        self.mix_dry_gain = (1.0 - b) * out_scale;
        self.mix_wet_gain = pol_sign * b * wet_trim_scale;
    }

    pub fn sync_tempo(
        &mut self,
        bpm: f32,
        subdivision_override: Option<TempoSubdivision>,
        fold_octave: bool,
    ) -> usize {
        self.tempo_state.set_bpm(bpm);
        let Some(ref hdr) = self.header else {
            return 0;
        };
        let mut updates = [(0usize, 0.0f32); 512];
        let count = TempoSyncState::resolve_sync_updates_into(
            hdr,
            self.tempo_state.current_bpm,
            subdivision_override,
            fold_octave,
            &mut updates,
        );
        for &(idx, val) in &updates[..count] {
            self.set_control_immediate(idx, val);
        }
        count
    }

    /// Registers a tap-tempo event at the current audio frame counter, updates the host BPM,
    /// and synchronizes any `DelayTimeMs` or `LfoRateHz` controls.
    pub fn register_tap_tempo(
        &mut self,
        subdivision_override: Option<TempoSubdivision>,
        fold_octave: bool,
    ) -> (f32, usize) {
        let bpm = self.tempo_state.register_tap();
        let count = self.sync_tempo(bpm, subdivision_override, fold_octave);
        (bpm, count)
    }

    pub fn apply_preset(&mut self, preset_id: &str) -> Result<(), DnamError> {
        let Some(ref hdr) = self.header else {
            return Err(DnamError::SchemaValidationError("header not loaded".into()));
        };
        let Some(preset) = hdr.presets.iter().find(|p| p.id == preset_id) else {
            return Err(DnamError::SchemaValidationError(alloc::format!(
                "unknown preset id: {}",
                preset_id
            )));
        };
        let mut updates = [(0usize, 0.0f32); 256];
        let mut count = 0;
        // First pass: apply switch controls so conditional range bounds are active before range controls
        for pass in 0..2 {
            for (id, val) in &preset.controls {
                if let Some((idx, def)) = hdr
                    .controls_schema
                    .iter()
                    .enumerate()
                    .find(|(_, c)| &c.id == id)
                {
                    let is_switch = def.kind == crate::schema::ControlKind::Switch;
                    if (pass == 0 && !is_switch) || (pass == 1 && is_switch) {
                        continue;
                    }
                    if let Some(f) = val.as_f64() {
                        if count < updates.len() {
                            updates[count] = (idx, f as f32);
                            count += 1;
                        }
                    } else if let Some(s) = val.as_str()
                        && let Some(ref opts) = def.declared_options
                        && let Some(opt_idx) = opts.iter().position(|o| o == s)
                        && count < updates.len()
                    {
                        updates[count] = (idx, opt_idx as f32);
                        count += 1;
                    }
                }
            }
        }
        for &(idx, val) in &updates[..count] {
            self.set_control_immediate(idx, val);
        }
        Ok(())
    }

    pub fn telemetry_snapshot(&self) -> DnamTelemetrySnapshot {
        DnamTelemetrySnapshot {
            active_tier: self.effective_tier() as u8,
            stereo_out: u8::from(self.preamble.is_stereo_out()),
            bypassed: u8::from(self.bypassed),
            is_frozen: u8::from(self.is_frozen()),
            active_clamp_mask: self.control_bus.active_clamp_mask,
            control_decimation_host: self.zoh.control_decimation_host as u16,
            max_op_n_drive: 1.0,
            max_pole_radius: 0.999,
            max_effective_loop_gain: self.stage_pipeline.stage4.g_loop,
            tail_time_ms: self.preamble.tail_time_ms,
            z_head: self.control_bus.z_head,
            env_states: self.control_bus.env_states,
            lfo_harmonics: self.control_bus.lfo_harmonics,
            clamped_controls: self.control_bus.telemetry_clamped_controls(),
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
        let mut header = DnamJsonHeader::default_reference();
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

        let mut cap_engine = DnamEngine::<ScalarBackend>::with_max_capacity();
        assert!(cap_engine.apply_preset("lead_boost").is_err());
        cap_engine
            .reload_model_in_place(&container, Tier::Full, 48_000)
            .unwrap();
        assert_eq!(cap_engine.tier(), Tier::Full);

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

        let mut owned_eng =
            DnamEngine::<ScalarBackend>::from_bytes_owned(&container, Tier::Lite, 48_000).unwrap();
        assert_eq!(owned_eng.tier(), Tier::Lite);
        assert!(
            owned_eng
                .reload_model_in_place_owned(&enc_flag, Tier::Lite, 48_000)
                .is_err()
        );

        // Exercise set_max_supported_tier clamping active tier
        engine.set_max_supported_tier(Tier::Lite);
        assert_eq!(engine.tier(), Tier::Lite);

        // Exercise configure_host_mix_and_leveling & sync_tempo on empty engine (header == None)
        let mut empty_eng = DnamEngine::<ScalarBackend>::new_empty(Tier::Lite);
        empty_eng.configure_host_mix_and_leveling(0.5, 0.0, false);
        assert_eq!(empty_eng.sync_tempo(120.0, None, false), 0);
    }

    #[test]
    fn test_dnam_engine_leveling_blend_trails_tempo_and_stereo_mode() {
        let mut header = DnamJsonHeader::default_reference();
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

        engine.configure_host_mix_and_leveling(0.5, 0.0, true);
        assert!((engine.mix_dry_gain - 0.5).abs() < 1e-4);
        assert!((engine.mix_wet_gain - (-1.0)).abs() < 1e-3);
        engine.configure_host_mix_and_leveling(0.5, 0.0, false);

        assert!((engine.stereo_width() - 1.0).abs() < 1e-6);
        engine.set_control_immediate(2, 0.0);
        assert!(engine.stereo_width().abs() < 1e-6);
        // Process block with stereo_width == 0.0 (stereo width folding active)
        let mut l_fold = [0.0_f32; 4];
        let mut r_fold = [0.0_f32; 4];
        engine.process_block(
            &[0.25, -0.5, 0.75, -0.25],
            &[-0.25, 0.5, -0.75, 0.25],
            &mut l_fold,
            &mut r_fold,
        );
        engine.reset_to_dc_equilibrium();
        engine.set_control_immediate(2, 1.0);
        assert!((engine.stereo_width() - 1.0).abs() < 1e-6);

        let updated = engine.sync_tempo(120.0, Some(TempoSubdivision::HalfDotted), false);
        assert_eq!(updated, 2);
        let snap = engine.telemetry_snapshot();
        assert!((snap.clamped_controls[0] - 5.0).abs() < 1e-3);
        assert!((snap.clamped_controls[1] - 2.0).abs() < 1e-3);

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
}
