//! Initialization, model hot-reload, sample-rate reconfiguration, tier/bounce modes, and
//! wait-free telemetry inspection for [`DnamHostHandle`].

use alloc::sync::Arc;

use crate::container::ZohRescaledParams;
use crate::dynamic::DynamicDnamEngine;
use crate::host::spsc::spsc_control_channel;
use crate::host::swap::realtime_engine_swap;
use crate::host::telemetry::{RawTelemetryRecord, TelemetryReader, telemetry_channel};
use crate::math::BackendKind;
use crate::provenance::VerificationStatus;
use crate::schema::{DnamError, DnamProtectionProfile};
use crate::types::{DnamTelemetrySnapshot, Tier};

use super::{AudioContext, DnamHostHandle, HostMutex, ProducerContext};

impl DnamHostHandle {
    /// Wraps an owned [`DynamicDnamEngine<'static>`] in a [`DnamHostHandle`].
    pub fn new(engine: DynamicDnamEngine<'static>) -> Self {
        let initial_record = RawTelemetryRecord::from_engine(&engine, 0);
        let (audio_telemetry_writer, audio_telemetry_reader) = telemetry_channel(initial_record);
        let (shadow_telemetry_writer, shadow_telemetry_reader) = telemetry_channel(initial_record);
        let (producer, consumer) = spsc_control_channel(&engine);
        let (swap_controller, swap_audio) = realtime_engine_swap(engine);

        Self {
            audio: Arc::new(HostMutex::new(AudioContext {
                swap_audio,
                consumer,
                telemetry_writer: audio_telemetry_writer,
            })),
            producer: Arc::new(HostMutex::new(ProducerContext {
                producer,
                swap_controller,
                shadow_telemetry_writer,
            })),
            audio_telemetry_reader,
            shadow_telemetry_reader,
        }
    }

    /// Creates a [`DnamHostHandle`] from `.dnam` container bytes using runtime CPU feature detection.
    pub fn from_bytes_owned(
        bytes: &[u8],
        tier: Tier,
        sample_rate_hz: u32,
    ) -> Result<Self, DnamError> {
        let engine = DynamicDnamEngine::from_bytes_owned(bytes, tier, sample_rate_hz)?;
        Ok(Self::new(engine))
    }

    /// Creates a [`DnamHostHandle`] capped at `max_supported_tier` from `.dnam` container bytes.
    pub fn from_bytes_owned_with_max_tier(
        bytes: &[u8],
        tier: Tier,
        max_supported_tier: Tier,
        sample_rate_hz: u32,
    ) -> Result<Self, DnamError> {
        let engine = DynamicDnamEngine::from_bytes_owned_with_max_tier(
            bytes,
            tier,
            max_supported_tier,
            sample_rate_hz,
        )?;
        Ok(Self::new(engine))
    }

    #[inline]
    pub(crate) fn try_drain_if_audio_idle(&self) {
        if let Some(mut audio_ctx) = self.audio.try_lock() {
            audio_ctx.drain_and_publish();
        }
    }

    /// Explicitly drains any pending SPSC control updates into the audio engine if the audio
    /// lock can be acquired (blocks until the current audio block completes).
    pub fn flush_pending_controls(&self) {
        let mut audio_ctx = self.audio.lock();
        audio_ctx.drain_and_publish();
    }

    /// Returns a cloneable wait-free [`TelemetryReader`] for the audio thread's Seqlock cell.
    #[inline]
    pub fn telemetry_reader(&self) -> TelemetryReader {
        self.audio_telemetry_reader.clone()
    }

    /// Reads the latest [`RawTelemetryRecord`] wait-free without acquiring any mutex.
    ///
    /// If control mutations have been accepted on the producer side more recently than the
    /// last audio block drain (`shadow.causal_epoch > audio.causal_epoch`), overlays the
    /// accepted control-plane state onto the audio thread's real-time DSP telemetry.
    #[inline]
    pub fn telemetry_record(&self) -> RawTelemetryRecord {
        let audio_rec = self.audio_telemetry_reader.read();
        let shadow_rec = self.shadow_telemetry_reader.read();
        if shadow_rec.causal_epoch > audio_rec.causal_epoch {
            let mut merged = shadow_rec;
            merged.snapshot.max_op_n_drive = audio_rec.snapshot.max_op_n_drive;
            merged.snapshot.max_pole_radius = audio_rec.snapshot.max_pole_radius;
            merged.snapshot.max_effective_loop_gain = audio_rec.snapshot.max_effective_loop_gain;
            merged.snapshot.z_head = audio_rec.snapshot.z_head;
            merged.snapshot.env_states = audio_rec.snapshot.env_states;
            merged.snapshot.lfo_harmonics = audio_rec.snapshot.lfo_harmonics;
            merged.snapshot.overflow_or_saturation_flags =
                audio_rec.snapshot.overflow_or_saturation_flags;
            merged
        } else {
            audio_rec
        }
    }

    /// Reads the latest C-ABI [`DnamTelemetrySnapshot`] wait-free without acquiring any mutex.
    #[inline]
    pub fn telemetry_snapshot(&self) -> DnamTelemetrySnapshot {
        self.telemetry_record().snapshot
    }

    /// Returns the active [`BackendKind`] wait-free without acquiring any mutex.
    #[inline]
    pub fn backend_kind(&self) -> BackendKind {
        self.telemetry_record().backend_kind
    }

    /// Returns the container's [`VerificationStatus`].
    #[inline]
    pub fn verification_status(&self) -> VerificationStatus {
        self.producer
            .lock()
            .producer
            .shadow()
            .verification_status()
            .clone()
    }

    /// Returns the container's [`DnamProtectionProfile`].
    #[inline]
    pub fn protection_profile(&self) -> DnamProtectionProfile {
        self.producer.lock().producer.shadow().protection_profile()
    }

    /// Returns the requested compute [`Tier`] wait-free.
    #[inline]
    pub fn tier(&self) -> Tier {
        self.telemetry_record().requested_tier
    }

    /// Switches the active compute [`Tier`] without blocking on an active audio block.
    pub fn set_tier(&self, tier: Tier) {
        {
            let mut prod = self.producer.lock();
            prod.producer.set_tier(tier);
            prod.publish_shadow_telemetry(&self.audio_telemetry_reader);
        }
        self.try_drain_if_audio_idle();
    }

    /// Returns the highest [`Tier`] present in the loaded `.dnam` container wait-free.
    #[inline]
    pub fn max_available_tier(&self) -> Tier {
        self.telemetry_record().max_available_tier
    }

    /// Returns the host ceiling [`Tier`] (`max_supported_tier`) wait-free.
    #[inline]
    pub fn max_supported_tier(&self) -> Tier {
        self.telemetry_record().max_supported_tier
    }

    /// Sets the host ceiling [`Tier`] (`max_supported_tier`) without blocking on an active audio block.
    pub fn set_max_supported_tier(&self, max_tier: Tier) {
        {
            let mut prod = self.producer.lock();
            prod.producer.set_max_supported_tier(max_tier);
            prod.publish_shadow_telemetry(&self.audio_telemetry_reader);
        }
        self.try_drain_if_audio_idle();
    }

    /// Returns the effective [`Tier`] wait-free.
    #[inline]
    pub fn effective_tier(&self) -> Tier {
        let rec = self.telemetry_record();
        Tier::from_u8(rec.snapshot.active_tier)
    }

    /// Returns `true` if $4\times$ oversampled Offline Bounce mode is enabled wait-free.
    #[inline]
    pub fn offline_bounce(&self) -> bool {
        self.telemetry_record().offline_bounce
    }

    /// Enables or disables $4\times$ oversampled Offline Bounce mode without blocking on an active audio block.
    pub fn set_offline_bounce(&self, enabled: bool) {
        {
            let mut prod = self.producer.lock();
            prod.producer.set_offline_bounce(enabled);
            prod.publish_shadow_telemetry(&self.audio_telemetry_reader);
        }
        self.try_drain_if_audio_idle();
    }

    /// Returns `true` if `maximize_tier_on_bounce` is enabled wait-free.
    #[inline]
    pub fn maximize_tier_on_bounce(&self) -> bool {
        self.telemetry_record().maximize_tier_on_bounce
    }

    /// Enables or disables automatic tier promotion during Offline Bounce without blocking on an active audio block.
    pub fn set_maximize_tier_on_bounce(&self, enabled: bool) {
        {
            let mut prod = self.producer.lock();
            prod.producer.set_maximize_tier_on_bounce(enabled);
            prod.publish_shadow_telemetry(&self.audio_telemetry_reader);
        }
        self.try_drain_if_audio_idle();
    }

    /// Returns the current Plugin Delay Compensation (`PDC`) latency in host samples (`0` or `12`) wait-free.
    #[inline]
    pub fn latency_samples(&self) -> usize {
        self.telemetry_record().latency_samples
    }

    /// Returns the active host sample rate in Hz wait-free.
    #[inline]
    pub fn sample_rate_hz(&self) -> u32 {
        self.telemetry_record().sample_rate_hz
    }

    /// Returns the active [`ZohRescaledParams`] wait-free.
    #[inline]
    pub fn zoh(&self) -> ZohRescaledParams {
        self.telemetry_record().zoh
    }

    /// Rescales discrete-time parameters for a new host sample rate.
    pub fn set_sample_rate(&self, sample_rate_hz: u32) -> Result<(), DnamError> {
        let mut audio_ctx = self.audio.lock();
        let mut prod = self.producer.lock();
        audio_ctx.poll_and_drain();
        audio_ctx
            .swap_audio
            .engine_mut()
            .set_sample_rate(sample_rate_hz)?;
        prod.producer
            .resync_from_engine(audio_ctx.swap_audio.engine());
        audio_ctx
            .consumer
            .advance_causal_epoch(prod.producer.causal_epoch());
        audio_ctx.publish_telemetry();
        prod.publish_shadow_telemetry(&self.audio_telemetry_reader);
        Ok(())
    }

    /// Hot-reloads a new `.dnam` model by pre-allocating and initializing the replacement
    /// [`DynamicDnamEngine<'static>`] outside the audio lock, then swapping it via
    /// [`crate::host::RealtimeEngineSwap`] and reclaiming the retired engine off the audio thread.
    pub fn reload_model_in_place_owned(
        &self,
        bytes: &[u8],
        tier: Tier,
        sample_rate_hz: u32,
    ) -> Result<(), DnamError> {
        let (max_sup, backend) = {
            let prod = self.producer.lock();
            let s = prod.producer.shadow();
            (s.max_supported_tier(), s.backend_kind())
        };
        let mut next_engine = DynamicDnamEngine::from_bytes_owned_with_backend(
            bytes,
            tier.min(max_sup),
            sample_rate_hz,
            backend,
        )?;
        next_engine.set_max_supported_tier(max_sup);
        {
            let mut prod = self.producer.lock();
            prod.producer.resync_from_engine(&next_engine);
            let epoch = prod.producer.causal_epoch();
            prod.swap_controller
                .publish_engine_at_epoch(next_engine, epoch);
            prod.publish_shadow_telemetry(&self.audio_telemetry_reader);
        }
        if let Some(mut audio_ctx) = self.audio.try_lock() {
            audio_ctx.drain_and_publish();
            let mut prod = self.producer.lock();
            let _ = prod.swap_controller.reclaim_retired();
            prod.publish_shadow_telemetry(&self.audio_telemetry_reader);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::{ARENA_OFFSET_ULTRA_END, build_dnam_container};
    use crate::schema::DnamJsonHeader;
    use alloc::vec;
    use alloc::vec::Vec;
    use proptest::prelude::*;

    fn build_lifecycle_container() -> Vec<u8> {
        let mut header = DnamJsonHeader::default_reference();
        let arena = vec![0_u8; ARENA_OFFSET_ULTRA_END as usize];
        build_dnam_container(&mut header, &arena, &arena, false).unwrap()
    }

    #[test]
    fn test_host_lifecycle_constructors_tiers_sample_rate_and_reload() {
        // verify[dsp.ctrl.op-m.spsc-ingress]
        // verify[arch.telemetry.seqlock-publication]
        let container = build_lifecycle_container();

        assert!(DnamHostHandle::from_bytes_owned(&[0u8; 8], Tier::Lite, 48_000).is_err());
        assert!(
            DnamHostHandle::from_bytes_owned_with_max_tier(
                &[0u8; 8],
                Tier::Lite,
                Tier::Pro,
                48_000
            )
            .is_err()
        );

        let host_full = DnamHostHandle::from_bytes_owned(&container, Tier::Lite, 48_000).unwrap();
        assert_eq!(host_full.max_available_tier(), Tier::Ultra);

        let host = DnamHostHandle::from_bytes_owned_with_max_tier(
            &container,
            Tier::Pro,
            Tier::Pro,
            48_000,
        )
        .unwrap();
        assert_eq!(host.tier(), Tier::Pro);
        assert_eq!(host.max_available_tier(), Tier::Pro);
        assert_eq!(host.max_supported_tier(), Tier::Pro);
        assert_eq!(host.effective_tier(), Tier::Pro);
        assert_eq!(host.sample_rate_hz(), 48_000);
        let _ = host.zoh();
        let _ = host.verification_status();
        let _ = host.protection_profile();
        let _ = host.backend_kind();
        let _ = host.telemetry_reader();

        host.set_max_supported_tier(Tier::Ultra);
        host.set_tier(Tier::Ultra);
        assert_eq!(host.tier(), Tier::Pro);
        host_full.set_tier(Tier::Ultra);
        assert_eq!(host_full.tier(), Tier::Ultra);
        host.set_offline_bounce(true);
        host.set_maximize_tier_on_bounce(false);
        assert!(host.offline_bounce());
        assert!(!host.maximize_tier_on_bounce());
        assert_eq!(host.latency_samples(), 12);

        host.set_sample_rate(96_000).unwrap();
        assert_eq!(host.sample_rate_hz(), 96_000);
        assert!(host.set_sample_rate(1_000).is_err());

        host.reload_model_in_place_owned(&container, Tier::Lite, 48_000)
            .unwrap();
        assert_eq!(host.tier(), Tier::Lite);
        assert_eq!(host.sample_rate_hz(), 48_000);
        assert!(
            host.reload_model_in_place_owned(&[0u8; 8], Tier::Lite, 48_000)
                .is_err()
        );

        // Hold audio lock to exercise shadow_rec.causal_epoch > audio_rec.causal_epoch overlay
        {
            let _guard = host.audio.lock();
            host.set_tier(Tier::Pro);
            let rec = host.telemetry_record();
            assert_eq!(rec.requested_tier, Tier::Pro);
        }
        host.flush_pending_controls();
        assert_eq!(host.telemetry_snapshot().active_tier, Tier::Pro as u8);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(16))]

        #[test]
        fn prop_host_lifecycle_tiers_and_sample_rates(
            tier_u8 in 0u8..4,
            max_sup_u8 in 0u8..4,
            bounce in any::<bool>(),
            max_on_bounce in any::<bool>(),
        ) {
            // verify[arch.telemetry.seqlock-publication]
            let container = build_lifecycle_container();
            let tier = Tier::from_u8(tier_u8);
            let max_sup = Tier::from_u8(max_sup_u8);
            let host = DnamHostHandle::from_bytes_owned_with_max_tier(
                &container,
                tier,
                max_sup,
                48_000,
            )
            .unwrap();
            host.set_offline_bounce(bounce);
            host.set_maximize_tier_on_bounce(max_on_bounce);

            prop_assert_eq!(host.tier(), tier.min(max_sup));
            prop_assert_eq!(host.max_supported_tier(), max_sup);
            prop_assert_eq!(host.offline_bounce(), bounce);
            prop_assert_eq!(host.maximize_tier_on_bounce(), max_on_bounce);
            prop_assert_eq!(host.latency_samples(), if bounce { 12 } else { 0 });
        }
    }
}
