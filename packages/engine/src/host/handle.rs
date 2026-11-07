//! Unified real-time host handle (`DnamHostHandle`) combining the 3-Lane SPSC control ingress,
//! [`crate::host::AcceptedStateShadow`], wait-free [`crate::host::TelemetryCell`], and
//! non-blocking [`crate::host::RealtimeEngineSwap`].
//!
//! Decomposed into focused submodules:
//! - [`lifecycle`]: Initialization, hot-reload, sample-rate reconfiguration, and wait-free telemetry.
//! - [`controls`]: Non-blocking SPSC control, preset, tempo, bypass, freeze, and stereo-width delegation.
//! - [`process`]: Sub-block audio rendering (`process_*`) and offline bounce (`render_offline_bounce*`).

pub mod controls;
pub mod lifecycle;
pub mod process;

use alloc::sync::Arc;
use std::sync::{Mutex, MutexGuard, TryLockError};

use crate::host::spsc::{ControlConsumer, ControlProducer};
use crate::host::swap::{EngineSwapAudioHandle, EngineSwapController};
use crate::host::telemetry::{TelemetryReader, TelemetryWriter};

/// Default sub-block quantum (`512` frames $\approx 10.67\text{ ms}$ at $48\text{ kHz}$) at which
/// long `process_*` or `render_offline_bounce` calls drain the SPSC control ring and publish
/// updated telemetry snapshots.
pub const DEFAULT_HOST_SUBBLOCK_FRAMES: usize = 512;

#[derive(Debug)]
pub(crate) struct HostMutex<T> {
    inner: Mutex<T>,
}

impl<T> HostMutex<T> {
    #[inline]
    pub(crate) fn new(value: T) -> Self {
        Self {
            inner: Mutex::new(value),
        }
    }

    #[inline]
    pub(crate) fn lock(&self) -> MutexGuard<'_, T> {
        self.inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    #[inline]
    pub(crate) fn try_lock(&self) -> Option<MutexGuard<'_, T>> {
        match self.inner.try_lock() {
            Ok(guard) => Some(guard),
            Err(TryLockError::Poisoned(p)) => Some(p.into_inner()),
            Err(TryLockError::WouldBlock) => None,
        }
    }
}

#[derive(Debug)]
pub(crate) struct AudioContext {
    pub(crate) swap_audio: EngineSwapAudioHandle,
    pub(crate) consumer: ControlConsumer,
    pub(crate) telemetry_writer: TelemetryWriter,
}

impl AudioContext {
    #[inline]
    pub(crate) fn poll_and_drain(&mut self) {
        if let Some(swap_epoch) = self.swap_audio.poll_swap_epoch() {
            self.consumer.advance_causal_epoch(swap_epoch);
        }
        let engine = self.swap_audio.engine_mut();
        let _ = self.consumer.drain_into(engine);
    }

    #[inline]
    pub(crate) fn publish_telemetry(&mut self) {
        let epoch = self.consumer.drained_causal_epoch();
        let engine = self.swap_audio.engine();
        self.consumer.publish_total_frames(engine);
        self.telemetry_writer.publish_from_engine(engine, epoch);
    }

    #[inline]
    pub(crate) fn drain_and_publish(&mut self) {
        self.poll_and_drain();
        self.publish_telemetry();
    }
}

#[derive(Debug)]
pub(crate) struct ProducerContext {
    pub(crate) producer: ControlProducer,
    pub(crate) swap_controller: EngineSwapController,
    pub(crate) shadow_telemetry_writer: TelemetryWriter,
}

impl ProducerContext {
    #[inline]
    pub(crate) fn publish_shadow_telemetry(&mut self, audio_reader: &TelemetryReader) {
        let base = audio_reader.read();
        let epoch = self.producer.causal_epoch();
        let rec = self.producer.shadow().overlay_telemetry(base, epoch);
        self.shadow_telemetry_writer.publish(rec);
    }
}

/// Unified lock-free real-time host handle (`dnam_engine::host::DnamHostHandle`).
///
/// Decouples real-time audio block processing (`AudioContext`) from control mutations
/// (`ProducerContext` + 3-Lane [`crate::host::SpscControlRing`]) and wait-free telemetry
/// inspection ([`TelemetryReader`]), so UI, Python, and JS host threads never block on an
/// active `process_*` or `render_offline_bounce` call (`ref[dsp.ctrl.op-m.spsc-ingress]`,
/// `ref[arch.telemetry.seqlock-publication]`).
#[derive(Clone, Debug)]
pub struct DnamHostHandle {
    pub(crate) audio: Arc<HostMutex<AudioContext>>,
    pub(crate) producer: Arc<HostMutex<ProducerContext>>,
    pub(crate) audio_telemetry_reader: TelemetryReader,
    pub(crate) shadow_telemetry_reader: TelemetryReader,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_host_mutex_lock_try_lock_and_poison_recovery() {
        // verify[dsp.ctrl.op-m.spsc-ingress]
        let mtx = Arc::new(HostMutex::new(10_u32));
        {
            let guard = mtx.lock();
            assert_eq!(*guard, 10);
            assert!(mtx.try_lock().is_none());
        }
        {
            let mut guard = mtx.try_lock().unwrap();
            *guard = 25;
        }
        assert_eq!(*mtx.lock(), 25);

        let mtx_poison = Arc::clone(&mtx);
        let _ = std::thread::spawn(move || {
            let mut guard = mtx_poison.lock();
            *guard = 42;
            panic!("intentional poison");
        })
        .join();

        assert_eq!(*mtx.lock(), 42);
        assert_eq!(*mtx.try_lock().unwrap(), 42);
    }
}
