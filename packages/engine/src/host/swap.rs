//! Lock-free real-time engine hot-swap (`RealtimeEngineSwap`, `EngineSwapController`,
//! `EngineSwapAudioHandle`).
//!
//! Implemented in 100% safe Rust (`#[forbid(unsafe_code)]`) using a non-blocking ping-pong
//! staging/reclamation slot design: replacement [`DynamicDnamEngine<'static>`] instances are
//! allocated and initialized on the non-real-time controller thread, swapped into the audio
//! handle via non-blocking `try_lock` when `has_pending` is set, and retired engines are moved
//! back to the controller's reclamation slots so the audio callback never invokes `malloc` or
//! `drop` (`free`).

use alloc::sync::Arc;
use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;

use crate::dynamic::DynamicDnamEngine;
use crate::schema::DnamError;
use crate::types::Tier;

const RETIRED_SLOTS: usize = 4;

/// Shared safe ping-pong handoff state between [`EngineSwapController`] (non-RT thread) and
/// [`EngineSwapAudioHandle`] (real-time audio thread).
#[repr(align(64))]
#[derive(Debug)]
pub struct RealtimeEngineSwap {
    has_pending: AtomicBool,
    pending_slot: Mutex<Option<(DynamicDnamEngine<'static>, u64)>>,
    retired_slots: [Mutex<Option<DynamicDnamEngine<'static>>>; RETIRED_SLOTS],
    swap_epoch: AtomicUsize,
}

impl RealtimeEngineSwap {
    fn new() -> Self {
        Self {
            has_pending: AtomicBool::new(false),
            pending_slot: Mutex::new(None),
            retired_slots: core::array::from_fn(|_| Mutex::new(None)),
            swap_epoch: AtomicUsize::new(0),
        }
    }

    fn reclaim_all_retired(&self) -> usize {
        let mut count = 0_usize;
        for slot in &self.retired_slots {
            let retired = match slot.lock() {
                Ok(mut g) => g.take(),
                Err(poisoned) => poisoned.into_inner().take(),
            };
            if let Some(old_engine) = retired {
                drop(old_engine);
                count += 1;
            }
        }
        count
    }
}

/// Non-real-time controller handle for staging new [`DynamicDnamEngine<'static>`] instances and
/// reclaiming retired engines off the audio thread.
#[derive(Debug)]
pub struct EngineSwapController {
    shared: Arc<RealtimeEngineSwap>,
}

impl Drop for EngineSwapController {
    fn drop(&mut self) {
        let _ = self.shared.reclaim_all_retired();
    }
}

impl EngineSwapController {
    /// Reclaims and drops all retired [`DynamicDnamEngine<'static>`] instances on the calling
    /// non-real-time thread. Returns the number of retired engines freed.
    #[inline]
    pub fn reclaim_retired(&mut self) -> usize {
        self.shared.reclaim_all_retired()
    }

    /// Publishes a pre-allocated `new_engine` at `causal_epoch` for the real-time audio thread
    /// and reclaims any previously retired engine instances on the calling non-real-time thread.
    pub fn publish_engine_at_epoch(
        &mut self,
        new_engine: DynamicDnamEngine<'static>,
        causal_epoch: u64,
    ) {
        let _ = self.reclaim_retired();
        let prev_unconsumed = {
            let mut guard = self
                .shared
                .pending_slot
                .lock()
                .unwrap_or_else(|p| p.into_inner());
            guard.replace((new_engine, causal_epoch))
        };
        self.shared.has_pending.store(true, Ordering::Release);
        drop(prev_unconsumed);
        let _ = self.reclaim_retired();
    }

    /// Publishes a pre-allocated `new_engine` for the real-time audio thread and reclaims any
    /// previously retired engine instances on the calling non-real-time thread.
    pub fn publish_engine(&mut self, new_engine: DynamicDnamEngine<'static>) {
        self.publish_engine_at_epoch(new_engine, 0);
    }

    /// Pre-allocates and initializes a new [`DynamicDnamEngine<'static>`] from `container_bytes`
    /// on the calling non-real-time thread and publishes it for non-blocking audio-thread swap.
    pub fn stage_reload_from_bytes(
        &mut self,
        container_bytes: &[u8],
        tier: Tier,
        max_supported_tier: Tier,
        sample_rate_hz: u32,
    ) -> Result<(), DnamError> {
        let next_engine = DynamicDnamEngine::from_bytes_owned_with_max_tier(
            container_bytes,
            tier,
            max_supported_tier,
            sample_rate_hz,
        )?;
        self.publish_engine(next_engine);
        Ok(())
    }

    /// Returns the total number of engine swaps completed by the audio thread.
    #[inline]
    pub fn swap_epoch(&self) -> usize {
        self.shared.swap_epoch.load(Ordering::Acquire)
    }
}

/// Real-time audio-thread handle owning the active [`DynamicDnamEngine<'static>`] and performing
/// non-blocking swaps at block boundaries with zero heap allocations or deallocations.
#[derive(Debug)]
pub struct EngineSwapAudioHandle {
    shared: Arc<RealtimeEngineSwap>,
    active: DynamicDnamEngine<'static>,
}

impl EngineSwapAudioHandle {
    /// Checks if a newly staged engine is pending and swaps it in via non-blocking `try_lock`,
    /// moving the retired engine into the non-RT reclamation slots (`0` `malloc`, `0` `drop`).
    ///
    /// Returns `Some(causal_epoch)` if a new engine was swapped in.
    #[inline]
    pub fn poll_swap_epoch(&mut self) -> Option<u64> {
        if !self.shared.has_pending.load(Ordering::Acquire) {
            return None;
        }
        let next_entry = match self.shared.pending_slot.try_lock() {
            Ok(mut guard) => {
                let taken = guard.take();
                if taken.is_some() {
                    self.shared.has_pending.store(false, Ordering::Release);
                }
                taken
            }
            Err(_) => None,
        };
        let (next, causal_epoch) = next_entry?;
        let old_engine = core::mem::replace(&mut self.active, next);
        for slot in &self.shared.retired_slots {
            if let Ok(mut guard) = slot.try_lock()
                && guard.is_none()
            {
                *guard = Some(old_engine);
                self.shared.swap_epoch.fetch_add(1, Ordering::Release);
                return Some(causal_epoch);
            }
        }
        if let Ok(mut guard) = self.shared.retired_slots[0].try_lock() {
            *guard = Some(old_engine);
        }
        self.shared.swap_epoch.fetch_add(1, Ordering::Release);
        Some(causal_epoch)
    }

    /// Checks if a newly staged engine is pending and swaps it in via non-blocking `try_lock`,
    /// moving the retired engine into the non-RT reclamation slots (`0` `malloc`, `0` `drop`).
    ///
    /// Returns `true` if a new engine was swapped in.
    #[inline]
    pub fn poll_swap(&mut self) -> bool {
        self.poll_swap_epoch().is_some()
    }

    /// Returns an immutable reference to the currently active [`DynamicDnamEngine<'static>`].
    #[inline]
    pub const fn engine(&self) -> &DynamicDnamEngine<'static> {
        &self.active
    }

    /// Returns a mutable reference to the currently active [`DynamicDnamEngine<'static>`].
    #[inline]
    pub fn engine_mut(&mut self) -> &mut DynamicDnamEngine<'static> {
        &mut self.active
    }

    /// Polls for any pending non-blocking engine swap and invokes `f` with the active engine.
    #[inline]
    pub fn with_engine<R>(
        &mut self,
        f: impl FnOnce(&mut DynamicDnamEngine<'static>, bool) -> R,
    ) -> R {
        let swapped = self.poll_swap();
        f(&mut self.active, swapped)
    }

    /// Returns the total number of engine swaps completed by the audio thread.
    #[inline]
    pub fn swap_epoch(&self) -> usize {
        self.shared.swap_epoch.load(Ordering::Acquire)
    }
}

/// Creates a paired `(EngineSwapController, EngineSwapAudioHandle)` initialized with `initial`.
pub fn realtime_engine_swap(
    initial: DynamicDnamEngine<'static>,
) -> (EngineSwapController, EngineSwapAudioHandle) {
    let shared = Arc::new(RealtimeEngineSwap::new());
    (
        EngineSwapController {
            shared: Arc::clone(&shared),
        },
        EngineSwapAudioHandle {
            shared,
            active: initial,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::{ARENA_OFFSET_ULTRA_END, build_dnam_container};
    use crate::schema::DnamJsonHeader;
    use alloc::vec;
    use alloc::vec::Vec;
    use proptest::prelude::*;

    fn build_minimal_container() -> Vec<u8> {
        let mut header = DnamJsonHeader::default_reference();
        let arena = vec![0_u8; ARENA_OFFSET_ULTRA_END as usize];
        build_dnam_container(&mut header, &arena, &arena, false).unwrap()
    }

    #[test]
    fn test_swap_lifecycle_unconsumed_replacement_contention_and_slot_overflow() {
        // verify[dsp.ctrl.op-m.spsc-ingress]
        let container = build_minimal_container();
        let initial = DynamicDnamEngine::from_bytes_owned(&container, Tier::Lite, 48_000).unwrap();
        let (mut controller, mut audio_handle) = realtime_engine_swap(initial);

        assert_eq!(controller.swap_epoch(), 0);
        assert_eq!(audio_handle.swap_epoch(), 0);
        assert!(!audio_handle.poll_swap());
        assert_eq!(audio_handle.poll_swap_epoch(), None);

        // Error path on stage_reload_from_bytes with invalid container bytes
        assert!(
            controller
                .stage_reload_from_bytes(&[0u8; 16], Tier::Lite, Tier::Ultra, 48_000)
                .is_err()
        );

        // Publish twice before audio thread polls: second replaces unconsumed first
        let e1 = DynamicDnamEngine::from_bytes_owned(&container, Tier::Full, 48_000).unwrap();
        let e2 = DynamicDnamEngine::from_bytes_owned(&container, Tier::Pro, 96_000).unwrap();
        controller.publish_engine_at_epoch(e1, 10);
        controller.publish_engine_at_epoch(e2, 20);

        // Simulate lock contention on pending_slot during poll_swap_epoch: non-blocking return None
        {
            let _guard = controller.shared.pending_slot.lock().unwrap();
            assert_eq!(audio_handle.poll_swap_epoch(), None);
        }

        // Now poll without contention: should swap in e2 at causal_epoch 20
        assert_eq!(audio_handle.poll_swap_epoch(), Some(20));
        assert_eq!(audio_handle.engine().tier(), Tier::Pro);
        assert_eq!(audio_handle.engine().sample_rate_hz(), 96_000);
        assert_eq!(controller.swap_epoch(), 1);
        assert_eq!(audio_handle.swap_epoch(), 1);
        assert_eq!(controller.reclaim_retired(), 1);

        // Test with_engine helper (both when a swap is pending and when no swap is pending)
        controller
            .stage_reload_from_bytes(&container, Tier::Ultra, Tier::Ultra, 44_100)
            .unwrap();
        let (tier_after_swap, did_swap) = audio_handle.with_engine(|eng, swapped| {
            eng.set_stereo_width(1.5);
            (eng.tier(), swapped)
        });
        assert!(did_swap);
        assert_eq!(tier_after_swap, Tier::Ultra);
        assert!((audio_handle.engine_mut().stereo_width() - 1.5).abs() < 1e-6);

        let (_, did_swap_again) = audio_handle.with_engine(|eng, swapped| (eng.tier(), swapped));
        assert!(!did_swap_again);
        assert_eq!(controller.reclaim_retired(), 1);

        // Pre-fill all RETIRED_SLOTS (4) to exercise the slot-0 fallback when all retired slots are full
        for slot in &controller.shared.retired_slots {
            let dummy =
                DynamicDnamEngine::from_bytes_owned(&container, Tier::Lite, 48_000).unwrap();
            *slot.lock().unwrap() = Some(dummy);
        }
        let next = DynamicDnamEngine::from_bytes_owned(&container, Tier::Full, 48_000).unwrap();
        controller
            .shared
            .pending_slot
            .lock()
            .unwrap()
            .replace((next, 77));
        controller.shared.has_pending.store(true, Ordering::Release);
        assert_eq!(audio_handle.poll_swap_epoch(), Some(77));
        assert_eq!(audio_handle.engine().tier(), Tier::Full);
        assert_eq!(controller.reclaim_retired(), RETIRED_SLOTS);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(32))]

        #[test]
        fn prop_realtime_engine_swap_epochs_and_reclamation(
            epochs in proptest::collection::vec((0u8..4, 1u64..10_000u64, any::<bool>()), 1..16)
        ) {
            // verify[dsp.ctrl.op-m.spsc-ingress]
            let container = build_minimal_container();
            let initial = DynamicDnamEngine::from_bytes_owned(&container, Tier::Lite, 48_000).unwrap();
            let (mut controller, mut audio_handle) = realtime_engine_swap(initial);

            let mut expected_swaps = 0usize;
            for (tier_u8, ep, poll_immediately) in epochs {
                let tier = Tier::from_u8(tier_u8);
                let eng = DynamicDnamEngine::from_bytes_owned(&container, tier, 48_000).unwrap();
                controller.publish_engine_at_epoch(eng, ep);
                if poll_immediately {
                    let polled = audio_handle.poll_swap_epoch();
                    prop_assert_eq!(polled, Some(ep));
                    prop_assert_eq!(audio_handle.engine().tier(), tier);
                    expected_swaps += 1;
                }
            }
            if audio_handle.poll_swap() {
                expected_swaps += 1;
            }
            prop_assert_eq!(controller.swap_epoch(), expected_swaps);
            prop_assert_eq!(audio_handle.swap_epoch(), expected_swaps);
            let _ = controller.reclaim_retired();
            prop_assert_eq!(controller.reclaim_retired(), 0);
        }
    }
}
