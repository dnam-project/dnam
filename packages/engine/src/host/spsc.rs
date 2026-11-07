//! 3-Lane Lock-Free SPSC Control Ingress (`SpscControlRing`, `ControlProducer`, `ControlConsumer`)
//! and synchronous producer-side state shadow (`AcceptedStateShadow`) (`ref[dsp.ctrl.op-m.spsc-ingress]`).
//!
//! Implemented in 100% safe Rust (`#[forbid(unsafe_code)]`) using pure `core::sync::atomic` primitives:
//! - **Lane 1 (`spsc::ring`):** 32 atomic `u64` slots (`MAX_CONTROLS = 32`, plus overflow slots for
//!   models up to 255 controls) packing `(epoch_u29, mode_u2, kind_u1, payload_u32)`.
//! - **Lane 2 (`spsc::ring`):** Fixed 64-slot ring of `[AtomicU64; 2]` entries for discrete state
//!   transitions (`ApplyPreset`, `SyncTempo`, `TapTempo`, `SetBypass`, `SetFreeze`, `SetTier`, etc.).
//! - **Lane 3 (`spsc::ring`):** Fixed `[AtomicU64; 20]` Seqlock latch so if more than 64 discrete
//!   commands arrive before the audio thread ticks once, state coalesces without dropping commands
//!   or blocking the producer.
//! - **Shadow (`spsc::shadow`):** [`AcceptedStateShadow`] mirrors clamping, preset resolution,
//!   tempo-sync, and host configuration on the producer side.
//! - **Producer & Consumer (`spsc::producer`, `spsc::consumer`):** [`ControlProducer`] and
//!   [`ControlConsumer`].

pub mod consumer;
pub mod producer;
pub mod ring;
pub mod shadow;

use alloc::sync::Arc;

use crate::dynamic::DynamicDnamEngine;

pub use consumer::{ControlConsumer, DrainedBounceOverrides};
pub use producer::ControlProducer;
pub use ring::{
    ControlCommand, ControlUpdateMode, ControlValueKind, FallbackSnapshot, Lane1Entry,
    MAX_CONTROLS, RingEntry, SPSC_RING_CAPACITY, SpscControlRing, pack_lane1_slot,
    unpack_lane1_slot,
};
pub use shadow::AcceptedStateShadow;

/// Creates a paired `(ControlProducer, ControlConsumer)` initialized from `engine`
/// (`ref[dsp.ctrl.op-m.spsc-ingress]`).
pub fn spsc_control_channel(engine: &DynamicDnamEngine<'_>) -> (ControlProducer, ControlConsumer) {
    let ring = Arc::new(SpscControlRing::new(
        engine.num_controls(),
        engine.tempo_total_frames(),
    ));
    let shadow = AcceptedStateShadow::from_engine(engine);
    (
        ControlProducer {
            ring: Arc::clone(&ring),
            shadow,
            next_epoch_29: 1,
            causal_epoch: 0,
        },
        ControlConsumer {
            ring,
            drained_causal_epoch: 0,
        },
    )
}
