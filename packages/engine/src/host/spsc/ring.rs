//! 3-Lane SPSC atomic ring memory (`SpscControlRing`) and word-packing codecs for Lane 1
//! (`Lane1Entry`), Lane 2 (`ControlCommand`, `RingEntry`), and Lane 3 (`FallbackSnapshot`)
//! (`ref[dsp.ctrl.op-m.spsc-ingress]`).

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};

use crate::schema::TempoSubdivision;
use crate::types::Tier;

/// Number of fast inline atomic parameter mailbox slots in Lane 1 (`ref[dsp.ctrl.op-m.spsc-ingress]`).
pub const MAX_CONTROLS: usize = 32;

/// Capacity of the cache-line-aligned Lane 2 SPSC command ring (`ref[dsp.ctrl.op-m.spsc-ingress]`).
pub const SPSC_RING_CAPACITY: usize = 64;

pub(crate) const SPSC_RING_MASK: usize = SPSC_RING_CAPACITY - 1;
pub(crate) const MAX_EPOCH_29: u32 = (1 << 29) - 2;
pub(crate) const FALLBACK_WORDS: usize = 20;

/// Parameter update interpolation mode for Lane 1 (`Slewed` vs `Immediate`).
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ControlUpdateMode {
    /// Smooth $3\text{ kHz}$ ZOH anti-zipper interpolation (`set_control_value`).
    Slewed = 1,
    /// Immediate snap of both target and smoothed state (`set_control_immediate`).
    Immediate = 2,
}

/// Parameter value encoding kind for Lane 1 (`Range` float bits vs `Switch` option index).
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ControlValueKind {
    /// Continuous physical range value (`f32::to_bits()`).
    Range = 0,
    /// Discrete switch option index (`u32`).
    Switch = 1,
}

/// Unpacked Lane 1 atomic parameter mailbox word (`(epoch_u29, mode_u2, kind_u1, payload_u32)`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Lane1Entry {
    /// 29-bit monotonic causal epoch (`1..=(1 << 29) - 1`).
    pub epoch: u32,
    /// Slew vs Immediate update mode.
    pub mode: ControlUpdateMode,
    /// Continuous range vs discrete switch kind.
    pub kind: ControlValueKind,
    /// Raw 32-bit payload (`f32::to_bits()` for `Range`, or `u32` option index for `Switch`).
    pub payload_bits: u32,
}

impl Lane1Entry {
    /// Converts the payload back into the physical `f32` value expected by `DynamicDnamEngine`.
    #[inline]
    pub fn physical_value(self) -> f32 {
        match self.kind {
            ControlValueKind::Range => f32::from_bits(self.payload_bits),
            ControlValueKind::Switch => self.payload_bits as f32,
        }
    }
}

/// Packs `(epoch_u29, mode_u2, kind_u1, payload_u32)` into a single 64-bit Lane 1 atomic word.
#[inline]
pub const fn pack_lane1_slot(
    epoch: u32,
    mode: ControlUpdateMode,
    kind: ControlValueKind,
    payload_bits: u32,
) -> u64 {
    let ep = (epoch & 0x1FFF_FFFF) as u64;
    let md = (mode as u64) & 0x3;
    let kd = (kind as u64) & 0x1;
    (ep << 35) | (md << 33) | (kd << 32) | (payload_bits as u64)
}

/// Unpacks a 64-bit Lane 1 atomic word into a [`Lane1Entry`], returning `None` if `word == 0`.
#[inline]
pub const fn unpack_lane1_slot(word: u64) -> Option<Lane1Entry> {
    if word == 0 {
        return None;
    }
    let epoch = ((word >> 35) & 0x1FFF_FFFF) as u32;
    let mode_bits = ((word >> 33) & 0x3) as u8;
    let kind_bits = ((word >> 32) & 0x1) as u8;
    let payload_bits = (word & 0xFFFF_FFFF) as u32;
    let mode = match mode_bits {
        2 => ControlUpdateMode::Immediate,
        _ => ControlUpdateMode::Slewed,
    };
    let kind = if kind_bits == 1 {
        ControlValueKind::Switch
    } else {
        ControlValueKind::Range
    };
    Some(Lane1Entry {
        epoch,
        mode,
        kind,
        payload_bits,
    })
}

/// Discrete state-transition commands queued in Lane 2 (`64`-slot SPSC ring).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ControlCommand {
    /// Applies a pre-validated preset by its index in `header.presets`.
    ApplyPreset {
        /// Index into `DnamJsonHeader::presets`.
        preset_index: u16,
    },
    /// Synchronizes tempo-calibrated controls to `bpm` and optional `subdivision`.
    SyncTempo {
        /// Target tempo in BPM.
        bpm: f32,
        /// Optional rhythmic subdivision override.
        subdivision: Option<TempoSubdivision>,
        /// Whether to fold out-of-range delays/rates by octaves.
        fold_octave: bool,
    },
    /// Applies a tap-tempo update resolved on the producer side.
    TapTempo {
        /// Resolved BPM from the tap-tempo estimator.
        resolved_bpm: f32,
        /// Optional rhythmic subdivision override.
        subdivision: Option<TempoSubdivision>,
        /// Whether to fold out-of-range delays/rates by octaves.
        fold_octave: bool,
    },
    /// Sets bypass state and spillover trails flag.
    SetBypass {
        /// Whether the engine is bypassed.
        bypassed: bool,
        /// Whether wet reverb/delay trails continue ringing while bypassed.
        trails: bool,
    },
    /// Engages or disengages Unitary Spectral Freeze.
    SetFreeze {
        /// Whether freeze is engaged.
        frozen: bool,
    },
    /// Sets continuous Unitary Spectral Freeze amount $\alpha_{\text{freeze}} \in [0.0, 1.0]$.
    SetFreezeAmount {
        /// Freeze blend factor (`> 0.5` engages freeze).
        alpha: f32,
    },
    /// Sets output Mid/Side stereo width $\alpha_{\text{width}} \in [0.0, 2.0]$.
    SetStereoWidth {
        /// Stereo width factor.
        width: f32,
    },
    /// Configures host dry/wet blend, output gain trim (dB), and unity loudness leveling.
    ConfigureHostMix {
        /// Wet blend ratio in `[0.0, 1.0]`.
        blend_wet: f32,
        /// Output gain trim in dB.
        output_gain_db: f32,
        /// Whether to apply calibrated unity loudness trim.
        apply_unity_leveling: bool,
    },
    /// Switches the active compute [`Tier`].
    SetTier {
        /// Target compute tier.
        tier: Tier,
    },
    /// Sets the host ceiling [`Tier`] (`max_supported_tier`).
    SetMaxSupportedTier {
        /// Maximum supported compute tier.
        tier: Tier,
    },
    /// Enables or disables $4\times$ oversampled Offline Bounce mode.
    SetOfflineBounce {
        /// Whether offline bounce mode is enabled.
        enabled: bool,
    },
    /// Enables or disables automatic tier promotion during Offline Bounce.
    SetMaximizeTierOnBounce {
        /// Whether to promote to `min(max_available_tier, max_supported_tier)` on bounce.
        enabled: bool,
    },
    /// Resets a single control `index` immediately to its schema default.
    ResetControl {
        /// Control index.
        index: u16,
    },
    /// Resets all controls immediately to their schema defaults (`reset_controls_to_defaults`).
    ResetAllControls,
    /// Resets all DSP states to analytical DC resting equilibrium (`reset_to_dc_equilibrium`).
    ResetEngine,
}

#[inline]
pub(crate) fn encode_subdiv(sub: Option<TempoSubdivision>) -> u8 {
    match sub {
        None => 0,
        Some(s) => TempoSubdivision::ALL
            .iter()
            .position(|&x| x == s)
            .map_or(0, |i| (i + 1) as u8),
    }
}

#[inline]
pub(crate) fn decode_subdiv(code: u8) -> Option<TempoSubdivision> {
    if code == 0 {
        None
    } else {
        TempoSubdivision::ALL.get((code - 1) as usize).copied()
    }
}

/// Single entry in the Lane 2 64-slot SPSC command ring.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RingEntry {
    /// Monotonic 29-bit causal epoch of this command.
    pub epoch: u32,
    /// Discrete control command payload.
    pub cmd: ControlCommand,
}

impl RingEntry {
    #[inline]
    pub(crate) fn to_words(self) -> [u64; 2] {
        let (tag, arg_u8, arg_u16, f32_a, f32_b): (u8, u8, u16, f32, f32) = match self.cmd {
            ControlCommand::ApplyPreset { preset_index } => (1, 0, preset_index, 0.0, 0.0),
            ControlCommand::SyncTempo {
                bpm,
                subdivision,
                fold_octave,
            } => (
                2,
                encode_subdiv(subdivision),
                u16::from(fold_octave),
                bpm,
                0.0,
            ),
            ControlCommand::TapTempo {
                resolved_bpm,
                subdivision,
                fold_octave,
            } => (
                3,
                encode_subdiv(subdivision),
                u16::from(fold_octave),
                resolved_bpm,
                0.0,
            ),
            ControlCommand::SetBypass { bypassed, trails } => {
                (4, u8::from(bypassed), u16::from(trails), 0.0, 0.0)
            }
            ControlCommand::SetFreeze { frozen } => (5, u8::from(frozen), 0, 0.0, 0.0),
            ControlCommand::SetFreezeAmount { alpha } => (6, 0, 0, alpha, 0.0),
            ControlCommand::SetStereoWidth { width } => (7, 0, 0, width, 0.0),
            ControlCommand::ConfigureHostMix {
                blend_wet,
                output_gain_db,
                apply_unity_leveling,
            } => (
                8,
                u8::from(apply_unity_leveling),
                0,
                blend_wet,
                output_gain_db,
            ),
            ControlCommand::SetTier { tier } => (9, tier as u8, 0, 0.0, 0.0),
            ControlCommand::SetMaxSupportedTier { tier } => (10, tier as u8, 0, 0.0, 0.0),
            ControlCommand::SetOfflineBounce { enabled } => (11, u8::from(enabled), 0, 0.0, 0.0),
            ControlCommand::SetMaximizeTierOnBounce { enabled } => {
                (12, u8::from(enabled), 0, 0.0, 0.0)
            }
            ControlCommand::ResetControl { index } => (13, 0, index, 0.0, 0.0),
            ControlCommand::ResetAllControls => (14, 0, 0, 0.0, 0.0),
            ControlCommand::ResetEngine => (15, 0, 0, 0.0, 0.0),
        };
        let w0 = ((self.epoch as u64) << 32)
            | ((tag as u64) << 24)
            | ((arg_u8 as u64) << 16)
            | (arg_u16 as u64);
        let w1 = (f32_a.to_bits() as u64) | ((f32_b.to_bits() as u64) << 32);
        [w0, w1]
    }

    #[inline]
    pub(crate) fn from_words(w: [u64; 2]) -> Self {
        let epoch = (w[0] >> 32) as u32;
        let tag = ((w[0] >> 24) & 0xFF) as u8;
        let arg_u8 = ((w[0] >> 16) & 0xFF) as u8;
        let arg_u16 = (w[0] & 0xFFFF) as u16;
        let f32_a = f32::from_bits(w[1] as u32);
        let f32_b = f32::from_bits((w[1] >> 32) as u32);

        let cmd = match tag {
            1 => ControlCommand::ApplyPreset {
                preset_index: arg_u16,
            },
            2 => ControlCommand::SyncTempo {
                bpm: f32_a,
                subdivision: decode_subdiv(arg_u8),
                fold_octave: arg_u16 != 0,
            },
            3 => ControlCommand::TapTempo {
                resolved_bpm: f32_a,
                subdivision: decode_subdiv(arg_u8),
                fold_octave: arg_u16 != 0,
            },
            4 => ControlCommand::SetBypass {
                bypassed: arg_u8 != 0,
                trails: arg_u16 != 0,
            },
            5 => ControlCommand::SetFreeze {
                frozen: arg_u8 != 0,
            },
            6 => ControlCommand::SetFreezeAmount { alpha: f32_a },
            7 => ControlCommand::SetStereoWidth { width: f32_a },
            8 => ControlCommand::ConfigureHostMix {
                blend_wet: f32_a,
                output_gain_db: f32_b,
                apply_unity_leveling: arg_u8 != 0,
            },
            9 => ControlCommand::SetTier {
                tier: Tier::from_u8(arg_u8),
            },
            10 => ControlCommand::SetMaxSupportedTier {
                tier: Tier::from_u8(arg_u8),
            },
            11 => ControlCommand::SetOfflineBounce {
                enabled: arg_u8 != 0,
            },
            12 => ControlCommand::SetMaximizeTierOnBounce {
                enabled: arg_u8 != 0,
            },
            13 => ControlCommand::ResetControl { index: arg_u16 },
            15 => ControlCommand::ResetEngine,
            _ => ControlCommand::ResetAllControls,
        };
        Self { epoch, cmd }
    }
}

#[derive(Debug, Default)]
pub(crate) struct AtomicRingSlot {
    words: [AtomicU64; 2],
}

impl AtomicRingSlot {
    #[inline]
    pub(crate) fn store(&self, entry: RingEntry) {
        let [w0, w1] = entry.to_words();
        self.words[0].store(w0, Ordering::Relaxed);
        self.words[1].store(w1, Ordering::Relaxed);
    }

    #[inline]
    pub(crate) fn load(&self) -> RingEntry {
        let w0 = self.words[0].load(Ordering::Relaxed);
        let w1 = self.words[1].load(Ordering::Relaxed);
        RingEntry::from_words([w0, w1])
    }
}

const FB_FLAG_RESET_ENGINE: u16 = 1 << 0;
const FB_FLAG_OFFLINE_BOUNCE: u16 = 1 << 1;
const FB_FLAG_MAXIMIZE_ON_BOUNCE: u16 = 1 << 2;
const FB_FLAG_BYPASSED: u16 = 1 << 3;
const FB_FLAG_TRAILS: u16 = 1 << 4;
const FB_FLAG_FROZEN: u16 = 1 << 5;
const FB_FLAG_HAS_HOST_MIX: u16 = 1 << 6;
const FB_FLAG_UNITY_LEVELING: u16 = 1 << 7;

/// Lane 3 overflow-proof full-state fallback snapshot (`FallbackLatch`).
#[derive(Clone, Copy, Debug)]
pub struct FallbackSnapshot {
    /// Causal epoch at which this fallback snapshot was latched.
    pub epoch: u32,
    /// Number of active controls mirrored in `raw_values`.
    pub num_controls: u16,
    /// Whether `reset_to_dc_equilibrium()` was requested during the overflow window.
    pub reset_engine_requested: bool,
    /// Raw physical control values for the first `MAX_CONTROLS` controls.
    pub raw_values: [f32; MAX_CONTROLS],
    /// Bitmask indicating which controls in `0..MAX_CONTROLS` were last set with `Immediate` mode.
    pub immediate_mask: u32,
    /// Active requested compute tier.
    pub tier: Tier,
    /// Host ceiling compute tier.
    pub max_supported_tier: Tier,
    /// Offline bounce flag.
    pub offline_bounce: bool,
    /// Maximize tier on bounce flag.
    pub maximize_tier_on_bounce: bool,
    /// Bypassed flag.
    pub bypassed: bool,
    /// Spillover trails flag.
    pub trails_enabled: bool,
    /// Unitary Spectral Freeze flag.
    pub is_frozen: bool,
    /// Stereo width in `[0.0, 2.0]`.
    pub stereo_width: f32,
    /// Optional host mix configuration `(blend_wet, output_gain_db, apply_unity_leveling)`.
    pub host_mix: Option<(f32, f32, bool)>,
    /// Active host BPM.
    pub current_bpm: f32,
    /// Additional clamp bitmask from tempo synchronization.
    pub tempo_clamp_mask: u16,
}

impl FallbackSnapshot {
    pub(crate) fn to_words(self) -> [u64; FALLBACK_WORDS] {
        let mut w = [0_u64; FALLBACK_WORDS];
        for i in 0..(MAX_CONTROLS / 2) {
            let a = self.raw_values[i * 2].to_bits() as u64;
            let b = self.raw_values[i * 2 + 1].to_bits() as u64;
            w[i] = a | (b << 32);
        }
        let mut flags = 0_u16;
        if self.reset_engine_requested {
            flags |= FB_FLAG_RESET_ENGINE;
        }
        if self.offline_bounce {
            flags |= FB_FLAG_OFFLINE_BOUNCE;
        }
        if self.maximize_tier_on_bounce {
            flags |= FB_FLAG_MAXIMIZE_ON_BOUNCE;
        }
        if self.bypassed {
            flags |= FB_FLAG_BYPASSED;
        }
        if self.trails_enabled {
            flags |= FB_FLAG_TRAILS;
        }
        if self.is_frozen {
            flags |= FB_FLAG_FROZEN;
        }
        let (blend_wet, output_gain_db) = if let Some((bw, og, unity)) = self.host_mix {
            flags |= FB_FLAG_HAS_HOST_MIX;
            if unity {
                flags |= FB_FLAG_UNITY_LEVELING;
            }
            (bw, og)
        } else {
            (1.0, 0.0)
        };
        let meta_hi = (self.num_controls as u32)
            | ((self.tier as u32) << 16)
            | ((self.max_supported_tier as u32) << 24);
        w[16] = (self.epoch as u64) | ((meta_hi as u64) << 32);
        w[17] = (self.immediate_mask as u64) | ((self.stereo_width.to_bits() as u64) << 32);
        w[18] = (self.current_bpm.to_bits() as u64) | ((blend_wet.to_bits() as u64) << 32);
        let flags_and_clamp = (flags as u32) | ((self.tempo_clamp_mask as u32) << 16);
        w[19] = (output_gain_db.to_bits() as u64) | ((flags_and_clamp as u64) << 32);
        w
    }

    pub(crate) fn from_words(w: &[u64; FALLBACK_WORDS]) -> Self {
        let mut raw_values = [0.0_f32; MAX_CONTROLS];
        for i in 0..(MAX_CONTROLS / 2) {
            raw_values[i * 2] = f32::from_bits(w[i] as u32);
            raw_values[i * 2 + 1] = f32::from_bits((w[i] >> 32) as u32);
        }
        let epoch = w[16] as u32;
        let meta_hi = (w[16] >> 32) as u32;
        let num_controls = (meta_hi & 0xFFFF) as u16;
        let tier = Tier::from_u8(((meta_hi >> 16) & 0xFF) as u8);
        let max_supported_tier = Tier::from_u8(((meta_hi >> 24) & 0xFF) as u8);
        let immediate_mask = w[17] as u32;
        let stereo_width = f32::from_bits((w[17] >> 32) as u32);
        let current_bpm = f32::from_bits(w[18] as u32);
        let blend_wet = f32::from_bits((w[18] >> 32) as u32);
        let output_gain_db = f32::from_bits(w[19] as u32);
        let flags_and_clamp = (w[19] >> 32) as u32;
        let flags = (flags_and_clamp & 0xFFFF) as u16;
        let tempo_clamp_mask = (flags_and_clamp >> 16) as u16;

        let host_mix = if (flags & FB_FLAG_HAS_HOST_MIX) != 0 {
            Some((
                blend_wet,
                output_gain_db,
                (flags & FB_FLAG_UNITY_LEVELING) != 0,
            ))
        } else {
            None
        };

        Self {
            epoch,
            num_controls,
            reset_engine_requested: (flags & FB_FLAG_RESET_ENGINE) != 0,
            raw_values,
            immediate_mask,
            tier,
            max_supported_tier,
            offline_bounce: (flags & FB_FLAG_OFFLINE_BOUNCE) != 0,
            maximize_tier_on_bounce: (flags & FB_FLAG_MAXIMIZE_ON_BOUNCE) != 0,
            bypassed: (flags & FB_FLAG_BYPASSED) != 0,
            trails_enabled: (flags & FB_FLAG_TRAILS) != 0,
            is_frozen: (flags & FB_FLAG_FROZEN) != 0,
            stereo_width,
            host_mix,
            current_bpm,
            tempo_clamp_mask,
        }
    }
}

/// 64-byte cache-line-aligned wrapper preventing false sharing between producer and consumer atomics.
#[repr(align(64))]
#[derive(Debug, Default)]
pub(crate) struct CachePadded<T> {
    pub(crate) value: T,
}

impl<T> CachePadded<T> {
    #[inline]
    pub(crate) const fn new(value: T) -> Self {
        Self { value }
    }
}

/// Shared 3-Lane SPSC control ring memory between [`super::ControlProducer`] and
/// [`super::ControlConsumer`] (`ref[dsp.ctrl.op-m.spsc-ingress]`).
#[repr(align(64))]
#[derive(Debug)]
pub struct SpscControlRing {
    /// Lane 1: 32 fast inline atomic parameter target slots.
    pub(crate) lane1_slots: [AtomicU64; MAX_CONTROLS],
    /// Extended Lane 1 slots for models with `> 32` controls.
    pub(crate) lane1_ext: Box<[AtomicU64]>,
    /// Producer-owned write index for Lane 2.
    pub(crate) head: CachePadded<AtomicUsize>,
    /// Consumer-owned read index for Lane 2.
    pub(crate) tail: CachePadded<AtomicUsize>,
    /// Lane 2: 64-slot SPSC ring buffer for discrete commands.
    pub(crate) lane2_slots: [AtomicRingSlot; SPSC_RING_CAPACITY],
    /// Lane 3: Seqlock counter for the overflow-proof fallback latch.
    pub(crate) fallback_seq: CachePadded<AtomicUsize>,
    /// Lane 3: Epoch of the pending fallback latch (`0` = none).
    pub(crate) fallback_epoch: AtomicU32,
    /// Lane 3: Cumulative flag tracking whether `ResetEngine` occurred in the latched state.
    pub(crate) fallback_reset_engine: AtomicU32,
    /// Lane 3: Atomic words storing [`FallbackSnapshot`].
    pub(crate) fallback_words: [AtomicU64; FALLBACK_WORDS],
    /// Base 64-bit causal epoch offset added to 29-bit relative epochs (`next_epoch_29`).
    pub(crate) epoch_base: AtomicU64,
    /// Monotonic 64-bit causal epoch counter of the last command drained by the audio thread.
    pub(crate) drained_causal_epoch: CachePadded<AtomicU64>,
    /// Latest cumulative audio frame count published by the audio thread for tap-tempo delta calculation.
    pub(crate) published_total_frames: CachePadded<AtomicU64>,
    /// Diagnostic counter of Lane 3 overflow latches triggered.
    pub(crate) overflow_latch_count: AtomicUsize,
}

impl SpscControlRing {
    pub(crate) fn new(num_controls: usize, initial_total_frames: u64) -> Self {
        let ext_len = num_controls.saturating_sub(MAX_CONTROLS);
        let mut ext_vec = Vec::with_capacity(ext_len);
        for _ in 0..ext_len {
            ext_vec.push(AtomicU64::new(0));
        }
        Self {
            lane1_slots: core::array::from_fn(|_| AtomicU64::new(0)),
            lane1_ext: ext_vec.into_boxed_slice(),
            head: CachePadded::new(AtomicUsize::new(0)),
            tail: CachePadded::new(AtomicUsize::new(0)),
            lane2_slots: core::array::from_fn(|_| AtomicRingSlot::default()),
            fallback_seq: CachePadded::new(AtomicUsize::new(0)),
            fallback_epoch: AtomicU32::new(0),
            fallback_reset_engine: AtomicU32::new(0),
            fallback_words: core::array::from_fn(|_| AtomicU64::new(0)),
            epoch_base: AtomicU64::new(0),
            drained_causal_epoch: CachePadded::new(AtomicU64::new(0)),
            published_total_frames: CachePadded::new(AtomicU64::new(initial_total_frames)),
            overflow_latch_count: AtomicUsize::new(0),
        }
    }

    #[inline]
    pub(crate) fn lane1_slot(&self, ctrl_idx: usize) -> Option<&AtomicU64> {
        if ctrl_idx < MAX_CONTROLS {
            Some(&self.lane1_slots[ctrl_idx])
        } else {
            self.lane1_ext.get(ctrl_idx - MAX_CONTROLS)
        }
    }

    #[inline]
    pub(crate) fn clear_all_lane1(&self) {
        for s in &self.lane1_slots {
            s.store(0, Ordering::Release);
        }
        for s in &self.lane1_ext {
            s.store(0, Ordering::Release);
        }
    }

    pub(crate) fn tombstone_unconsumed_lane2(&self) -> bool {
        let mut saw_reset_engine = false;
        let tail = self.tail.value.load(Ordering::Acquire);
        let head = self.head.value.load(Ordering::Relaxed);
        let mut cursor = tail;
        while cursor != head {
            let slot = &self.lane2_slots[cursor & SPSC_RING_MASK];
            if matches!(slot.load().cmd, ControlCommand::ResetEngine) {
                saw_reset_engine = true;
            }
            slot.store(RingEntry {
                epoch: 0,
                cmd: ControlCommand::ResetAllControls,
            });
            cursor = cursor.wrapping_add(1);
        }
        saw_reset_engine
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_lane1_lane2_lane3_word_packing_and_ring_slots() {
        // verify[dsp.ctrl.op-m.spsc-ingress]
        assert_eq!(unpack_lane1_slot(0), None);
        let e_range = Lane1Entry {
            epoch: 7,
            mode: ControlUpdateMode::Slewed,
            kind: ControlValueKind::Range,
            payload_bits: 3.5_f32.to_bits(),
        };
        assert!((e_range.physical_value() - 3.5).abs() < 1e-6);
        let e_sw = Lane1Entry {
            epoch: 8,
            mode: ControlUpdateMode::Immediate,
            kind: ControlValueKind::Switch,
            payload_bits: 2,
        };
        assert!((e_sw.physical_value() - 2.0).abs() < 1e-6);

        for &sub in TempoSubdivision::ALL.iter() {
            let code = encode_subdiv(Some(sub));
            assert_eq!(decode_subdiv(code), Some(sub));
        }
        assert_eq!(encode_subdiv(None), 0);
        assert_eq!(decode_subdiv(0), None);
        assert_eq!(decode_subdiv(250), None);

        let commands = [
            ControlCommand::ApplyPreset { preset_index: 3 },
            ControlCommand::SyncTempo {
                bpm: 120.0,
                subdivision: Some(TempoSubdivision::EighthDotted),
                fold_octave: true,
            },
            ControlCommand::TapTempo {
                resolved_bpm: 135.5,
                subdivision: None,
                fold_octave: false,
            },
            ControlCommand::SetBypass {
                bypassed: true,
                trails: false,
            },
            ControlCommand::SetFreeze { frozen: true },
            ControlCommand::SetFreezeAmount { alpha: 0.75 },
            ControlCommand::SetStereoWidth { width: 1.4 },
            ControlCommand::ConfigureHostMix {
                blend_wet: 0.8,
                output_gain_db: -1.5,
                apply_unity_leveling: true,
            },
            ControlCommand::SetTier { tier: Tier::Pro },
            ControlCommand::SetMaxSupportedTier { tier: Tier::Full },
            ControlCommand::SetOfflineBounce { enabled: true },
            ControlCommand::SetMaximizeTierOnBounce { enabled: false },
            ControlCommand::ResetControl { index: 5 },
            ControlCommand::ResetAllControls,
            ControlCommand::ResetEngine,
        ];
        let slot = AtomicRingSlot::default();
        for (i, cmd) in commands.into_iter().enumerate() {
            let entry = RingEntry {
                epoch: (i as u32) + 1,
                cmd,
            };
            slot.store(entry);
            assert_eq!(slot.load(), entry);
        }

        let ring = SpscControlRing::new(36, 1024);
        assert!(ring.lane1_slot(0).is_some());
        assert!(ring.lane1_slot(35).is_some());
        assert!(ring.lane1_slot(36).is_none());
        ring.lane1_slot(0).unwrap().store(1, Ordering::Release);
        ring.lane1_slot(35).unwrap().store(2, Ordering::Release);
        ring.clear_all_lane1();
        assert_eq!(ring.lane1_slot(0).unwrap().load(Ordering::Acquire), 0);
        assert_eq!(ring.lane1_slot(35).unwrap().load(Ordering::Acquire), 0);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(32))]

        #[test]
        fn prop_lane1_and_fallback_snapshot_word_bijection(
            epoch in 1u32..MAX_EPOCH_29,
            is_immediate in any::<bool>(),
            is_switch in any::<bool>(),
            payload_bits in any::<u32>(),
            flags_bits in any::<u8>(),
            has_mix in any::<bool>(),
            unity_lvl in any::<bool>(),
            blend_wet in 0.0f32..1.0f32,
            output_gain_db in -24.0f32..24.0f32,
            stereo_width in 0.0f32..2.0f32,
            current_bpm in 30.0f32..300.0f32,
            immediate_mask in any::<u32>(),
            tempo_clamp_mask in any::<u16>(),
        ) {
            // verify[dsp.ctrl.op-m.spsc-ingress]
            let mode = if is_immediate {
                ControlUpdateMode::Immediate
            } else {
                ControlUpdateMode::Slewed
            };
            let kind = if is_switch {
                ControlValueKind::Switch
            } else {
                ControlValueKind::Range
            };
            let word = pack_lane1_slot(epoch, mode, kind, payload_bits);
            let unpacked = unpack_lane1_slot(word).unwrap();
            prop_assert_eq!(unpacked.epoch, epoch);
            prop_assert_eq!(unpacked.mode, mode);
            prop_assert_eq!(unpacked.kind, kind);
            prop_assert_eq!(unpacked.payload_bits, payload_bits);

            let fb = FallbackSnapshot {
                epoch,
                num_controls: 8,
                reset_engine_requested: (flags_bits & 1) != 0,
                raw_values: [1.5; MAX_CONTROLS],
                immediate_mask,
                tier: Tier::Pro,
                max_supported_tier: Tier::Ultra,
                offline_bounce: (flags_bits & 2) != 0,
                maximize_tier_on_bounce: (flags_bits & 4) != 0,
                bypassed: (flags_bits & 8) != 0,
                trails_enabled: (flags_bits & 16) != 0,
                is_frozen: (flags_bits & 32) != 0,
                stereo_width,
                host_mix: if has_mix {
                    Some((blend_wet, output_gain_db, unity_lvl))
                } else {
                    None
                },
                current_bpm,
                tempo_clamp_mask,
            };
            let words = fb.to_words();
            let fb_out = FallbackSnapshot::from_words(&words);
            prop_assert_eq!(fb_out.epoch, fb.epoch);
            prop_assert_eq!(fb_out.reset_engine_requested, fb.reset_engine_requested);
            prop_assert_eq!(fb_out.offline_bounce, fb.offline_bounce);
            prop_assert_eq!(fb_out.maximize_tier_on_bounce, fb.maximize_tier_on_bounce);
            prop_assert_eq!(fb_out.bypassed, fb.bypassed);
            prop_assert_eq!(fb_out.trails_enabled, fb.trails_enabled);
            prop_assert_eq!(fb_out.is_frozen, fb.is_frozen);
            prop_assert_eq!(fb_out.host_mix, fb.host_mix);
            prop_assert_eq!(fb_out.tempo_clamp_mask, fb.tempo_clamp_mask);
        }
    }
}
