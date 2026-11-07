//! Wait-free Seqlock telemetry publication (`TelemetryCell`, `TelemetryWriter`, `TelemetryReader`)
//! (`ref[arch.telemetry.seqlock-publication]`).
//!
//! The real-time audio thread publishes a fixed-size [`RawTelemetryRecord`] at the end of every
//! audio block using an odd/even sequence counter (`Ordering::Release`), while UI, Python, and
//! JS host threads read consistent snapshots wait-free (`Ordering::Acquire` retry loop) without
//! ever locking the audio engine or allocating heap memory.

use alloc::sync::Arc;
use core::sync::atomic::{AtomicU64, Ordering, fence};

use crate::container::ZohRescaledParams;
use crate::control::TELEMETRY_CONTROL_SLOTS;
use crate::dynamic::DynamicDnamEngine;
use crate::math::BackendKind;
use crate::types::{DnamTelemetrySnapshot, Tier};

/// Number of 64-bit atomic words used to store a [`RawTelemetryRecord`] inside [`TelemetryCell`].
const TELEMETRY_WORDS: usize = 29;

/// Maximum wait-free Seqlock read retries before returning the latest snapshot (`ref[arch.telemetry.seqlock-publication]`).
const MAX_SEQLOCK_READ_RETRIES: usize = 64;

const FLAG_OFFLINE_BOUNCE: u8 = 1 << 0;
const FLAG_MAXIMIZE_TIER_ON_BOUNCE: u8 = 1 << 1;
const FLAG_SUPPORTS_FREEZE: u8 = 1 << 2;
const FLAG_TRAILS_ENABLED: u8 = 1 << 3;

#[inline]
const fn backend_to_u8(kind: BackendKind) -> u8 {
    match kind {
        BackendKind::Scalar => 0,
        BackendKind::ArmNeon => 1,
        BackendKind::Neon => 2,
        BackendKind::ArmSme => 3,
        BackendKind::X86Sse42 => 4,
        BackendKind::X86Avx2Fma => 5,
        BackendKind::X86Simd => 6,
        BackendKind::WasmSimd128 => 7,
    }
}

#[inline]
const fn backend_from_u8(val: u8) -> BackendKind {
    match val {
        1 => BackendKind::ArmNeon,
        2 => BackendKind::Neon,
        3 => BackendKind::ArmSme,
        4 => BackendKind::X86Sse42,
        5 => BackendKind::X86Avx2Fma,
        6 => BackendKind::X86Simd,
        7 => BackendKind::WasmSimd128,
        _ => BackendKind::Scalar,
    }
}

/// Fixed-size, `Copy` telemetry record published wait-free by the audio thread after each block
/// (`ref[arch.telemetry.seqlock-publication]`).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RawTelemetryRecord {
    /// Core C-ABI 160-byte telemetry snapshot (`z_head`, `env_states`, `lfo_harmonics`, `clamped_controls`).
    pub snapshot: DnamTelemetrySnapshot,
    /// Active zero-order-hold sample-rate rescaling parameters.
    pub zoh: ZohRescaledParams,
    /// Output Mid/Side stereo width $\alpha_{\text{width}} \in [0.0, 2.0]$.
    pub stereo_width: f32,
    /// Active host tempo in BPM.
    pub current_bpm: f32,
    /// Current Plugin Delay Compensation (`PDC`) latency in host samples (`0` or `12`).
    pub latency_samples: usize,
    /// Number of active user controls in the loaded container (`0..=255`).
    pub num_controls: u8,
    /// Active hardware math backend kind.
    pub backend_kind: BackendKind,
    /// Active host sample rate in Hz.
    pub sample_rate_hz: u32,
    /// Requested compute tier (`Lite`, `Full`, `Pro`, `Ultra`).
    pub requested_tier: Tier,
    /// Highest tier available in the loaded `.dnam` container.
    pub max_available_tier: Tier,
    /// Host ceiling tier (`max_supported_tier`).
    pub max_supported_tier: Tier,
    /// Whether $4\times$ oversampled offline bounce mode is enabled.
    pub offline_bounce: bool,
    /// Whether offline bounce automatically promotes to `min(max_available_tier, max_supported_tier)`.
    pub maximize_tier_on_bounce: bool,
    /// Whether the loaded model supports Unitary Spectral Freeze.
    pub supports_freeze: bool,
    /// Whether bypass spillover trails are enabled.
    pub trails_enabled: bool,
    /// Monotonic causal control epoch reflected in this record.
    pub causal_epoch: u64,
}

impl RawTelemetryRecord {
    /// Captures a complete [`RawTelemetryRecord`] from a [`DynamicDnamEngine`] at `causal_epoch`.
    #[inline]
    pub fn from_engine(engine: &DynamicDnamEngine<'_>, causal_epoch: u64) -> Self {
        let (_, trails_enabled) = engine.bypass_state();
        Self {
            snapshot: engine.telemetry_snapshot(),
            zoh: *engine.zoh(),
            stereo_width: engine.stereo_width(),
            current_bpm: engine.current_bpm(),
            latency_samples: engine.latency_samples(),
            num_controls: engine.num_controls().min(u8::MAX as usize) as u8,
            backend_kind: engine.backend_kind(),
            sample_rate_hz: engine.sample_rate_hz(),
            requested_tier: engine.tier(),
            max_available_tier: engine.max_available_tier(),
            max_supported_tier: engine.max_supported_tier(),
            offline_bounce: engine.offline_bounce(),
            maximize_tier_on_bounce: engine.maximize_tier_on_bounce(),
            supports_freeze: engine.supports_freeze(),
            trails_enabled,
            causal_epoch,
        }
    }

    #[inline]
    fn pack_f32_pair(a: f32, b: f32) -> u64 {
        (a.to_bits() as u64) | ((b.to_bits() as u64) << 32)
    }

    #[inline]
    fn unpack_f32_pair(word: u64) -> (f32, f32) {
        (
            f32::from_bits(word as u32),
            f32::from_bits((word >> 32) as u32),
        )
    }

    #[inline]
    fn pack_u32_pair(lo: u32, hi: u32) -> u64 {
        (lo as u64) | ((hi as u64) << 32)
    }

    #[inline]
    fn unpack_u32_pair(word: u64) -> (u32, u32) {
        (word as u32, (word >> 32) as u32)
    }

    #[inline]
    fn to_words(self) -> [u64; TELEMETRY_WORDS] {
        let mut w = [0_u64; TELEMETRY_WORDS];
        let s = &self.snapshot;
        let hdr_lo = (s.active_tier as u32)
            | ((s.stereo_out as u32) << 8)
            | ((s.bypassed as u32) << 16)
            | ((s.is_frozen as u32) << 24);
        let hdr_hi = (s.active_clamp_mask as u32) | ((s.control_decimation_host as u32) << 16);
        w[0] = Self::pack_u32_pair(hdr_lo, hdr_hi);
        w[1] = Self::pack_f32_pair(s.max_op_n_drive, s.max_pole_radius);
        w[2] = Self::pack_f32_pair(s.max_effective_loop_gain, s.tail_time_ms);
        w[3] = Self::pack_f32_pair(s.z_head[0], s.z_head[1]);
        w[4] = Self::pack_f32_pair(s.z_head[2], s.z_head[3]);
        w[5] = Self::pack_f32_pair(s.env_states[0], s.env_states[1]);
        w[6] = Self::pack_f32_pair(s.env_states[2], s.env_states[3]);
        for i in 0..4 {
            w[7 + i] = Self::pack_f32_pair(s.lfo_harmonics[i * 2], s.lfo_harmonics[i * 2 + 1]);
        }
        for i in 0..(TELEMETRY_CONTROL_SLOTS / 2) {
            w[11 + i] =
                Self::pack_f32_pair(s.clamped_controls[i * 2], s.clamped_controls[i * 2 + 1]);
        }
        w[19] = Self::pack_u32_pair(s.overflow_or_saturation_flags, self.sample_rate_hz);
        w[20] = Self::pack_u32_pair(
            self.zoh.sample_rate_host_hz,
            self.zoh.switch_xfade_samples_host,
        );
        w[21] = self.zoh.rho_sr.to_bits();
        w[22] = self.zoh.rho_ctrl.to_bits();
        w[23] = self.zoh.control_decimation_host as u64;
        w[24] = Self::pack_f32_pair(self.zoh.alpha_ramp_host, self.zoh.r_ac_host);
        w[25] = Self::pack_f32_pair(self.zoh.r_dc_host, self.zoh.tau_ref_samples_host);
        w[26] = Self::pack_f32_pair(self.stereo_width, self.current_bpm);

        let mut flags = 0_u8;
        if self.offline_bounce {
            flags |= FLAG_OFFLINE_BOUNCE;
        }
        if self.maximize_tier_on_bounce {
            flags |= FLAG_MAXIMIZE_TIER_ON_BOUNCE;
        }
        if self.supports_freeze {
            flags |= FLAG_SUPPORTS_FREEZE;
        }
        if self.trails_enabled {
            flags |= FLAG_TRAILS_ENABLED;
        }
        let meta_lo = ((self.latency_samples.min(u16::MAX as usize)) as u32)
            | ((self.num_controls as u32) << 16)
            | ((backend_to_u8(self.backend_kind) as u32) << 24);
        let meta_hi = (self.requested_tier as u32)
            | ((self.max_available_tier as u32) << 8)
            | ((self.max_supported_tier as u32) << 16)
            | ((flags as u32) << 24);
        w[27] = Self::pack_u32_pair(meta_lo, meta_hi);
        w[28] = self.causal_epoch;
        w
    }

    #[inline]
    fn from_words(w: &[u64; TELEMETRY_WORDS]) -> Self {
        let (hdr_lo, hdr_hi) = Self::unpack_u32_pair(w[0]);
        let active_tier = (hdr_lo & 0xFF) as u8;
        let stereo_out = ((hdr_lo >> 8) & 0xFF) as u8;
        let bypassed = ((hdr_lo >> 16) & 0xFF) as u8;
        let is_frozen = ((hdr_lo >> 24) & 0xFF) as u8;
        let active_clamp_mask = (hdr_hi & 0xFFFF) as u16;
        let control_decimation_host = ((hdr_hi >> 16) & 0xFFFF) as u16;

        let (max_op_n_drive, max_pole_radius) = Self::unpack_f32_pair(w[1]);
        let (max_effective_loop_gain, tail_time_ms) = Self::unpack_f32_pair(w[2]);
        let (z0, z1) = Self::unpack_f32_pair(w[3]);
        let (z2, z3) = Self::unpack_f32_pair(w[4]);
        let (e0, e1) = Self::unpack_f32_pair(w[5]);
        let (e2, e3) = Self::unpack_f32_pair(w[6]);

        let mut lfo_harmonics = [0.0_f32; 8];
        for i in 0..4 {
            let (a, b) = Self::unpack_f32_pair(w[7 + i]);
            lfo_harmonics[i * 2] = a;
            lfo_harmonics[i * 2 + 1] = b;
        }
        let mut clamped_controls = [0.0_f32; TELEMETRY_CONTROL_SLOTS];
        for i in 0..(TELEMETRY_CONTROL_SLOTS / 2) {
            let (a, b) = Self::unpack_f32_pair(w[11 + i]);
            clamped_controls[i * 2] = a;
            clamped_controls[i * 2 + 1] = b;
        }
        let (overflow_or_saturation_flags, sample_rate_hz) = Self::unpack_u32_pair(w[19]);
        let (sample_rate_host_hz, switch_xfade_samples_host) = Self::unpack_u32_pair(w[20]);
        let rho_sr = f64::from_bits(w[21]);
        let rho_ctrl = f64::from_bits(w[22]);
        let k_ctrl_usize = w[23] as usize;
        let (alpha_ramp_host, r_ac_host) = Self::unpack_f32_pair(w[24]);
        let (r_dc_host, tau_ref_samples_host) = Self::unpack_f32_pair(w[25]);
        let (stereo_width, current_bpm) = Self::unpack_f32_pair(w[26]);
        let (meta_lo, meta_hi) = Self::unpack_u32_pair(w[27]);
        let causal_epoch = w[28];

        let latency_u16 = (meta_lo & 0xFFFF) as usize;
        let num_controls = ((meta_lo >> 16) & 0xFF) as u8;
        let backend_kind = backend_from_u8(((meta_lo >> 24) & 0xFF) as u8);

        let requested_tier = Tier::from_u8((meta_hi & 0xFF) as u8);
        let max_available_tier = Tier::from_u8(((meta_hi >> 8) & 0xFF) as u8);
        let max_supported_tier = Tier::from_u8(((meta_hi >> 16) & 0xFF) as u8);
        let flags = ((meta_hi >> 24) & 0xFF) as u8;

        Self {
            snapshot: DnamTelemetrySnapshot {
                active_tier,
                stereo_out,
                bypassed,
                is_frozen,
                active_clamp_mask,
                control_decimation_host,
                max_op_n_drive,
                max_pole_radius,
                max_effective_loop_gain,
                tail_time_ms,
                z_head: [z0, z1, z2, z3],
                env_states: [e0, e1, e2, e3],
                lfo_harmonics,
                clamped_controls,
                overflow_or_saturation_flags,
            },
            zoh: ZohRescaledParams {
                sample_rate_host_hz,
                rho_sr,
                control_decimation_host: k_ctrl_usize,
                rho_ctrl,
                alpha_ramp_host,
                r_ac_host,
                r_dc_host,
                tau_ref_samples_host,
                switch_xfade_samples_host,
            },
            stereo_width,
            current_bpm,
            latency_samples: latency_u16,
            num_controls,
            backend_kind,
            sample_rate_hz,
            requested_tier,
            max_available_tier,
            max_supported_tier,
            offline_bounce: (flags & FLAG_OFFLINE_BOUNCE) != 0,
            maximize_tier_on_bounce: (flags & FLAG_MAXIMIZE_TIER_ON_BOUNCE) != 0,
            supports_freeze: (flags & FLAG_SUPPORTS_FREEZE) != 0,
            trails_enabled: (flags & FLAG_TRAILS_ENABLED) != 0,
            causal_epoch,
        }
    }
}

/// Cache-line-aligned (`64 B`) wait-free Seqlock cell holding a [`RawTelemetryRecord`]
/// (`ref[arch.telemetry.seqlock-publication]`).
#[repr(align(64))]
#[derive(Debug)]
pub struct TelemetryCell {
    seq: AtomicU64,
    words: [AtomicU64; TELEMETRY_WORDS],
}

impl TelemetryCell {
    /// Creates a new [`TelemetryCell`] initialized with `initial`.
    #[inline]
    pub fn new(initial: RawTelemetryRecord) -> Self {
        let packed = initial.to_words();
        Self {
            seq: AtomicU64::new(0),
            words: core::array::from_fn(|i| AtomicU64::new(packed[i])),
        }
    }

    /// Publishes a new [`RawTelemetryRecord`] using the single-writer Seqlock odd/even protocol.
    #[inline]
    pub fn publish(&self, record: RawTelemetryRecord) {
        let packed = record.to_words();
        let s0 = self.seq.load(Ordering::Relaxed);
        let s_odd = s0.wrapping_add(1) | 1;
        self.seq.store(s_odd, Ordering::Release);
        fence(Ordering::AcqRel);
        for (slot, &word) in self.words.iter().zip(packed.iter()) {
            slot.store(word, Ordering::Relaxed);
        }
        self.seq.store(s_odd.wrapping_add(1), Ordering::Release);
    }

    /// Reads a consistent [`RawTelemetryRecord`] snapshot using a bounded `Ordering::Acquire` retry loop
    /// (`64` iterations max before returning the latest snapshot, `ref[arch.telemetry.seqlock-publication]`).
    #[inline]
    pub fn read(&self) -> RawTelemetryRecord {
        let mut buf = [0_u64; TELEMETRY_WORDS];
        for _ in 0..MAX_SEQLOCK_READ_RETRIES {
            let s1 = self.seq.load(Ordering::Acquire);
            if (s1 & 1) != 0 {
                core::hint::spin_loop();
                continue;
            }
            for (dst, slot) in buf.iter_mut().zip(self.words.iter()) {
                *dst = slot.load(Ordering::Relaxed);
            }
            fence(Ordering::Acquire);
            let s2 = self.seq.load(Ordering::Relaxed);
            if s1 == s2 {
                return RawTelemetryRecord::from_words(&buf);
            }
            core::hint::spin_loop();
        }
        for (dst, slot) in buf.iter_mut().zip(self.words.iter()) {
            *dst = slot.load(Ordering::Acquire);
        }
        RawTelemetryRecord::from_words(&buf)
    }

    /// Returns the current even/odd sequence number for diagnostic and test inspection.
    #[inline]
    pub fn sequence(&self) -> usize {
        self.seq.load(Ordering::Acquire) as usize
    }
}

/// Single-writer handle for publishing [`RawTelemetryRecord`] snapshots from the real-time thread.
#[derive(Debug)]
pub struct TelemetryWriter {
    cell: Arc<TelemetryCell>,
}

impl TelemetryWriter {
    /// Publishes `record` wait-free to all [`TelemetryReader`] handles.
    #[inline]
    pub fn publish(&mut self, record: RawTelemetryRecord) {
        self.cell.publish(record);
    }

    /// Captures and publishes a [`RawTelemetryRecord`] directly from `engine` at `causal_epoch`.
    #[inline]
    pub fn publish_from_engine(&mut self, engine: &DynamicDnamEngine<'_>, causal_epoch: u64) {
        self.cell
            .publish(RawTelemetryRecord::from_engine(engine, causal_epoch));
    }
}

/// Cloneable wait-free reader handle for inspecting [`RawTelemetryRecord`] snapshots from any thread.
#[derive(Clone, Debug)]
pub struct TelemetryReader {
    cell: Arc<TelemetryCell>,
}

impl TelemetryReader {
    /// Reads a consistent [`RawTelemetryRecord`] wait-free without locking the audio thread.
    #[inline]
    pub fn read(&self) -> RawTelemetryRecord {
        self.cell.read()
    }

    /// Reads the C-ABI [`DnamTelemetrySnapshot`] wait-free without locking the audio thread.
    #[inline]
    pub fn read_snapshot(&self) -> DnamTelemetrySnapshot {
        self.cell.read().snapshot
    }

    /// Returns the current sequence counter of the underlying [`TelemetryCell`].
    #[inline]
    pub fn sequence(&self) -> usize {
        self.cell.sequence()
    }
}

/// Creates a paired `(TelemetryWriter, TelemetryReader)` backed by a shared 64B-aligned [`TelemetryCell`].
#[inline]
pub fn telemetry_channel(initial: RawTelemetryRecord) -> (TelemetryWriter, TelemetryReader) {
    let cell = Arc::new(TelemetryCell::new(initial));
    (
        TelemetryWriter {
            cell: Arc::clone(&cell),
        },
        TelemetryReader { cell },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::{ARENA_OFFSET_ULTRA_END, build_dnam_container};
    use crate::schema::DnamJsonHeader;
    use alloc::vec;
    use proptest::prelude::*;

    fn sample_record() -> RawTelemetryRecord {
        RawTelemetryRecord {
            snapshot: DnamTelemetrySnapshot {
                active_tier: Tier::Pro as u8,
                stereo_out: 1,
                bypassed: 0,
                is_frozen: 1,
                active_clamp_mask: 0x05A5,
                control_decimation_host: 16,
                max_op_n_drive: 3.5,
                max_pole_radius: 0.995,
                max_effective_loop_gain: 0.85,
                tail_time_ms: 420.0,
                z_head: [0.1, -0.2, 0.3, -0.4],
                env_states: [0.01, 0.02, 0.03, 0.04],
                lfo_harmonics: [1.0, 0.0, 0.5, -0.5, 0.25, -0.25, 0.125, -0.125],
                clamped_controls: [
                    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                    15.0, 16.0,
                ],
                overflow_or_saturation_flags: 7,
            },
            zoh: ZohRescaledParams {
                sample_rate_host_hz: 96_000,
                rho_sr: 0.5,
                control_decimation_host: 32,
                rho_ctrl: 1.0,
                alpha_ramp_host: 0.03125,
                r_ac_host: 0.9975,
                r_dc_host: 0.998,
                tau_ref_samples_host: 64.0,
                switch_xfade_samples_host: 128,
            },
            stereo_width: 1.25,
            current_bpm: 128.5,
            latency_samples: 12,
            num_controls: 6,
            backend_kind: BackendKind::Neon,
            sample_rate_hz: 96_000,
            requested_tier: Tier::Pro,
            max_available_tier: Tier::Ultra,
            max_supported_tier: Tier::Pro,
            offline_bounce: true,
            maximize_tier_on_bounce: true,
            supports_freeze: true,
            trails_enabled: true,
            causal_epoch: 42,
        }
    }

    #[test]
    fn test_backend_kind_u8_roundtrip_and_unknown_fallback() {
        // verify[arch.telemetry.seqlock-publication]
        let all_backends = [
            BackendKind::Scalar,
            BackendKind::ArmNeon,
            BackendKind::Neon,
            BackendKind::ArmSme,
            BackendKind::X86Sse42,
            BackendKind::X86Avx2Fma,
            BackendKind::X86Simd,
            BackendKind::WasmSimd128,
        ];
        for kind in all_backends {
            let code = backend_to_u8(kind);
            assert_eq!(backend_from_u8(code), kind);
        }
        assert_eq!(backend_from_u8(255), BackendKind::Scalar);
    }

    #[test]
    fn test_seqlock_telemetry_cell_publish_read_and_odd_retry_fallback() {
        // verify[arch.telemetry.seqlock-publication]
        let mut rec = sample_record();
        let (mut writer, reader) = telemetry_channel(rec);
        assert_eq!(reader.sequence(), 0);
        assert_eq!(reader.read(), rec);
        assert_eq!(reader.read_snapshot(), rec.snapshot);

        // Toggle all boolean flags false and publish
        rec.offline_bounce = false;
        rec.maximize_tier_on_bounce = false;
        rec.supports_freeze = false;
        rec.trails_enabled = false;
        rec.causal_epoch = 99;
        writer.publish(rec);
        assert_eq!(reader.sequence(), 2);
        assert_eq!(reader.read(), rec);

        // Test publish_from_engine
        let mut header = DnamJsonHeader::default_reference();
        let arena = vec![0_u8; ARENA_OFFSET_ULTRA_END as usize];
        let container = build_dnam_container(&mut header, &arena, &arena, false).unwrap();
        let engine = DynamicDnamEngine::from_bytes_owned(&container, Tier::Lite, 48_000).unwrap();
        writer.publish_from_engine(&engine, 123);
        assert_eq!(reader.sequence(), 4);
        let from_eng = reader.read();
        assert_eq!(from_eng.causal_epoch, 123);
        assert_eq!(from_eng.sample_rate_hz, 48_000);
        assert_eq!(from_eng.requested_tier, Tier::Lite);

        // Exercise bounded retry fallback when sequence is observed odd
        reader.cell.seq.store(5, Ordering::Release);
        assert_eq!(reader.cell.sequence(), 5);
        let fallback_read = reader.read();
        assert_eq!(fallback_read.causal_epoch, 123);

        // Recover smoothly on next publish even if seq was odd
        writer.publish(rec);
        assert_eq!(reader.read(), rec);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn prop_raw_telemetry_record_word_packing_bijection(
            active_tier in 0u8..4,
            stereo_out in 0u8..2,
            bypassed in 0u8..2,
            is_frozen in 0u8..2,
            active_clamp_mask in any::<u16>(),
            control_decimation_host in any::<u16>(),
            max_op_n_drive in -100.0f32..100.0f32,
            max_pole_radius in 0.0f32..1.0f32,
            max_effective_loop_gain in 0.0f32..10.0f32,
            tail_time_ms in 0.0f32..10_000.0f32,
            stereo_width in 0.0f32..2.0f32,
            current_bpm in 20.0f32..300.0f32,
            latency_samples in 0usize..1024,
            num_controls in any::<u8>(),
            backend_idx in 0u8..8,
            sample_rate_hz in 44_100u32..192_000u32,
            req_tier in 0u8..4,
            max_avail_tier in 0u8..4,
            max_sup_tier in 0u8..4,
            flags_bits in 0u8..16,
            causal_epoch in any::<u64>(),
        ) {
            // verify[arch.telemetry.seqlock-publication]
            let mut rec = sample_record();
            rec.snapshot.active_tier = active_tier;
            rec.snapshot.stereo_out = stereo_out;
            rec.snapshot.bypassed = bypassed;
            rec.snapshot.is_frozen = is_frozen;
            rec.snapshot.active_clamp_mask = active_clamp_mask;
            rec.snapshot.control_decimation_host = control_decimation_host;
            rec.snapshot.max_op_n_drive = max_op_n_drive;
            rec.snapshot.max_pole_radius = max_pole_radius;
            rec.snapshot.max_effective_loop_gain = max_effective_loop_gain;
            rec.snapshot.tail_time_ms = tail_time_ms;
            rec.stereo_width = stereo_width;
            rec.current_bpm = current_bpm;
            rec.latency_samples = latency_samples;
            rec.num_controls = num_controls;
            rec.backend_kind = backend_from_u8(backend_idx);
            rec.sample_rate_hz = sample_rate_hz;
            rec.requested_tier = Tier::from_u8(req_tier);
            rec.max_available_tier = Tier::from_u8(max_avail_tier);
            rec.max_supported_tier = Tier::from_u8(max_sup_tier);
            rec.offline_bounce = (flags_bits & 1) != 0;
            rec.maximize_tier_on_bounce = (flags_bits & 2) != 0;
            rec.supports_freeze = (flags_bits & 4) != 0;
            rec.trails_enabled = (flags_bits & 8) != 0;
            rec.causal_epoch = causal_epoch;

            let words = rec.to_words();
            let unpacked = RawTelemetryRecord::from_words(&words);
            prop_assert_eq!(unpacked, rec);

            let cell = TelemetryCell::new(rec);
            prop_assert_eq!(cell.read(), rec);
        }
    }
}
