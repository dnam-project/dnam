//! Core execution types (`Tier`, `FtzDazGuard`, and `DnamTelemetrySnapshot`).

use crate::control::TELEMETRY_CONTROL_SLOTS;

/// Slimmable execution tier (`Lite`, `Full`, `Pro`, or `Ultra`).
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Tier {
    /// **`Lite` ($C=4$ channels, $M=4$ tank lines, $D_h=4$ SVD modes, `10,752 B` `Arena 0`):**
    /// Mix-Ready / Live Grade (`168` cache lines).
    Lite = 0,
    /// **`Full` ($C=8$ channels, $M=8$ tank lines, $D_h=4$ SVD modes, `22,336 B` `Arena 0..1`):**
    /// True Studio Quality (`349` cache lines).
    Full = 1,
    /// **`Pro` ($C=12$ channels, $M=8$ tank lines, $D_h=4$ SVD modes, `36,224 B` `Arena 0..2`):**
    /// Mastering / Reference Grade (`566` cache lines, `8 + 4` SIMD loop).
    Pro = 2,
    /// **`Ultra` ($C=16$ channels, $M=16$ tank lines, $D_h=8$ SVD modes, `64,000 B` `Arena 0..3`):**
    /// Unconstrained Workstation Max (`1,000` cache lines, Rank-8 FM + 16-line Householder/Givens tank).
    Ultra = 3,
}

impl Tier {
    /// Converts a raw `u8` (`0`, `1`, `2`, `3`) into a [`Tier`], clamping values $\ge 3$ to [`Tier::Ultra`].
    #[inline]
    pub const fn from_u8(val: u8) -> Self {
        match val {
            0 => Self::Lite,
            1 => Self::Full,
            2 => Self::Pro,
            _ => Self::Ultra,
        }
    }

    /// Returns the canonical lowercase string identifier (`"lite"`, `"full"`, `"pro"`, `"ultra"`).
    #[inline]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Lite => "lite",
            Self::Full => "full",
            Self::Pro => "pro",
            Self::Ultra => "ultra",
        }
    }

    /// Parses a tier name (case-insensitive `"lite"`, `"full"`, `"pro"`, `"ultra"`).
    #[inline]
    pub fn from_str_name(s: &str) -> Option<Self> {
        if s.eq_ignore_ascii_case("lite") {
            Some(Self::Lite)
        } else if s.eq_ignore_ascii_case("full") {
            Some(Self::Full)
        } else if s.eq_ignore_ascii_case("pro") {
            Some(Self::Pro)
        } else if s.eq_ignore_ascii_case("ultra") {
            Some(Self::Ultra)
        } else {
            None
        }
    }

    /// Returns the number of active channels ($C \in \{4, 8, 12, 16\}$).
    #[inline]
    pub const fn channels(self) -> usize {
        match self {
            Self::Lite => 4,
            Self::Full => 8,
            Self::Pro => 12,
            Self::Ultra => 16,
        }
    }

    /// Alias for `channels`.
    #[inline]
    pub const fn active_channels(self) -> usize {
        self.channels()
    }

    /// Returns the number of 4-channel SIMD tier groups (`1`, `2`, `3`, or `4`).
    #[inline]
    pub const fn simd_groups(self) -> usize {
        match self {
            Self::Lite => 1,
            Self::Full => 2,
            Self::Pro => 3,
            Self::Ultra => 4,
        }
    }

    /// Returns the number of 4-line SIMD groups in Stage IV tank (`1` for `Lite`, `2` for `Full`/`Pro`, `4` for `Ultra`).
    #[inline]
    pub const fn tank_groups(self) -> usize {
        match self {
            Self::Lite => 1,
            Self::Full | Self::Pro => 2,
            Self::Ultra => 4,
        }
    }

    /// Returns the number of active Stage IV tank delay lines ($M \in \{4, 8, 8, 16\}$).
    #[inline]
    pub const fn tank_lines(self) -> usize {
        self.tank_groups() * 4
    }

    /// Returns the active SVD control subspace rank ($D_h = 4$ for `Lite`/`Full`/`Pro`, $D_h = 8$ for `Ultra`).
    #[inline]
    pub const fn svd_control_modes(self) -> usize {
        match self {
            Self::Lite | Self::Full | Self::Pro => 4,
            Self::Ultra => 8,
        }
    }

    /// Returns the active Factorization Machine (FM) interaction rank ($R = 4$ for `Lite`/`Full`/`Pro`, $R = 8$ for `Ultra`).
    #[inline]
    pub const fn fm_interaction_rank(self) -> usize {
        match self {
            Self::Lite | Self::Full | Self::Pro => 4,
            Self::Ultra => 8,
        }
    }

    /// Returns the cumulative arena byte end offset (`10,752`, `22,336`, `36,224`, or `64,000`).
    #[inline]
    pub const fn expected_arena_bytes(self) -> usize {
        match self {
            Self::Lite => 10_752,
            Self::Full => 22_336,
            Self::Pro => 36_224,
            Self::Ultra => 64_000,
        }
    }

    /// Returns the cumulative number of 64-byte cache lines (`168`, `349`, `566`, or `1,000`).
    #[inline]
    pub const fn expected_cache_lines(self) -> usize {
        self.expected_arena_bytes() / 64
    }
}

#[cfg(any(test, target_arch = "aarch64"))]
#[inline(always)]
const fn audio_fpcr(previous: u64) -> u64 {
    let fiz_if_ah = (previous >> 1) & 1;
    (previous & !((0b11_u64 << 22) | 0x9f00)) | (1_u64 << 24) | fiz_if_ah
}

#[cfg(any(test, target_arch = "x86_64"))]
#[inline(always)]
const fn audio_mxcsr(previous: u32) -> u32 {
    (previous & !(0x6000 | 0x003f)) | 0x8040 | 0x1f80
}

/// Thread-bound RAII guard that selects round-to-nearest, ties-to-even, non-trapping
/// floating-point exceptions, and hardware denormal protection for an audio block
/// on `aarch64` and `x86_64`. On AArch64, it preserves `FPCR.AH` and sets
/// `FPCR.FIZ` when `AH` is set, enabling input flushing for most floating-point
/// instructions on CPUs with `FEAT_AFP`.
/// It restores the caller's control register on drop. On `x86_64`, entry clears
/// the `MXCSR` sticky exception flags, and drop restores their saved values.
/// On `aarch64`, exception status is held separately in `FPSR` and is not restored.
///
/// Engine block methods create this guard automatically. When calling per-sample
/// engine, operator, or stage methods, or control ticks directly, hold one guard
/// on the processing thread for the entire batch. Drop it before yielding or
/// calling unrelated host code. On other architectures this guard is a no-op;
/// the target must supply the required rounding behavior.
///
/// ```
/// use dnam_engine::{FtzDazGuard, operators::OpTDelayLine};
///
/// let mut delay = OpTDelayLine::<2048>::new();
/// let input = [0.0_f32; 16];
/// let mut output = [0.0_f32; 16];
/// {
///     let _audio = FtzDazGuard::new();
///     for (out, sample) in output.iter_mut().zip(input) {
///         *out = delay.step(sample);
///     }
/// }
/// ```
#[derive(Debug)]
pub struct FtzDazGuard {
    #[cfg(target_arch = "aarch64")]
    prev_fpcr: u64,
    #[cfg(target_arch = "x86_64")]
    prev_mxcsr: u32,
    _thread_bound: core::marker::PhantomData<*mut ()>,
}

impl FtzDazGuard {
    /// Selects `RNE`, non-trapping exceptions and hardware `FTZ`/`DAZ` where available.
    #[inline(always)]
    pub fn new() -> Self {
        #[cfg(target_arch = "aarch64")]
        {
            let prev_fpcr: u64;
            // SAFETY: The ARM64 FPCR is saved; only rounding, trap-enable, FZ and conditional FIZ bits change.
            unsafe {
                core::arch::asm!("mrs {}, fpcr", out(reg) prev_fpcr, options(nomem, nostack, preserves_flags));
                let new_fpcr = audio_fpcr(prev_fpcr);
                if new_fpcr != prev_fpcr {
                    core::arch::asm!("msr fpcr, {}", in(reg) new_fpcr, options(nomem, nostack, preserves_flags));
                }
            }
            Self {
                prev_fpcr,
                _thread_bound: core::marker::PhantomData,
            }
        }
        #[cfg(target_arch = "x86_64")]
        {
            let mut prev_mxcsr: u32 = 0;
            // SAFETY: `stmxcsr` and `ldmxcsr` read/write the user-space SSE MXCSR control word.
            unsafe {
                core::arch::asm!("stmxcsr [{}]", in(reg) &mut prev_mxcsr, options(nostack, preserves_flags));
                let new_mxcsr = audio_mxcsr(prev_mxcsr);
                if new_mxcsr != prev_mxcsr {
                    core::arch::asm!("ldmxcsr [{}]", in(reg) &new_mxcsr, options(nostack, preserves_flags));
                }
            }
            Self {
                prev_mxcsr,
                _thread_bound: core::marker::PhantomData,
            }
        }
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            Self {
                _thread_bound: core::marker::PhantomData,
            }
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
            // SAFETY: Restores the saved FPCR even if it changed after entry.
            unsafe {
                core::arch::asm!("msr fpcr, {}", in(reg) self.prev_fpcr, options(nomem, nostack, preserves_flags));
            }
        }
        #[cfg(target_arch = "x86_64")]
        {
            // SAFETY: Restores the saved MXCSR, including flags raised after entry.
            unsafe {
                core::arch::asm!("ldmxcsr [{}]", in(reg) &self.prev_mxcsr, options(nostack, preserves_flags));
            }
        }
    }
}

/// Lock-free, copyable `#[repr(C)]` telemetry snapshot for 60 fps UI visualizers
/// (`ToraTelemetry.tsx`, `packages/plugin`, and `packages/js`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DnamTelemetrySnapshot {
    /// Active slimmable tier (`0 = Lite`, `1 = Full`, `2 = Pro`, `3 = Ultra`).
    pub active_tier: u8,
    /// `1` if stereo output head is active, `0` if mono output mode.
    pub stereo_out: u8,
    /// `1` if the engine is currently bypassed, `0` otherwise.
    pub bypassed: u8,
    /// `1` if Stage IV infinite freeze is currently engaged, `0` otherwise.
    pub is_frozen: u8,
    /// Bitmask of controls currently clamped by `[CLAMP-RANGE]`, `[CLAMP-SWITCH]`, or `[CLAMP-STABILITY]`.
    pub active_clamp_mask: u16,
    /// Current host control-bus decimation stride $K_{\text{ctrl}}(f_{s,\text{host}})$.
    pub control_decimation_host: u16,
    /// Peak `Op-N` normalized drive $|u_c| \in [0, 1]$ in the latest block.
    pub max_op_n_drive: f32,
    /// Peak `Op-M` modal pole radius $r_c[n] \le 0.9995$ in the latest block.
    pub max_pole_radius: f32,
    /// Peak Stage IV feedback loop gain $g_{\text{eff},m} \le 0.9990$ in the latest block.
    pub max_effective_loop_gain: f32,
    /// Reported $T_{60}$ decay tail time in milliseconds.
    pub tail_time_ms: f32,
    /// Current collapsed SVD head coordinates $\mathbf{z}_{\text{head}} \in \mathbb{R}^4$ ($\mathbf{z}_{0..3}$).
    pub z_head: [f32; 4],
    /// Current 3-band envelope follower states (`[Low, Mid, High, 0.0]`).
    pub env_states: [f32; 4],
    /// Current 4-harmonic Chebyshev LFO waves ($T_1..T_4$ for LFO 0 and LFO 1).
    pub lfo_harmonics: [f32; 8],
    /// Current clamped physical control values across the leading 16 C-ABI telemetry slots.
    pub clamped_controls: [f32; TELEMETRY_CONTROL_SLOTS],
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_fp_control_bit_transforms() {
        let fpcr_changed = (0b11_u64 << 22) | 0x9f00 | (1_u64 << 24) | 1;
        let fpcr_varied = fpcr_changed | 2;
        for preserved in [0, !fpcr_varied] {
            for traps in 0_u64..64 {
                for rounding in 0_u64..4 {
                    for denormal in 0_u64..2 {
                        for alternate in 0_u64..4 {
                            let host = preserved
                                | ((traps & 0x1f) << 8)
                                | ((traps & 0x20) << 10)
                                | (rounding << 22)
                                | (denormal << 24)
                                | alternate;
                            let active = audio_fpcr(host);
                            assert_eq!(active & 0x9f00, 0);
                            assert_eq!(active & (0b11 << 22), 0);
                            assert_ne!(active & (1 << 24), 0);
                            assert_eq!(
                                active & 1,
                                u64::from(alternate != 0),
                                "incoming AH/FIZ: {alternate:#04b}"
                            );
                            assert_eq!(active & !fpcr_changed, host & !fpcr_changed);
                            assert_eq!(audio_fpcr(active), active);
                        }
                    }
                }
            }
        }

        for masks in 0_u32..64 {
            for sticky in 0_u32..64 {
                for rounding in 0_u32..4 {
                    for denormal in 0_u32..4 {
                        let host = 0xa5a5_0000
                            | (masks << 7)
                            | sticky
                            | (rounding << 13)
                            | ((denormal & 1) << 6)
                            | ((denormal & 2) << 14);
                        let active = audio_mxcsr(host);
                        assert_eq!(active & 0x003f, 0);
                        assert_eq!(active & 0x1f80, 0x1f80);
                        assert_eq!(active & 0x6000, 0);
                        assert_eq!(active & 0x8040, 0x8040);
                        assert_eq!(active & 0xffff_0000, host & 0xffff_0000);
                    }
                }
            }
        }
    }

    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    fn read_fp_control() -> u64 {
        #[cfg(target_arch = "aarch64")]
        {
            let value: u64;
            // SAFETY: Reading the current thread's user-space FPCR does not change it.
            unsafe {
                core::arch::asm!("mrs {}, fpcr", out(reg) value, options(nomem, nostack, preserves_flags));
            }
            value
        }
        #[cfg(target_arch = "x86_64")]
        {
            let mut value = 0_u32;
            // SAFETY: The pointer refers to a writable MXCSR-sized local value.
            unsafe {
                core::arch::asm!("stmxcsr [{}]", in(reg) &mut value, options(nostack, preserves_flags));
            }
            u64::from(value)
        }
    }

    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    fn write_fp_control(value: u64) {
        #[cfg(target_arch = "aarch64")]
        {
            // SAFETY: The value is derived from the current FPCR with defined bits changed.
            unsafe {
                core::arch::asm!("msr fpcr, {}", in(reg) value, options(nomem, nostack, preserves_flags));
            }
        }
        #[cfg(target_arch = "x86_64")]
        {
            let value = value as u32;
            // SAFETY: The value is derived from the current MXCSR with defined bits changed.
            unsafe {
                core::arch::asm!("ldmxcsr [{}]", in(reg) &value, options(nostack, preserves_flags));
            }
        }
    }

    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    struct RestoreFpControl(u64);

    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    impl Drop for RestoreFpControl {
        fn drop(&mut self) {
            write_fp_control(self.0);
        }
    }

    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    fn expected_audio_fp_control(host: u64) -> u64 {
        #[cfg(target_arch = "aarch64")]
        {
            let mut expected = (host & !((0b11_u64 << 22) | 0x9f00)) | (1_u64 << 24);
            if host & 0b10 != 0 {
                expected |= 0b01;
            }
            expected
        }
        #[cfg(target_arch = "x86_64")]
        {
            (host & !(0x6000 | 0x003f)) | 0x8040 | 0x1f80
        }
    }

    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    #[test]
    fn test_fp_guard_selects_rne_and_restores_all_control_bits() {
        #[cfg(target_arch = "aarch64")]
        let (round_shift, denormal_bits, denormal_options, safe_original) = (
            22,
            1_u64 << 24,
            [0, 0, 1_u64 << 24, 1_u64 << 24],
            read_fp_control() & !0x9f00,
        );
        #[cfg(target_arch = "x86_64")]
        let (round_shift, denormal_bits, denormal_options, safe_original) = (
            13,
            0x8040_u64,
            [0, 0x0040, 0x8000, 0x8040],
            read_fp_control() | 0x1f80,
        );
        let _restore = RestoreFpControl(read_fp_control());
        let round_mask = 0b11_u64 << round_shift;

        for mode in 0..4 {
            for denormal in denormal_options {
                let requested = (safe_original & !(round_mask | denormal_bits))
                    | (mode << round_shift)
                    | denormal;
                write_fp_control(requested);
                let host = read_fp_control();
                assert_eq!(
                    host & (round_mask | denormal_bits),
                    requested & (round_mask | denormal_bits)
                );
                {
                    let _guard = FtzDazGuard::new();
                    let expected = expected_audio_fp_control(host);
                    assert_eq!(read_fp_control(), expected);
                    #[cfg(target_arch = "x86_64")]
                    write_fp_control(expected | 0x003f);
                    let outer = read_fp_control();
                    {
                        let _nested = FtzDazGuard::new();
                        assert_eq!(read_fp_control(), expected_audio_fp_control(outer));
                    }
                    assert_eq!(read_fp_control(), outer);

                    let one = std::hint::black_box(1.0_f32);
                    let half_ulp = std::hint::black_box(f32::from_bits(0x3380_0000));
                    assert_eq!(
                        std::hint::black_box(one + half_ulp).to_bits(),
                        1.0_f32.to_bits()
                    );
                    let min_normal = std::hint::black_box(f32::MIN_POSITIVE);
                    let half = std::hint::black_box(0.5_f32);
                    assert_eq!(std::hint::black_box(min_normal * half).to_bits(), 0);
                    let subnormal = std::hint::black_box(f32::from_bits(0x007f_ffff));
                    let two = std::hint::black_box(2.0_f32);
                    assert_eq!(std::hint::black_box(subnormal * two).to_bits(), 0);
                }
                assert_eq!(read_fp_control(), host);
            }
        }

        write_fp_control(safe_original);
        let host = read_fp_control();
        let unwound = std::panic::catch_unwind(|| {
            let _guard = FtzDazGuard::new();
            panic!("exercise floating-point guard unwinding");
        });
        assert!(unwound.is_err());
        assert_eq!(read_fp_control(), host);
    }

    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    #[test]
    fn test_fp_guard_restores_changes_after_unchanged_entry() {
        let original = read_fp_control();
        let _restore = RestoreFpControl(original);
        write_fp_control(expected_audio_fp_control(original));
        let host = read_fp_control();
        assert_eq!(expected_audio_fp_control(host), host);

        {
            let _guard = FtzDazGuard::new();
            assert_eq!(read_fp_control(), host);
            #[cfg(target_arch = "x86_64")]
            {
                let one = std::hint::black_box(1.0_f32);
                let half_ulp = std::hint::black_box(f32::from_bits(0x3380_0000));
                assert_eq!(
                    std::hint::black_box(one + half_ulp).to_bits(),
                    1.0_f32.to_bits()
                );
                assert_ne!(read_fp_control() & 0x0020, 0);
            }
            #[cfg(target_arch = "aarch64")]
            {
                write_fp_control(host ^ (1 << 24));
                assert_ne!(read_fp_control(), host);
            }
        }
        assert_eq!(read_fp_control(), host);

        {
            let _guard = FtzDazGuard::new();
            #[cfg(target_arch = "aarch64")]
            let changed = host ^ (1 << 22);
            #[cfg(target_arch = "x86_64")]
            let changed = host ^ (1 << 13);
            write_fp_control(changed);
            assert_eq!(read_fp_control(), changed);
        }
        assert_eq!(read_fp_control(), host);
    }

    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    #[test]
    fn test_standalone_sample_batch_uses_caller_fp_guard() {
        use crate::engine::DnamEngine;
        use crate::math::ScalarBackend;
        use crate::operators::SymplecticModal4Lane;
        use crate::stages::TankDcBlocker;

        let original = read_fp_control();
        let _restore = RestoreFpControl(original);
        let mut engine = DnamEngine::<ScalarBackend>::new_empty(Tier::Lite);
        let mut modal = std::hint::black_box(SymplecticModal4Lane::new());
        let mut dc = std::hint::black_box(TankDcBlocker::new());
        #[cfg(target_arch = "aarch64")]
        let requested = (original & !(0x9f00 | (0b11 << 22) | (1 << 24))) | (0b01 << 22);
        #[cfg(target_arch = "x86_64")]
        let requested = ((original | 0x1f80) & !(0x8040 | 0x003f | (0b11 << 13))) | (0b10 << 13);
        write_fp_control(requested);
        let host = read_fp_control();

        {
            let _guard = FtzDazGuard::new();
            for _ in 0..4 {
                engine.ctrl_sub_counter = 0;
                engine.ctrl_energy_accum = std::hint::black_box(1.0);
                let half_ulp = std::hint::black_box(f32::from_bits(0x3380_0000));
                let _ = engine.step_frame_tora(half_ulp, half_ulp, false);
                assert_eq!(engine.ctrl_energy_accum.to_bits(), 1.0_f32.to_bits());

                engine.control_bus.env_poles[0] = std::hint::black_box(0.5);
                engine.control_bus.env_states[0] = std::hint::black_box(2.0);
                let _ = engine
                    .control_bus
                    .tick_3khz(std::hint::black_box(f32::EPSILON), &[]);
                assert_eq!(
                    engine.control_bus.env_states[0].to_bits(),
                    1.0_f32.to_bits()
                );

                let tiny = std::hint::black_box(f32::from_bits(1));
                assert_eq!(modal.step_lane(0, tiny).0.to_bits(), 0);
                dc.y_prev = std::hint::black_box(f32::MIN_POSITIVE);
                dc.r_dc = std::hint::black_box(0.5);
                assert_eq!(dc.step(std::hint::black_box(0.0)).to_bits(), 0);
            }
            let active = read_fp_control();
            #[cfg(target_arch = "aarch64")]
            assert_eq!(active, expected_audio_fp_control(host));
            #[cfg(target_arch = "x86_64")]
            assert_eq!(active & !0x003f, expected_audio_fp_control(host));
        }
        assert_eq!(read_fp_control(), host);
    }

    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    #[test]
    fn test_fp_guard_masks_exceptions_and_restores_host_status() {
        const CHILD_ENV: &str = "DNAM_FP_GUARD_EXCEPTION_PROBE";
        if std::env::var_os(CHILD_ENV).is_none() {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "types::tests::test_fp_guard_masks_exceptions_and_restores_host_status",
                    "--nocapture",
                ])
                .env(CHILD_ENV, "1")
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "floating-point exception probe failed: {}\n{}\n{}",
                output.status,
                std::string::String::from_utf8_lossy(&output.stdout),
                std::string::String::from_utf8_lossy(&output.stderr)
            );
            std::eprint!("{}", std::string::String::from_utf8_lossy(&output.stderr));
            return;
        }

        let original = read_fp_control();
        let _restore = RestoreFpControl(original);
        #[cfg(target_arch = "aarch64")]
        let (safe, mask_bits) = (
            original & !0x9f00,
            [0x0100, 0x0200, 0x0400, 0x0800, 0x1000, 0x8000],
        );
        #[cfg(target_arch = "x86_64")]
        let (safe, mask_bits) = (
            original | 0x1f80,
            [0x0080, 0x0100, 0x0200, 0x0400, 0x0800, 0x1000],
        );

        let mut writable_traps = 0;
        for pattern in 0_u64..64 {
            let mask = mask_bits
                .iter()
                .enumerate()
                .fold(0, |acc, (i, bit)| acc | (bit * ((pattern >> i) & 1)));
            #[cfg(target_arch = "aarch64")]
            let requested = safe | mask;
            #[cfg(target_arch = "x86_64")]
            let requested = ((safe & !0x003f) | pattern) & !mask;
            write_fp_control(requested);
            let host = read_fp_control();
            let (active, after_arithmetic, nested_active, nested_restored, results) = {
                let _guard = FtzDazGuard::new();
                let active = read_fp_control();
                let zero = std::hint::black_box(0.0_f32);
                let one = std::hint::black_box(1.0_f32);
                let two = std::hint::black_box(2.0_f32);
                let min = std::hint::black_box(f32::MIN_POSITIVE);
                let tiny = std::hint::black_box(f32::from_bits(1));
                let tenth = std::hint::black_box(0.1_f32);
                let half_ulp = std::hint::black_box(f32::from_bits(0x3380_0000));
                let maximum = std::hint::black_box(f32::MAX);
                let results = [
                    std::hint::black_box(zero / zero).to_bits(),
                    std::hint::black_box(one / zero).to_bits(),
                    std::hint::black_box(maximum * two).to_bits(),
                    std::hint::black_box(min * tenth).to_bits(),
                    std::hint::black_box(one + half_ulp).to_bits(),
                    std::hint::black_box(tiny * two).to_bits(),
                ];
                let after_arithmetic = read_fp_control();
                let nested_active = {
                    let _nested = FtzDazGuard::new();
                    read_fp_control()
                };
                (
                    active,
                    after_arithmetic,
                    nested_active,
                    read_fp_control(),
                    results,
                )
            };
            let restored = read_fp_control();
            write_fp_control(safe);

            assert_eq!(active, expected_audio_fp_control(host));
            assert_eq!(nested_active, expected_audio_fp_control(after_arithmetic));
            assert_eq!(nested_restored, after_arithmetic);
            assert_eq!(restored, host);
            #[cfg(target_arch = "x86_64")]
            {
                writable_traps |= !host & mask;
                assert_eq!(host & 0x003f, pattern);
                assert_eq!(active & 0x003f, 0);
                assert_ne!(after_arithmetic & 0x003f, 0);
            }
            #[cfg(target_arch = "aarch64")]
            {
                writable_traps |= host & 0x9f00;
            }
            assert_eq!(results[0] & 0x7f80_0000, 0x7f80_0000);
            assert_ne!(results[0] & 0x007f_ffff, 0);
            assert_eq!(results[1], f32::INFINITY.to_bits());
            assert_eq!(results[2], f32::INFINITY.to_bits());
            assert_eq!(results[3], 0);
            assert_eq!(results[4], 1.0_f32.to_bits());
            assert_eq!(results[5], 0);
        }
        #[cfg(target_arch = "aarch64")]
        std::eprintln!("writable AArch64 FPCR trap-enable bits: {writable_traps:#06x}/0x9f00");
        #[cfg(target_arch = "x86_64")]
        std::eprintln!("writable x86-64 MXCSR trap-enable bits: {writable_traps:#06x}/0x1f80");

        let hook = std::panic::take_hook();
        std::panic::set_hook(std::boxed::Box::new(|_| {}));
        #[cfg(target_arch = "aarch64")]
        let requested = safe | 0x9f00;
        #[cfg(target_arch = "x86_64")]
        let requested = (safe | 0x003f) & !0x1f80;
        write_fp_control(requested);
        let host = read_fp_control();
        let unwound = std::panic::catch_unwind(|| {
            let _guard = FtzDazGuard::new();
            panic!("exercise floating-point guard unwinding with host traps enabled");
        });
        let restored = read_fp_control();
        write_fp_control(safe);
        std::panic::set_hook(hook);
        assert!(unwound.is_err());
        assert_eq!(restored, host);

        fn check_callbacks<B: crate::math::MathBackend>(safe: u64) {
            use crate::engine::DnamEngine;

            let mut engine = DnamEngine::<B>::new_empty(Tier::Lite);
            for expired in [false, true] {
                for entry in 0..4 {
                    engine.preview_guard.is_expired = expired;
                    engine.ctrl_sub_counter = 0;
                    engine.ctrl_energy_accum = 0.0;
                    let input = [std::hint::black_box(f32::MAX), std::hint::black_box(0.1)];
                    let mut left = input;
                    let mut right = input;
                    #[cfg(target_arch = "aarch64")]
                    let requested = safe | 0x9f00;
                    #[cfg(target_arch = "x86_64")]
                    let requested = (safe | 0x003f) & !0x1f80;
                    write_fp_control(requested);
                    let host = read_fp_control();
                    match entry {
                        0 => engine.process_block(&input, &input, &mut left, &mut right),
                        1 => engine.process_mono_to_stereo(&input, &mut left, &mut right),
                        2 => engine.process_mono(&input, &mut left),
                        _ => engine.process_block_in_place(&mut left, &mut right),
                    }
                    let restored = read_fp_control();
                    write_fp_control(safe);
                    assert_eq!(restored, host);
                }
            }
        }

        check_callbacks::<crate::math::ScalarBackend>(safe);
        check_callbacks::<crate::math::DefaultMathBackend>(safe);
    }

    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    #[test]
    fn test_block_entry_points_restore_fp_control_on_normal_and_early_exit() {
        use crate::engine::DnamEngine;
        use crate::math::ScalarBackend;

        #[cfg(target_arch = "aarch64")]
        let requested = (read_fp_control() & !(0x9f00 | (0b11 << 22) | (1 << 24))) | (0b01 << 22);
        #[cfg(target_arch = "x86_64")]
        let requested = ((read_fp_control() | 0x1f80) & !(0x8040 | (0b11 << 13))) | (0b01 << 13);
        let _restore = RestoreFpControl(read_fp_control());
        let mut engine = DnamEngine::<ScalarBackend>::new_empty(Tier::Lite);
        engine.set_bypass(true, false);
        let input = [0.25_f32, -0.5];
        let mut left = [0.0; 2];
        let mut right = [0.0; 2];

        for expired in [false, true] {
            engine.preview_guard.is_expired = expired;
            write_fp_control(requested);
            let host = read_fp_control();
            engine.process_block(&input, &input, &mut left, &mut right);
            assert_eq!(read_fp_control(), host);
            engine.process_mono_to_stereo(&input, &mut left, &mut right);
            assert_eq!(read_fp_control(), host);
            engine.process_mono(&input, &mut left);
            assert_eq!(read_fp_control(), host);
            engine.process_block_in_place(&mut left, &mut right);
            assert_eq!(read_fp_control(), host);
        }
    }

    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    #[test]
    fn test_active_block_entry_points_select_rne_and_denormal_mode() {
        use crate::engine::DnamEngine;
        use crate::math::{DefaultMathBackend, MathBackend, ScalarBackend};

        fn check_backend<B: MathBackend>(requested: u64) {
            let mut engine = DnamEngine::<B>::new_empty(Tier::Lite);
            for entry in 0..4 {
                for (sample, initial, expected_bits) in [
                    (f32::from_bits(0x3380_0000), 1.0_f32, 1.0_f32.to_bits()),
                    (f32::from_bits(0x0040_0000), 0.0_f32, 0_u32),
                ] {
                    engine.ctrl_sub_counter = 0;
                    engine.ctrl_energy_accum = std::hint::black_box(initial);
                    let input = [std::hint::black_box(sample)];
                    let mut left = input;
                    let mut right = input;
                    write_fp_control(requested);
                    let host = read_fp_control();
                    match entry {
                        0 => engine.process_block(&input, &input, &mut left, &mut right),
                        1 => engine.process_mono_to_stereo(&input, &mut left, &mut right),
                        2 => engine.process_mono(&input, &mut left),
                        _ => engine.process_block_in_place(&mut left, &mut right),
                    }
                    assert_eq!(read_fp_control(), host);
                    assert_eq!(engine.ctrl_energy_accum.to_bits(), expected_bits);
                }
            }
        }

        #[cfg(target_arch = "aarch64")]
        let requested = (read_fp_control() & !(0x9f00 | (0b11 << 22) | (1 << 24))) | (0b01 << 22);
        #[cfg(target_arch = "x86_64")]
        let requested = ((read_fp_control() | 0x1f80) & !(0x8040 | (0b11 << 13))) | (0b10 << 13);
        let _restore = RestoreFpControl(read_fp_control());
        check_backend::<ScalarBackend>(requested);
        check_backend::<DefaultMathBackend>(requested);
    }

    #[test]
    fn test_tier_conversions_and_ftz_daz_guard() {
        assert_eq!(Tier::from_u8(0), Tier::Lite);
        assert_eq!(Tier::from_u8(1), Tier::Full);
        assert_eq!(Tier::from_u8(2), Tier::Pro);
        assert_eq!(Tier::from_u8(3), Tier::Ultra);
        assert_eq!(Tier::from_u8(99), Tier::Ultra);

        assert_eq!(Tier::Lite.as_str(), "lite");
        assert_eq!(Tier::Full.as_str(), "full");
        assert_eq!(Tier::Pro.as_str(), "pro");
        assert_eq!(Tier::Ultra.as_str(), "ultra");

        assert_eq!(Tier::from_str_name("LITE"), Some(Tier::Lite));
        assert_eq!(Tier::from_str_name("Full"), Some(Tier::Full));
        assert_eq!(Tier::from_str_name("pro"), Some(Tier::Pro));
        assert_eq!(Tier::from_str_name("ultra"), Some(Tier::Ultra));
        assert_eq!(Tier::from_str_name("invalid"), None);

        assert_eq!(Tier::Lite.channels(), 4);
        assert_eq!(Tier::Full.channels(), 8);
        assert_eq!(Tier::Pro.channels(), 12);
        assert_eq!(Tier::Ultra.channels(), 16);

        assert_eq!(Tier::Lite.active_channels(), 4);
        assert_eq!(Tier::Ultra.active_channels(), 16);

        assert_eq!(Tier::Lite.simd_groups(), 1);
        assert_eq!(Tier::Full.simd_groups(), 2);
        assert_eq!(Tier::Pro.simd_groups(), 3);
        assert_eq!(Tier::Ultra.simd_groups(), 4);

        assert_eq!(Tier::Lite.tank_groups(), 1);
        assert_eq!(Tier::Full.tank_groups(), 2);
        assert_eq!(Tier::Pro.tank_groups(), 2);
        assert_eq!(Tier::Ultra.tank_groups(), 4);

        assert_eq!(Tier::Lite.tank_lines(), 4);
        assert_eq!(Tier::Full.tank_lines(), 8);
        assert_eq!(Tier::Pro.tank_lines(), 8);
        assert_eq!(Tier::Ultra.tank_lines(), 16);

        assert_eq!(Tier::Lite.svd_control_modes(), 4);
        assert_eq!(Tier::Pro.svd_control_modes(), 4);
        assert_eq!(Tier::Ultra.svd_control_modes(), 8);
        assert_eq!(Tier::Lite.fm_interaction_rank(), 4);
        assert_eq!(Tier::Full.fm_interaction_rank(), 4);
        assert_eq!(Tier::Pro.fm_interaction_rank(), 4);
        assert_eq!(Tier::Ultra.fm_interaction_rank(), 8);

        assert_eq!(Tier::Lite.expected_arena_bytes(), 10_752);
        assert_eq!(Tier::Full.expected_arena_bytes(), 22_336);
        assert_eq!(Tier::Pro.expected_arena_bytes(), 36_224);
        assert_eq!(Tier::Ultra.expected_arena_bytes(), 64_000);

        assert_eq!(Tier::Lite.expected_cache_lines(), 168);
        assert_eq!(Tier::Full.expected_cache_lines(), 349);
        assert_eq!(Tier::Pro.expected_cache_lines(), 566);
        assert_eq!(Tier::Ultra.expected_cache_lines(), 1000);

        let guard1 = FtzDazGuard::default();
        let guard2 = FtzDazGuard::new();
        drop(guard2);
        drop(guard1);
    }
}
