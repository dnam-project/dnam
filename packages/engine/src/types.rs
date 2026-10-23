//! Core execution types (`Tier`, `FtzDazGuard`, and `DnamTelemetrySnapshot`).

use crate::control::MAX_CONTROLS;

/// Slimmable execution tier (`Lite`, `Full`, or `Ultra`).
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Tier {
    /// **`Lite` ($C=4$ Core 1 channels, $M=4$ FDN lines, `10.5 KB` `Arena 0`):**
    /// Ultra-low-power tier (`~920`–`1,850 MACs/sample`).
    Lite = 0,
    /// **`Full` ($C=8$ Core 1 channels, $M=8$ FDN lines, `21.8 KB` `Arena 0..1`):**
    /// Standard real-time studio/pedalboard tier (`~3,150`–`4,625 MACs/sample`).
    Full = 1,
    /// **`Ultra` ($C=12$ Core 1 channels, $M=8$ FDN lines, `35.4 KB` `Arena 0..2`):**
    /// Mastering-grade tier (`~6,300`–`7,775 MACs/sample`, `8 + 4` SIMD loop).
    Ultra = 2,
}

impl Tier {
    /// Converts a raw `u8` (`0`, `1`, `2`) into a [`Tier`], clamping values $\ge 2$ to [`Tier::Ultra`].
    #[inline]
    pub const fn from_u8(val: u8) -> Self {
        match val {
            0 => Self::Lite,
            1 => Self::Full,
            _ => Self::Ultra,
        }
    }

    /// Returns the number of active Core 1 channels ($C \in \{4, 8, 12\}$).
    #[inline]
    pub const fn channels(self) -> usize {
        match self {
            Self::Lite => 4,
            Self::Full => 8,
            Self::Ultra => 12,
        }
    }

    /// Returns the number of 4-channel SIMD tier groups in Core 1 (`1`, `2`, or `3`).
    #[inline]
    pub const fn core1_groups(self) -> usize {
        match self {
            Self::Lite => 1,
            Self::Full => 2,
            Self::Ultra => 3,
        }
    }

    /// Returns the number of 4-line SIMD groups in Core 2 `M-UFDN` (`1` for `Lite`, `2` for `Full`/`Ultra`).
    #[inline]
    pub const fn fdn_groups(self) -> usize {
        match self {
            Self::Lite => 1,
            Self::Full | Self::Ultra => 2,
        }
    }
}

/// Scoped RAII guard that enables hardware **Flush-to-Zero (`FTZ`)** and **Denormals-Are-Zero (`DAZ`)**
/// on `aarch64` (`FPCR` bit 24) and `x86_64` (`MXCSR` bits 6 & 15 = `0x8040`) for the duration of an
/// audio block, restoring the caller's previous floating-point control register on drop.
#[derive(Debug)]
pub struct FtzDazGuard {
    #[cfg(target_arch = "aarch64")]
    prev_fpcr: u64,
    #[cfg(target_arch = "x86_64")]
    prev_mxcsr: u32,
}

impl FtzDazGuard {
    /// Enables hardware `FTZ`/`DAZ` for the current scope.
    #[inline(always)]
    pub fn new() -> Self {
        #[cfg(target_arch = "aarch64")]
        {
            let prev_fpcr: u64;
            // SAFETY: Reading and setting bit 24 (FZ) of the ARM64 user-space FPCR register is safe
            // and prevents subnormal traps inside recursive IIR/RG-LRU/FDN filters.
            unsafe {
                core::arch::asm!("mrs {}, fpcr", out(reg) prev_fpcr, options(nomem, nostack, preserves_flags));
                let new_fpcr = prev_fpcr | (1_u64 << 24);
                if new_fpcr != prev_fpcr {
                    core::arch::asm!("msr fpcr, {}", in(reg) new_fpcr, options(nomem, nostack, preserves_flags));
                }
            }
            Self { prev_fpcr }
        }
        #[cfg(target_arch = "x86_64")]
        {
            let mut prev_mxcsr: u32 = 0;
            // SAFETY: `stmxcsr` and `ldmxcsr` read/write the user-space SSE MXCSR control word.
            unsafe {
                core::arch::asm!("stmxcsr [{}]", in(reg) &mut prev_mxcsr, options(nostack, preserves_flags));
                let new_mxcsr = prev_mxcsr | 0x8040;
                if new_mxcsr != prev_mxcsr {
                    core::arch::asm!("ldmxcsr [{}]", in(reg) &new_mxcsr, options(nostack, preserves_flags));
                }
            }
            Self { prev_mxcsr }
        }
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            Self {}
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
            // SAFETY: Restores the caller's original FPCR value saved in `FtzDazGuard::new`.
            unsafe {
                core::arch::asm!("msr fpcr, {}", in(reg) self.prev_fpcr, options(nomem, nostack, preserves_flags));
            }
        }
        #[cfg(target_arch = "x86_64")]
        {
            // SAFETY: Restores the caller's original MXCSR value saved in `FtzDazGuard::new`.
            unsafe {
                core::arch::asm!("ldmxcsr [{}]", in(reg) &self.prev_mxcsr, options(nostack, preserves_flags));
            }
        }
    }
}

/// Lock-free, copyable `#[repr(C)]` telemetry snapshot for 60 fps UI visualizers
/// (`TriPhysicsTelemetry.tsx`, `packages/plugin`, and `packages/nodejs`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DnamTelemetrySnapshot {
    /// Active slimmable tier (`0 = Lite`, `1 = Full`, `2 = Ultra`).
    pub active_tier: u8,
    /// `1` if Core 2 (`M-UFDN`) is active, `0` if bypassed (`0 KB` delay RAM).
    pub core2_active: u8,
    /// `1` if $3\text{ kHz}$ LFOs and envelope followers are active, `0` otherwise.
    pub lfo_env_active: u8,
    /// `1` if stereo output head is active, `0` if mono output mode.
    pub stereo_out: u8,
    /// Bitmask of controls currently clamped by `[CLAMP-RANGE]`, `[CLAMP-SWITCH]`, or `[CLAMP-STABILITY]`.
    pub active_clamp_mask: u16,
    /// Current host control-bus decimation stride $K_{\text{ctrl}}(f_{s,\text{host}})$.
    pub control_decimation_host: u16,
    /// Peak `PolyADAA` normalized drive $|u_c| \in [0, 1]$ in the latest block.
    pub max_adaa_drive: f32,
    /// Peak `SelectiveRgLru` pole radius $r_c[n] \le 0.9995$ in the latest block.
    pub max_pole_radius: f32,
    /// Peak `M-UFDN` feedback loop gain $g_{\text{eff},m} \le 0.9990$ in the latest block.
    pub max_effective_loop_gain: f32,
    /// Reported $T_{60}$ decay tail time in milliseconds.
    pub tail_time_ms: f32,
    /// Current collapsed SVD head coordinates $\mathbf{z}_{\text{head}} \in \mathbb{R}^4$.
    pub z_head: [f32; 4],
    /// Current 3-band envelope follower states (`[Low, Mid, High, 0.0]`).
    pub env_states: [f32; 4],
    /// Current 4-harmonic Chebyshev LFO waves ($T_1..T_4$ for LFO 0 and LFO 1).
    pub lfo_harmonics: [f32; 8],
    /// Current clamped physical control values across all 16 slots.
    pub clamped_controls: [f32; MAX_CONTROLS],
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tier_conversions_and_ftz_daz_guard() {
        assert_eq!(Tier::from_u8(0), Tier::Lite);
        assert_eq!(Tier::from_u8(1), Tier::Full);
        assert_eq!(Tier::from_u8(2), Tier::Ultra);
        assert_eq!(Tier::from_u8(99), Tier::Ultra);

        assert_eq!(Tier::Lite.channels(), 4);
        assert_eq!(Tier::Full.channels(), 8);
        assert_eq!(Tier::Ultra.channels(), 12);

        assert_eq!(Tier::Lite.core1_groups(), 1);
        assert_eq!(Tier::Full.core1_groups(), 2);
        assert_eq!(Tier::Ultra.core1_groups(), 3);

        assert_eq!(Tier::Lite.fdn_groups(), 1);
        assert_eq!(Tier::Full.fdn_groups(), 2);
        assert_eq!(Tier::Ultra.fdn_groups(), 2);

        let guard1 = FtzDazGuard::default();
        let guard2 = FtzDazGuard::new();
        drop(guard2);
        drop(guard1);
    }
}
