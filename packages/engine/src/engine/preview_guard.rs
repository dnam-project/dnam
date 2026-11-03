//! Evaluation Preview & Trial Guard (`PreviewGuard`).
//!
//! Provides sample-accurate trial timeouts, evaluation periodic mutes, and trial watermark flags.

use crate::schema::DnamProtectionProfile;

/// Real-time evaluation preview & trial protection guard state.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PreviewGuard {
    /// Active protection profile.
    pub profile: DnamProtectionProfile,
    /// Accumulated evaluation frames processed.
    pub elapsed_frames: u64,
    /// Host sample rate in Hz.
    pub sample_rate_hz: u32,
    /// Trial evaluation limit in seconds (0 = unlimited).
    pub limit_seconds: u32,
    /// Whether trial timeout has expired (forces complete mute).
    pub is_expired: bool,
    /// Periodic trial beep / drop counter.
    pub beep_phase: u32,
}

impl Default for PreviewGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl PreviewGuard {
    /// Creates a default unrestricted [`PreviewGuard`].
    pub const fn new() -> Self {
        Self {
            profile: DnamProtectionProfile::Open,
            elapsed_frames: 0,
            sample_rate_hz: 48_000,
            limit_seconds: 0,
            is_expired: false,
            beep_phase: 0,
        }
    }

    /// Configures the guard for a given protection profile and sample rate.
    pub fn configure(&mut self, profile: DnamProtectionProfile, sample_rate_hz: u32) {
        self.profile = profile;
        self.sample_rate_hz = sample_rate_hz.max(1);
        self.limit_seconds = match profile {
            DnamProtectionProfile::Open => 0,
            DnamProtectionProfile::DeviceAllowlisted => 0,
            DnamProtectionProfile::EnclaveSealed => 0,
        };
        self.elapsed_frames = 0;
        self.is_expired = false;
        self.beep_phase = 0;
    }

    /// Resets runtime counters to sample 0.
    pub fn reset(&mut self) {
        self.elapsed_frames = 0;
        self.is_expired = false;
        self.beep_phase = 0;
    }

    /// Advances the guard by `num_frames` and returns whether audio should be passed (`true`) or muted (`false`).
    #[inline]
    pub fn advance_block(&mut self, num_frames: usize) -> bool {
        if self.is_expired {
            return false;
        }

        self.elapsed_frames = self.elapsed_frames.saturating_add(num_frames as u64);
        if self.limit_seconds > 0 {
            let limit_frames = (self.limit_seconds as u64) * (self.sample_rate_hz as u64);
            if self.elapsed_frames >= limit_frames {
                self.is_expired = true;
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preview_guard_lifecycle() {
        let mut guard = PreviewGuard::default();
        assert!(guard.advance_block(1024));

        guard.configure(DnamProtectionProfile::EnclaveSealed, 48_000);
        assert_eq!(guard.limit_seconds, 0);

        guard.configure(DnamProtectionProfile::DeviceAllowlisted, 48_000);
        guard.limit_seconds = 1; // 1 second = 48,000 frames
        assert!(guard.advance_block(40_000));
        assert!(!guard.is_expired);
        assert!(!guard.advance_block(10_000));
        assert!(guard.is_expired);
        // Subsequent advance_block calls when already expired return false immediately
        assert!(!guard.advance_block(128));

        guard.reset();
        assert!(!guard.is_expired);
    }
}
