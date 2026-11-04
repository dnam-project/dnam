//! Host Tempo & Musical Subdivision Synchronization (`TempoSyncState`).

use crate::schema::{DnamJsonHeader, TempoSubdivision};

/// Host tempo synchronization and tap-tempo tracking state.
#[derive(Clone, Debug, PartialEq)]
pub struct TempoSyncState {
    /// Active host tempo in Beats Per Minute (BPM).
    pub current_bpm: f32,
    /// Sample rate in Hz.
    pub sample_rate_hz: u32,
    /// Tap tempo circular buffer of frame intervals.
    tap_intervals: [u32; 4],
    tap_idx: usize,
    last_tap_frame: Option<u64>,
    pub(crate) total_frames: u64,
}

impl Default for TempoSyncState {
    fn default() -> Self {
        Self::new()
    }
}

impl TempoSyncState {
    /// Creates a default [`TempoSyncState`] at 120.0 BPM and 48,000 Hz.
    pub const fn new() -> Self {
        Self {
            current_bpm: 120.0,
            sample_rate_hz: 48_000,
            tap_intervals: [0; 4],
            tap_idx: 0,
            last_tap_frame: None,
            total_frames: 0,
        }
    }

    /// Resets tap-tempo history.
    pub fn reset(&mut self) {
        self.tap_intervals = [0; 4];
        self.tap_idx = 0;
        self.last_tap_frame = None;
        self.total_frames = 0;
    }

    /// Sets the host BPM.
    pub fn set_bpm(&mut self, bpm: f32) {
        self.current_bpm = bpm.clamp(20.0, 400.0);
    }

    /// Advances frame counter by `num_frames`.
    #[inline]
    pub fn advance_frames(&mut self, num_frames: usize) {
        self.total_frames = self.total_frames.saturating_add(num_frames as u64);
    }

    /// Registers a tap event and updates `current_bpm` from the interval moving average.
    pub fn register_tap(&mut self) -> f32 {
        let now = self.total_frames;
        if let Some(last) = self.last_tap_frame
            && now > last
        {
            let interval = (now - last).min(u32::MAX as u64) as u32;
            let min_interval = self.sample_rate_hz / 7; // Max ~420 BPM
            let max_interval = self.sample_rate_hz.saturating_mul(3); // Min ~20 BPM
            if interval > max_interval {
                // Pause > 3s starts a fresh tap-tempo sequence without stale interval history
                self.tap_intervals = [0; 4];
                self.tap_idx = 0;
            } else if interval >= min_interval {
                self.tap_intervals[self.tap_idx] = interval;
                self.tap_idx = (self.tap_idx + 1) % 4;

                let mut sum = 0u32;
                let mut count = 0u32;
                for &intv in &self.tap_intervals {
                    if intv > 0 {
                        sum += intv;
                        count += 1;
                    }
                }
                if count > 0 {
                    let avg_interval = (sum as f32) / (count as f32);
                    let bpm = (60.0 * (self.sample_rate_hz as f32)) / avg_interval;
                    self.current_bpm = bpm.clamp(20.0, 400.0);
                }
            }
        }
        self.last_tap_frame = Some(now);
        self.current_bpm
    }

    /// Synchronizes all calibrated controls in `header` matching `(bpm, subdivision_override)`.
    pub fn resolve_sync_updates(
        header: &DnamJsonHeader,
        bpm: f32,
        subdivision_override: Option<TempoSubdivision>,
        fold_octave: bool,
    ) -> alloc::vec::Vec<(usize, f32)> {
        let mut buf = [(0usize, 0.0f32); 512];
        let n = Self::resolve_sync_updates_into(
            header,
            bpm,
            subdivision_override,
            fold_octave,
            &mut buf,
        );
        buf[..n].to_vec()
    }

    /// Zero-allocation helper that writes `(ctrl_idx, physical_value)` tempo updates into `out`.
    pub fn resolve_sync_updates_into(
        header: &DnamJsonHeader,
        bpm: f32,
        subdivision_override: Option<TempoSubdivision>,
        fold_octave: bool,
        out: &mut [(usize, f32)],
    ) -> usize {
        let mut clamp_mask = 0u16;
        Self::resolve_sync_updates_with_clamp_mask_into(
            header,
            bpm,
            subdivision_override,
            fold_octave,
            out,
            &mut clamp_mask,
        )
    }

    /// Zero-allocation helper that writes `(ctrl_idx, physical_value)` tempo updates into `out`
    /// and sets bits in `clamp_mask` for any control whose target was clamped to its reachable bound
    /// (`ref[dsp.ctrl.range.f32-rank-bounds]`, `ref[dsp.tempo.physical-inversion]`).
    pub fn resolve_sync_updates_with_clamp_mask_into(
        header: &DnamJsonHeader,
        bpm: f32,
        subdivision_override: Option<TempoSubdivision>,
        fold_octave: bool,
        out: &mut [(usize, f32)],
        clamp_mask: &mut u16,
    ) -> usize {
        let mut count = 0;
        for (ctrl_idx, ctrl_def) in header.controls_schema.iter().enumerate() {
            let Some(ref cal) = ctrl_def.calibration else {
                continue;
            };
            let Some(res) = cal.resolve_tempo_sync(bpm, subdivision_override, fold_octave) else {
                continue;
            };
            if res.was_clamped && ctrl_idx < 16 {
                *clamp_mask |= 1u16 << ctrl_idx;
            }
            if let Some((sw_id, opt_label)) = res.switch_selection
                && let Some((sw_idx, sw_def)) = header
                    .controls_schema
                    .iter()
                    .enumerate()
                    .find(|(_, c)| c.id == sw_id)
                && let Some(ref opts) = sw_def.declared_options
                && let Some(opt_idx) = opts.iter().position(|o| o == &opt_label)
                && count < out.len()
            {
                out[count] = (sw_idx, opt_idx as f32);
                count += 1;
            }
            if count < out.len() {
                let decl = ctrl_def.declared_range.unwrap_or([0.0, 10.0]);
                let phys = decl[0] + 0.5 * (res.normalized_u + 1.0) * (decl[1] - decl[0]);
                out[count] = (ctrl_idx, phys);
                count += 1;
            }
        }
        count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tempo_sync_tap_and_bpm() {
        let mut tempo = TempoSyncState::default();
        tempo.set_bpm(130.0);
        assert_eq!(tempo.current_bpm, 130.0);
        // BPM clamping [20.0, 400.0]
        tempo.set_bpm(5.0);
        assert_eq!(tempo.current_bpm, 20.0);
        tempo.set_bpm(999.0);
        assert_eq!(tempo.current_bpm, 400.0);

        tempo.reset();
        tempo.sample_rate_hz = 48_000;
        // Tap at 120 BPM: interval = 48,000 * 60 / 120 = 24,000 frames (4 taps wrap circular buffer)
        tempo.register_tap();
        for _ in 0..5 {
            tempo.advance_frames(24_000);
            let bpm = tempo.register_tap();
            assert!((bpm - 120.0).abs() < 1e-2);
        }

        // Out-of-bounds tap interval (e.g. 10 frames, > 420 BPM) is ignored and preserves BPM
        tempo.advance_frames(10);
        let bpm_after_glitch = tempo.register_tap();
        assert!((bpm_after_glitch - 120.0).abs() < 1e-2);

        // Pause > 3s (> 144,000 frames) clears stale tap_intervals so new sequence locks immediately to 90 BPM
        tempo.advance_frames(200_000);
        tempo.register_tap();
        tempo.advance_frames(32_000); // 90 BPM at 48 kHz = 32,000 frames
        let bpm_new_seq = tempo.register_tap();
        assert!((bpm_new_seq - 90.0).abs() < 1e-2);

        tempo.reset();
        assert_eq!(tempo.total_frames, 0);
        assert_eq!(tempo.last_tap_frame, None);
    }
}
