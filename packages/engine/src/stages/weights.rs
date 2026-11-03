//! Zero-copy 64B-aligned views over 64,000 B Arena 0..3 directory.
//!
//! Canonical Reference: `docs/containers/dnam_binary_format.md` and `docs/overview/tora_architecture.md`.

use crate::types::Tier;

/// Cumulative footprint of Arena 0 (Lite tier): 10,752 B (168 cache lines).
pub const ARENA_LITE_BYTES: usize = 10_752;

/// Cumulative footprint of Arena 0..1 (Full tier): 22,336 B (349 cache lines).
pub const ARENA_FULL_BYTES: usize = 22_336;

/// Cumulative footprint of Arena 0..2 (Pro tier): 36,224 B (566 cache lines).
pub const ARENA_PRO_BYTES: usize = 36_224;

/// Cumulative footprint of Arena 0..3 (Ultra tier): 64,000 B (1,000 cache lines).
pub const ARENA_ULTRA_BYTES: usize = 64_000;

// ============================================================================
// Arena 0 (Lite): Canonical Tensor Offsets & Cache-Line Directory (0..10,752 B, 168 CL)
// Reference: docs/containers/dnam_binary_format.md §2.2
// ============================================================================

/// Offset of `arena0_lite.router_and_pre_bus` (0..256 B, 4 CL):
/// - CL 00 (0..64 B): 4x4 S_router scattering matrix
/// - CL 01 (64..128 B): Dual Catmull-Rom latency alignment & normaling
/// - CL 02 (128..192 B): 4th-Order Bezout 200 Hz crossover & phase compensator
/// - CL 03 (192..256 B): WDW output mixer & dedicated dry out
pub const ARENA0_ROUTER_AND_PRE_BUS_OFFSET: usize = 0;
pub const ARENA0_ROUTER_AND_PRE_BUS_BYTES: usize = 256;
pub const ARENA0_ROUTER_AND_PRE_BUS_CL: usize = 4;

/// Offset of `arena0_lite.stage1_pre_params` (256..1,024 B, 12 CL):
pub const ARENA0_STAGE1_PRE_PARAMS_OFFSET: usize = 256;
pub const ARENA0_STAGE1_PRE_PARAMS_BYTES: usize = 768;
pub const ARENA0_STAGE1_PRE_PARAMS_CL: usize = 12;

/// Offset of `arena0_lite.stage2_cascade_w` (1,024..4,096 B, 48 CL):
/// Shared 3-stage commuted nonlinear cascade projection matrices W_stage^(2.1..2.3) [3, 16, 4, 4]
pub const ARENA0_STAGE2_CASCADE_W_OFFSET: usize = 1_024;
pub const ARENA0_STAGE2_CASCADE_W_BYTES: usize = 3_072;
pub const ARENA0_STAGE2_CASCADE_W_CL: usize = 48;
pub const ARENA0_STAGE2_CASCADE_W_WEIGHTS: usize = 768;

/// Offset of `arena0_lite.stage2_filter_params` (4,096..4,288 B, 3 CL):
/// LTI Tone/Sag Poles [3, 4, 4] (4 tone cutoff poles + 4 sag recovery poles).
pub const ARENA0_STAGE2_FILTER_PARAMS_OFFSET: usize = 4_096;
pub const ARENA0_STAGE2_FILTER_PARAMS_BYTES: usize = 192;
pub const ARENA0_STAGE2_FILTER_PARAMS_CL: usize = 3;

/// Offset of `arena0_lite.stage3_early_fir_kautz` (4,288..4,672 B, 6 CL):
/// Packs 16-tap Early FIR + 4-pair Kautz modal bank poles.
pub const ARENA0_STAGE3_EARLY_FIR_KAUTZ_OFFSET: usize = 4_288;
pub const ARENA0_STAGE3_EARLY_FIR_KAUTZ_BYTES: usize = 384;
pub const ARENA0_STAGE3_EARLY_FIR_KAUTZ_CL: usize = 6;

/// Offset of `arena0_lite.stage4_tank_spatial` (4,672..4,928 B, 4 CL):
/// FDN Tank parameters Lite (tau_0, Schroeder all-pass, DC blocker pole R_dc = 0.995).
pub const ARENA0_STAGE4_TANK_SPATIAL_OFFSET: usize = 4_672;
pub const ARENA0_STAGE4_TANK_SPATIAL_BYTES: usize = 256;
pub const ARENA0_STAGE4_TANK_SPATIAL_CL: usize = 4;

/// Offset of `arena0_lite.stereo_head_svd` (4,928..7,488 B, 40 CL):
/// 256-D Mid/Side Kronecker Head SVD factors [5, 2, 16, 4] (Dh = 4).
pub const ARENA0_STEREO_HEAD_SVD_OFFSET: usize = 4_928;
pub const ARENA0_STEREO_HEAD_SVD_BYTES: usize = 2_560;
pub const ARENA0_STEREO_HEAD_SVD_CL: usize = 40;

/// Offset of `arena0_lite.control_bus_and_clamps` (7,488..10,752 B, 51 CL):
/// - CL 00..15 (7,488..8,512 B): 4-band envelope alphas, pitchfork hysteresis, Chebyshev T1..T4, U_int FM matrix
/// - CL 16..23 (8,512..9,024 B): Multi-range switch table (discrete mode mapping)
/// - CL 24..31 (9,024..9,536 B): 12-subdivision tempo sync & PLL reciprocal multipliers
/// - CL 32..47 (9,536..10,560 B): 16x16 Stage parameter dispatch matrix W_ctrl
/// - CL 48..50 (10,560..10,752 B): rho_head trust bound (CL 48), tier_gain_trim: [f32; 16] (CL 49), anti_zipper_micro_ramp_scale = 0.0625f32 (CL 50)
pub const ARENA0_CONTROL_BUS_AND_CLAMPS_OFFSET: usize = 7_488;
pub const ARENA0_CONTROL_BUS_AND_CLAMPS_BYTES: usize = 3_264;
pub const ARENA0_CONTROL_BUS_AND_CLAMPS_CL: usize = 51;

/// Specific cache-line offsets inside `arena0_lite.control_bus_and_clamps`:
pub const ARENA0_W_CTRL_OFFSET: usize = 9_536; // CL 32..47 (16 x 16 floats = 1024 bytes)
pub const ARENA0_W_CTRL_BYTES: usize = 1_024;
pub const ARENA0_RHO_HEAD_OFFSET: usize = 10_560; // CL 48 (16 floats = 64 bytes)
pub const ARENA0_TIER_GAIN_TRIM_OFFSET: usize = 10_624; // CL 49 (16 floats = 64 bytes)
pub const ARENA0_ANTI_ZIPPER_RAMP_SCALE_OFFSET: usize = 10_688; // CL 50 (16 floats = 64 bytes)
pub const ANTI_ZIPPER_MICRO_RAMP_SCALE_DEFAULT: f32 = 0.0625; // 1/16

const _: () = assert!(
    ARENA0_ROUTER_AND_PRE_BUS_BYTES
        + ARENA0_STAGE1_PRE_PARAMS_BYTES
        + ARENA0_STAGE2_CASCADE_W_BYTES
        + ARENA0_STAGE2_FILTER_PARAMS_BYTES
        + ARENA0_STAGE3_EARLY_FIR_KAUTZ_BYTES
        + ARENA0_STAGE4_TANK_SPATIAL_BYTES
        + ARENA0_STEREO_HEAD_SVD_BYTES
        + ARENA0_CONTROL_BUS_AND_CLAMPS_BYTES
        == ARENA_LITE_BYTES
);

const _: () = assert!(
    ARENA0_ROUTER_AND_PRE_BUS_CL
        + ARENA0_STAGE1_PRE_PARAMS_CL
        + ARENA0_STAGE2_CASCADE_W_CL
        + ARENA0_STAGE2_FILTER_PARAMS_CL
        + ARENA0_STAGE3_EARLY_FIR_KAUTZ_CL
        + ARENA0_STAGE4_TANK_SPATIAL_CL
        + ARENA0_STEREO_HEAD_SVD_CL
        + ARENA0_CONTROL_BUS_AND_CLAMPS_CL
        == 168
);

// ============================================================================
// Arena 1 (Full): Delta Tensors (10,752..22,336 B, 181 CL)
// ============================================================================
pub const ARENA1_STAGE1_DELTA_PRE_OFFSET: usize = 10_752;
pub const ARENA1_STAGE1_DELTA_PRE_BYTES: usize = 512;
pub const ARENA1_STAGE1_DELTA_PRE_CL: usize = 8;

pub const ARENA1_STAGE2_CASCADE_W_DELTA_OFFSET: usize = 11_264;
pub const ARENA1_STAGE2_CASCADE_W_DELTA_BYTES: usize = 6_144;
pub const ARENA1_STAGE2_CASCADE_W_DELTA_CL: usize = 96;

pub const ARENA1_STAGE3_KAUTZ_DELTA_OFFSET: usize = 17_408;
pub const ARENA1_STAGE3_KAUTZ_DELTA_BYTES: usize = 384;
pub const ARENA1_STAGE3_KAUTZ_DELTA_CL: usize = 6;

pub const ARENA1_STAGE4_TANK_DELTA_OFFSET: usize = 17_792;
pub const ARENA1_STAGE4_TANK_DELTA_BYTES: usize = 256;
pub const ARENA1_STAGE4_TANK_DELTA_CL: usize = 4;

pub const ARENA1_STEREO_HEAD_SVD_DELTA_OFFSET: usize = 18_048;
pub const ARENA1_STEREO_HEAD_SVD_DELTA_BYTES: usize = 2_560;
pub const ARENA1_STEREO_HEAD_SVD_DELTA_CL: usize = 40;

pub const ARENA1_CONTROL_BUS_PROJ_DELTA_OFFSET: usize = 20_608;
pub const ARENA1_CONTROL_BUS_PROJ_DELTA_BYTES: usize = 1_728;
pub const ARENA1_CONTROL_BUS_PROJ_DELTA_CL: usize = 27;

pub const ARENA1_BYTES: usize = 11_584;
pub const ARENA1_CL: usize = 181;

const _: () = assert!(
    ARENA1_STAGE1_DELTA_PRE_BYTES
        + ARENA1_STAGE2_CASCADE_W_DELTA_BYTES
        + ARENA1_STAGE3_KAUTZ_DELTA_BYTES
        + ARENA1_STAGE4_TANK_DELTA_BYTES
        + ARENA1_STEREO_HEAD_SVD_DELTA_BYTES
        + ARENA1_CONTROL_BUS_PROJ_DELTA_BYTES
        == ARENA1_BYTES
);
const _: () = assert!(ARENA_LITE_BYTES + ARENA1_BYTES == ARENA_FULL_BYTES);

// ============================================================================
// Arena 2 (Pro): Delta Tensors (22,336..36,224 B, 217 CL)
// ============================================================================
pub const ARENA2_STAGE2_CASCADE_W_DELTA_OFFSET: usize = 22_336;
pub const ARENA2_STAGE2_CASCADE_W_DELTA_BYTES: usize = 9_216;
pub const ARENA2_STAGE2_CASCADE_W_DELTA_CL: usize = 144;

pub const ARENA2_STAGE3_KAUTZ_DELTA_OFFSET: usize = 31_552;
pub const ARENA2_STAGE3_KAUTZ_DELTA_BYTES: usize = 384;
pub const ARENA2_STAGE3_KAUTZ_DELTA_CL: usize = 6;

pub const ARENA2_STEREO_HEAD_SVD_DELTA_OFFSET: usize = 31_936;
pub const ARENA2_STEREO_HEAD_SVD_DELTA_BYTES: usize = 2_560;
pub const ARENA2_STEREO_HEAD_SVD_DELTA_CL: usize = 40;

pub const ARENA2_CONTROL_BUS_PROJ_DELTA_OFFSET: usize = 34_496;
pub const ARENA2_CONTROL_BUS_PROJ_DELTA_BYTES: usize = 1_728;
pub const ARENA2_CONTROL_BUS_PROJ_DELTA_CL: usize = 27;

pub const ARENA2_BYTES: usize = 13_888;
pub const ARENA2_CL: usize = 217;

const _: () = assert!(
    ARENA2_STAGE2_CASCADE_W_DELTA_BYTES
        + ARENA2_STAGE3_KAUTZ_DELTA_BYTES
        + ARENA2_STEREO_HEAD_SVD_DELTA_BYTES
        + ARENA2_CONTROL_BUS_PROJ_DELTA_BYTES
        == ARENA2_BYTES
);
const _: () = assert!(ARENA_FULL_BYTES + ARENA2_BYTES == ARENA_PRO_BYTES);

// ============================================================================
// Arena 3 (Ultra): Delta Tensors (36,224..64,000 B, 434 CL)
// ============================================================================
pub const ARENA3_STAGE2_CASCADE_W_DELTA_OFFSET: usize = 36_224;
pub const ARENA3_STAGE2_CASCADE_W_DELTA_BYTES: usize = 12_288;
pub const ARENA3_STAGE2_CASCADE_W_DELTA_CL: usize = 192;

pub const ARENA3_STAGE3_KAUTZ_DELTA_OFFSET: usize = 48_512;
pub const ARENA3_STAGE3_KAUTZ_DELTA_BYTES: usize = 384;
pub const ARENA3_STAGE3_KAUTZ_DELTA_CL: usize = 6;

pub const ARENA3_STAGE4_TANK_SPATIAL_16_OFFSET: usize = 48_896;
pub const ARENA3_STAGE4_TANK_SPATIAL_16_BYTES: usize = 512;
pub const ARENA3_STAGE4_TANK_SPATIAL_16_CL: usize = 8;

pub const ARENA3_STEREO_HEAD_MODES5TO8_OFFSET: usize = 49_408;
pub const ARENA3_STEREO_HEAD_MODES5TO8_BYTES: usize = 8_192;
pub const ARENA3_STEREO_HEAD_MODES5TO8_CL: usize = 128;

pub const ARENA3_STEREO_HEAD_DELTA_OFFSET: usize = 57_600;
pub const ARENA3_STEREO_HEAD_DELTA_BYTES: usize = 2_560;
pub const ARENA3_STEREO_HEAD_DELTA_CL: usize = 40;

pub const ARENA3_CONTROL_BUS_PROJ_ULTRA_OFFSET: usize = 60_160;
pub const ARENA3_CONTROL_BUS_PROJ_ULTRA_BYTES: usize = 3_840;
pub const ARENA3_CONTROL_BUS_PROJ_ULTRA_CL: usize = 60;

pub const ARENA3_BYTES: usize = 27_776;
pub const ARENA3_CL: usize = 434;

const _: () = assert!(
    ARENA3_STAGE2_CASCADE_W_DELTA_BYTES
        + ARENA3_STAGE3_KAUTZ_DELTA_BYTES
        + ARENA3_STAGE4_TANK_SPATIAL_16_BYTES
        + ARENA3_STEREO_HEAD_MODES5TO8_BYTES
        + ARENA3_STEREO_HEAD_DELTA_BYTES
        + ARENA3_CONTROL_BUS_PROJ_ULTRA_BYTES
        == ARENA3_BYTES
);
const _: () = assert!(ARENA_PRO_BYTES + ARENA3_BYTES == ARENA_ULTRA_BYTES);
const _: () = assert!(168 + ARENA1_CL + ARENA2_CL + ARENA3_CL == 1_000);

#[inline(always)]
pub fn read_f32_le(slice: &[u8], offset: usize) -> f32 {
    if offset + 4 <= slice.len() {
        f32::from_le_bytes([
            slice[offset],
            slice[offset + 1],
            slice[offset + 2],
            slice[offset + 3],
        ])
    } else {
        0.0
    }
}

/// Zero-copy directory view into the 64,000 B tensor arena.
#[derive(Clone, Copy, Debug)]
pub struct ArenaDirectoryView<'a> {
    pub raw: &'a [u8],
    pub tier: Tier,
}

impl<'a> ArenaDirectoryView<'a> {
    /// Validates 64-byte alignment and tier prefix capacity.
    pub fn new(bytes: &'a [u8], tier: Tier) -> Result<Self, &'static str> {
        let min_len = match tier {
            Tier::Lite => ARENA_LITE_BYTES,
            Tier::Full => ARENA_FULL_BYTES,
            Tier::Pro => ARENA_PRO_BYTES,
            Tier::Ultra => ARENA_ULTRA_BYTES,
        };
        if bytes.len() < min_len {
            return Err("[DNAM-WEIGHTS] Arena buffer smaller than tier requirement");
        }
        Ok(Self { raw: bytes, tier })
    }

    /// Reads an f32 parameter at `byte_offset`.
    #[inline(always)]
    pub fn read_f32(&self, byte_offset: usize) -> f32 {
        read_f32_le(self.raw, byte_offset)
    }

    /// Copies `count` floats starting at `byte_offset` into `dst`.
    pub fn read_f32_into(&self, byte_offset: usize, dst: &mut [f32]) {
        for (i, d) in dst.iter_mut().enumerate() {
            *d = self.read_f32(byte_offset + i * 4);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate alloc;
    use alloc::vec;

    #[test]
    fn test_arena_directory_view_all_tiers_and_reads() {
        let mut buf = vec![0u8; ARENA_ULTRA_BYTES];
        let val: f32 = 42.5;
        buf[64..68].copy_from_slice(&val.to_le_bytes());
        buf[68..72].copy_from_slice(&(13.25_f32).to_le_bytes());

        for tier in [Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra] {
            let view = ArenaDirectoryView::new(&buf, tier).unwrap();
            assert_eq!(view.tier, tier);
            assert_eq!(view.read_f32(64), 42.5);

            let mut out = [0.0f32; 2];
            view.read_f32_into(64, &mut out);
            assert_eq!(out, [42.5, 13.25]);
        }

        // Test error on too-small buffer
        let small = vec![0u8; 100];
        assert!(ArenaDirectoryView::new(&small, Tier::Lite).is_err());

        // Out of bounds read_f32_le returns 0.0
        assert_eq!(read_f32_le(&[1, 2, 3], 0), 0.0);
    }

    #[test]
    fn test_arena0_canonical_directory_layout() {
        assert_eq!(ARENA0_ROUTER_AND_PRE_BUS_OFFSET, 0);
        assert_eq!(ARENA0_ROUTER_AND_PRE_BUS_BYTES, 256);
        assert_eq!(ARENA0_ROUTER_AND_PRE_BUS_CL, 4);

        assert_eq!(ARENA0_STAGE1_PRE_PARAMS_OFFSET, 256);
        assert_eq!(ARENA0_STAGE1_PRE_PARAMS_BYTES, 768);
        assert_eq!(ARENA0_STAGE1_PRE_PARAMS_CL, 12);

        assert_eq!(ARENA0_STAGE2_CASCADE_W_OFFSET, 1024);
        assert_eq!(ARENA0_STAGE2_CASCADE_W_BYTES, 3072);
        assert_eq!(ARENA0_STAGE2_CASCADE_W_CL, 48);
        assert_eq!(ARENA0_STAGE2_CASCADE_W_WEIGHTS, 768);

        assert_eq!(ARENA0_STAGE2_FILTER_PARAMS_OFFSET, 4096);
        assert_eq!(ARENA0_STAGE2_FILTER_PARAMS_BYTES, 192);
        assert_eq!(ARENA0_STAGE2_FILTER_PARAMS_CL, 3);

        assert_eq!(ARENA0_STAGE3_EARLY_FIR_KAUTZ_OFFSET, 4288);
        assert_eq!(ARENA0_STAGE3_EARLY_FIR_KAUTZ_BYTES, 384);
        assert_eq!(ARENA0_STAGE3_EARLY_FIR_KAUTZ_CL, 6);

        assert_eq!(ARENA0_STAGE4_TANK_SPATIAL_OFFSET, 4672);
        assert_eq!(ARENA0_STAGE4_TANK_SPATIAL_BYTES, 256);
        assert_eq!(ARENA0_STAGE4_TANK_SPATIAL_CL, 4);

        assert_eq!(ARENA0_STEREO_HEAD_SVD_OFFSET, 4928);
        assert_eq!(ARENA0_STEREO_HEAD_SVD_BYTES, 2560);
        assert_eq!(ARENA0_STEREO_HEAD_SVD_CL, 40);

        assert_eq!(ARENA0_CONTROL_BUS_AND_CLAMPS_OFFSET, 7488);
        assert_eq!(ARENA0_CONTROL_BUS_AND_CLAMPS_BYTES, 3264);
        assert_eq!(ARENA0_CONTROL_BUS_AND_CLAMPS_CL, 51);

        assert_eq!(ARENA0_W_CTRL_OFFSET, 9536);
        assert_eq!(ARENA0_W_CTRL_BYTES, 1024);
        assert_eq!(ARENA0_RHO_HEAD_OFFSET, 10560);
        assert_eq!(ARENA0_TIER_GAIN_TRIM_OFFSET, 10624);
        assert_eq!(ARENA0_ANTI_ZIPPER_RAMP_SCALE_OFFSET, 10688);
        assert_eq!(ANTI_ZIPPER_MICRO_RAMP_SCALE_DEFAULT, 0.0625);

        // Arena 1..3 delta offsets and byte lengths
        assert_eq!(ARENA1_STAGE1_DELTA_PRE_OFFSET, 10752);
        assert_eq!(ARENA1_STAGE2_CASCADE_W_DELTA_OFFSET, 11264);
        assert_eq!(ARENA1_STAGE3_KAUTZ_DELTA_OFFSET, 17408);
        assert_eq!(ARENA1_STAGE4_TANK_DELTA_OFFSET, 17792);
        assert_eq!(ARENA1_STEREO_HEAD_SVD_DELTA_OFFSET, 18048);
        assert_eq!(ARENA1_CONTROL_BUS_PROJ_DELTA_OFFSET, 20608);

        assert_eq!(ARENA2_STAGE2_CASCADE_W_DELTA_OFFSET, 22336);
        assert_eq!(ARENA2_STAGE3_KAUTZ_DELTA_OFFSET, 31552);
        assert_eq!(ARENA2_STEREO_HEAD_SVD_DELTA_OFFSET, 31936);
        assert_eq!(ARENA2_CONTROL_BUS_PROJ_DELTA_OFFSET, 34496);

        assert_eq!(ARENA3_STAGE2_CASCADE_W_DELTA_OFFSET, 36224);
        assert_eq!(ARENA3_STAGE3_KAUTZ_DELTA_OFFSET, 48512);
        assert_eq!(ARENA3_STAGE4_TANK_SPATIAL_16_OFFSET, 48896);
        assert_eq!(ARENA3_STEREO_HEAD_MODES5TO8_OFFSET, 49408);
        assert_eq!(ARENA3_STEREO_HEAD_DELTA_OFFSET, 57600);
        assert_eq!(ARENA3_CONTROL_BUS_PROJ_ULTRA_OFFSET, 60160);

        assert_eq!(ARENA_LITE_BYTES, 10752);
        assert_eq!(ARENA_FULL_BYTES, 22336);
        assert_eq!(ARENA_PRO_BYTES, 36224);
        assert_eq!(ARENA_ULTRA_BYTES, 64000);
    }
}
