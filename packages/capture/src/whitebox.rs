#![forbid(unsafe_code)]
//! `64 B` `DNWB` Binary Pipe Protocol, GPL-Isolated `FaustRunner` & `SpiceRunner`
//! (`PCHIP` Variable-Step `4x` Interpolation + `309`-Tap Kaiser-Sinc `4x -> 1x` Decimation),
//! and Analytical Rust `f64` Reference Twins (`whitebox`).
//!
//! Canonical Reference: `ADR-0018` and `docs/specs/training/four_stage_pipeline.md` (`r[train.pipeline.fixtures-matrix]`).

mod faust;
mod ipc;
mod runners;
mod session;

pub use faust::{FaustControlSpec, FaustRunner, parse_faust_json_controls};
pub use ipc::{
    execute_dnwb_subprocess_request, read_dnwb_stem_response, serve_dnwb_single_request,
    serve_dnwb_single_request_with_switches, write_dnwb_stem_response,
};
pub use runners::{
    SPICE_CAUSAL_DELAY_SAMPLES, SpiceRunner, format_spice_tran_directives,
    format_spice_tran_directives_for_rate, kaiser_sinc_decimate_4x_to_48khz,
    pchip_resample_to_192khz, pchip_resample_to_rate,
};
pub use session::{DnwbWorkerSession, serve_dnwb_stream, serve_dnwb_stream_with_switches};

use crate::targets::StemStimulus;
use std::io::{Read, Write};

/// Magic bytes for the `64 B`-aligned White-Box Distillation Pipe Protocol (`b"DNWB"`).
pub const DNWB_MAGIC: [u8; 4] = *b"DNWB";

/// Current `DNWB` wire protocol version (`1`).
pub const DNWB_VERSION: u16 = 1;

/// Maximum permitted `DNWB` payload byte length (`64 MiB`) to prevent OOM allocation on corrupt frames.
pub const MAX_DNWB_PAYLOAD_BYTES: u64 = 64 * 1024 * 1024;

/// Maximum SPICE solver timestep at `48 kHz` ($\Delta t_{\max} = 1 / 192{,}000\text{ s} \approx 5.208333\times 10^{-6}\text{ s}$).
pub const SPICE_MAXSTEP_SEC: f64 = 1.0 / 192_000.0;

const DNWB_FLAG_STRUCTURED_V1: u16 = 1 << 0;
const DNWB_FLAG_HAS_TRAJECTORIES: u16 = 1 << 1;
const DNWB_FLAG_HAS_FREEZE_WINDOW: u16 = 1 << 2;
const DNWB_CHANNEL_MASK_SHIFT: u16 = 8;
const DNWB_KNOWN_REQ_FLAGS: u16 = DNWB_FLAG_STRUCTURED_V1
    | DNWB_FLAG_HAS_TRAJECTORIES
    | DNWB_FLAG_HAS_FREEZE_WINDOW
    | (0x0f << DNWB_CHANNEL_MASK_SHIFT);

/// Command opcodes for [`DnwbHeader64B`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u16)]
pub enum DnwbCommand {
    /// Reset internal circuit/DSP state to DC equilibrium (`0`).
    Reset = 0,
    /// Render a full batch stem (`1`).
    RenderStem = 1,
    /// Render a single streaming frame without resetting state (`2`).
    RenderFrame = 2,
}

impl DnwbCommand {
    #[must_use]
    pub const fn from_u16(v: u16) -> Option<Self> {
        match v {
            0 => Some(Self::Reset),
            1 => Some(Self::RenderStem),
            2 => Some(Self::RenderFrame),
            _ => None,
        }
    }
}

/// Errors emitted by the `DNWB` binary protocol, `FaustRunner`, or `SpiceRunner`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DnwbError {
    InvalidMagic([u8; 4]),
    UnsupportedVersion(u16),
    InvalidCommand(u16),
    NonZeroReservedBytes,
    PayloadTooLarge { actual: u64, max: u64 },
    TruncatedPacket { expected: usize, actual: usize },
    InvalidStimulusLayout(String),
    InvalidJsonMetadata(String),
    WatchdogTimeout { timeout_ms: u64 },
    IoFailure(String),
}

/// `64 B`-aligned binary header for out-of-process white-box distillation IPC (`b"DNWB"`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DnwbHeader64B {
    pub magic: [u8; 4],
    pub version: u16,
    pub command: DnwbCommand,
    pub sample_rate_hz: u32,
    pub num_frames: u32,
    pub num_input_channels: u16,
    pub num_output_channels: u16,
    pub num_controls: u16,
    pub num_switches: u16,
    pub num_internal_nodes: u16,
    pub flags: u16,
    pub payload_byte_len: u64,
}

impl DnwbHeader64B {
    /// Serializes this header into an exact `64 B` little-endian array (`bytes[36..64]` zero-padded).
    #[must_use]
    pub fn to_bytes(&self) -> [u8; 64] {
        let mut out = [0u8; 64];
        out[0..4].copy_from_slice(&self.magic);
        out[4..6].copy_from_slice(&self.version.to_le_bytes());
        out[6..8].copy_from_slice(&(self.command as u16).to_le_bytes());
        out[8..12].copy_from_slice(&self.sample_rate_hz.to_le_bytes());
        out[12..16].copy_from_slice(&self.num_frames.to_le_bytes());
        out[16..18].copy_from_slice(&self.num_input_channels.to_le_bytes());
        out[18..20].copy_from_slice(&self.num_output_channels.to_le_bytes());
        out[20..22].copy_from_slice(&self.num_controls.to_le_bytes());
        out[22..24].copy_from_slice(&self.num_switches.to_le_bytes());
        out[24..26].copy_from_slice(&self.num_internal_nodes.to_le_bytes());
        out[26..28].copy_from_slice(&self.flags.to_le_bytes());
        out[28..36].copy_from_slice(&self.payload_byte_len.to_le_bytes());
        out
    }

    /// Parses and validates a `64 B` little-endian `DNWB` header, enforcing zeroed reserved bytes
    /// and the `MAX_DNWB_PAYLOAD_BYTES` OOM guard.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, DnwbError> {
        if bytes.len() < 64 {
            return Err(DnwbError::TruncatedPacket {
                expected: 64,
                actual: bytes.len(),
            });
        }
        let magic = [bytes[0], bytes[1], bytes[2], bytes[3]];
        if magic != DNWB_MAGIC {
            return Err(DnwbError::InvalidMagic(magic));
        }
        let version = u16::from_le_bytes([bytes[4], bytes[5]]);
        if version != DNWB_VERSION {
            return Err(DnwbError::UnsupportedVersion(version));
        }
        let cmd_raw = u16::from_le_bytes([bytes[6], bytes[7]]);
        let command = DnwbCommand::from_u16(cmd_raw).ok_or(DnwbError::InvalidCommand(cmd_raw))?;
        if bytes[36..64].iter().any(|&b| b != 0) {
            return Err(DnwbError::NonZeroReservedBytes);
        }
        let sample_rate_hz = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        let num_frames = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);
        let num_input_channels = u16::from_le_bytes([bytes[16], bytes[17]]);
        let num_output_channels = u16::from_le_bytes([bytes[18], bytes[19]]);
        let num_controls = u16::from_le_bytes([bytes[20], bytes[21]]);
        let num_switches = u16::from_le_bytes([bytes[22], bytes[23]]);
        let num_internal_nodes = u16::from_le_bytes([bytes[24], bytes[25]]);
        let flags = u16::from_le_bytes([bytes[26], bytes[27]]);
        let payload_byte_len = u64::from_le_bytes([
            bytes[28], bytes[29], bytes[30], bytes[31], bytes[32], bytes[33], bytes[34], bytes[35],
        ]);
        if payload_byte_len > MAX_DNWB_PAYLOAD_BYTES {
            return Err(DnwbError::PayloadTooLarge {
                actual: payload_byte_len,
                max: MAX_DNWB_PAYLOAD_BYTES,
            });
        }
        Ok(Self {
            magic,
            version,
            command,
            sample_rate_hz,
            num_frames,
            num_input_channels,
            num_output_channels,
            num_controls,
            num_switches,
            num_internal_nodes,
            flags,
            payload_byte_len,
        })
    }
}

fn checked_add_usize(a: usize, b: usize) -> Result<usize, DnwbError> {
    a.checked_add(b)
        .ok_or_else(|| DnwbError::InvalidStimulusLayout(String::from("byte length overflow")))
}

fn checked_mul_usize(a: usize, b: usize) -> Result<usize, DnwbError> {
    a.checked_mul(b).ok_or_else(|| {
        DnwbError::InvalidStimulusLayout(String::from("byte multiplication overflow"))
    })
}

fn read_u16_le(buf: &[u8], off: &mut usize) -> Result<u16, DnwbError> {
    let end = checked_add_usize(*off, 2)?;
    if buf.len() < end {
        return Err(DnwbError::TruncatedPacket {
            expected: end,
            actual: buf.len(),
        });
    }
    let val = u16::from_le_bytes([buf[*off], buf[*off + 1]]);
    *off = end;
    Ok(val)
}

fn read_u32_le(buf: &[u8], off: &mut usize) -> Result<u32, DnwbError> {
    let end = checked_add_usize(*off, 4)?;
    if buf.len() < end {
        return Err(DnwbError::TruncatedPacket {
            expected: end,
            actual: buf.len(),
        });
    }
    let val = u32::from_le_bytes([buf[*off], buf[*off + 1], buf[*off + 2], buf[*off + 3]]);
    *off = end;
    Ok(val)
}

fn read_u64_le(buf: &[u8], off: &mut usize) -> Result<u64, DnwbError> {
    let end = checked_add_usize(*off, 8)?;
    if buf.len() < end {
        return Err(DnwbError::TruncatedPacket {
            expected: end,
            actual: buf.len(),
        });
    }
    let val = u64::from_le_bytes([
        buf[*off],
        buf[*off + 1],
        buf[*off + 2],
        buf[*off + 3],
        buf[*off + 4],
        buf[*off + 5],
        buf[*off + 6],
        buf[*off + 7],
    ]);
    *off = end;
    Ok(val)
}

pub(super) fn ensure_finite_f32_slice(xs: &[f32], field: &str) -> Result<(), DnwbError> {
    if xs.iter().any(|v| !v.is_finite()) {
        return Err(DnwbError::InvalidStimulusLayout(format!(
            "non-finite f32 sample in DNWB {field}"
        )));
    }
    Ok(())
}

fn read_f32_vec(buf: &[u8], off: &mut usize, count: usize) -> Result<Vec<f32>, DnwbError> {
    let byte_len = checked_mul_usize(count, 4)?;
    let end = checked_add_usize(*off, byte_len)?;
    if buf.len() < end {
        return Err(DnwbError::TruncatedPacket {
            expected: end,
            actual: buf.len(),
        });
    }
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let idx = *off + i * 4;
        let val = f32::from_le_bytes([buf[idx], buf[idx + 1], buf[idx + 2], buf[idx + 3]]);
        if !val.is_finite() {
            return Err(DnwbError::InvalidStimulusLayout(String::from(
                "non-finite f32 sample in DNWB payload",
            )));
        }
        out.push(val);
    }
    *off = end;
    Ok(out)
}

/// Writes a `DNWB` stimulus request (`64 B` header + `stem_id` + controls + switches + all active
/// dry audio channels `in_1..in_4` + optional `[P][N]` trajectories & freeze window) to `writer`.
pub fn write_dnwb_stimulus_request<W: Write>(
    writer: &mut W,
    command: DnwbCommand,
    stimulus: &StemStimulus,
) -> Result<DnwbHeader64B, DnwbError> {
    write_dnwb_stimulus_request_with_switches(writer, command, stimulus, &[])
}

/// Writes a `DNWB` stimulus request including discrete switch states (`switches`).
pub fn write_dnwb_stimulus_request_with_switches<W: Write>(
    writer: &mut W,
    command: DnwbCommand,
    stimulus: &StemStimulus,
    switches: &[u8],
) -> Result<DnwbHeader64B, DnwbError> {
    write_dnwb_stimulus_request_dyn(writer, command, stimulus, switches)
}

fn write_dnwb_stimulus_request_dyn(
    writer: &mut dyn Write,
    command: DnwbCommand,
    stimulus: &StemStimulus,
    switches: &[u8],
) -> Result<DnwbHeader64B, DnwbError> {
    let n_frames = stimulus.in_1.len();
    if n_frames > u32::MAX as usize {
        return Err(DnwbError::InvalidStimulusLayout(String::from(
            "num_frames exceeds u32::MAX",
        )));
    }
    let stem_bytes = stimulus.stem_id.as_bytes();
    if stem_bytes.len() > u16::MAX as usize {
        return Err(DnwbError::InvalidStimulusLayout(String::from(
            "stem_id byte length exceeds u16::MAX",
        )));
    }
    if stimulus.control_values.len() > u16::MAX as usize || switches.len() > u16::MAX as usize {
        return Err(DnwbError::InvalidStimulusLayout(String::from(
            "control or switch count exceeds u16::MAX",
        )));
    }

    ensure_finite_f32_slice(&stimulus.control_values, "control_values")?;
    ensure_finite_f32_slice(&stimulus.in_1, "in_1")?;

    let opt_channels = [
        (1u16, stimulus.in_2.as_ref()),
        (2u16, stimulus.in_3.as_ref()),
        (3u16, stimulus.in_4.as_ref()),
    ];
    let mut ch_mask: u16 = 1 << 0;
    let mut n_in_ch: usize = 1;
    for (bit, ch_opt) in opt_channels {
        if let Some(ch) = ch_opt {
            if ch.len() != n_frames {
                return Err(DnwbError::InvalidStimulusLayout(format!(
                    "input channel {} length {} != in_1 length {}",
                    bit + 1,
                    ch.len(),
                    n_frames
                )));
            }
            ensure_finite_f32_slice(ch, "optional input channel")?;
            ch_mask |= 1 << bit;
            n_in_ch += 1;
        }
    }

    let mut flags = DNWB_FLAG_STRUCTURED_V1 | (ch_mask << DNWB_CHANNEL_MASK_SHIFT);
    let mut traj_count: usize = 0;
    if let Some(ref trajs) = stimulus.control_trajectories {
        if trajs.len() > u16::MAX as usize {
            return Err(DnwbError::InvalidStimulusLayout(String::from(
                "trajectory control count exceeds u16::MAX",
            )));
        }
        if trajs.len() != stimulus.control_values.len() {
            return Err(DnwbError::InvalidStimulusLayout(format!(
                "control_trajectories len {} != control_values len {}",
                trajs.len(),
                stimulus.control_values.len()
            )));
        }
        for (p, t) in trajs.iter().enumerate() {
            if t.len() != n_frames {
                return Err(DnwbError::InvalidStimulusLayout(format!(
                    "control_trajectories[{p}] len {} != num_frames {n_frames}",
                    t.len()
                )));
            }
            ensure_finite_f32_slice(t, "control_trajectories")?;
        }
        flags |= DNWB_FLAG_HAS_TRAJECTORIES;
        traj_count = trajs.len();
    }
    if let Some([start, end]) = stimulus.freeze_transition_window {
        if start > end || end > n_frames {
            return Err(DnwbError::InvalidStimulusLayout(format!(
                "invalid freeze_transition_window [{start}, {end}] for num_frames {n_frames}"
            )));
        }
        flags |= DNWB_FLAG_HAS_FREEZE_WINDOW;
    }

    let mut payload_bytes = checked_add_usize(2, stem_bytes.len())?;
    payload_bytes = checked_add_usize(
        payload_bytes,
        checked_mul_usize(stimulus.control_values.len(), 4)?,
    )?;
    payload_bytes = checked_add_usize(payload_bytes, switches.len())?;
    let audio_samples = checked_mul_usize(n_in_ch, n_frames)?;
    payload_bytes = checked_add_usize(payload_bytes, checked_mul_usize(audio_samples, 4)?)?;
    if flags & DNWB_FLAG_HAS_TRAJECTORIES != 0 {
        let traj_samples = checked_mul_usize(traj_count, n_frames)?;
        let traj_bytes = checked_add_usize(2, checked_mul_usize(traj_samples, 4)?)?;
        payload_bytes = checked_add_usize(payload_bytes, traj_bytes)?;
    }
    if flags & DNWB_FLAG_HAS_FREEZE_WINDOW != 0 {
        payload_bytes = checked_add_usize(payload_bytes, 16)?;
    }

    let payload_byte_len = payload_bytes as u64;
    if payload_byte_len > MAX_DNWB_PAYLOAD_BYTES {
        return Err(DnwbError::PayloadTooLarge {
            actual: payload_byte_len,
            max: MAX_DNWB_PAYLOAD_BYTES,
        });
    }

    let hdr = DnwbHeader64B {
        magic: DNWB_MAGIC,
        version: DNWB_VERSION,
        command,
        sample_rate_hz: stimulus.sample_rate_hz,
        num_frames: n_frames as u32,
        num_input_channels: n_in_ch as u16,
        num_output_channels: 0,
        num_controls: stimulus.control_values.len() as u16,
        num_switches: switches.len() as u16,
        num_internal_nodes: 0,
        flags,
        payload_byte_len,
    };

    let mut payload = Vec::with_capacity(payload_bytes);
    payload.extend_from_slice(&(stem_bytes.len() as u16).to_le_bytes());
    payload.extend_from_slice(stem_bytes);
    for &c in &stimulus.control_values {
        payload.extend_from_slice(&c.to_le_bytes());
    }
    payload.extend_from_slice(switches);
    for &x in &stimulus.in_1 {
        payload.extend_from_slice(&x.to_le_bytes());
    }
    for (_, ch_opt) in opt_channels {
        if let Some(ch) = ch_opt {
            for &x in ch {
                payload.extend_from_slice(&x.to_le_bytes());
            }
        }
    }
    if let Some(ref trajs) = stimulus.control_trajectories {
        payload.extend_from_slice(&(trajs.len() as u16).to_le_bytes());
        for t in trajs {
            for &v in t {
                payload.extend_from_slice(&v.to_le_bytes());
            }
        }
    }
    if let Some([start, end]) = stimulus.freeze_transition_window {
        payload.extend_from_slice(&(start as u64).to_le_bytes());
        payload.extend_from_slice(&(end as u64).to_le_bytes());
    }

    writer
        .write_all(&hdr.to_bytes())
        .and_then(|()| writer.write_all(&payload))
        .map_err(|e| DnwbError::IoFailure(e.to_string()))?;
    Ok(hdr)
}

/// Reads a `DNWB` stimulus request from `reader` (discarding optional discrete switches).
pub fn read_dnwb_stimulus_request<R: Read>(
    reader: &mut R,
) -> Result<(DnwbHeader64B, StemStimulus), DnwbError> {
    let (hdr, stim, _switches) = read_dnwb_stimulus_request_with_switches(reader)?;
    Ok((hdr, stim))
}

/// Reads a `DNWB` stimulus request and discrete switch vector from `reader`.
pub fn read_dnwb_stimulus_request_with_switches<R: Read>(
    reader: &mut R,
) -> Result<(DnwbHeader64B, StemStimulus, Vec<u8>), DnwbError> {
    read_dnwb_stimulus_request_dyn(reader)
}

fn read_dnwb_stimulus_request_dyn(
    reader: &mut dyn Read,
) -> Result<(DnwbHeader64B, StemStimulus, Vec<u8>), DnwbError> {
    let mut hdr_buf = [0u8; 64];
    reader
        .read_exact(&mut hdr_buf)
        .map_err(|e| DnwbError::IoFailure(e.to_string()))?;
    let hdr = DnwbHeader64B::from_bytes(&hdr_buf)?;
    if hdr.flags & !DNWB_KNOWN_REQ_FLAGS != 0 {
        return Err(DnwbError::InvalidStimulusLayout(format!(
            "unknown DNWB request flags {:#06x}",
            hdr.flags
        )));
    }
    let payload_len =
        usize::try_from(hdr.payload_byte_len).map_err(|_| DnwbError::PayloadTooLarge {
            actual: hdr.payload_byte_len,
            max: MAX_DNWB_PAYLOAD_BYTES,
        })?;
    let mut payload = vec![0u8; payload_len];
    reader
        .read_exact(&mut payload)
        .map_err(|e| DnwbError::IoFailure(e.to_string()))?;

    let mut off = 0usize;
    let stem_id = if hdr.flags & DNWB_FLAG_STRUCTURED_V1 != 0 {
        let id_len = read_u16_le(&payload, &mut off)? as usize;
        let end = checked_add_usize(off, id_len)?;
        if payload.len() < end {
            return Err(DnwbError::TruncatedPacket {
                expected: end,
                actual: payload.len(),
            });
        }
        let s = std::str::from_utf8(&payload[off..end])
            .map_err(|e| DnwbError::InvalidStimulusLayout(e.to_string()))?
            .to_string();
        off = end;
        s
    } else {
        String::from("dnwb_stem")
    };

    let n_ctrl = hdr.num_controls as usize;
    let n_sw = hdr.num_switches as usize;
    let n_frames = hdr.num_frames as usize;
    let control_values = read_f32_vec(&payload, &mut off, n_ctrl)?;

    let sw_end = checked_add_usize(off, n_sw)?;
    if payload.len() < sw_end {
        return Err(DnwbError::TruncatedPacket {
            expected: sw_end,
            actual: payload.len(),
        });
    }
    let switches = payload[off..sw_end].to_vec();
    off = sw_end;

    let ch_mask = if hdr.flags & DNWB_FLAG_STRUCTURED_V1 != 0 {
        let mask = (hdr.flags >> DNWB_CHANNEL_MASK_SHIFT) & 0x0f;
        if mask.count_ones() as u16 != hdr.num_input_channels {
            return Err(DnwbError::InvalidStimulusLayout(format!(
                "channel mask popcount {} != num_input_channels {}",
                mask.count_ones(),
                hdr.num_input_channels
            )));
        }
        mask
    } else {
        if (hdr.flags >> DNWB_CHANNEL_MASK_SHIFT) & 0x0f != 0 {
            return Err(DnwbError::InvalidStimulusLayout(String::from(
                "channel mask bits set without DNWB_FLAG_STRUCTURED_V1",
            )));
        }
        match hdr.num_input_channels {
            0 => 0,
            1 => 0b0001,
            2 => 0b0011,
            3 => 0b0111,
            4 => 0b1111,
            n => {
                return Err(DnwbError::InvalidStimulusLayout(format!(
                    "unsupported legacy num_input_channels {n} (max 4)"
                )));
            }
        }
    };
    if (ch_mask & 1 == 0) && (hdr.num_input_channels > 0 || n_frames > 0) {
        return Err(DnwbError::InvalidStimulusLayout(String::from(
            "primary input channel in_1 bit missing in request channel mask",
        )));
    }

    let in_1 = if ch_mask & (1 << 0) != 0 {
        read_f32_vec(&payload, &mut off, n_frames)?
    } else {
        Vec::new()
    };
    let in_2 = if ch_mask & (1 << 1) != 0 {
        Some(read_f32_vec(&payload, &mut off, n_frames)?)
    } else {
        None
    };
    let in_3 = if ch_mask & (1 << 2) != 0 {
        Some(read_f32_vec(&payload, &mut off, n_frames)?)
    } else {
        None
    };
    let in_4 = if ch_mask & (1 << 3) != 0 {
        Some(read_f32_vec(&payload, &mut off, n_frames)?)
    } else {
        None
    };

    let control_trajectories = if hdr.flags & DNWB_FLAG_HAS_TRAJECTORIES != 0 {
        let p_cnt = read_u16_le(&payload, &mut off)? as usize;
        if p_cnt != n_ctrl {
            return Err(DnwbError::InvalidStimulusLayout(format!(
                "control_trajectories count {p_cnt} != num_controls {n_ctrl}"
            )));
        }
        let mut trajs = Vec::with_capacity(p_cnt);
        for _ in 0..p_cnt {
            trajs.push(read_f32_vec(&payload, &mut off, n_frames)?);
        }
        Some(trajs)
    } else {
        None
    };

    let freeze_transition_window = if hdr.flags & DNWB_FLAG_HAS_FREEZE_WINDOW != 0 {
        let start_u64 = read_u64_le(&payload, &mut off)?;
        let end_u64 = read_u64_le(&payload, &mut off)?;
        let start = usize::try_from(start_u64).map_err(|_| {
            DnwbError::InvalidStimulusLayout(String::from("freeze window start overflow"))
        })?;
        let end = usize::try_from(end_u64).map_err(|_| {
            DnwbError::InvalidStimulusLayout(String::from("freeze window end overflow"))
        })?;
        if start > end || end > n_frames {
            return Err(DnwbError::InvalidStimulusLayout(format!(
                "invalid freeze_transition_window [{start}, {end}] for num_frames {n_frames}"
            )));
        }
        Some([start, end])
    } else {
        None
    };

    if off != payload.len() {
        return Err(DnwbError::InvalidStimulusLayout(format!(
            "trailing unconsumed stimulus bytes: consumed {off} of {}",
            payload.len()
        )));
    }

    Ok((
        hdr,
        StemStimulus {
            stem_id,
            sample_rate_hz: hdr.sample_rate_hz,
            in_1,
            in_2,
            in_3,
            in_4,
            control_values,
            control_trajectories,
            freeze_transition_window,
        },
        switches,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_dnwb_protocol_roundtrip_multichannel_trajectories_and_guards() {
        assert_eq!(DnwbCommand::from_u16(0), Some(DnwbCommand::Reset));
        assert_eq!(DnwbCommand::from_u16(1), Some(DnwbCommand::RenderStem));
        assert_eq!(DnwbCommand::from_u16(2), Some(DnwbCommand::RenderFrame));
        assert_eq!(DnwbCommand::from_u16(99), None);

        assert!(DnwbHeader64B::from_bytes(&[0u8; 12]).is_err());
        let mut bad_magic = [0u8; 64];
        bad_magic[0..4].copy_from_slice(b"XXXX");
        assert!(matches!(
            DnwbHeader64B::from_bytes(&bad_magic),
            Err(DnwbError::InvalidMagic(_))
        ));
        let mut bad_ver = [0u8; 64];
        bad_ver[0..4].copy_from_slice(&DNWB_MAGIC);
        bad_ver[4..6].copy_from_slice(&99u16.to_le_bytes());
        assert!(matches!(
            DnwbHeader64B::from_bytes(&bad_ver),
            Err(DnwbError::UnsupportedVersion(99))
        ));
        let mut bad_cmd = [0u8; 64];
        bad_cmd[0..4].copy_from_slice(&DNWB_MAGIC);
        bad_cmd[4..6].copy_from_slice(&DNWB_VERSION.to_le_bytes());
        bad_cmd[6..8].copy_from_slice(&77u16.to_le_bytes());
        assert!(matches!(
            DnwbHeader64B::from_bytes(&bad_cmd),
            Err(DnwbError::InvalidCommand(77))
        ));

        let mut bad_rsvd = [0u8; 64];
        bad_rsvd[0..4].copy_from_slice(&DNWB_MAGIC);
        bad_rsvd[4..6].copy_from_slice(&DNWB_VERSION.to_le_bytes());
        bad_rsvd[6..8].copy_from_slice(&1u16.to_le_bytes());
        bad_rsvd[63] = 0xAA;
        assert_eq!(
            DnwbHeader64B::from_bytes(&bad_rsvd),
            Err(DnwbError::NonZeroReservedBytes)
        );

        let mut bad_oom = [0u8; 64];
        bad_oom[0..4].copy_from_slice(&DNWB_MAGIC);
        bad_oom[4..6].copy_from_slice(&DNWB_VERSION.to_le_bytes());
        bad_oom[6..8].copy_from_slice(&1u16.to_le_bytes());
        bad_oom[28..36].copy_from_slice(&(MAX_DNWB_PAYLOAD_BYTES + 1).to_le_bytes());
        assert!(matches!(
            DnwbHeader64B::from_bytes(&bad_oom),
            Err(DnwbError::PayloadTooLarge { .. })
        ));

        let stim = StemStimulus {
            stem_id: String::from("stem_4ch_96k"),
            sample_rate_hz: 96_000,
            in_1: vec![0.1, -0.2, 0.3, -0.4],
            in_2: Some(vec![-0.1, 0.2, -0.3, 0.4]),
            in_3: Some(vec![0.05, 0.05, -0.05, -0.05]),
            in_4: Some(vec![0.0, 0.1, 0.0, -0.1]),
            control_values: vec![0.25, -0.5],
            control_trajectories: Some(vec![vec![-0.5, -0.2, 0.2, 0.5], vec![0.4, 0.2, 0.0, -0.2]]),
            freeze_transition_window: Some([1, 3]),
        };
        let switches = vec![1u8, 2u8];
        let mut buf = Vec::new();
        let hdr = write_dnwb_stimulus_request_with_switches(
            &mut buf,
            DnwbCommand::RenderStem,
            &stim,
            &switches,
        )
        .unwrap();
        assert_eq!(hdr.num_input_channels, 4);
        assert_eq!(hdr.num_switches, 2);

        let mut cursor = Cursor::new(buf.clone());
        let (hdr_rt, stim_rt, sw_rt) =
            read_dnwb_stimulus_request_with_switches(&mut cursor).unwrap();
        assert_eq!(hdr, hdr_rt);
        assert_eq!(stim_rt, stim);
        assert_eq!(sw_rt, switches);

        // Truncated payload should fail cleanly
        let mut trunc_cursor = Cursor::new(&buf[..68]);
        assert!(read_dnwb_stimulus_request(&mut trunc_cursor).is_err());

        // Mismatched secondary channel length should fail cleanly
        let mut bad_stim = stim.clone();
        bad_stim.in_2 = Some(vec![0.1]);
        assert!(matches!(
            write_dnwb_stimulus_request(&mut Vec::new(), DnwbCommand::RenderStem, &bad_stim),
            Err(DnwbError::InvalidStimulusLayout(_))
        ));

        // Trajectory count & length validation guards
        let mut bad_traj_cnt = stim.clone();
        bad_traj_cnt.control_trajectories = Some(vec![vec![0.0; 4]]);
        assert!(matches!(
            write_dnwb_stimulus_request(&mut Vec::new(), DnwbCommand::RenderStem, &bad_traj_cnt),
            Err(DnwbError::InvalidStimulusLayout(_))
        ));
        let mut bad_traj_len = stim.clone();
        bad_traj_len.control_trajectories = Some(vec![vec![0.0; 2], vec![0.0; 4]]);
        assert!(matches!(
            write_dnwb_stimulus_request(&mut Vec::new(), DnwbCommand::RenderStem, &bad_traj_len),
            Err(DnwbError::InvalidStimulusLayout(_))
        ));
        let mut bad_freeze = stim.clone();
        bad_freeze.freeze_transition_window = Some([3, 1]);
        assert!(matches!(
            write_dnwb_stimulus_request(&mut Vec::new(), DnwbCommand::RenderStem, &bad_freeze),
            Err(DnwbError::InvalidStimulusLayout(_))
        ));

        // Truncated u16 / u64 / f32_vec & wire trajectory/freeze layout guards
        assert!(matches!(
            read_u16_le(&[0u8; 1], &mut 0),
            Err(DnwbError::TruncatedPacket { .. })
        ));
        assert!(matches!(
            read_u32_le(&[0u8; 3], &mut 0),
            Err(DnwbError::TruncatedPacket { .. })
        ));
        assert!(matches!(
            read_u64_le(&[0u8; 7], &mut 0),
            Err(DnwbError::TruncatedPacket { .. })
        ));
        assert!(matches!(
            read_f32_vec(&[0u8; 3], &mut 0, 1),
            Err(DnwbError::TruncatedPacket { .. })
        ));

        // Corrupt freeze_transition_window [3, 1] on wire + trajectory p_cnt mismatch on wire
        let mut wire_bad_freeze = buf.clone();
        let wlen = wire_bad_freeze.len();
        wire_bad_freeze[wlen - 16..wlen - 8].copy_from_slice(&3u64.to_le_bytes());
        wire_bad_freeze[wlen - 8..wlen].copy_from_slice(&1u64.to_le_bytes());
        assert!(matches!(
            read_dnwb_stimulus_request(&mut Cursor::new(wire_bad_freeze)),
            Err(DnwbError::InvalidStimulusLayout(_))
        ));
        let mut wire_bad_pcnt = buf.clone();
        let pcnt_off = wlen - 16 - 2 * 4 * 4 - 2;
        wire_bad_pcnt[pcnt_off..pcnt_off + 2].copy_from_slice(&99u16.to_le_bytes());
        assert!(matches!(
            read_dnwb_stimulus_request(&mut Cursor::new(wire_bad_pcnt)),
            Err(DnwbError::InvalidStimulusLayout(_))
        ));

        // Legacy flags=0 header parsing across 0..=4 channels + channel mask & truncation guards
        for n_ch in [0u16, 1, 2, 3, 4] {
            let mut legacy_hdr = hdr;
            legacy_hdr.flags = 0;
            legacy_hdr.num_input_channels = n_ch;
            legacy_hdr.num_frames = if n_ch == 0 { 0 } else { 4 };
            legacy_hdr.num_controls = 1;
            legacy_hdr.num_switches = 1;
            let mut legacy_payload = Vec::new();
            legacy_payload.extend_from_slice(&0.5_f32.to_le_bytes()); // 1 control
            legacy_payload.push(1u8); // 1 switch
            for _ in 0..((n_ch as usize) * 4) {
                legacy_payload.extend_from_slice(&0.25_f32.to_le_bytes());
            }
            legacy_hdr.payload_byte_len = legacy_payload.len() as u64;
            let mut legacy_wire = Vec::from(legacy_hdr.to_bytes());
            legacy_wire.extend_from_slice(&legacy_payload);
            let (_, legacy_stim, legacy_sw) =
                read_dnwb_stimulus_request_with_switches(&mut Cursor::new(&legacy_wire)).unwrap();
            assert_eq!(legacy_stim.stem_id, "dnwb_stem");
            assert_eq!(legacy_stim.in_2.is_some(), n_ch >= 2);
            assert_eq!(legacy_stim.in_3.is_some(), n_ch >= 3);
            assert_eq!(legacy_stim.in_4.is_some(), n_ch >= 4);
            assert_eq!(legacy_sw, vec![1u8]);
        }
        let mut zero_ch_nonzero_frames = hdr;
        zero_ch_nonzero_frames.flags = 0;
        zero_ch_nonzero_frames.num_input_channels = 0;
        zero_ch_nonzero_frames.num_controls = 0;
        zero_ch_nonzero_frames.num_switches = 0;
        zero_ch_nonzero_frames.payload_byte_len = 0;
        assert!(matches!(
            read_dnwb_stimulus_request(&mut Cursor::new(zero_ch_nonzero_frames.to_bytes())),
            Err(DnwbError::InvalidStimulusLayout(_))
        ));

        // Primary channel bit missing in structured v1 mask
        let mut missing_in1_hdr = hdr;
        missing_in1_hdr.num_input_channels = 1;
        missing_in1_hdr.flags = DNWB_FLAG_STRUCTURED_V1 | (0b0010 << DNWB_CHANNEL_MASK_SHIFT);
        missing_in1_hdr.payload_byte_len = 2 + 2 + 8 + 2; // id="s0" + 2 ctrls + 2 sw
        let mut missing_in1_wire = Vec::from(missing_in1_hdr.to_bytes());
        missing_in1_wire.extend_from_slice(&2u16.to_le_bytes());
        missing_in1_wire.extend_from_slice(b"s0");
        missing_in1_wire.extend_from_slice(&[0u8; 10]);
        assert!(matches!(
            read_dnwb_stimulus_request(&mut Cursor::new(missing_in1_wire)),
            Err(DnwbError::InvalidStimulusLayout(_))
        ));

        let mut trailing_buf = buf.clone();
        let new_len = hdr.payload_byte_len + 4;
        trailing_buf[28..36].copy_from_slice(&new_len.to_le_bytes());
        trailing_buf.extend_from_slice(&[0u8; 4]);
        assert!(matches!(
            read_dnwb_stimulus_request(&mut Cursor::new(trailing_buf)),
            Err(DnwbError::InvalidStimulusLayout(_))
        ));

        let mut bad_mask_buf = buf.clone();
        bad_mask_buf[16..18].copy_from_slice(&1u16.to_le_bytes()); // popcount(0b1111)=4 != 1
        assert!(matches!(
            read_dnwb_stimulus_request(&mut Cursor::new(bad_mask_buf)),
            Err(DnwbError::InvalidStimulusLayout(_))
        ));
    }
}
