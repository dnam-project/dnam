#![forbid(unsafe_code)]
//! `DNWB` Stem Response Framing (`out_1..out_4` + `internal_nodes`) & GPL-Isolated Subprocess Pipe IPC.

use super::{
    DNWB_CHANNEL_MASK_SHIFT, DNWB_FLAG_STRUCTURED_V1, DNWB_MAGIC, DNWB_VERSION, DnwbCommand,
    DnwbError, DnwbHeader64B, DnwbWorkerSession, MAX_DNWB_PAYLOAD_BYTES, checked_add_usize,
    checked_mul_usize, ensure_finite_f32_slice, read_dnwb_stimulus_request_with_switches,
    read_f32_vec, read_u16_le, read_u32_le,
};
use crate::targets::{DistillationTarget, StemResponse, StemStimulus, StimulusFrame};
use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::time::Duration;

/// Writes a `DNWB` stem response (`64 B` header + `stem_id` + `out_1..out_4` + `internal_nodes`) to `writer`.
pub fn write_dnwb_stem_response<W: Write>(
    writer: &mut W,
    command: DnwbCommand,
    response: &StemResponse,
) -> Result<DnwbHeader64B, DnwbError> {
    write_dnwb_stem_response_dyn(writer, command, response)
}

fn write_dnwb_stem_response_dyn(
    writer: &mut dyn Write,
    command: DnwbCommand,
    response: &StemResponse,
) -> Result<DnwbHeader64B, DnwbError> {
    let n_frames = response.out_1.len();
    if n_frames > u32::MAX as usize {
        return Err(DnwbError::InvalidStimulusLayout(String::from(
            "response num_frames exceeds u32::MAX",
        )));
    }
    let stem_bytes = response.stem_id.as_bytes();
    if stem_bytes.len() > u16::MAX as usize {
        return Err(DnwbError::InvalidStimulusLayout(String::from(
            "response stem_id exceeds u16::MAX",
        )));
    }
    if response.internal_nodes.len() > u16::MAX as usize {
        return Err(DnwbError::InvalidStimulusLayout(String::from(
            "internal_nodes count exceeds u16::MAX",
        )));
    }

    ensure_finite_f32_slice(&response.out_1, "out_1")?;
    let opt_outputs = [
        (1u16, response.out_2.as_ref()),
        (2u16, response.out_3.as_ref()),
        (3u16, response.out_4.as_ref()),
    ];
    let mut ch_mask: u16 = 1 << 0;
    let mut n_out_ch: usize = 1;
    for (bit, ch_opt) in opt_outputs {
        if let Some(ch) = ch_opt {
            if ch.len() != n_frames {
                return Err(DnwbError::InvalidStimulusLayout(format!(
                    "output channel {} length {} != out_1 length {n_frames}",
                    bit + 1,
                    ch.len()
                )));
            }
            ensure_finite_f32_slice(ch, "optional output channel")?;
            ch_mask |= 1 << bit;
            n_out_ch += 1;
        }
    }

    let mut payload_bytes = checked_add_usize(2, stem_bytes.len())?;
    let out_samples = checked_mul_usize(n_out_ch, n_frames)?;
    payload_bytes = checked_add_usize(payload_bytes, checked_mul_usize(out_samples, 4)?)?;
    for (name, trace) in &response.internal_nodes {
        if name.len() > u16::MAX as usize || trace.len() > u32::MAX as usize {
            return Err(DnwbError::InvalidStimulusLayout(String::from(
                "internal node name or trace length overflow",
            )));
        }
        ensure_finite_f32_slice(trace, "internal_nodes")?;
        let entry_hdr = checked_add_usize(6, name.len())?;
        let entry_data = checked_mul_usize(trace.len(), 4)?;
        payload_bytes =
            checked_add_usize(payload_bytes, checked_add_usize(entry_hdr, entry_data)?)?;
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
        sample_rate_hz: response.sample_rate_hz,
        num_frames: n_frames as u32,
        num_input_channels: 0,
        num_output_channels: n_out_ch as u16,
        num_controls: 0,
        num_switches: 0,
        num_internal_nodes: response.internal_nodes.len() as u16,
        flags: DNWB_FLAG_STRUCTURED_V1 | (ch_mask << DNWB_CHANNEL_MASK_SHIFT),
        payload_byte_len,
    };

    let mut payload = Vec::with_capacity(payload_bytes);
    payload.extend_from_slice(&(stem_bytes.len() as u16).to_le_bytes());
    payload.extend_from_slice(stem_bytes);
    for &y in &response.out_1 {
        payload.extend_from_slice(&y.to_le_bytes());
    }
    for (_, ch_opt) in opt_outputs {
        if let Some(ch) = ch_opt {
            for &y in ch {
                payload.extend_from_slice(&y.to_le_bytes());
            }
        }
    }
    for (name, trace) in &response.internal_nodes {
        let nb = name.as_bytes();
        payload.extend_from_slice(&(nb.len() as u16).to_le_bytes());
        payload.extend_from_slice(nb);
        payload.extend_from_slice(&(trace.len() as u32).to_le_bytes());
        for &v in trace {
            payload.extend_from_slice(&v.to_le_bytes());
        }
    }

    writer
        .write_all(&hdr.to_bytes())
        .and_then(|()| writer.write_all(&payload))
        .map_err(|e| DnwbError::IoFailure(e.to_string()))?;
    Ok(hdr)
}

/// Reads a `DNWB` stem response (`out_1..out_4` + `internal_nodes`) from `reader`.
pub fn read_dnwb_stem_response<R: Read>(
    reader: &mut R,
) -> Result<(DnwbHeader64B, StemResponse), DnwbError> {
    read_dnwb_stem_response_dyn(reader)
}

fn read_dnwb_stem_response_dyn(
    reader: &mut dyn Read,
) -> Result<(DnwbHeader64B, StemResponse), DnwbError> {
    let mut hdr_buf = [0u8; 64];
    reader
        .read_exact(&mut hdr_buf)
        .map_err(|e| DnwbError::IoFailure(e.to_string()))?;
    let hdr = DnwbHeader64B::from_bytes(&hdr_buf)?;
    const KNOWN_RESP_FLAGS: u16 = DNWB_FLAG_STRUCTURED_V1 | (0x0f << DNWB_CHANNEL_MASK_SHIFT);
    if hdr.flags & DNWB_FLAG_STRUCTURED_V1 == 0 || (hdr.flags & !KNOWN_RESP_FLAGS != 0) {
        return Err(DnwbError::InvalidStimulusLayout(format!(
            "invalid or unsupported DNWB response flags {:#06x}",
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
    let id_len = read_u16_le(&payload, &mut off)? as usize;
    let id_end = checked_add_usize(off, id_len)?;
    if payload.len() < id_end {
        return Err(DnwbError::TruncatedPacket {
            expected: id_end,
            actual: payload.len(),
        });
    }
    let stem_id = std::str::from_utf8(&payload[off..id_end])
        .map_err(|e| DnwbError::InvalidStimulusLayout(e.to_string()))?
        .to_string();
    off = id_end;

    let n_frames = hdr.num_frames as usize;
    let ch_mask = (hdr.flags >> DNWB_CHANNEL_MASK_SHIFT) & 0x0f;
    if ch_mask.count_ones() as u16 != hdr.num_output_channels {
        return Err(DnwbError::InvalidStimulusLayout(format!(
            "output channel mask popcount {} != num_output_channels {}",
            ch_mask.count_ones(),
            hdr.num_output_channels
        )));
    }
    if (ch_mask & 1 == 0) && (hdr.num_output_channels > 0 || n_frames > 0) {
        return Err(DnwbError::InvalidStimulusLayout(String::from(
            "primary output channel out_1 bit missing in response channel mask",
        )));
    }

    let out_1 = if ch_mask & (1 << 0) != 0 {
        read_f32_vec(&payload, &mut off, n_frames)?
    } else {
        Vec::new()
    };
    let out_2 = if ch_mask & (1 << 1) != 0 {
        Some(read_f32_vec(&payload, &mut off, n_frames)?)
    } else {
        None
    };
    let out_3 = if ch_mask & (1 << 2) != 0 {
        Some(read_f32_vec(&payload, &mut off, n_frames)?)
    } else {
        None
    };
    let out_4 = if ch_mask & (1 << 3) != 0 {
        Some(read_f32_vec(&payload, &mut off, n_frames)?)
    } else {
        None
    };

    let mut internal_nodes = BTreeMap::new();
    for _ in 0..hdr.num_internal_nodes {
        let name_len = read_u16_le(&payload, &mut off)? as usize;
        let name_end = checked_add_usize(off, name_len)?;
        if payload.len() < name_end {
            return Err(DnwbError::TruncatedPacket {
                expected: name_end,
                actual: payload.len(),
            });
        }
        let name = std::str::from_utf8(&payload[off..name_end])
            .map_err(|e| DnwbError::InvalidStimulusLayout(e.to_string()))?
            .to_string();
        off = name_end;
        let trace_len = read_u32_le(&payload, &mut off)? as usize;
        let trace = read_f32_vec(&payload, &mut off, trace_len)?;
        if internal_nodes.insert(name.clone(), trace).is_some() {
            return Err(DnwbError::InvalidStimulusLayout(format!(
                "duplicate internal_nodes key '{name}' in DNWB response"
            )));
        }
    }

    if off != payload.len() {
        return Err(DnwbError::InvalidStimulusLayout(format!(
            "trailing unconsumed response bytes: consumed {off} of {}",
            payload.len()
        )));
    }

    Ok((
        hdr,
        StemResponse {
            stem_id,
            sample_rate_hz: hdr.sample_rate_hz,
            out_1,
            out_2,
            out_3,
            out_4,
            internal_nodes,
        },
    ))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn validate_runner_stimulus_layout(
    sample_rate_hz: u32,
    in_1: &[f32],
    in_2: Option<&[f32]>,
    in_3: Option<&[f32]>,
    in_4: Option<&[f32]>,
    control_values: &[f32],
    control_trajectories: Option<&Vec<Vec<f32>>>,
    freeze_window: Option<[usize; 2]>,
) -> Result<(), DnwbError> {
    if !(8_000..=1_536_000).contains(&sample_rate_hz) {
        return Err(DnwbError::InvalidStimulusLayout(format!(
            "sample_rate_hz {sample_rate_hz} outside supported [8000, 1536000]"
        )));
    }
    let n_frames = in_1.len();
    for (ch_num, opt) in [(2usize, in_2), (3, in_3), (4, in_4)] {
        if let Some(ch) = opt
            && ch.len() != n_frames
        {
            return Err(DnwbError::InvalidStimulusLayout(format!(
                "input channel {ch_num} length {} != in_1 length {n_frames}",
                ch.len()
            )));
        }
    }
    if let Some(trajs) = control_trajectories {
        if trajs.len() > u16::MAX as usize || trajs.len() != control_values.len() {
            return Err(DnwbError::InvalidStimulusLayout(format!(
                "control_trajectories len {} != control_values len {}",
                trajs.len(),
                control_values.len()
            )));
        }
        for (p, t) in trajs.iter().enumerate() {
            if t.len() != n_frames {
                return Err(DnwbError::InvalidStimulusLayout(format!(
                    "control_trajectories[{p}] len {} != num_frames {n_frames}",
                    t.len()
                )));
            }
        }
    }
    if let Some([start, end]) = freeze_window
        && (start > end || end > n_frames)
    {
        return Err(DnwbError::InvalidStimulusLayout(format!(
            "invalid freeze_transition_window [{start}, {end}] for num_frames {n_frames}"
        )));
    }
    Ok(())
}

/// Serves a single `DNWB` pipe request from `reader`, invoking `on_switches` before dispatching
/// to `target` and writing the `DNWB` response to `writer`.
pub fn serve_dnwb_single_request_with_switches<R, W, T, F>(
    reader: &mut R,
    writer: &mut W,
    target: &mut T,
    on_switches: F,
) -> Result<(), DnwbError>
where
    R: Read,
    W: Write,
    T: DistillationTarget,
    F: FnOnce(&mut T, &[u8]),
{
    let (hdr, stim, switches) = read_dnwb_stimulus_request_with_switches(reader)?;
    on_switches(target, &switches);
    let resp = match hdr.command {
        DnwbCommand::Reset => {
            target.reset_state();
            StemResponse {
                stem_id: stim.stem_id,
                sample_rate_hz: stim.sample_rate_hz,
                out_1: Vec::new(),
                out_2: None,
                out_3: None,
                out_4: None,
                internal_nodes: BTreeMap::new(),
            }
        }
        DnwbCommand::RenderStem => target
            .render_stem(&stim)
            .map_err(|e| DnwbError::IoFailure(format!("{e:?}")))?,
        DnwbCommand::RenderFrame => {
            let req_stem_id = stim.stem_id;
            let frame_index = req_stem_id
                .strip_prefix("frame_")
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(0);
            let frame = StimulusFrame {
                frame_index,
                sample_rate_hz: stim.sample_rate_hz,
                in_1: stim.in_1,
                in_2: stim.in_2,
                in_3: stim.in_3,
                in_4: stim.in_4,
                control_values: stim.control_values,
                control_trajectories: stim.control_trajectories,
            };
            let mut r = target
                .render_frame(&frame)
                .map_err(|e| DnwbError::IoFailure(format!("{e:?}")))?;
            r.stem_id = req_stem_id;
            r
        }
    };
    write_dnwb_stem_response(writer, hdr.command, &resp)?;
    writer
        .flush()
        .map_err(|e| DnwbError::IoFailure(e.to_string()))?;
    Ok(())
}

/// Serves a single `DNWB` pipe request from `reader`, dispatching to `target` and writing the
/// `DNWB` response to `writer`.
pub fn serve_dnwb_single_request<R: Read, W: Write, T: DistillationTarget>(
    reader: &mut R,
    writer: &mut W,
    target: &mut T,
) -> Result<(), DnwbError> {
    serve_dnwb_single_request_with_switches(reader, writer, target, |_, _| {})
}

/// Executes an out-of-process GPL-isolated worker command (`cmd[0] cmd[1..]`) over `stdin`/`stdout`
/// using `64 B` `DNWB` pipe framing with concurrent non-deadlocking `stdin`/`stdout` pump threads
/// and a wall-clock watchdog `timeout`.
pub fn execute_dnwb_subprocess_request(
    cmd: &[String],
    command: DnwbCommand,
    stimulus: &StemStimulus,
    switches: &[u8],
    timeout: Duration,
) -> Result<StemResponse, DnwbError> {
    let mut session = DnwbWorkerSession::new();
    session.execute(cmd, command, stimulus, switches, timeout)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::whitebox::{
        FaustRunner, SpiceRunner, write_dnwb_stimulus_request,
        write_dnwb_stimulus_request_with_switches,
    };
    use std::io::Cursor;
    use std::time::Instant;

    #[test]
    fn test_ipc_response_guards_and_server_error_propagation() {
        let empty_resp = StemResponse {
            stem_id: String::from("r0"),
            sample_rate_hz: 48_000,
            out_1: Vec::new(),
            out_2: None,
            out_3: None,
            out_4: None,
            internal_nodes: BTreeMap::new(),
        };
        let mut wire = Vec::new();
        let hdr = write_dnwb_stem_response(&mut wire, DnwbCommand::Reset, &empty_resp).unwrap();

        // Zero output channels (ch_mask = 0, num_output_channels = 0) and missing STRUCTURED_V1 flag
        let mut zero_ch_hdr = hdr;
        zero_ch_hdr.num_output_channels = 0;
        zero_ch_hdr.flags = DNWB_FLAG_STRUCTURED_V1;
        let mut zero_ch_wire = Vec::from(zero_ch_hdr.to_bytes());
        zero_ch_wire.extend_from_slice(&wire[64..]);
        let (_, z_resp) = read_dnwb_stem_response(&mut Cursor::new(zero_ch_wire)).unwrap();
        assert!(z_resp.out_1.is_empty());

        let mut unflagged_hdr = hdr;
        unflagged_hdr.flags = 0;
        assert!(matches!(
            read_dnwb_stem_response(&mut Cursor::new(unflagged_hdr.to_bytes())),
            Err(DnwbError::InvalidStimulusLayout(_))
        ));

        let mono_resp = StemResponse {
            stem_id: String::from("mono_only"),
            sample_rate_hz: 48_000,
            out_1: vec![0.25, -0.5],
            out_2: None,
            out_3: None,
            out_4: None,
            internal_nodes: BTreeMap::new(),
        };
        let mut mono_wire = Vec::new();
        write_dnwb_stem_response(&mut mono_wire, DnwbCommand::RenderStem, &mono_resp).unwrap();
        let (_, mono_rt) = read_dnwb_stem_response(&mut Cursor::new(mono_wire)).unwrap();
        assert_eq!(mono_rt, mono_resp);

        // Long stem_id (> u16::MAX) and invalid UTF-8 stem_id / node_name guards
        let mut huge_id_resp = empty_resp.clone();
        huge_id_resp.stem_id = "x".repeat(usize::from(u16::MAX) + 1);
        assert!(
            write_dnwb_stem_response(&mut Vec::new(), DnwbCommand::Reset, &huge_id_resp).is_err()
        );
        let mut huge_node_resp = empty_resp.clone();
        huge_node_resp
            .internal_nodes
            .insert("n".repeat(usize::from(u16::MAX) + 1), vec![0.0]);
        assert!(
            write_dnwb_stem_response(&mut Vec::new(), DnwbCommand::Reset, &huge_node_resp).is_err()
        );
        let mut bad_utf8_id = zero_ch_hdr;
        bad_utf8_id.payload_byte_len = 4;
        let mut bad_utf8_wire = Vec::from(bad_utf8_id.to_bytes());
        bad_utf8_wire.extend_from_slice(&2u16.to_le_bytes());
        bad_utf8_wire.extend_from_slice(&[0xff, 0xff]);
        assert!(read_dnwb_stem_response(&mut Cursor::new(bad_utf8_wire)).is_err());

        // serve_dnwb_single_request_with_switches on FaustRunner across Reset, RenderStem & RenderFrame
        let faust_json = r#"{"ui":[{"items":[{"type":"hslider","label":"Drive","init":0.5,"min":0.0,"max":1.0,"step":0.01},{"type":"checkbox","label":"Bright"}]}]}"#;
        let mut faust = FaustRunner::from_faust_json("ipc.dsp", faust_json, 2_000).unwrap();
        let ok_stim = StemStimulus {
            stem_id: String::from("ok_faust"),
            sample_rate_hz: 48_000,
            in_1: vec![0.2, -0.2],
            in_2: None,
            in_3: None,
            in_4: None,
            control_values: vec![0.3],
            control_trajectories: None,
            freeze_transition_window: None,
        };
        for cmd in [
            DnwbCommand::Reset,
            DnwbCommand::RenderStem,
            DnwbCommand::RenderFrame,
        ] {
            let mut req = Vec::new();
            write_dnwb_stimulus_request_with_switches(&mut req, cmd, &ok_stim, &[1]).unwrap();
            let mut out = Vec::new();
            serve_dnwb_single_request_with_switches(
                &mut Cursor::new(req),
                &mut out,
                &mut faust,
                |t, sw| t.set_switches(sw),
            )
            .unwrap();
            let (_, r) = read_dnwb_stem_response(&mut Cursor::new(out)).unwrap();
            if cmd != DnwbCommand::Reset {
                assert_eq!(r.out_1.len(), 2);
            }
        }

        // Output channel mask popcount mismatch & missing primary out_1 bit & duplicate internal_nodes
        let mut bad_mask_wire = wire.clone();
        bad_mask_wire[18..20].copy_from_slice(&3u16.to_le_bytes());
        assert!(matches!(
            read_dnwb_stem_response(&mut Cursor::new(bad_mask_wire)),
            Err(DnwbError::InvalidStimulusLayout(_))
        ));
        let mut no_out1_hdr = hdr;
        no_out1_hdr.num_output_channels = 1;
        no_out1_hdr.flags = DNWB_FLAG_STRUCTURED_V1 | (0b0010 << DNWB_CHANNEL_MASK_SHIFT);
        let mut no_out1_wire = Vec::from(no_out1_hdr.to_bytes());
        no_out1_wire.extend_from_slice(&wire[64..]);
        assert!(matches!(
            read_dnwb_stem_response(&mut Cursor::new(no_out1_wire)),
            Err(DnwbError::InvalidStimulusLayout(_))
        ));
        let mut dup_node_hdr = zero_ch_hdr;
        dup_node_hdr.num_internal_nodes = 2;
        let mut dup_payload = Vec::new();
        dup_payload.extend_from_slice(&2u16.to_le_bytes());
        dup_payload.extend_from_slice(b"r0");
        for _ in 0..2 {
            dup_payload.extend_from_slice(&2u16.to_le_bytes());
            dup_payload.extend_from_slice(b"nd");
            dup_payload.extend_from_slice(&0u32.to_le_bytes());
        }
        dup_node_hdr.payload_byte_len = dup_payload.len() as u64;
        let mut dup_wire = Vec::from(dup_node_hdr.to_bytes());
        dup_wire.extend_from_slice(&dup_payload);
        assert!(matches!(
            read_dnwb_stem_response(&mut Cursor::new(dup_wire)),
            Err(DnwbError::InvalidStimulusLayout(_))
        ));

        // Truncated stem_id length inside payload
        let mut trunc_id_hdr = hdr;
        trunc_id_hdr.payload_byte_len = 3;
        let mut trunc_id_wire = Vec::from(trunc_id_hdr.to_bytes());
        trunc_id_wire.extend_from_slice(&10u16.to_le_bytes());
        trunc_id_wire.push(b'a');
        assert!(matches!(
            read_dnwb_stem_response(&mut Cursor::new(trunc_id_wire)),
            Err(DnwbError::TruncatedPacket { .. })
        ));

        // Truncated internal_nodes name_len inside payload
        let mut trunc_node_hdr = zero_ch_hdr;
        trunc_node_hdr.num_internal_nodes = 1;
        let mut node_payload = Vec::new();
        node_payload.extend_from_slice(&2u16.to_le_bytes());
        node_payload.extend_from_slice(b"r0");
        node_payload.extend_from_slice(&12u16.to_le_bytes()); // claims 12-byte node name
        node_payload.push(b'x');
        trunc_node_hdr.payload_byte_len = node_payload.len() as u64;
        let mut trunc_node_wire = Vec::from(trunc_node_hdr.to_bytes());
        trunc_node_wire.extend_from_slice(&node_payload);
        assert!(matches!(
            read_dnwb_stem_response(&mut Cursor::new(trunc_node_wire)),
            Err(DnwbError::TruncatedPacket { .. })
        ));

        // serve_dnwb_single_request propagates target render_stem / render_frame errors cleanly
        let stim = StemStimulus {
            stem_id: String::from("err_stem"),
            sample_rate_hz: 48_000,
            in_1: vec![0.1, -0.1],
            in_2: None,
            in_3: None,
            in_4: None,
            control_values: vec![0.0],
            control_trajectories: None,
            freeze_transition_window: None,
        };
        let mut failing_spice = SpiceRunner::new("timeout.cir", 0);
        for cmd in [DnwbCommand::RenderStem, DnwbCommand::RenderFrame] {
            let mut req = Vec::new();
            write_dnwb_stimulus_request(&mut req, cmd, &stim).unwrap();
            assert!(
                serve_dnwb_single_request(
                    &mut Cursor::new(req),
                    &mut Vec::new(),
                    &mut failing_spice
                )
                .is_err()
            );
        }

        // Direct in-process validate_runner_stimulus_layout branches + subprocess error branches
        assert!(
            validate_runner_stimulus_layout(4_000, &[0.0], None, None, None, &[], None, None)
                .is_err()
        );
        assert!(
            validate_runner_stimulus_layout(
                48_000,
                &[0.0, 0.1],
                Some(&[0.0]),
                None,
                None,
                &[],
                None,
                None,
            )
            .is_err()
        );
        let empty_trajs: Vec<Vec<f32>> = Vec::new();
        let short_trajs: Vec<Vec<f32>> = vec![vec![0.5]];
        assert!(
            validate_runner_stimulus_layout(
                48_000,
                &[0.0, 0.1],
                None,
                None,
                None,
                &[0.5],
                Some(&empty_trajs),
                None,
            )
            .is_err()
        );
        assert!(
            validate_runner_stimulus_layout(
                48_000,
                &[0.0, 0.1],
                None,
                None,
                None,
                &[0.5],
                Some(&short_trajs),
                None,
            )
            .is_err()
        );
        assert!(
            validate_runner_stimulus_layout(
                48_000,
                &[0.0, 0.1],
                None,
                None,
                None,
                &[],
                None,
                Some([2, 1]),
            )
            .is_err()
        );

        assert!(matches!(
            execute_dnwb_subprocess_request(
                &[String::from("sh")],
                DnwbCommand::Reset,
                &stim,
                &[],
                Duration::ZERO,
            ),
            Err(DnwbError::WatchdogTimeout { timeout_ms: 0 })
        ));
        assert!(
            execute_dnwb_subprocess_request(
                &[],
                DnwbCommand::Reset,
                &stim,
                &[],
                Duration::from_millis(50),
            )
            .is_err()
        );
        #[cfg(unix)]
        {
            assert!(
                execute_dnwb_subprocess_request(
                    &[
                        String::from("sh"),
                        String::from("-c"),
                        String::from("exit 1")
                    ],
                    DnwbCommand::Reset,
                    &stim,
                    &[],
                    Duration::from_millis(500),
                )
                .is_err()
            );
            // Grandchild inheriting stdout must NOT hang WatchdogTimeout
            let t0 = Instant::now();
            assert!(matches!(
                execute_dnwb_subprocess_request(
                    &[
                        String::from("sh"),
                        String::from("-c"),
                        String::from("sleep 2 & sleep 2"),
                    ],
                    DnwbCommand::Reset,
                    &stim,
                    &[],
                    Duration::from_millis(25),
                ),
                Err(DnwbError::WatchdogTimeout { .. })
            ));
            assert!(t0.elapsed() < Duration::from_millis(500));
        }

        // RenderFrame stem_id ("frame_5") preservation, unknown flags rejection, and Spice end-freeze extension
        let mut frame5_stim = stim.clone();
        frame5_stim.stem_id = String::from("frame_5");
        let mut req_f5 = Vec::new();
        write_dnwb_stimulus_request(&mut req_f5, DnwbCommand::RenderFrame, &frame5_stim).unwrap();
        let mut out_f5 = Vec::new();
        let mut spice = SpiceRunner::new("f5.cir", 2_000);
        serve_dnwb_single_request(&mut Cursor::new(&req_f5), &mut out_f5, &mut spice).unwrap();
        let (_hdr_f5, resp_f5) = read_dnwb_stem_response(&mut Cursor::new(&out_f5)).unwrap();
        assert_eq!(resp_f5.stem_id, "frame_5");

        let mut bad_req_flags = req_f5.clone();
        bad_req_flags[26..28].copy_from_slice(&0x8001_u16.to_le_bytes());
        assert!(
            crate::whitebox::read_dnwb_stimulus_request(&mut Cursor::new(bad_req_flags)).is_err()
        );

        let mut bad_legacy_ch = [0u8; 64];
        bad_legacy_ch[0..4].copy_from_slice(b"DNWB");
        bad_legacy_ch[4..6].copy_from_slice(&1u16.to_le_bytes());
        bad_legacy_ch[8..12].copy_from_slice(&48_000u32.to_le_bytes());
        bad_legacy_ch[16..18].copy_from_slice(&5u16.to_le_bytes());
        assert!(
            crate::whitebox::read_dnwb_stimulus_request(&mut Cursor::new(bad_legacy_ch)).is_err()
        );
        // Legacy header with channel mask bits set without STRUCTURED_V1 must fail
        let mut bad_legacy_mask = bad_legacy_ch;
        bad_legacy_mask[16..18].copy_from_slice(&1u16.to_le_bytes());
        bad_legacy_mask[26..28].copy_from_slice(&0x0100_u16.to_le_bytes());
        assert!(
            crate::whitebox::read_dnwb_stimulus_request(&mut Cursor::new(bad_legacy_mask)).is_err()
        );
        // Oversized stem_id and switches (> u16::MAX) rejected cleanly
        let mut huge_id_stim = stim.clone();
        huge_id_stim.stem_id = "s".repeat(65_536);
        assert!(
            write_dnwb_stimulus_request(&mut Vec::new(), DnwbCommand::Reset, &huge_id_stim)
                .is_err()
        );
        assert!(
            crate::whitebox::write_dnwb_stimulus_request_with_switches(
                &mut Vec::new(),
                DnwbCommand::Reset,
                &stim,
                &vec![0u8; 65_536],
            )
            .is_err()
        );

        let mut bad_resp_flags = out_f5.clone();
        bad_resp_flags[26..28].copy_from_slice(&0x8101_u16.to_le_bytes());
        assert!(read_dnwb_stem_response(&mut Cursor::new(bad_resp_flags)).is_err());

        // SpiceRunner zero-phase render_stem with freeze_transition_window ending at n holds cap_voltage
        let mut fw_end_stim = StemStimulus {
            stem_id: String::from("fw_end"),
            sample_rate_hz: 48_000,
            in_1: vec![0.6_f32; 96],
            in_2: None,
            in_3: None,
            in_4: None,
            control_values: vec![0.5],
            control_trajectories: None,
            freeze_transition_window: Some([16, 96]),
        };
        fw_end_stim.in_1[16..].fill(0.0);
        let fw_resp = spice.render_stem(&fw_end_stim).unwrap();
        // With freeze extended through the 39-sample tail, out_1[80..96] stays constant at the frozen plateau
        let spread = fw_resp.out_1[80..96]
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &v| {
                (lo.min(v), hi.max(v))
            });
        assert!((spread.1 - spread.0).abs() < 1.0e-5, "spread={spread:?}");

        // Single-request pipe server round-trip for Reset, RenderStem, and RenderFrame at 96 kHz
        let stim_96k = StemStimulus {
            stem_id: String::from("stem_96k"),
            sample_rate_hz: 96_000,
            in_1: vec![0.2, -0.4, 0.6, -0.8],
            in_2: None,
            in_3: None,
            in_4: None,
            control_values: vec![0.5],
            control_trajectories: None,
            freeze_transition_window: None,
        };
        for cmd in [
            DnwbCommand::Reset,
            DnwbCommand::RenderStem,
            DnwbCommand::RenderFrame,
        ] {
            let mut req_bytes = Vec::new();
            write_dnwb_stimulus_request(&mut req_bytes, cmd, &stim_96k).unwrap();
            let mut out_bytes = Vec::new();
            serve_dnwb_single_request(&mut Cursor::new(req_bytes), &mut out_bytes, &mut spice)
                .unwrap();
            let (_h, r) = read_dnwb_stem_response(&mut Cursor::new(out_bytes)).unwrap();
            if cmd == DnwbCommand::Reset {
                assert!(r.out_1.is_empty());
            } else {
                assert_eq!(r.out_1.len(), 4);
                assert_eq!(r.sample_rate_hz, 96_000);
            }
        }

        // Response round-trip with 4 outputs + internal nodes + non-finite f32 fail-closed guards
        let mut internal_nodes = BTreeMap::new();
        internal_nodes.insert(String::from("node_a"), vec![0.25, -0.25, 0.5, -0.5]);
        let resp = StemResponse {
            stem_id: String::from("stem_4ch_96k"),
            sample_rate_hz: 96_000,
            out_1: vec![0.2, -0.4, 0.6, -0.8],
            out_2: Some(vec![-0.2, 0.4, -0.6, 0.8]),
            out_3: Some(vec![0.1, 0.1, -0.1, -0.1]),
            out_4: Some(vec![-0.1, -0.1, 0.1, 0.1]),
            internal_nodes,
        };
        let mut resp_wire = Vec::new();
        let resp_hdr =
            write_dnwb_stem_response(&mut resp_wire, DnwbCommand::RenderStem, &resp).unwrap();
        let (resp_hdr_rt, resp_rt) =
            read_dnwb_stem_response(&mut Cursor::new(resp_wire.clone())).unwrap();
        assert_eq!(resp_hdr, resp_hdr_rt);
        assert_eq!(resp_rt, resp);

        let mut bad_resp = resp.clone();
        bad_resp.out_2 = Some(vec![0.1]);
        assert!(
            write_dnwb_stem_response(&mut Vec::new(), DnwbCommand::RenderStem, &bad_resp).is_err()
        );
        let mut nan_resp = resp.clone();
        nan_resp.out_1[0] = f32::NAN;
        assert!(
            write_dnwb_stem_response(&mut Vec::new(), DnwbCommand::RenderStem, &nan_resp).is_err()
        );
        let mut nan_stim = stim_96k.clone();
        nan_stim.in_1[1] = f32::INFINITY;
        assert!(
            write_dnwb_stimulus_request(&mut Vec::new(), DnwbCommand::RenderStem, &nan_stim)
                .is_err()
        );
        // Corrupt wire f32 in resp_wire to IEEE-754 NaN and verify read_dnwb_stem_response fails closed
        let mut nan_wire = resp_wire.clone();
        let first_sample_off = 64 + 2 + resp.stem_id.len();
        nan_wire[first_sample_off..first_sample_off + 4].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(read_dnwb_stem_response(&mut Cursor::new(nan_wire)).is_err());

        assert!(read_dnwb_stem_response(&mut Cursor::new(&resp_wire[..68])).is_err());
        let mut resp_trailing = resp_wire.clone();
        let r_len = resp_hdr.payload_byte_len + 2;
        resp_trailing[28..36].copy_from_slice(&r_len.to_le_bytes());
        resp_trailing.extend_from_slice(&[0u8; 2]);
        assert!(read_dnwb_stem_response(&mut Cursor::new(resp_trailing)).is_err());
    }
}
