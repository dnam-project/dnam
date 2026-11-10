#![forbid(unsafe_code)]
//! Persistent Stateful `DNWB` Subprocess Session & Multi-Packet Stream Server (`session`).
//!
//! Keeps GPL-isolated `FaustRunner` and `SpiceRunner` worker processes alive across sequential
//! [`DistillationTarget::render_frame`], [`DistillationTarget::render_stem`], and
//! [`DistillationTarget::reset_state`] calls over `stdin`/`stdout` pipes while enforcing
//! wall-clock watchdog timeouts and deterministic child reaping.

use super::{
    DnwbCommand, DnwbError, DnwbHeader64B, read_dnwb_stem_response,
    serve_dnwb_single_request_with_switches, write_dnwb_stimulus_request_with_switches,
};
use crate::targets::{DistillationTarget, StemResponse, StemStimulus};
use std::io::{Read, Write};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, SyncSender};
use std::time::{Duration, Instant};

fn read_first_byte_or_eof(reader: &mut dyn Read) -> Result<Option<u8>, DnwbError> {
    let mut first = [0u8; 1];
    loop {
        match reader.read(&mut first) {
            Ok(0) => return Ok(None),
            Ok(_) => return Ok(Some(first[0])),
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(DnwbError::IoFailure(e.to_string())),
        }
    }
}

/// Serves sequential `DNWB` requests from `reader` onto a single persistent `target` instance
/// until clean `EOF` at a packet boundary (`UnexpectedEof` mid-header or mid-payload returns `Err`).
pub fn serve_dnwb_stream_with_switches<R, W, T, F>(
    reader: &mut R,
    writer: &mut W,
    target: &mut T,
    mut on_switches: F,
) -> Result<usize, DnwbError>
where
    R: Read,
    W: Write,
    T: DistillationTarget,
    F: FnMut(&mut T, &[u8]),
{
    let reader_dyn: &mut dyn Read = reader;
    let mut writer_dyn: &mut dyn Write = writer;
    let mut served = 0usize;
    while let Some(first_byte) = read_first_byte_or_eof(reader_dyn)? {
        let prefix = [first_byte];
        let mut chained = (&prefix[..]).chain(&mut *reader_dyn);
        serve_dnwb_single_request_with_switches(
            &mut chained,
            &mut writer_dyn,
            target,
            &mut on_switches,
        )?;
        served += 1;
    }
    Ok(served)
}

/// Serves sequential `DNWB` requests from `reader` onto a single persistent `target` until clean `EOF`.
pub fn serve_dnwb_stream<R: Read, W: Write, T: DistillationTarget>(
    reader: &mut R,
    writer: &mut W,
    target: &mut T,
) -> Result<usize, DnwbError> {
    serve_dnwb_stream_with_switches(reader, writer, target, |_, _| {})
}

struct LiveDnwbWorker {
    cmd: Vec<String>,
    child: Child,
    req_tx: Option<SyncSender<Vec<u8>>>,
    write_ack_rx: Receiver<Result<(), String>>,
    resp_rx: Receiver<Result<(DnwbHeader64B, StemResponse), DnwbError>>,
}

impl LiveDnwbWorker {
    fn spawn(cmd: &[String]) -> Result<Self, DnwbError> {
        let Some((program, args)) = cmd.split_first() else {
            return Err(DnwbError::IoFailure(String::from(
                "empty subprocess command",
            )));
        };
        let mut child = Command::new(program)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| DnwbError::IoFailure(e.to_string()))?;

        let mut child_stdin = child
            .stdin
            .take()
            .ok_or_else(|| DnwbError::IoFailure(String::from("missing child stdin")))?;
        let mut child_stdout = child
            .stdout
            .take()
            .ok_or_else(|| DnwbError::IoFailure(String::from("missing child stdout")))?;

        let (req_tx, req_rx) = mpsc::sync_channel::<Vec<u8>>(1);
        let (write_ack_tx, write_ack_rx) = mpsc::sync_channel::<Result<(), String>>(1);
        std::thread::spawn(move || {
            while let Ok(wire) = req_rx.recv() {
                let res = child_stdin
                    .write_all(&wire)
                    .and_then(|()| child_stdin.flush())
                    .map_err(|e| e.to_string());
                let failed = res.is_err();
                if write_ack_tx.send(res).is_err() || failed {
                    break;
                }
            }
        });

        let (resp_tx, resp_rx) =
            mpsc::sync_channel::<Result<(DnwbHeader64B, StemResponse), DnwbError>>(1);
        std::thread::spawn(move || {
            loop {
                match read_first_byte_or_eof(&mut child_stdout) {
                    Ok(None) => break,
                    Ok(Some(first)) => {
                        let prefix = [first];
                        let mut chained = (&prefix[..]).chain(&mut child_stdout);
                        let res = read_dnwb_stem_response(&mut chained);
                        let is_err = res.is_err();
                        if resp_tx.send(res).is_err() || is_err {
                            break;
                        }
                    }
                    Err(e) => {
                        let _ = resp_tx.send(Err(e));
                        break;
                    }
                }
            }
        });

        Ok(Self {
            cmd: cmd.to_vec(),
            child,
            req_tx: Some(req_tx),
            write_ack_rx,
            resp_rx,
        })
    }

    fn terminate(&mut self) {
        self.req_tx.take();
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Drop for LiveDnwbWorker {
    fn drop(&mut self) {
        self.terminate();
    }
}

/// Reusable persistent `DNWB` worker session maintaining state across `render_frame` and `reset_state`
/// calls while automatically reaping child processes on timeout, I/O failure, `Clone`, or `Drop`.
#[derive(Default)]
pub struct DnwbWorkerSession {
    live: Option<LiveDnwbWorker>,
}

impl Clone for DnwbWorkerSession {
    fn clone(&self) -> Self {
        Self { live: None }
    }
}

impl std::fmt::Debug for DnwbWorkerSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DnwbWorkerSession")
            .field("is_live", &self.live.is_some())
            .finish()
    }
}

impl DnwbWorkerSession {
    /// Creates an idle `DnwbWorkerSession` that spawns its child lazily on first request.
    #[must_use]
    pub const fn new() -> Self {
        Self { live: None }
    }

    /// Returns `true` if a live worker process is currently attached to this session.
    #[must_use]
    pub const fn is_live(&self) -> bool {
        self.live.is_some()
    }

    /// Terminates and reaps any attached worker process.
    pub fn terminate(&mut self) {
        if let Some(mut worker) = self.live.take() {
            worker.terminate();
        }
    }

    /// Sends a `DnwbCommand::Reset` packet if a worker is currently live; on any error, terminates
    /// the worker so the next request spawns a clean DC-equilibrium worker.
    pub fn reset_if_live(&mut self, cmd: &[String], switches: &[u8], timeout: Duration) {
        if !self.is_live() {
            return;
        }
        let reset_stim = StemStimulus {
            stem_id: String::from("reset"),
            sample_rate_hz: 48_000,
            in_1: Vec::new(),
            in_2: None,
            in_3: None,
            in_4: None,
            control_values: Vec::new(),
            control_trajectories: None,
            freeze_transition_window: None,
        };
        if self
            .execute(cmd, DnwbCommand::Reset, &reset_stim, switches, timeout)
            .is_err()
        {
            self.terminate();
        }
    }

    /// Executes a `DNWB` request over the persistent worker process (spawning lazily if needed).
    pub fn execute(
        &mut self,
        cmd: &[String],
        command: DnwbCommand,
        stimulus: &StemStimulus,
        switches: &[u8],
        timeout: Duration,
    ) -> Result<StemResponse, DnwbError> {
        if timeout.is_zero() {
            self.terminate();
            return Err(DnwbError::WatchdogTimeout { timeout_ms: 0 });
        }
        let mut req_wire = Vec::new();
        write_dnwb_stimulus_request_with_switches(&mut req_wire, command, stimulus, switches)?;

        let needs_respawn = match self.live.as_mut() {
            None => true,
            Some(w) => w.cmd != cmd || !matches!(w.child.try_wait(), Ok(None)),
        };
        if needs_respawn {
            self.terminate();
            self.live = Some(LiveDnwbWorker::spawn(cmd)?);
        }

        let start = Instant::now();
        let res = (|| -> Result<StemResponse, DnwbError> {
            let worker = self.live.as_mut().expect("worker spawned above");
            worker
                .req_tx
                .as_ref()
                .ok_or_else(|| DnwbError::IoFailure(String::from("DNWB stdin channel closed")))?
                .send(req_wire)
                .map_err(|_| {
                    DnwbError::IoFailure(String::from("DNWB worker stdin disconnected"))
                })?;

            let rem_write = timeout.saturating_sub(start.elapsed());
            if rem_write.is_zero() {
                return Err(DnwbError::WatchdogTimeout {
                    timeout_ms: timeout.as_millis() as u64,
                });
            }
            match worker.write_ack_rx.recv_timeout(rem_write) {
                Ok(Ok(())) => {}
                Ok(Err(msg)) => return Err(DnwbError::IoFailure(msg)),
                Err(RecvTimeoutError::Timeout) => {
                    return Err(DnwbError::WatchdogTimeout {
                        timeout_ms: timeout.as_millis() as u64,
                    });
                }
                Err(RecvTimeoutError::Disconnected) => {
                    return Err(DnwbError::IoFailure(String::from(
                        "DNWB stdin writer thread disconnected",
                    )));
                }
            }

            let rem_read = timeout.saturating_sub(start.elapsed());
            if rem_read.is_zero() {
                return Err(DnwbError::WatchdogTimeout {
                    timeout_ms: timeout.as_millis() as u64,
                });
            }
            let (resp_hdr, resp) = match worker.resp_rx.recv_timeout(rem_read) {
                Ok(r) => r?,
                Err(RecvTimeoutError::Timeout) => {
                    return Err(DnwbError::WatchdogTimeout {
                        timeout_ms: timeout.as_millis() as u64,
                    });
                }
                Err(RecvTimeoutError::Disconnected) => {
                    let detail = match worker.child.try_wait() {
                        Ok(Some(st)) => format!("DNWB worker exited unexpectedly with {st}"),
                        _ => String::from("DNWB worker closed stdout before responding"),
                    };
                    return Err(DnwbError::IoFailure(detail));
                }
            };

            if resp_hdr.command != command {
                return Err(DnwbError::InvalidStimulusLayout(format!(
                    "response command {:?} != request command {:?}",
                    resp_hdr.command, command
                )));
            }
            match command {
                DnwbCommand::Reset => {
                    if !resp.out_1.is_empty() {
                        return Err(DnwbError::InvalidStimulusLayout(format!(
                            "Reset response contained {} unexpected output frames",
                            resp.out_1.len()
                        )));
                    }
                }
                DnwbCommand::RenderStem | DnwbCommand::RenderFrame => {
                    if resp.sample_rate_hz != stimulus.sample_rate_hz
                        || resp.out_1.len() != stimulus.in_1.len()
                    {
                        return Err(DnwbError::InvalidStimulusLayout(format!(
                            "response rate/len ({} Hz, {} frames) != stimulus ({} Hz, {} frames)",
                            resp.sample_rate_hz,
                            resp.out_1.len(),
                            stimulus.sample_rate_hz,
                            stimulus.in_1.len()
                        )));
                    }
                }
            }
            Ok(resp)
        })();

        if res.is_err() {
            self.terminate();
        }
        res
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::whitebox::FaustRunner;
    use std::io::Cursor;

    #[test]
    fn test_serve_dnwb_stream_stateful_multi_packet_and_truncated_tail_error() {
        let json = r#"{"ui":[{"items":[{"type":"hslider","label":"Drive","init":0.5,"min":0.0,"max":1.0,"step":0.01}]}]}"#;
        let mut server_target = FaustRunner::from_faust_json("s.dsp", json, 2_000).unwrap();
        let mut empty_out = Vec::new();
        assert_eq!(
            serve_dnwb_stream(
                &mut Cursor::new(Vec::<u8>::new()),
                &mut empty_out,
                &mut server_target
            )
            .unwrap(),
            0
        );

        let f0 = StemStimulus {
            stem_id: String::from("frame_0"),
            sample_rate_hz: 48_000,
            in_1: vec![0.4, -0.2, 0.3, 0.1],
            in_2: None,
            in_3: None,
            in_4: None,
            control_values: vec![0.5],
            control_trajectories: None,
            freeze_transition_window: None,
        };
        let f1 = StemStimulus {
            stem_id: String::from("frame_1"),
            ..f0.clone()
        };
        let mut req_stream = Vec::new();
        write_dnwb_stimulus_request_with_switches(
            &mut req_stream,
            DnwbCommand::RenderFrame,
            &f0,
            &[1],
        )
        .unwrap();
        write_dnwb_stimulus_request_with_switches(
            &mut req_stream,
            DnwbCommand::RenderFrame,
            &f1,
            &[1],
        )
        .unwrap();

        let mut resp_stream = Vec::new();
        let count = serve_dnwb_stream_with_switches(
            &mut Cursor::new(&req_stream),
            &mut resp_stream,
            &mut server_target,
            |t, sw| t.set_switches(sw),
        )
        .unwrap();
        assert_eq!(count, 2);

        let mut r_cur = Cursor::new(resp_stream);
        let (_, r0) = read_dnwb_stem_response(&mut r_cur).unwrap();
        let (_, r1) = read_dnwb_stem_response(&mut r_cur).unwrap();
        assert_ne!(
            r0.out_1, r1.out_1,
            "stateful LP filter must carry across frames"
        );

        // Mid-header truncation must fail rather than masquerading as clean EOF
        let mut truncated = req_stream[..20].to_vec();
        assert!(
            serve_dnwb_stream(
                &mut Cursor::new(&mut truncated),
                &mut Vec::new(),
                &mut server_target
            )
            .is_err()
        );

        // Session error, timeout, clone & drop lifecycle
        let mut sess = DnwbWorkerSession::new();
        assert!(!sess.is_live());
        assert!(format!("{sess:?}").contains("DnwbWorkerSession"));
        sess.reset_if_live(&[], &[], Duration::from_millis(50));
        assert!(
            sess.execute(
                &[],
                DnwbCommand::RenderStem,
                &f0,
                &[],
                Duration::from_millis(50)
            )
            .is_err()
        );
        assert!(
            sess.execute(
                &[String::from("cat")],
                DnwbCommand::RenderStem,
                &f0,
                &[],
                Duration::ZERO,
            )
            .is_err()
        );
        let cloned = sess.clone();
        assert!(!cloned.is_live());
        #[cfg(unix)]
        {
            assert!(
                sess.execute(
                    &[String::from("cat")],
                    DnwbCommand::RenderStem,
                    &f0,
                    &[],
                    Duration::from_millis(500),
                )
                .is_err()
            );
            assert!(
                sess.execute(
                    &[String::from("true")],
                    DnwbCommand::RenderStem,
                    &f0,
                    &[],
                    Duration::from_millis(500),
                )
                .is_err()
            );
            assert!(matches!(
                sess.execute(
                    &[String::from("sleep"), String::from("5")],
                    DnwbCommand::RenderStem,
                    &f0,
                    &[],
                    Duration::from_millis(20),
                ),
                Err(DnwbError::WatchdogTimeout { .. })
            ));
            // Mismatched sample_rate_hz / num_frames from worker response fails closed
            let mut bogus_wire = Vec::new();
            super::super::write_dnwb_stem_response(
                &mut bogus_wire,
                DnwbCommand::RenderStem,
                &StemResponse {
                    stem_id: String::from("bad"),
                    sample_rate_hz: 96_000,
                    out_1: vec![0.0_f32; 2],
                    out_2: None,
                    out_3: None,
                    out_4: None,
                    internal_nodes: std::collections::BTreeMap::new(),
                },
            )
            .unwrap();
            let oct: String = bogus_wire.iter().map(|b| format!("\\{:03o}", b)).collect();
            assert!(matches!(
                sess.execute(
                    &[
                        String::from("sh"),
                        String::from("-c"),
                        format!("head -c 64 >/dev/null; printf '{oct}'"),
                    ],
                    DnwbCommand::RenderStem,
                    &f0,
                    &[],
                    Duration::from_millis(500),
                ),
                Err(DnwbError::InvalidStimulusLayout(_))
            ));
        }

        // Multi-channel SpiceRunner + empty subprocess_cmd error propagation
        let mut spice = crate::whitebox::SpiceRunner::new("traj.cir", 2_000);
        let traj_stim = StemStimulus {
            stem_id: String::from("spice_traj"),
            sample_rate_hz: 48_000,
            in_1: vec![0.2, -0.2, 0.3, -0.3],
            in_2: Some(vec![0.3, -0.3, 0.2, -0.2]),
            in_3: Some(vec![0.1, -0.1, 0.1, -0.1]),
            in_4: Some(vec![-0.1, 0.1, -0.1, 0.1]),
            control_values: vec![f32::NAN],
            control_trajectories: Some(vec![vec![-0.5, f32::NAN, 0.5, 1.0]]),
            freeze_transition_window: None,
        };
        let traj_resp = spice.render_stem(&traj_stim).unwrap();
        assert_eq!(traj_resp.out_1.len(), 4);
        assert!(traj_resp.out_2.as_ref().unwrap().iter().any(|&v| v != 0.0));
        for ch_opt in [&traj_resp.out_3, &traj_resp.out_4] {
            assert!(ch_opt.as_ref().unwrap().iter().all(|v| v.is_finite()));
        }
        let mut spice_sub =
            crate::whitebox::SpiceRunner::new("sub.cir", 50).with_subprocess_command(Vec::new());
        spice_sub.reset_state();
        assert!(spice_sub.render_stem(&traj_stim).is_err());
        assert!(
            spice_sub
                .render_frame(&crate::targets::StimulusFrame {
                    frame_index: 0,
                    sample_rate_hz: 48_000,
                    in_1: vec![0.1; 4],
                    in_2: None,
                    in_3: None,
                    in_4: None,
                    control_values: vec![0.0],
                    control_trajectories: Some(vec![vec![0.0; 4]]),
                })
                .is_err()
        );
    }
}
