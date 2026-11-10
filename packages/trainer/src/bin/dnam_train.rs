#![forbid(unsafe_code)]
//! `dnam-train` — Command-line entrypoint for the DNAM Analytical Two-Sweep Trainer.

use std::process::ExitCode;

fn main() -> ExitCode {
    match dnam_trainer::run_cli_from_args(std::env::args_os()) {
        Ok(report) => {
            if !report.binary_pipe_mode {
                println!(
                    "dnam-train ready (max_tier={:?}, feature_dim={}, packed_gram_len={})",
                    report.max_tier, report.feature_dim, report.packed_gram_len
                );
            }
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("{err}");
            ExitCode::FAILURE
        }
    }
}
