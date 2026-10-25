//! Subsystem 4: `uv` Auto-Bootstrap Venv Manager & JSON-Job/JSONL Subprocess Bridge (`dnam-train`).

pub mod bridge;
pub mod env_manager;

pub use bridge::{
    TrainerBridge, TrainerJobOptions, TrainingEventMessage, build_training_request_json,
    parse_jsonl_events,
};
pub use env_manager::{TrainerEnvManager, TrainerEnvStatus};
