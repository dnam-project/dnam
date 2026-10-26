"""4-Stage Deterministic Training Pipeline, Hardware Runtime, Surrogate Delay & `.dnam-ckpt`
Resume."""

from __future__ import annotations

from dnam_trainer.training.arena import (
    ARENA_OFFSET_FULL_END,
    ARENA_OFFSET_LITE_END,
    ARENA_OFFSET_ULTRA_END,
    ArenaWeightLayout,
    build_signed_dnam_containers,
    default_tensor_directory,
    rglru_soa_byte_offset,
    sign_header_with_creator_key,
)
from dnam_trainer.training.backends import (
    LocalTrainingBackend,
    RemoteTrainingBackend,
    TrainingBackend,
    get_training_backend,
    list_training_backends,
    prepare_request_for_remote,
    register_training_backend,
)
from dnam_trainer.training.hardware import (
    HardwareRuntimeProfile,
    move_to_host_float64,
    resolve_hardware_profile,
    setup_accelerator_environment,
    torch_dtype_from_profile_name,
)
from dnam_trainer.training.kronecker_head import (
    ARENA3_STEM_BYTE_LEN,
    MultiTierKroneckerHeadSolution,
    StemSufficientStatistics,
    compute_stem_sufficient_statistics,
    compute_svd_control_projector,
    evaluate_kronecker_head_features,
    merge_stem_statistics_for_resume,
    solve_all_tiers_kronecker_head_float64,
    solve_kronecker_head_float64,
)
from dnam_trainer.training.pipeline import (
    DnamTrainer,
    DnamTriPhysicsModel,
    ResumeExecutionResult,
    detect_core2_activity,
    resolve_stem_audio_pcm,
)
from dnam_trainer.training.surrogate_delay import (
    SurrogateDelayReadout,
    optimize_delay_with_surrogate,
    surrogate_delay_readout,
)
from dnam_trainer.training.telemetry import (
    TrainingStepRecord,
    build_training_telemetry_df,
    plot_esr_convergence_chart,
    save_telemetry_html_report,
    save_telemetry_parquet,
)

__all__ = [
    "ARENA3_STEM_BYTE_LEN",
    "ARENA_OFFSET_FULL_END",
    "ARENA_OFFSET_LITE_END",
    "ARENA_OFFSET_ULTRA_END",
    "ArenaWeightLayout",
    "DnamTrainer",
    "DnamTriPhysicsModel",
    "HardwareRuntimeProfile",
    "LocalTrainingBackend",
    "MultiTierKroneckerHeadSolution",
    "RemoteTrainingBackend",
    "ResumeExecutionResult",
    "StemSufficientStatistics",
    "SurrogateDelayReadout",
    "TrainingBackend",
    "TrainingStepRecord",
    "build_signed_dnam_containers",
    "build_training_telemetry_df",
    "compute_stem_sufficient_statistics",
    "compute_svd_control_projector",
    "default_tensor_directory",
    "detect_core2_activity",
    "evaluate_kronecker_head_features",
    "get_training_backend",
    "list_training_backends",
    "merge_stem_statistics_for_resume",
    "move_to_host_float64",
    "optimize_delay_with_surrogate",
    "plot_esr_convergence_chart",
    "prepare_request_for_remote",
    "register_training_backend",
    "resolve_hardware_profile",
    "resolve_stem_audio_pcm",
    "rglru_soa_byte_offset",
    "save_telemetry_html_report",
    "save_telemetry_parquet",
    "setup_accelerator_environment",
    "sign_header_with_creator_key",
    "solve_all_tiers_kronecker_head_float64",
    "solve_kronecker_head_float64",
    "surrogate_delay_readout",
    "torch_dtype_from_profile_name",
]
