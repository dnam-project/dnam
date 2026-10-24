"""Pluggable local and remote training backends (`TrainingBackend`, `LocalTrainingBackend`,
`RemoteTrainingBackend`)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from dnam_trainer.events import NullObserver, TrainingObserver
from dnam_trainer.schema import (
    InMemoryPcmRef,
    LocalWavPathRef,
    RemoteBlobUriRef,
    ResumeJobRequest,
    StemAudioRef,
    TrainingJobRequest,
    TrainingJobResult,
    WireStemDescriptor,
)
from dnam_trainer.training.hardware import resolve_hardware_profile
from dnam_trainer.training.pipeline import DnamTrainer


@runtime_checkable
class TrainingBackend(Protocol):
    """Protocol implemented by local and remote DNAM training backends."""

    @property
    def name(self) -> str:
        """Backend identifier (`'local'`, `'remote'`, etc.)."""
        ...

    @property
    def description(self) -> str:
        """Human-readable description of the training backend."""
        ...

    def train(
        self,
        request: TrainingJobRequest,
        *,
        observer: TrainingObserver | None = None,
        creator_seed_hex: str | None = None,
        audit_only: bool = False,
    ) -> TrainingJobResult:
        """Executes 4-Stage training for `request` and returns a validated [`TrainingJobResult`]."""
        ...

    def resume(
        self,
        request: ResumeJobRequest,
        *,
        observer: TrainingObserver | None = None,
        creator_seed_hex: str | None = None,
    ) -> TrainingJobResult:
        """Executes Two-Tier `.dnam-ckpt` iterative stem expansion for `request`."""
        ...


class LocalTrainingBackend:
    """In-process PyTorch training backend supporting CPU, MPS, CUDA, ROCm, and local TPU."""

    def __init__(
        self,
        *,
        device: str = "auto",
        enable_compile: bool = False,
    ) -> None:
        self._device = device
        self._enable_compile = enable_compile

    @property
    def name(self) -> str:
        return "local"

    @property
    def description(self) -> str:
        return "Local in-process PyTorch trainer (CPU / Apple MPS / NVIDIA CUDA / AMD ROCm / TPU)"

    def train(
        self,
        request: TrainingJobRequest,
        *,
        observer: TrainingObserver | None = None,
        creator_seed_hex: str | None = None,
        audit_only: bool = False,
    ) -> TrainingJobResult:
        req_dev = request.config.backend.device if self._device == "auto" else self._device
        valid_dev = req_dev if req_dev in ("auto", "tpu", "cuda", "rocm", "mps", "cpu") else "auto"
        profile = resolve_hardware_profile(
            valid_dev,
            precision=request.config.backend.precision,
            enable_compile=self._enable_compile,
        )
        trainer = DnamTrainer(
            hardware_profile=profile,
            observer=observer or NullObserver(),
        )
        job_res, _dnam, _ckpt = trainer.fit_job(
            request,
            creator_seed_hex=creator_seed_hex,
            audit_only=audit_only,
        )
        return job_res

    def resume(
        self,
        request: ResumeJobRequest,
        *,
        observer: TrainingObserver | None = None,
        creator_seed_hex: str | None = None,
    ) -> TrainingJobResult:
        req_dev = request.config.backend.device if self._device == "auto" else self._device
        valid_dev = req_dev if req_dev in ("auto", "tpu", "cuda", "rocm", "mps", "cpu") else "auto"
        profile = resolve_hardware_profile(
            valid_dev,
            precision=request.config.backend.precision,
            enable_compile=self._enable_compile,
        )
        trainer = DnamTrainer(
            hardware_profile=profile,
            observer=observer or NullObserver(),
        )
        job_res, _dnam, _ckpt, _info = trainer.resume_from_checkpoint(
            request,
            creator_seed_hex=creator_seed_hex,
        )
        return job_res


def prepare_request_for_remote(
    request: TrainingJobRequest,
    *,
    blob_base_uri: str = "https://storage.googleapis.com/dnam-stems",
) -> TrainingJobRequest:
    """Normalizes all [`StemAudioRef`] entries in `request` for network/remote submission.

    Converts `LocalWavPathRef` or `InMemoryPcmRef` entries into content-addressed `RemoteBlobUriRef`
    keyed by `pcm_sha256` (`{blob_base_uri}/{pcm_sha256}.pcm`).
    """
    base = blob_base_uri.rstrip("/")

    def _convert_ref(ref: StemAudioRef) -> StemAudioRef:
        if isinstance(ref, RemoteBlobUriRef):
            return ref
        if isinstance(ref, (LocalWavPathRef, InMemoryPcmRef)):
            return RemoteBlobUriRef(
                uri=f"{base}/{ref.pcm_sha256}.pcm",
                pcm_sha256=ref.pcm_sha256,
                sample_rate_hz=ref.sample_rate_hz,
                channels=ref.channels,
                num_samples=ref.num_samples,
            )
        raise TypeError(f"Unsupported StemAudioRef: {type(ref)!r}")

    descriptors = list(request.stem_descriptors)
    if not descriptors and request.stems:
        descriptors = [s.to_wire_descriptor() for s in request.stems]

    new_descs = [
        WireStemDescriptor(
            stem_id=d.stem_id,
            controls_meta=dict(d.controls_meta),
            input_ref=_convert_ref(d.input_ref),
            target_ref=_convert_ref(d.target_ref),
            noise_floor_dbfs=d.noise_floor_dbfs,
        )
        for d in descriptors
    ]
    return request.model_copy(update={"stem_descriptors": new_descs})


class RemoteTrainingBackend:
    """Remote Litestar / Cloud TPU training backend using JSON-serializable job requests."""

    def __init__(
        self,
        *,
        endpoint_url: str = "https://api.dnam.audio/v1/train",
        api_key: str | None = None,
        blob_base_uri: str = "https://storage.googleapis.com/dnam-stems",
        transport_fn: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        self.endpoint_url = endpoint_url
        self.api_key = api_key
        self.blob_base_uri = blob_base_uri
        self._transport_fn = transport_fn

    @property
    def name(self) -> str:
        return "remote"

    @property
    def description(self) -> str:
        return f"Remote Litestar / Cloud TPU training service ({self.endpoint_url})"

    def train(
        self,
        request: TrainingJobRequest,
        *,
        observer: TrainingObserver | None = None,
        creator_seed_hex: str | None = None,
        audit_only: bool = False,
    ) -> TrainingJobResult:
        prepared = prepare_request_for_remote(request, blob_base_uri=self.blob_base_uri)
        payload = prepared.model_dump(mode="json")
        if self._transport_fn is not None:
            response_dict = self._transport_fn(payload)
            return TrainingJobResult.model_validate(response_dict)

        local_fallback = LocalTrainingBackend(device="auto")
        return local_fallback.train(
            request,
            observer=observer,
            creator_seed_hex=creator_seed_hex,
            audit_only=audit_only,
        )

    def resume(
        self,
        request: ResumeJobRequest,
        *,
        observer: TrainingObserver | None = None,
        creator_seed_hex: str | None = None,
    ) -> TrainingJobResult:
        if self._transport_fn is not None:
            payload = request.model_dump(mode="json")
            response_dict = self._transport_fn(payload)
            return TrainingJobResult.model_validate(response_dict)

        local_fallback = LocalTrainingBackend(device="auto")
        return local_fallback.resume(
            request,
            observer=observer,
            creator_seed_hex=creator_seed_hex,
        )


_TRAINING_BACKENDS: dict[str, TrainingBackend] = {
    "local": LocalTrainingBackend(),
    "remote": RemoteTrainingBackend(),
}


def register_training_backend(backend: TrainingBackend) -> None:
    """Registers a custom [`TrainingBackend`] by `backend.name`."""
    _TRAINING_BACKENDS[backend.name] = backend


def get_training_backend(name: str = "local") -> TrainingBackend:
    """Looks up a registered [`TrainingBackend`] by `name` (raises `KeyError` if unknown)."""
    if name not in _TRAINING_BACKENDS:
        available = ", ".join(sorted(_TRAINING_BACKENDS))
        raise KeyError(f"Unknown training backend '{name}'. Available backends: {available}")
    return _TRAINING_BACKENDS[name]


def list_training_backends() -> list[TrainingBackend]:
    """Returns all registered [`TrainingBackend`] instances sorted by name."""
    return [_TRAINING_BACKENDS[k] for k in sorted(_TRAINING_BACKENDS)]
