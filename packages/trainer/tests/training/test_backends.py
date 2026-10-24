"""Unit tests for `dnam_trainer.training.backends` (`LocalTrainingBackend`,
`RemoteTrainingBackend`)."""

from __future__ import annotations

from typing import Any

import pytest

from dnam_trainer.schema import (
    RemoteBlobUriRef,
    ResumeJobRequest,
    TrainingConfig,
    TrainingJobRequest,
    TrainingStem,
)
from dnam_trainer.training.backends import (
    LocalTrainingBackend,
    RemoteTrainingBackend,
    get_training_backend,
    list_training_backends,
    prepare_request_for_remote,
    register_training_backend,
)


def test_local_and_remote_training_backends(
    synthetic_training_stems: list[TrainingStem],
) -> None:
    backends = list_training_backends()
    names = [b.name for b in backends]
    assert "local" in names
    assert "remote" in names

    local = get_training_backend("local")
    assert isinstance(local, LocalTrainingBackend)

    req = TrainingJobRequest(
        model_id="m_backend",
        model_name="Backend Model",
        config=TrainingConfig(epochs=1, topology="core1_only"),
        stems=synthetic_training_stems[:2],
    )
    res_local = local.train(req, audit_only=True)
    assert res_local.model_id == "m_backend"

    # Prepare request for remote submission converts refs to RemoteBlobUriRef
    prep = prepare_request_for_remote(req, blob_base_uri="https://storage.googleapis.com/dnam-test")
    assert len(prep.stem_descriptors) == 2
    assert isinstance(prep.stem_descriptors[0].input_ref, RemoteBlobUriRef)
    assert prep.stem_descriptors[0].input_ref.uri.startswith(
        "https://storage.googleapis.com/dnam-test/"
    )

    # Remote backend with custom transport callable
    captured_payloads: list[dict[str, Any]] = []

    def _fake_transport(payload: dict[str, Any]) -> dict[str, Any]:
        captured_payloads.append(payload)
        return res_local.model_dump(mode="json")

    remote = RemoteTrainingBackend(transport_fn=_fake_transport)
    register_training_backend(remote)
    res_rem = remote.train(req)
    assert res_rem.model_id == "m_backend"
    assert len(captured_payloads) == 1

    # Resume via local and remote
    resume_req = ResumeJobRequest(
        checkpoint_bytes=res_local.checkpoint_bytes,
        new_stems=synthetic_training_stems,
        config=TrainingConfig(epochs=1, topology="core1_only"),
    )
    res_resume_local = local.resume(resume_req)
    assert res_resume_local.resume_tier_executed == 1

    res_resume_remote = remote.resume(resume_req)
    assert res_resume_remote.model_id == "m_backend"

    # RemoteTrainingBackend fallback when transport_fn is None
    remote_fallback = RemoteTrainingBackend(transport_fn=None)
    assert "Remote" in remote_fallback.description
    assert "Local" in local.description
    res_fb = remote_fallback.train(req, audit_only=True)
    assert res_fb.model_id == "m_backend"
    res_fb_res = remote_fallback.resume(resume_req)
    assert res_fb_res.resume_tier_executed == 1

    with pytest.raises(KeyError, match="Unknown training backend"):
        get_training_backend("nonexistent_backend")
