"""Unit tests for `dnam_trainer.training.pipeline` (`DnamTriPhysicsModel`, `DnamTrainer`,
topology auto-detection, 4-Stage training, and Two-Tier `.dnam-ckpt` iterative resume)."""

from __future__ import annotations

import numpy as np
import pytest

from dnam_trainer.schema import (
    PresetDef,
    ResumeJobRequest,
    TrainingConfig,
    TrainingJobRequest,
    TrainingStem,
    parse_compact_range_spec,
    parse_compact_switch_spec,
)
from dnam_trainer.training.pipeline import (
    DnamTrainer,
    DnamTriPhysicsModel,
    compute_esr_db,
    detect_core2_activity,
)


def test_detect_core2_activity_modes() -> None:
    rng = np.random.default_rng(10)
    x = rng.standard_normal((8192, 2)).astype(np.float32) * 0.2
    y_dry = (x * 0.8).astype(np.float32)

    # core1_only always bypasses Core 2
    act_c1, tail_c1 = detect_core2_activity(
        [x], [y_dry], sample_rate_hz=48_000, topology="core1_only"
    )
    assert not act_c1
    assert tail_c1 == 0.0

    # full always enables Core 2
    act_full, tail_full = detect_core2_activity(
        [x], [y_dry], sample_rate_hz=48_000, topology="full"
    )
    assert act_full
    assert tail_full >= 120.0

    # auto on dry instant signal -> False
    act_auto_dry, _ = detect_core2_activity([x], [y_dry], sample_rate_hz=48_000, topology="auto")
    assert not act_auto_dry

    # auto on signal with strong late delay echo at +2400 samples (50 ms) -> True
    y_echo = y_dry.copy()
    y_echo[2400:] += 0.65 * x[:-2400]
    act_auto_wet, tail_wet = detect_core2_activity(
        [x], [y_echo], sample_rate_hz=48_000, topology="auto"
    )
    assert act_auto_wet
    assert tail_wet > 0.0


def test_dnam_trainer_4stage_fit_and_two_tier_ckpt_resume(
    synthetic_training_stems: list[TrainingStem],
) -> None:
    ctrls = [
        parse_compact_range_spec("drive=0..10:5"),
        parse_compact_switch_spec("attack=cut,flat,boost:flat"),
    ]
    req = TrainingJobRequest(
        model_id="b7k_test",
        model_name="B7K Test Model",
        author="Pytest",
        controls_schema=ctrls,
        presets=[PresetDef(id="p0", name="Noon", controls={"drive": 5.0, "attack": "flat"})],
        config=TrainingConfig(epochs=2, topology="core1_only"),
        stems=synthetic_training_stems[:2],
    )

    trainer = DnamTrainer()
    seed_hex = "33" * 32
    job_res, dnam_c, ckpt_c = trainer.fit_job(req, creator_seed_hex=seed_hex)

    assert job_res.model_id == "b7k_test"
    assert not dnam_c.is_checkpoint
    assert ckpt_c.is_checkpoint
    assert ckpt_c.preamble.num_ckpt_stems == 2
    assert str(dnam_c.verification) == "CreatorVerified"
    assert job_res.esr_ultra_db < 0.0

    # Two-Tier .dnam-ckpt iterative resume adding the 3rd stem (`stem_02_high`)
    resume_req = ResumeJobRequest(
        checkpoint_bytes=ckpt_c.to_bytes(),
        new_stems=synthetic_training_stems,  # 2 existing + 1 new
        config=TrainingConfig(epochs=2, topology="core1_only"),
    )
    res_job, res_dnam, res_ckpt, res_info = trainer.resume_from_checkpoint(
        resume_req,
        controls_schema=ctrls,
        creator_seed_hex=seed_hex,
    )
    assert res_info.tier_executed == 1
    assert res_info.reused_stem_count == 2
    assert res_info.evaluated_stem_count == 1
    assert res_ckpt.preamble.num_ckpt_stems == 3
    assert str(res_dnam.verification) == "CreatorVerified"
    assert res_job.resume_tier_executed == 1

    # Calling resume_from_checkpoint on a non-checkpoint (.dnam) raises ValueError
    with pytest.raises(ValueError, match=r"requires a \.dnam-ckpt"):
        trainer.resume_from_checkpoint(
            ResumeJobRequest(checkpoint_bytes=dnam_c.to_bytes(), new_stems=synthetic_training_stems)
        )

    # Also test DnamTriPhysicsModel with core2_active=True and fit_job with topology="full" and wire
    # descriptors
    from pathlib import Path

    from dnam_trainer.schema import (
        InMemoryPcmRef,
        LocalWavPathRef,
        RemoteBlobUriRef,
        WireStemDescriptor,
    )
    from dnam_trainer.training.pipeline import resolve_stem_audio_pcm
    from dnam_trainer.training.wav_io import write_wav_24bit

    m_full = DnamTriPhysicsModel(sample_rate_hz=48_000, core2_active=True)
    assert m_full.core2_ufdn is not None
    assert compute_esr_db(np.ones((16, 2)), np.ones((16, 2))) < -100.0

    # Test fit_job with topology="full" (`core2_active=True`)
    req_full_c2 = TrainingJobRequest(
        model_id="b7k_c2",
        model_name="B7K Core2",
        controls_schema=ctrls,
        config=TrainingConfig(epochs=1, topology="full"),
        stems=synthetic_training_stems[:2],
    )
    res_c2, dnam_c2, _ = trainer.fit_job(req_full_c2, audit_only=True)
    assert res_c2.core2_active
    assert dnam_c2.header.core2_mufdn.active

    # Test resolve_stem_audio_pcm and descriptor-based _extract_job_stems
    tmp_wav = Path("/tmp/dnam_test_resolve_stem.wav")
    write_wav_24bit(tmp_wav, synthetic_training_stems[0].input_audio, sample_rate_hz=48_000)
    sha_in = synthetic_training_stems[0].input_pcm_sha256
    n_samp = synthetic_training_stems[0].num_samples
    pcm_loc, sr_loc = resolve_stem_audio_pcm(
        LocalWavPathRef(path=str(tmp_wav), pcm_sha256=sha_in, num_samples=n_samp)
    )
    assert sr_loc == 48_000
    assert pcm_loc.shape[1] == 2
    pcm_rem, sr_rem = resolve_stem_audio_pcm(
        RemoteBlobUriRef(uri=f"file://{tmp_wav}", pcm_sha256=sha_in, num_samples=n_samp)
    )
    assert sr_rem == 48_000
    assert pcm_rem.shape[1] == 2

    with pytest.raises(ValueError, match="InMemoryPcmRef"):
        resolve_stem_audio_pcm(InMemoryPcmRef(pcm_sha256=sha_in, num_samples=n_samp))
    with pytest.raises(ValueError, match="Unsupported remote blob"):
        resolve_stem_audio_pcm(
            RemoteBlobUriRef(uri="https://example.com/a.wav", pcm_sha256=sha_in, num_samples=n_samp)
        )

    req_desc = TrainingJobRequest(
        model_id="b7k_desc",
        model_name="B7K Desc",
        config=TrainingConfig(epochs=1, topology="core1_only"),
        stem_descriptors=[
            WireStemDescriptor(
                stem_id="s_desc",
                controls_meta={},
                input_ref=LocalWavPathRef(path=str(tmp_wav), pcm_sha256=sha_in, num_samples=n_samp),
                target_ref=LocalWavPathRef(
                    path=str(tmp_wav), pcm_sha256=sha_in, num_samples=n_samp
                ),
            )
        ],
    )
    res_desc, _, _ = trainer.fit_job(req_desc, audit_only=True)
    assert res_desc.model_id == "b7k_desc"
    tmp_wav.unlink(missing_ok=True)
