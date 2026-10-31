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
from dnam_trainer.training.metrics import compute_floor_regularized_besr_db
from dnam_trainer.training.pipeline import (
    DnamTrainer,
    DnamTriPhysicsModel,
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
    assert compute_floor_regularized_besr_db(np.ones((16, 2)), np.ones((16, 2))) < -100.0

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


def test_pipeline_helpers_and_selective_cascade_e2e(tmp_path: pytest.TempPathFactory) -> None:
    import hashlib
    from pathlib import Path
    from typing import Any, cast

    import torch

    from dnam_trainer.training.cascade_primitives import (
        measure_and_shift_usb_jitter_tensor,
        solve_cholesky_ridge,
    )
    from dnam_trainer.training.metrics import compute_band_split_esr_db_torch
    from dnam_trainer.training.pipeline import (
        _calibrate_capabilities_and_leveling,
        _estimate_core2_ring_buffer_log2,
        _extract_job_stems,
        _to_frames_2ch,
        detect_wet_polarity_inverted,
        measure_gated_loudness_db,
        resolve_stem_audio_pcm,
    )

    assert _to_frames_2ch(np.ones((8, 1), dtype=np.float32)).shape == (8, 2)
    assert _to_frames_2ch(np.ones((8, 3), dtype=np.float32)).shape == (8, 2)
    with pytest.raises(TypeError, match="Unknown StemAudioRef"):
        resolve_stem_audio_pcm(cast(Any, "bad"))

    mdl_tmp = DnamTriPhysicsModel(sample_rate_hz=48_000)
    assert mdl_tmp(torch.zeros((1, 16, 1)), torch.zeros((1, 4))).shape == (1, 16, 2)
    assert mdl_tmp.forward_head_from_features(
        torch.zeros((1, 16, 192)), torch.zeros(4), tier="lite"
    ).shape == (1, 16, 2)
    from types import SimpleNamespace

    st_pad = SimpleNamespace(
        stem_id="pad",
        controls_meta={},
        input_audio=np.zeros(16, dtype=np.float32),
        target_audio=np.zeros(32, dtype=np.float32),
        input_pcm_sha256="0" * 64,
        target_pcm_sha256="1" * 64,
        sample_rate_hz=48_000,
    )
    _, _, in_p, _, _, _ = _extract_job_stems(cast(Any, [st_pad]), [], 48_000)
    assert in_p[0].shape == (32, 2)
    with pytest.raises(ValueError, match="sample rate"):
        _extract_job_stems(cast(Any, [st_pad]), [], 44_100)

    # 1. measure_gated_loudness_db (1D, (2, N), empty, k_weighted_lufs, quiet)
    sig = np.sin(np.linspace(0.0, 100.0, 9600, dtype=np.float64)) * 0.25
    assert measure_gated_loudness_db(sig, metric="k_weighted_lufs") > -30.0
    assert measure_gated_loudness_db(np.vstack([sig, sig]), metric="rms_dbfs") > -30.0
    assert measure_gated_loudness_db(np.zeros(0)) == -96.0
    assert measure_gated_loudness_db(np.zeros(9600)) == -96.0
    assert measure_gated_loudness_db(np.ones(32) * 0.1) > -30.0

    # 2. detect_wet_polarity_inverted & _estimate_core2_ring_buffer_log2
    assert not detect_wet_polarity_inverted([], [])
    assert not detect_wet_polarity_inverted(
        [np.ones((16, 2), dtype=np.float32)], [np.ones((16, 2), dtype=np.float32)]
    )
    assert detect_wet_polarity_inverted(
        [sig[:, None].astype(np.float32)], [(-sig[:, None]).astype(np.float32)]
    )
    assert _estimate_core2_ring_buffer_log2([], 0.0, 48_000, True) == 12
    assert _estimate_core2_ring_buffer_log2([], 250.0, 48_000, True) >= 12

    from types import SimpleNamespace

    fake_ctrl = SimpleNamespace(
        calibration=SimpleNamespace(
            role="delay_time_ms",
            lobatto_physical_anchors=[50.0, 200.0, 400.0],
            conditional_physical_anchors={"mode_a": [100.0, 500.0]},
        )
    )
    assert _estimate_core2_ring_buffer_log2([fake_ctrl], 0.0, 48_000, True) >= 14

    from dnam_trainer.training.kronecker_head import MultiTierKroneckerHeadSolution

    dummy_head = MultiTierKroneckerHeadSolution(
        w_lite_5x2x64=np.zeros((5, 2, 64), dtype=np.float64),
        w_full_5x2x128=np.zeros((5, 2, 128), dtype=np.float64),
        w_ultra_5x2x192=np.zeros((5, 2, 192), dtype=np.float64),
        ridge_lite=1e-6,
        ridge_full=1e-6,
        ridge_ultra=1e-6,
    )
    for b_mode in ("none", "modeled_control", "host_parallel"):
        cap_b, _, _, _ = _calibrate_capabilities_and_leveling(
            controls_schema=[],
            config=SimpleNamespace(blend_mode=b_mode),
            core2_active=False,
            head_sol=dummy_head,
            sample_rate_hz=48_000,
            inputs_list=[],
            targets_list=[],
            stems_controls=[],
            default_features_192=None,
            stem_stats=[],
        )
        assert cap_b.blend_mode == b_mode

    # 3. detect_core2_activity with trailing silent tail (>= 100 ms of silence at end of x)
    x_tail = np.zeros((12_000, 2), dtype=np.float32)
    x_tail[500:4000] = 0.3
    y_tail = x_tail.copy()
    y_tail[6000:9000] = 0.15
    act_t, _ = detect_core2_activity([x_tail], [y_tail], sample_rate_hz=48_000, topology="auto")
    assert act_t
    assert not detect_core2_activity(
        [np.zeros((512, 2), dtype=np.float32)],
        [np.zeros((512, 2), dtype=np.float32)],
        sample_rate_hz=48_000,
        topology="auto",
    )[0]

    # 4. solve_cholesky_ridge, measure_and_shift_usb_jitter_tensor, compute_band_split_esr_db_torch
    g_t = torch.eye(4, dtype=torch.float32)
    b_1d = torch.ones(4, dtype=torch.float32)
    w_1d = solve_cholesky_ridge(g_t, b_1d)
    assert w_1d.shape == (4,)

    x_j = torch.sin(torch.linspace(0.0, 80.0, 512, dtype=torch.float32))
    y_j = torch.roll(x_j, shifts=2)
    _, lag_j, _ = measure_and_shift_usb_jitter_tensor(x_j, y_j, min_len=64)
    assert abs(lag_j) > 0.5
    _, lag_c, _ = measure_and_shift_usb_jitter_tensor(x_j, y_j, min_len=64, cached_lag=1.5)
    assert lag_c == pytest.approx(1.5)
    _, lag_z, _ = measure_and_shift_usb_jitter_tensor(x_j, y_j, min_len=64, cached_lag=0.0)
    assert lag_z == 0.0

    # Fuzz fallback in measure_and_shift_usb_jitter_tensor
    y_uncorr = torch.cos(torch.linspace(0.0, 500.0, 512, dtype=torch.float32))
    _, lag_u, _ = measure_and_shift_usb_jitter_tensor(x_j, y_uncorr, min_len=64)
    assert isinstance(lag_u, float)

    # compute_band_split_esr_db_torch branches
    assert compute_band_split_esr_db_torch(torch.zeros(4), torch.zeros(4)) == (0.0, 0.0, 0.0, 0.0)
    b_t, _, _, _ = compute_band_split_esr_db_torch(
        x_j,
        x_j.unsqueeze(-1).expand(512, 2),
        band_floors_dbfs=(-90.0, -90.0, -90.0),
    )
    assert b_t <= -60.0

    # 5. End-to-end fit_job & resume_from_checkpoint with min_stem_len >= 32_768 (cascade=True)
    l_long = 32_768
    t_arr = np.linspace(0.0, 0.68, l_long, dtype=np.float32)
    x_long = (0.25 * np.sin(2.0 * np.pi * 220.0 * t_arr)).astype(np.float32)
    y_long_0 = np.vstack([np.tanh(2.0 * x_long), np.tanh(2.0 * x_long)]).astype(np.float32)
    y_long_1 = np.vstack([np.tanh(3.5 * x_long), np.tanh(3.5 * x_long)]).astype(np.float32)
    sha_x = hashlib.sha256(x_long.tobytes()).hexdigest()
    sha_y0 = hashlib.sha256(y_long_0.tobytes()).hexdigest()
    sha_y1 = hashlib.sha256(y_long_1.tobytes()).hexdigest()

    stems_long = [
        TrainingStem(
            stem_id="long_0",
            controls_meta={"drive": 2.0},
            input_audio=x_long,
            target_audio=y_long_0,
            input_pcm_sha256=sha_x,
            target_pcm_sha256=sha_y0,
            sample_rate_hz=48_000,
        ),
        TrainingStem(
            stem_id="long_1",
            controls_meta={"drive": 8.0},
            input_audio=x_long,
            target_audio=y_long_1,
            input_pcm_sha256=sha_x,
            target_pcm_sha256=sha_y1,
            sample_rate_hz=48_000,
        ),
    ]
    c_dir = Path(str(tmp_path)) / "long_cache"
    req_long = TrainingJobRequest(
        model_id="long_cascade",
        model_name="Long Cascade",
        controls_schema=[parse_compact_range_spec("drive=0..10:5")],
        config=TrainingConfig(
            epochs=1,
            topology="core1_only",
            cache_enabled=True,
            cache_dir=str(c_dir),
        ),
        stems=stems_long,
    )
    tr = DnamTrainer()
    res_l, _, ckpt_l = tr.fit_job(req_long)
    assert res_l.esr_ultra_db < -15.0
    assert ckpt_l.preamble.has_calib_stats is True

    # Audit-only mode with cache hit
    res_audit, _, _ = tr.fit_job(req_long, audit_only=True)
    assert res_audit.esr_ultra_db < 0.0

    # Resume with all_reused_long (both stems reused and >= 32,768 frames) + presets + user_metadata
    res_req_l = ResumeJobRequest(
        checkpoint_bytes=ckpt_l.to_bytes(),
        new_stems=stems_long,
        config=TrainingConfig(epochs=1, topology="core1_only"),
    )
    res_r, _, _, info_r = tr.resume_from_checkpoint(
        res_req_l,
        presets=[PresetDef(id="p1", name="P1", controls={"drive": 5.0})],
        user_metadata={"tag": "v1"},
    )
    assert info_r.reused_stem_count == 2
    assert res_r.esr_ultra_db < -5.0

    # Also test stereo 32-bit float WAV roundtrip & sample rate validation
    from dnam_trainer.training.wav_io import read_wav, write_wav_32bit_float

    w_st_path = write_wav_32bit_float(Path(str(tmp_path)) / "st32.wav", y_long_0[:, :64], 48_000)
    rd_st, sr_st = read_wav(w_st_path)
    assert sr_st == 48_000
    assert rd_st.shape == (2, 64)
    with pytest.raises(ValueError, match="Invalid sample_rate_hz"):
        write_wav_32bit_float(Path(str(tmp_path)) / "bad.wav", x_long[:16], 1000)
