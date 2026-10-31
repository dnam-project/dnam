"""Unit and property tests for `dnam_trainer.training.calib_cache` and `.dnam-ckpt` Arena 4."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from dnam_engine.container import DnamContainer
from dnam_engine.schema import (
    ControlBusConfig,
    Core1Config,
    Core2Config,
    DatasetProvenance,
    DnamJsonHeader,
    IoConfig,
    LevelingConfig,
    ModelCapabilities,
    ModelMetadata,
    SlimmableTiersConfig,
    TierProfile,
)
from dnam_trainer.events import CacheEvent, TrainingEvent
from dnam_trainer.training.arena import (
    ARENA_OFFSET_FULL_END,
    ARENA_OFFSET_LITE_END,
    ARENA_OFFSET_ULTRA_END,
    ArenaWeightLayout,
    build_signed_dnam_containers,
)
from dnam_trainer.training.calib_cache import (
    KIND_INPUT_ONLY,
    KIND_STAGE_BASIS,
    Arena4CalibrationData,
    InputCalibStats,
    SidecarCacheStore,
    StemCalibStats,
)
from dnam_trainer.training.kronecker_head import compute_stem_sufficient_statistics


class _CacheEventCollector:
    def __init__(self) -> None:
        self.events: list[TrainingEvent] = []

    def on_training_event(self, event: TrainingEvent) -> None:
        self.events.append(event)


def test_arena4_calibration_data_roundtrip_and_tamper_rejection() -> None:
    sha_in = "a1" * 32
    inp = InputCalibStats(
        input_pcm_sha256=sha_in,
        num_samples=48_000,
        win_bounds=((0, 24_000), (24_000, 48_000)),
        win_grams_f64=np.eye(8, dtype=np.float64)[None, :, :].repeat(2, axis=0),
        s1_proj_f32=np.ones((16, 8), dtype=np.float32),
    )
    st = StemCalibStats(
        stem_id="stem_00",
        input_pcm_sha256=sha_in,
        is_silent=False,
        jitter_lag_samples=-0.25,
        jitter_corr=0.999,
        noise_floor_dbfs_broadband=-91.2,
        noise_floor_dbfs_lf=-93.0,
        noise_floor_dbfs_mf=-92.0,
        noise_floor_dbfs_hf=-95.0,
        band_energy_broadband=-18.5,
        band_energy_lf=-21.0,
        band_energy_mf=-20.0,
        band_energy_hf=-25.0,
        seg_rhs_f64=np.ones((2, 8), dtype=np.float64),
        seg_y2_f64=np.array([1.5, 2.5], dtype=np.float64),
        lti_ir_f32=np.linspace(1.0, 0.0, 128, dtype=np.float32),
    )
    arena4 = Arena4CalibrationData(inputs={sha_in: inp}, stems={"stem_00": st})
    raw = arena4.to_bytes()
    assert len(raw) % 64 == 0

    parsed = Arena4CalibrationData.from_bytes(raw)
    assert sha_in in parsed.inputs
    assert "stem_00" in parsed.stems
    p_in = parsed.inputs[sha_in]
    assert p_in.num_samples == 48_000
    assert p_in.win_bounds == ((0, 24_000), (24_000, 48_000))
    assert p_in.s1_proj_f32 is not None
    assert inp.s1_proj_f32 is not None
    np.testing.assert_allclose(p_in.s1_proj_f32, inp.s1_proj_f32)

    p_st = parsed.stems["stem_00"]
    assert p_st.jitter_lag_samples == pytest.approx(-0.25)
    assert p_st.noise_floor_dbfs_broadband == pytest.approx(-91.2)
    assert p_st.lti_ir_f32 is not None
    assert st.lti_ir_f32 is not None
    np.testing.assert_allclose(p_st.lti_ir_f32, st.lti_ir_f32)

    # Tamper body -> CRC32C mismatch
    bad = bytearray(raw)
    bad[80] ^= 0xFF
    with pytest.raises(ValueError, match="CRC32C mismatch"):
        Arena4CalibrationData.from_bytes(bytes(bad))


def test_ckpt_container_arena4_roundtrip_and_dnam_stripping() -> None:
    sha_in = "b2" * 32
    sha_tg = "c3" * 32
    arena4 = Arena4CalibrationData(
        inputs={
            sha_in: InputCalibStats(
                input_pcm_sha256=sha_in,
                num_samples=1024,
                win_bounds=((0, 1024),),
                win_grams_f64=np.eye(4, dtype=np.float64)[None, :, :],
            )
        },
        stems={
            "s0": StemCalibStats(
                stem_id="s0",
                input_pcm_sha256=sha_in,
                is_silent=False,
                jitter_lag_samples=0.0,
                jitter_corr=1.0,
                noise_floor_dbfs_broadband=-90.0,
                noise_floor_dbfs_lf=-92.0,
                noise_floor_dbfs_mf=-92.0,
                noise_floor_dbfs_hf=-95.0,
                band_energy_broadband=-20.0,
                band_energy_lf=-23.0,
                band_energy_mf=-22.0,
                band_energy_hf=-26.0,
            )
        },
    )
    phi = np.ones((64, 192), dtype=np.float32) * 0.01
    y = np.ones((64, 2), dtype=np.float32) * 0.01
    stat = compute_stem_sufficient_statistics(
        phi,
        y,
        stem_id="s0",
        controls_meta={},
        input_pcm_sha256=sha_in,
        target_pcm_sha256=sha_tg,
    )
    header = DnamJsonHeader(
        version="1.0.0",
        container_kind="dnam",
        architecture="DNAM_TriPhysics",
        sample_rate_train_hz=48_000,
        metadata=ModelMetadata(
            name="Arena4 Test",
            created_at="2026-10-31T00:00:00Z",
            provenance=DatasetProvenance(
                dataset_manifest_sha256="d4" * 32,
                config_recipe_sha256="e5" * 32,
                epochs_trained=1,
            ),
        ),
        io_config=IoConfig(
            inputs=2,
            output_mode="stereo",
            latency_samples=0,
            tail_time_ms=0.0,
            capabilities=ModelCapabilities(),
            leveling=LevelingConfig(),
        ),
        slimmable_tiers=SlimmableTiersConfig(
            lite=TierProfile(
                channels=4,
                fdn_lines=0,
                early_taps=0,
                head_svd_rank=1,
                loudness_lufs=-18.0,
                output_trim_db=0.0,
                validation_esr_db=-30.0,
                arena_end_byte_offset=ARENA_OFFSET_LITE_END,
            ),
            full=TierProfile(
                channels=8,
                fdn_lines=0,
                early_taps=0,
                head_svd_rank=1,
                loudness_lufs=-18.0,
                output_trim_db=0.0,
                validation_esr_db=-32.0,
                arena_end_byte_offset=ARENA_OFFSET_FULL_END,
            ),
            ultra=TierProfile(
                channels=12,
                fdn_lines=0,
                early_taps=0,
                head_svd_rank=1,
                loudness_lufs=-18.0,
                output_trim_db=0.0,
                validation_esr_db=-35.0,
                arena_end_byte_offset=ARENA_OFFSET_ULTRA_END,
            ),
        ),
        control_bus=ControlBusConfig(),
        core1_hybrid_ssm=Core1Config(),
        core2_mufdn=Core2Config(active=False, ring_buffer_pow2_log2=12),
    )
    layout = ArenaWeightLayout()
    dnam_c, ckpt_c = build_signed_dnam_containers(
        header,
        layout,
        [stat],
        creator_seed_hex="11" * 32,
        arena4_calib=arena4,
    )

    # .dnam must strip Arena 4
    dnam_rt = DnamContainer.from_bytes(dnam_c.to_bytes())
    assert dnam_rt.preamble.has_calib_stats is False
    assert dnam_rt.arena4_bytes is None

    # .dnam-ckpt must preserve Arena 4
    ckpt_rt = DnamContainer.from_bytes(ckpt_c.to_bytes())
    assert ckpt_rt.preamble.has_calib_stats is True
    assert ckpt_rt.arena4_bytes is not None
    parsed_a4 = Arena4CalibrationData.from_bytes(ckpt_rt.arena4_bytes)
    assert sha_in in parsed_a4.inputs
    assert "s0" in parsed_a4.stems


def test_sidecar_cache_store_lifecycle_and_mmap(tmp_path: Path) -> None:
    obs = _CacheEventCollector()
    cache_dir = tmp_path / ".dnam-cache"
    store = SidecarCacheStore(
        cache_dir,
        num_stems=2,
        max_gb=0.5,
        device_kind="cpu",
        observer=obs,
    )

    assert store.should_materialize(payload_bytes=1_000_000, expected_passes=3, est_regen_ms=50.0)
    assert not store.should_materialize(
        payload_bytes=1_000_000, expected_passes=1, est_regen_ms=50.0
    )

    # Put and get float32, bfloat16, int32 tensors
    t_f32 = torch.arange(256, dtype=torch.float32).reshape(64, 4)
    p_f32 = store.put_tensor("stage1_basis", ["sha_a", 0], t_f32, kind=KIND_STAGE_BASIS)
    assert p_f32 is not None
    assert p_f32.exists()
    assert p_f32.stat().st_size % 64 == 0

    loaded_full = store.get_tensor("stage1_basis", ["sha_a", 0])
    assert loaded_full is not None
    torch.testing.assert_close(loaded_full, t_f32)

    loaded_slice = store.get_tensor("stage1_basis", ["sha_a", 0], row_slice=(10, 30))
    assert loaded_slice is not None
    torch.testing.assert_close(loaded_slice, t_f32[10:30])

    t_bf16 = t_f32.to(torch.bfloat16)
    store.put_tensor("bf16_basis", ["sha_a", 1], t_bf16, kind=KIND_INPUT_ONLY)
    loaded_bf16 = store.get_tensor("bf16_basis", ["sha_a", 1])
    assert loaded_bf16 is not None
    assert loaded_bf16.dtype == torch.bfloat16
    torch.testing.assert_close(loaded_bf16, t_bf16)

    t_i32 = torch.arange(64, dtype=torch.int32).reshape(16, 4)
    store.put_tensor("i32_basis", ["sha_a", 2], t_i32, kind=KIND_STAGE_BASIS)
    loaded_i32 = store.get_tensor("i32_basis", ["sha_a", 2])
    assert loaded_i32 is not None
    assert loaded_i32.dtype == torch.int32
    torch.testing.assert_close(loaded_i32, t_i32)

    # Stage marker round-trip
    store.write_stage_marker("stage1", ["sha_a"], {"regimes": [[0, 1024]]})
    marker = store.read_stage_marker("stage1", ["sha_a"])
    assert marker == {"regimes": [[0, 1024]]}

    # Corrupt file recovery -> returns None, deletes file, emits corrupt CacheEvent
    corrupt_bytes = bytearray(p_f32.read_bytes())
    corrupt_bytes[70] ^= 0xFF
    p_f32.write_bytes(bytes(corrupt_bytes))
    assert store.get_tensor("stage1_basis", ["sha_a", 0]) is None
    assert not p_f32.exists()
    assert any(isinstance(e, CacheEvent) and e.action == "corrupt_ignored" for e in obs.events)

    # Cleanup stage bases removes KIND_STAGE_BASIS, retains KIND_INPUT_ONLY
    store.cleanup_stage_bases()
    assert store.get_tensor("i32_basis", ["sha_a", 2]) is None
    assert store.get_tensor("bf16_basis", ["sha_a", 1]) is not None

    # Cleanup on success removes markers, retains KIND_INPUT_ONLY
    store.cleanup_on_success()
    assert store.read_stage_marker("stage1", ["sha_a"]) is None
    assert store.get_tensor("bf16_basis", ["sha_a", 1]) is not None
    store.cleanup_on_failure()


def test_sidecar_cache_store_maintenance_eviction_and_error_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os
    import time

    # 1. Disabled cache_dir=None no-op paths
    disabled = SidecarCacheStore(None)
    assert disabled.put_tensor("ns", ["k"], torch.ones(8, dtype=torch.float32)) is None
    assert disabled.get_tensor("ns", ["k"]) is None
    assert disabled.write_stage_marker("s1", ["k"], {"a": 1}) is None
    assert disabled.read_stage_marker("s1", ["k"]) is None
    assert disabled.cleanup_stage_bases() == 0
    assert disabled.cleanup_on_success() == 0
    assert disabled.startup_maintenance() == 0
    assert not disabled.should_materialize(1000, 2, 100.0)

    # 2. Quota via DNAM_CACHE_MAX_GB env var
    monkeypatch.setenv("DNAM_CACHE_MAX_GB", "not_a_number")
    SidecarCacheStore(tmp_path / "env_invalid", num_stems=1, max_gb=0.1)
    monkeypatch.setenv("DNAM_CACHE_MAX_GB", "0.0001")  # ~104 KB
    cdir = tmp_path / "env_cache"
    store = SidecarCacheStore(cdir, num_stems=1)
    assert store.quota_bytes == int(0.0001 * (1024**3))
    assert not store.should_materialize(store.quota_bytes + 1024, 3, 500.0)
    monkeypatch.delenv("DNAM_CACHE_MAX_GB", raising=False)

    # 3. 1D tensor, fallback dtype (int16 -> float32), and 3D tensor rejection
    p1d = store.put_tensor("vec", ["1d"], torch.ones(32, dtype=torch.int16), pin=False)
    assert p1d is not None
    t1d = store.get_tensor("vec", ["1d"])
    assert t1d is not None
    assert t1d.dtype == torch.float32
    with pytest.raises(ValueError, match="1D or 2D"):
        store.put_tensor("bad3d", ["3d"], torch.ones((2, 2, 2), dtype=torch.float32))

    # 4. Eviction on space pressure (unpinned entry evicted when writing larger tensor)
    big_t = torch.ones((256, 128), dtype=torch.float32)  # 128 KB > 104 KB quota
    store.put_tensor("big", ["b1"], big_t, pin=False)
    assert store.get_tensor("big", ["b1"]) is not None
    assert store.cleanup_stage_bases(stage_prefix="nonexistent_prefix") == 0

    # 5. Startup maintenance: orphan .tmp, unaligned .dnce, bad magic .dnce, expired .dnce, LRU
    store.put_tensor("valid1", ["v1"], torch.ones((16, 4), dtype=torch.float32), pin=False)
    store.put_tensor("valid2", ["v2"], torch.ones((16, 4), dtype=torch.float32), pin=False)
    (cdir / "orphan.tmp").write_bytes(b"1234")
    (cdir / "unaligned.dnce").write_bytes(b"short")
    (cdir / "bad_magic.dnce").write_bytes(b"XXXX" + b"\x00" * 60)
    expired_p = store.put_tensor("exp", ["e1"], torch.ones((16, 4), dtype=torch.float32), pin=False)
    assert expired_p is not None
    old_time = time.time() - 10 * 86_400.0
    os.utime(expired_p, (old_time, old_time))
    store.quota_bytes = 64  # Force LRU eviction loop inside startup_maintenance
    evicted = store.startup_maintenance()
    assert evicted > 0
    assert not (cdir / "orphan.tmp").exists()
    assert not (cdir / "unaligned.dnce").exists()
    assert not (cdir / "bad_magic.dnce").exists()
    assert not expired_p.exists()

    # 6. Corrupt stage marker recovery
    m_path = store.write_stage_marker("s2", ["k"], {"ok": True})
    assert m_path is not None
    m_path.write_bytes(b"corrupt")
    assert store.read_stage_marker("s2", ["k"]) is None
    assert not m_path.exists()

    # 7. Disk floor failure when free space is below floor
    monkeypatch.setattr(SidecarCacheStore, "_free_disk_bytes", lambda _self: 1024)
    with pytest.raises(RuntimeError, match="Insufficient disk space"):
        SidecarCacheStore(tmp_path / "low_disk", num_stems=2)


def test_arena4_validation_errors() -> None:
    from dnam_trainer.training.calib_cache import ARENA4_HEADER_STRUCT, ARENA4_MAGIC, crc32c

    with pytest.raises(ValueError, match="Invalid Arena 4 buffer length"):
        Arena4CalibrationData.from_bytes(b"short")
    with pytest.raises(ValueError, match="Invalid Arena 4 magic"):
        Arena4CalibrationData.from_bytes(b"XXXX" + b"\x00" * 60)
    bad_ver = ARENA4_HEADER_STRUCT.pack(ARENA4_MAGIC, 99, 0, 0, 0, 64, 0, b"\x00" * 44)
    with pytest.raises(ValueError, match="Unsupported Arena 4 version"):
        Arena4CalibrationData.from_bytes(bad_ver)
    bad_len = ARENA4_HEADER_STRUCT.pack(ARENA4_MAGIC, 1, 0, 0, 0, 128, 0, b"\x00" * 44)
    with pytest.raises(ValueError, match="total_byte_len"):
        Arena4CalibrationData.from_bytes(bad_len)
    empty_crc = crc32c(b"")
    trunc_in = ARENA4_HEADER_STRUCT.pack(ARENA4_MAGIC, 1, 0, 1, 0, 64, empty_crc, b"\x00" * 44)
    with pytest.raises(ValueError, match="Truncated Arena 4 input entry header"):
        Arena4CalibrationData.from_bytes(trunc_in)
    trunc_st = ARENA4_HEADER_STRUCT.pack(ARENA4_MAGIC, 1, 0, 0, 1, 64, empty_crc, b"\x00" * 44)
    with pytest.raises(ValueError, match="Truncated Arena 4 stem entry header"):
        Arena4CalibrationData.from_bytes(trunc_st)

    from dnam_trainer.training.calib_cache import (
        ARENA4_INPUT_ENTRY_HEADER,
        ARENA4_INPUT_RECORD_MAGIC,
        ARENA4_STEM_ENTRY_HEADER,
        ARENA4_STEM_RECORD_MAGIC,
    )

    def _wrap_body(n_in: int, n_st: int, body: bytes) -> bytes:
        hdr = ARENA4_HEADER_STRUCT.pack(
            ARENA4_MAGIC, 1, 0, n_in, n_st, 64 + len(body), crc32c(body), b"\x00" * 44
        )
        return hdr + body

    bad_in_m = ARENA4_INPUT_ENTRY_HEADER.pack(
        b"XXXX", b"\x00" * 32, 0, 0, 0, 0, 0, b"\x00" * 8, 64, 0
    )
    with pytest.raises(ValueError, match="Invalid Arena 4 input record magic"):
        Arena4CalibrationData.from_bytes(_wrap_body(1, 0, bad_in_m))

    bad_in_l = ARENA4_INPUT_ENTRY_HEADER.pack(
        ARENA4_INPUT_RECORD_MAGIC, b"\x00" * 32, 0, 0, 0, 0, 0, b"\x00" * 8, 128, 0
    )
    with pytest.raises(ValueError, match="Truncated or unaligned Arena 4 input entry payload"):
        Arena4CalibrationData.from_bytes(_wrap_body(1, 0, bad_in_l))

    bad_in_c = ARENA4_INPUT_ENTRY_HEADER.pack(
        ARENA4_INPUT_RECORD_MAGIC, b"\x00" * 32, 0, 0, 0, 0, 0, b"\x00" * 8, 64, 12345
    )
    with pytest.raises(ValueError, match="Arena 4 input entry CRC32C mismatch"):
        Arena4CalibrationData.from_bytes(_wrap_body(1, 0, bad_in_c))

    bad_st_m = ARENA4_STEM_ENTRY_HEADER.pack(
        b"XXXX", b"\x00" * 32, 0, 0, 0, 0, *([0.0] * 10), b"\x00" * 32, 128, 0, b"\x00" * 4
    )
    with pytest.raises(ValueError, match="Invalid Arena 4 stem record magic"):
        Arena4CalibrationData.from_bytes(_wrap_body(0, 1, bad_st_m))

    bad_st_l = ARENA4_STEM_ENTRY_HEADER.pack(
        ARENA4_STEM_RECORD_MAGIC,
        b"\x00" * 32,
        0,
        0,
        0,
        0,
        *([0.0] * 10),
        b"\x00" * 32,
        256,
        0,
        b"\x00" * 4,
    )
    with pytest.raises(ValueError, match="Truncated or unaligned Arena 4 stem entry payload"):
        Arena4CalibrationData.from_bytes(_wrap_body(0, 1, bad_st_l))

    bad_st_c = ARENA4_STEM_ENTRY_HEADER.pack(
        ARENA4_STEM_RECORD_MAGIC,
        b"\x00" * 32,
        0,
        0,
        0,
        0,
        *([0.0] * 10),
        b"\x00" * 32,
        128,
        12345,
        b"\x00" * 4,
    )
    with pytest.raises(ValueError, match="Arena 4 stem entry CRC32C mismatch"):
        Arena4CalibrationData.from_bytes(_wrap_body(0, 1, bad_st_c))
