"""TORA Stage 0-IV, Control-Bus, and `TensorSliceEntry` configurations
(`ref[sdk.python.schema.rust-parity]`).
"""

from __future__ import annotations

from pydantic import Field, field_validator

from dnam.errors import ValidationError
from dnam.schema._base import StrictModel

DEFAULT_ALLPASS_DELAYS: tuple[int, ...] = (
    19,
    31,
    47,
    73,
    103,
    139,
    179,
    223,
    29,
    53,
    83,
    113,
    151,
    191,
    233,
    251,
)


class ControlBusConfig(StrictModel):
    """$3\\text{ kHz}$ Parametric Control-Bus configuration (`control_bus`)."""

    envelopes: int = Field(default=4, ge=0, le=16)
    lfo_oscillators: int = Field(default=2, ge=0, le=8)
    lfo_harmonics: int = Field(default=4, ge=0, le=16)
    fm_interaction_rank: int = Field(default=4, ge=0, le=16)
    modulation_bus_dim: int = Field(default=16, ge=1, le=64)
    head_trust_region_radius: float = Field(default=2.5, gt=0.0)


class RouterConfig(StrictModel):
    """Stage 0 8-Port Matrix Router configuration (`stage0_router`)."""

    input_ports: int = Field(default=4, ge=1, le=16)
    output_ports: int = Field(default=4, ge=1, le=16)
    sub_crossover_hz: float = Field(default=200.0, gt=0.0)


class Stage1PreFxConfig(StrictModel):
    """Stage I Pre-Dynamics and Tone Shaping configuration (`stage1_pre`)."""

    topology_selector: int = Field(default=0, ge=0, le=15)
    noise_gate_enabled: bool = True
    gate_threshold_db: float = -60.0


class Stage2CascadeConfig(StrictModel):
    """Stage II Commuted Nonlinear Cascade configuration (`stage2_cascade`)."""

    num_stages: int = Field(default=3, ge=1, le=8)
    adaa_mode: str = "1x_bspline_algebraic"
    ac_coupling_pole: float = Field(default=0.9993, gt=0.0, lt=1.0)


class Stage3CabinetKautzConfig(StrictModel):
    """
    Stage III Cabinet Early FIR + Orthonormal Kautz Modal Bank configuration (`stage3_cabinet`).
    """

    fir_taps: int = Field(default=16, ge=0, le=256)
    kautz_modals: int = Field(default=8, ge=0, le=128)


class Stage4SpatialTankConfig(StrictModel):
    """Stage IV Spatial Tank configuration (`stage4_tank`)."""

    delay_storage: str = "fp16"
    interp: str = "catmull_rom_cubic"
    tzf_reference_tap_samples: float = Field(default=64.0, ge=0.0)
    switch_xfade_samples: int = Field(default=128, ge=1, le=65535)
    allpass_delays_samples: list[int] = Field(default_factory=lambda: list(DEFAULT_ALLPASS_DELAYS))
    dc_blocker_pole: float = Field(default=0.995, gt=0.0, lt=1.0)
    loop_ceiling: float = Field(default=0.985, gt=0.0, lt=1.0)

    @field_validator("allpass_delays_samples", mode="before")
    @classmethod
    def _validate_allpass_delays(cls, value: object) -> list[int]:
        if not isinstance(value, (list, tuple)):
            raise ValidationError("allpass_delays_samples must be a sequence of 8 or 16 integers")
        items = [int(x) for x in value]
        if len(items) == 8:
            return items + list(DEFAULT_ALLPASS_DELAYS[8:])
        if len(items) == 16:
            return items
        raise ValidationError(
            f"allpass_delays_samples must contain 8 or 16 elements, got {len(items)}"
        )


class TensorSliceEntry(StrictModel):
    """Tensor directory entry describing a 64-byte aligned slice in `Arena 0..3` or `Arena 4..5`."""

    arena: int = Field(ge=0, le=15)
    dtype: str
    packing: str | None = None
    feature_dim: int | None = Field(default=None, ge=1)
    packed_len: int | None = Field(default=None, ge=1)
    shape: list[int] | None = None
    byte_offset: int = Field(ge=0)
    byte_length: int = Field(ge=0)
