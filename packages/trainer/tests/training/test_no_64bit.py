"""Strict `TorchDispatchMode` guard verifying zero 64-bit (`float64`, `complex128`, `int64`)
PyTorch tensor allocations across the trainer numeric core, DSP modules, and selective cascade.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from dnam_trainer.control.basis import ParametricControlHyperBus
from dnam_trainer.dsp.modulated_ufdn import DifferentiableModulatedUfdn
from dnam_trainer.dsp.poly_adaa import DifferentiablePolyAdaa
from dnam_trainer.dsp.selective_rglru import DifferentiableSelectiveRgLru
from dnam_trainer.schema import parse_compact_range_spec, parse_compact_switch_spec
from dnam_trainer.training.metrics import (
    compute_band_split_esr_db_torch,
    torch_floor_regularized_besr_loss,
    torch_mr_stft_loss,
)
from dnam_trainer.training.numerics import ExactGramAccumulator, HostDenseSolver
from dnam_trainer.training.selective_cascade import calibrate_3stage_selective_cascade
from dnam_trainer.training.surrogate_delay import surrogate_delay_readout
from dnam_trainer.training.windowing import (
    CausalBsplineWindowSchedule,
    build_uniform_windows,
    compute_cumulative_energy_state,
    compute_segmentation_window_stats,
)

FORBIDDEN_DTYPES = (torch.float64, torch.complex128, torch.int64)


class _No64BitTensorGuard(TorchDispatchMode):
    """Raises AssertionError if any PyTorch op produces a `float64`, `complex128`, or `int64`
    tensor, except for internal `stft` / `fft` index helpers on CPU if any.
    """

    def __torch_dispatch__(
        self,
        func: Any,
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        op_name = str(func)
        # Check output tensors
        if isinstance(out, torch.Tensor):
            if out.dtype in FORBIDDEN_DTYPES:
                raise AssertionError(
                    f"Forbidden 64-bit tensor dtype {out.dtype} produced by {op_name}"
                )
        elif isinstance(out, (tuple, list)):
            for item in out:
                if isinstance(item, torch.Tensor) and item.dtype in FORBIDDEN_DTYPES:
                    raise AssertionError(
                        f"Forbidden 64-bit tensor dtype {item.dtype} in tuple produced by {op_name}"
                    )
        return out


def test_zero_64bit_tensors_in_numerics_windowing_dsp_and_cascade() -> None:
    rng = np.random.default_rng(77)
    x_np = (0.25 * rng.standard_normal(512)).astype(np.float32)
    y_np = np.tanh(2.0 * x_np).astype(np.float32)
    y_2ch = np.column_stack([y_np, y_np])

    ctrls = [
        parse_compact_range_spec("drive=0..10:5"),
        parse_compact_switch_spec("voice=clean,hot:clean"),
    ]
    bus = ParametricControlHyperBus(ctrls)
    adaa = DifferentiablePolyAdaa(num_channels=4)
    rglru = DifferentiableSelectiveRgLru(num_channels=4)
    ufdn = DifferentiableModulatedUfdn(num_lines=4)

    with _No64BitTensorGuard():
        # 1. ExactGramAccumulator & HostDenseSolver
        x_t = torch.from_numpy(x_np).unsqueeze(-1).repeat(1, 8)
        y_t = torch.from_numpy(y_2ch)
        acc = ExactGramAccumulator(8, 2, device="cpu")
        acc.add(x_t, y_t)
        solver = HostDenseSolver()
        w = solver.solve_single(acc)
        assert w.dtype == torch.float32

        # 2. Windowing & Segmentation
        e_cum = compute_cumulative_energy_state(torch.from_numpy(x_np), sr=48_000.0)
        wb = build_uniform_windows(512, win_len=256)
        sched = CausalBsplineWindowSchedule(e_cum, wb, overlap=32)
        dense_w = sched.to_dense()
        assert dense_w.dtype == torch.float32
        _seg = compute_segmentation_window_stats(torch.from_numpy(x_np), y_t, wb, sr=48_000.0)

        # 3. Control Bus & DSP modules
        psi = bus.encode_control_coords({"drive": 7.0, "voice": "hot"})
        assert psi.dtype == torch.float32
        u4 = torch.from_numpy(x_np[:64]).view(1, 1, 64).repeat(1, 4, 1)
        z_adaa, _ = adaa(u4)
        z_rglru, _ = rglru(z_adaa)
        u2 = torch.from_numpy(y_2ch[:64]).transpose(0, 1).unsqueeze(0)
        z_ufdn = ufdn(u2)
        tau_t = torch.tensor([[12.5]], dtype=torch.float32)
        z_del = surrogate_delay_readout(u2[:, 0, :], tau_t, sigma_grad=1.5)
        assert z_adaa.dtype == torch.float32
        assert z_rglru.dtype == torch.float32
        assert z_ufdn.dtype == torch.float32
        assert z_del.dtype == torch.float32

        # 4. Torch Losses
        pred_3d = torch.from_numpy(y_2ch).unsqueeze(0)
        tgt_3d = pred_3d * 0.9
        l_besr = torch_floor_regularized_besr_loss(pred_3d, tgt_3d)
        l_stft = torch_mr_stft_loss(pred_3d, tgt_3d, fft_sizes=(64, 128))
        _esr_lf, _esr_mf, _esr_hf, _res = compute_band_split_esr_db_torch(
            pred_3d.squeeze(0), tgt_3d.squeeze(0)
        )
        for loss_t in (l_besr, l_stft):
            assert loss_t.dtype == torch.float32

        # 5. Full 3-Stage Selective Cascade Calibration
        b_l, b_f, b_u = calibrate_3stage_selective_cascade(
            x_np,
            [y_2ch],
            sample_rate_hz=48_000,
            device="cpu",
            solver=solver,
        )
        assert b_l.dtype == np.float32
        assert b_f.dtype == np.float32
        assert b_u.dtype == np.float32
