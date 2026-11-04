"""Example 5: Vectorized TORA Reference DSP Kernels (`Op-T`, `Op-M`, `Op-N`, Control Bus & Mixing)
(`examples/reference_dsp_operators.py`).

Run with:
  uv run --no-sync python packages/python/examples/reference_dsp_operators.py
"""

from __future__ import annotations

import math

import numpy as np

from dnam import dsp


def main() -> None:
    """Demonstrates vectorized reference DSP kernels across `dnam.dsp`."""
    t = np.linspace(0.0, 1.0, 512, endpoint=False, dtype=np.float64)
    signal = 0.6 * np.sin(2.0 * np.pi * 5.0 * t)

    # 1. Op-N: Division-free algebraic PolyADAA slice scan
    adaa_out, u_prev, a_prev, m_dc = dsp.poly_adaa_slice(
        signal, b_conv=0.05, alpha=0.3, beta=0.9, delta=0.1, reset=True
    )
    print(
        f"Op-N PolyADAA     | peak={float(np.max(np.abs(adaa_out))):.6f} "
        f"state=({u_prev:.4f}, {a_prev:.4f}, {m_dc:.4f})"
    )

    # 2. Op-M: Symplectic modal resonator & contractive unit-circle rotor
    modal_out, z_re, z_im = dsp.modal_slice(adaa_out, r_low=0.96, dr_span=0.03, reset=True)
    rotors = dsp.unit_rotor(np.linspace(-np.pi * 0.9, np.pi * 0.9, 64, dtype=np.float64))
    assert isinstance(rotors, np.ndarray)
    max_norm = float(np.max(np.hypot(rotors[0], rotors[1])))
    print(
        f"Op-M Modal & Rotor| peak={float(np.max(np.abs(modal_out))):.6f} "
        f"state=({z_re:.4f}, {z_im:.4f}) max_rotor_norm={max_norm:.8f}"
    )

    # 3. Op-T: Coercive-saturating Lyapunov tank loop & Catmull-Rom cubic Hermite interpolation
    tank_out = dsp.tank_slice(modal_out, g_lin=0.88, sigma_knee=1.0, v_ceil=4.0)
    interp = dsp.catmull_rom_hermite(-0.2, 0.1, 0.5, 0.3, 0.35)
    print(
        f"Op-T Tank & Interp| peak={float(np.max(np.abs(tank_out))):.6f} "
        f"catmull_rom(mu=0.35)={float(interp):.6f}"
    )

    # 4. Control Bus: Lobatto basis, Rank-4 FM, Chebyshev LFO, Trust-Region Soft-Clipper
    u_grid = np.linspace(-1.0, 1.0, 128, dtype=np.float64)
    lobatto = dsp.lobatto_basis(u_grid, anchors=(0.0, -1.0, 1.0), degree=3)
    lfo_waves = dsp.chebyshev_lfo(np.cos(u_grid))
    clipped = dsp.trust_region_softclip(u_grid * 5.0, mh=2.5)
    fm_vec = dsp.rank4_fm(np.ones(8, dtype=np.float64), np.full((8, 4), 0.25, dtype=np.float64))
    assert isinstance(lobatto, np.ndarray)
    assert isinstance(lfo_waves, np.ndarray)
    assert isinstance(clipped, np.ndarray)
    print(
        f"Control Bus       | lobatto={lobatto.shape} lfo={lfo_waves.shape} "
        f"softclip_max={float(np.max(np.abs(clipped))):.4f} fm0={float(fm_vec[0]):.4f}"
    )

    # 5. Mixing: Cubic Lagrange weights & 8ch/16ch Orthogonal Givens-Householder mixing
    w_lag = dsp.cubic_lagrange_weights(0.4)
    assert isinstance(w_lag, tuple)
    x8 = np.linspace(-1.0, 1.0, 8, dtype=np.float64)
    y8 = dsp.givens_householder_8ch(x8, cos_psi=math.cos(0.4), sin_psi=math.sin(0.4))
    print(
        f"Mixing            | lagrange_sum={sum(w_lag):.12f} "
        f"energy_in={float(np.sum(x8 * x8)):.6f} energy_out={float(np.sum(y8 * y8)):.6f}"
    )


if __name__ == "__main__":
    main()
