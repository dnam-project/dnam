# `src/operators` — TORA DSP Kernel Operators (`Op-T`, `Op-M`, `Op-N`)

Implements the three fundamental real-time DSP kernel operators of TORA (`#[forbid(unsafe_code)]`):
1. **Op-T (Transport & Elastic Buffer):** Fractional delay, Dual-Mode slew/crossfade, and diffusers (`CL-06`–`CL-08`).
2. **Op-M (Modal Bank & Kautz):** Symplectic recurrence, Newton rotor, and Kautz filter (`CL-03`–`CL-05`).
3. **Op-N (Commuted Cascade & PolyADAA):** Multi-drive `1×` B-spline `PolyADAA` and wavefolders (`CL-01`, `CL-02`).

## 1. Submodule Map

| Operator / File | Responsibility | Canonical Spec (SSOT) |
| :--- | :--- | :--- |
| `op_t` | Catmull-Rom fractional delay, ring buffer mask, and Schroeder diffusers | [`op_t_transport.md`](../../../../docs/dsp/op_t_transport.md) |
| `op_m` | 4-lane SIMD symplectic recurrence, Newton unit rotor, and Kautz modal bank | [`op_m_modal_kautz.md`](../../../../docs/dsp/op_m_modal_kautz.md) |
| `op_n` | `PolyADAA`, resting DC equilibrium, and wavefolder kernels | [`op_n_commuted_cascade.md`](../../../../docs/dsp/op_n_commuted_cascade.md) |

See [`packages/engine/CONTRIBUTING.md`](../../CONTRIBUTING.md) for SIMD vector register and zero-allocation invariants.
