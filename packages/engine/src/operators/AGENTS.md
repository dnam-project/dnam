# `src/operators` — TORA DSP Kernel Operators (`Op-T`, `Op-M`, `Op-N`)

Implements the three fundamental real-time DSP kernel operators of TORA (`#[forbid(unsafe_code)]`):
1. **Op-T (Transport & Elastic Buffer):** Fractional delay, Dual-Mode slew/crossfade, and diffusers.
2. **Op-M (Modal Bank & Kautz):** Symplectic recurrence, Newton rotor, and Kautz filter.
3. **Op-N (Commuted Cascade & PolyADAA):** Multi-drive `1×` B-spline `PolyADAA` and wavefolders.

## 1. Submodule Map

| Operator / File | Responsibility | Canonical Spec (SSOT) |
| :--- | :--- | :--- |
| [`op_t.rs`](op_t.rs) ([`delay_and_diffuser.rs`](op_t/delay_and_diffuser.rs), [`ring_and_guard.rs`](op_t/ring_and_guard.rs)) | Catmull-Rom fractional delay, ring buffer mask, and Schroeder diffusers | [`op_t_transport.md`](../../../../docs/specs/dsp/op_t_transport.md) |
| [`op_m.rs`](op_m.rs) ([`kautz.rs`](op_m/kautz.rs), [`hilbert.rs`](op_m/hilbert.rs), [`tracker.rs`](op_m/tracker.rs)) | 4-lane SIMD symplectic recurrence, Kautz modal bank, Hilbert & pitch tracker | [`op_m_modal_kautz.md`](../../../../docs/specs/dsp/op_m_modal_kautz.md) |
| [`op_n.rs`](op_n.rs) ([`polyadaa.rs`](op_n/polyadaa.rs), [`wavefolder_and_bitcrusher.rs`](op_n/wavefolder_and_bitcrusher.rs), [`commuted_and_ctrl.rs`](op_n/commuted_and_ctrl.rs)) | `PolyADAA`, resting DC equilibrium, wavefolder & bitcrusher kernels | [`op_n_commuted_cascade.md`](../../../../docs/specs/dsp/op_n_commuted_cascade.md) |

See [`packages/engine/CONTRIBUTING.md`](../../CONTRIBUTING.md) for SIMD vector register and zero-allocation invariants.
