# `dnam_trainer/dsp/` — Differentiable PyTorch DSP & Reference Bridge (`CL-01`–`CL-11`)

**`dnam_trainer.dsp`** implements differentiable PyTorch `nn.Module` layers for Op-N (`PolyADAA`), Op-M (`SelectiveRgLru`), and Stage IV (`ModulatedUfdn`) with exact `.to_reference_dsp()` parity against `dnam_engine.dsp`.

## 1. Module Routing Map

| Module | Enforced Clamps | Key Exports & Optimizations | Canonical Spec (SSOT) |
| :--- | :--- | :--- | :--- |
| [`poly_adaa.py`](poly_adaa.py) | `CL-01`, `CL-02` | `poly_adaa_algebraic_torch`, `resting_dc_equilibrium`, `DifferentiablePolyAdaa` (`1×` B-spline `(0.875, 0.125)`, chunked 5 Hz AC scan `R_ac = 0.9993`) | [`op_n_commuted_cascade.md`](../../../../../docs/dsp/op_n_commuted_cascade.md) |
| [`selective_rglru.py`](selective_rglru.py) | `CL-03`, `CL-04`, `CL-05` | `log_octave_pole_init`, `algebraic_unit_rotor_torch`, `parallel_associative_scan_complex` (Two-Level Chunked Scan `L_chunk = 256`), `DifferentiableSelectiveRgLru` | [`op_m_modal_kautz.md`](../../../../../docs/dsp/op_m_modal_kautz.md) |
| [`modulated_ufdn.py`](modulated_ufdn.py) | `CL-06`–`CL-11` | `cubic_lagrange_read_torch`, `dual_pointer_switch_crossfade_torch`, `givens_householder_matrix_torch`, `coercive_lyapunov_loop_step_torch`, `DifferentiableModulatedUfdn` | [`op_t_transport.md`](../../../../../docs/dsp/op_t_transport.md) |

See [`packages/trainer/CONTRIBUTING.md`](../../../CONTRIBUTING.md) for mixed-precision (`torch.autocast`) rules.
