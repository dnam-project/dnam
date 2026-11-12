---
id: ADR-0017
title: "100% Pure-Rust 5-Stage Analytical Variable-Projection (VarPro) Solver, Planar Stage Decoupling & fp16 Coreset Storage"
status: accepted
date: "2026-03-30"
supersedes: null
superseded_by: null
specs:
  - ../specs/training/four_stage_pipeline.md
  - ../specs/training/iterative_resume_and_caching.md
  - ../specs/containers/project_container_format.md
requirements:
  - REQ-1.5
  - REQ-3.1
  - REQ-3.2
  - REQ-3.3
  - REQ-4.1
  - REQ-4.2
  - REQ-4.3
---

# ADR-0017: 100% Pure-Rust 5-Stage Analytical Variable-Projection (`VarPro`) Solver, Planar Stage Decoupling & `fp16` Coreset Storage

## 1. Context & Problem Statement

Maintaining a dual-language training stack (`Python`/`PyTorch` in `packages/trainer` + `PyO3` bindings in `packages/python` alongside `#![no_std]` Rust in `packages/engine`) and iterative first-order SGD introduced three structural bottlenecks (`REQ-1.5`, `REQ-3.1`–`REQ-3.3`, `REQ-4.1`–`REQ-4.3`):

1. **Cross-Language Operator Drift, Browser Incompatibility & Gauge/Watermark Interference (`Theorem 1`):** Duplicating `Op-T`, `Op-M`, `Op-N`, and `3 kHz` control-bus recurrences in Python/PyTorch risked IEEE-754 rounding divergence from `dnam-engine`, required a `1.5+ GB` Python/CUDA/MPS runtime distribution, prevented zero-GPU execution inside a browser `WebWorker` (`REQ-4.1`), and required strict byte-disjoint ordering between Mechanism A gauge transforms and Mechanism B spread-spectrum watermarking.
2. **Slow First-Order SGD & Naive Multi-Tier Prefix Degradation (`Theorem 2`, `Theorem 5`):** Backpropagating through `48 kHz` IIR recurrences via `AdamW` required hundreds of epochs (`45–180 s` on GPU) to converge on coupled nonlinear drive (`Stage II`), modal poles (`Stage III`), and spatial feedback (`Stage IV`), while naive independent per-tier head solves violated strict monotonic prefix improvement across `Lite` $\subset$ `Full` $\subset$ `Pro` $\subset$ `Ultra` (`REQ-4.1`, `REQ-5.4`, `REQ-5.6`).
3. **Unbounded Trajectory Cache Footprint (`Theorem 6`):** Storing full `f32` `256-D` feature trajectories across multi-stem captures exceeded `65–150 MB` per stem during iterative capture vault expansion, violating the `< 256 MiB` host and `< 64 MiB` browser worker memory ceilings (`REQ-4.3`).

---

## 2. Considered Candidates & Failure Modes

| Candidate | Operator, Gauge & Browser Parity | Interactive Solve Time | Per-Stem Coreset Capsule (`INPC`/`STMC`) | Failure Mode |
| :--- | :---: | :---: | :---: | :--- |
| **A. Python / PyTorch `AdamW` + `PyO3` FFI (`v3.3` Baseline)** | Dual-implementation drift; cannot run in browser `WASM` | `45–180 s` (GPU required) | `65.5 MB` (`f32` full trajectory) | Requires Python/`uv`/PyTorch toolchain; slow first-order gradient descent on stiff poles; violates `REQ-4.1` and `REQ-4.3` |
| **B. Rust First-Order AD (`burn` / `candle` Backprop) + Naive Per-Tier Solves** | Single Rust workspace, but heavy AD graph & sub-optimal tier stripping | `30–90 s` | `32.8 MB` (`fp16` full trajectory) | Ill-conditioned first-order gradients across commuted nonlinear cascades; naive tier truncation degrades stripped tiers; exceeds `< 64 MiB` worker ceiling |
| **C. Pure-Rust 5-Stage (`Stage A`–`Stage E`) Analytical `VarPro` + `Theorem 1/2/5/6` Schur/MM-KKT + `24 KB / 48 KB` `fp16` Coreset (Selected)** | **Bit-exact single `dnam-engine` crate (Native + `WASM`); `0.000000 ULP` Mechanism A gauge (`Theorem 1`)** | **`< 25 ms` typical (`< 100 ms` ceiling, zero GPU; `< 5 ms` Tier-1 resume)** | **`24,000 B` (`C_out=1`) / `48,000 B` (`C_out=2`) `fp16` (`12,000` frames)** | Zero identified failure modes; closed-form `256-D` `uplo='U'` Schur-complement Cholesky (`Theorem 2`), Loewner/SK `VarPro`, and `EMR`-weighted `MM-Cholesky` + `33`-band `LTAS` KKT (`Theorem 5`) eliminate first-order SGD |

---

## 3. Decision & Mathematical Formulation

We eliminate Python/PyTorch (`packages/python`) and adopt a **100% Pure-Rust (`#![forbid(unsafe_code)]`) 5-Stage (`Stage A`–`Stage E`) Analytical Variable-Projection (`VarPro`) & Schur-Complement Cholesky Solver** in `packages/capture` (`dnam-capture`, see [`ADR-0020`](0020-modular-monorepo-and-dnam-proj-container.md)) linking `dnam-engine` directly (`ref[arch.isomorphism.two-sweep]`, `ref[arch.isomorphism.audit-wiring]`):

1. **`Theorem 1` — Intrinsic `0.000000 ULP` Mechanism A Reciprocal-Gauge Invariance via `128 B` 8-Vec SoA Shared-Pole Kautz, Disjoint `SO(4) ⊕ SO(4)` Head/Stage Gauge, Layer 1+2 Watermarking & Mechanism A $\to$ B Ordering (`ref[sec.watermark.reciprocal-gauge]`, `ref[sec.watermark.orthogonal-gauge]`, `ref[sec.watermark.tardos-code]`, `ref[container.dnam.stage3-slices]`):**
   - Each 4-pair shared-pole Kautz block in `stage3_kautz_modals` (`STAGE3_KAUTZ_BASES = [4_416, 17_536, 31_680, 60_160]`, `STAGE3_KAUTZ_BLOCKS_PER_TIER = [1, 1, 2, 4]`) uses [`ADR-0011`](0011-end-to-end-stage-wiring-and-arena-slice-topology.md)'s `128 B` 8-Vec SoA layout `[r_low, dr, theta, b_gamma, c_re_l, c_im_l, c_re_r, c_im_r]`. Because `apply_isometric_gauge_watermark` scales $b_{\gamma,m}^{(\text{buyer})} = \eta_m b_{\gamma,m}^{(\text{master})}$ (`+48..+64`) and all four stereo readout vectors $c_{a,m}^{(\text{buyer})} = c_{a,m}^{(\text{master})} / \eta_m$ (`+64..+128`, $a \in \lbrace L, R \rbrace$), the bilinear factor $\eta_m \cdot \eta_m^{-1} \equiv 1$ cancels **inside Stage III** before entering the `256-D` Feature Bus (`0.000000 ULP` invariance across all 256 feature slots and `cos_sim == 1.0` on `extract_structural_hash_224d` with `1e-2 rad` pole-angle grid quantization `quantize_pole_angle_1e2`).
   - Enforces a **Disjoint `SO(4) ⊕ SO(4)` Head/Stage Gauge** across `CP1` contributors (`cp1_lanes_dir` at `Arena 0 [7,524..7,552]`): right-multiplying head rows `0..4` of `main_base`/`U_int^base` by $\mathbf{Q}_{\text{buyer}}^{\text{base}} \in SO(4)$ (and `main_delta`/`U_int,Delta` by $\mathbf{Q}_{\text{buyer}}^{\text{Ultra}} \in SO(4)$) while left-multiplying `W_head` Modes `1..8`, `b_head`, and Columns `0..4` of `W_ctrl^base` by $\mathbf{Q}_{\text{buyer}}^\top$, leaving the disjoint stage rows $\mathbf{z}_{\text{stage}}$ (`Columns 4..8` of `W_ctrl^base` and `W_ctrl,Delta^Ultra`) untouched.
   - Splits `stage2_cascade_w` into **Role A** (`b = 0`, 16 `f32` inter-stage `4×4` coupling matrix) and **Role B** (`b = 1..K_t-1`, `Vec4` readout blocks folded surjectively via $s = 16c + g = (4r + q) \bmod 32$ onto all `32` `Vec4` head slices per tier), pairs **Layer 1 Cascade-Head Gauge** (`apply_cascade_head_gauge`, exact `0.000000 ULP` dyadic invariance across Role B blocks and all 32 `Vec4` head slices) with **Layer 2 Pole-Angle Phase Coding** (`apply_phase_coding_watermark`, $\lvert\Delta\theta_m\rvert \le 8.726646 \times 10^{-4}\text{ rad} = 0.05^\circ$ via `"DNAM-WM-PHASE-V1"` on Stage III Kautz pole angles `STAGE3_KAUTZ_BASES` and Stage II inter-stage pole angles `4,180`/`4,212`, `0.00 dB` pole-radius change), and applies **Mechanism A first, then Mechanism B second** (`apply_tardos_watermark_7680` with `Role A = 0`, per-tier HMAC-SHA256 Arcsine bias $p_i$, and `EMR`-masked envelopes via `compute_emr_masked_tardos_envelopes` on `stage2_cascade_w` at `1,024..4,096`, `11,264..17,408`, `22,336..31,552`, `36,224..48,512`).

2. **`Theorem 2` — `256-D` Feature Bus Intra-Tier `64-D` Schur-Complement Shared Re-Balancing Inside Each Tier's `2,560 B` (`[5, 2, 64]`) Slice (`ref[container.dnam.tensor-directory]`):**
   - In `CANONICAL-256D-FEATURE-BUS-MAP`, each tier $t \in \lbrace 0, 1, 2, 3 \rbrace$ (`Lite`, `Full`, `Pro`, `Ultra`) owns a `64`-feature vector $\boldsymbol{\phi}_t[n] = [\boldsymbol{\phi}_{\text{s3},t}[n], \boldsymbol{\phi}_{\text{shared},t}[n]]^\top \in \mathbb{R}^{64}$ (`64t..64(t+1)`) and a `2,560 B` (`[5, 2, 64]`) head slice $\mathbf{w}_t = [\mathbf{w}_{\text{s3}}^{(t)}, \Delta\mathbf{w}_{\text{shared}}^{(t)}]^\top \in \mathbb{R}^{64}$ matching [`ADR-0011`](0011-end-to-end-stage-wiring-and-arena-slice-topology.md) (`HEAD_BASES = [4_928, 18_048, 31_936, 57_600]`).
   - At Tier $t = 0$ (`Lite`), solves $(\mathbf{G}_{0,0} + \boldsymbol{\Lambda}_0)\mathbf{w}_0 = \mathbf{c}_0^{(y)}$; at Tiers $t \in \lbrace 1, 2, 3 \rbrace$, solves the exact **intra-tier `64-D` Schur-complement update** $(\mathbf{G}_{\text{s3},t} + \lambda_t\mathbf{I} - \mathbf{G}_{\text{s3},\text{shared},t}(\mathbf{G}_{\text{shared},t} + \lambda_{\text{shared}}\mathbf{I})^{-1}\mathbf{G}_{\text{shared},\text{s3},t})\mathbf{w}_{\text{s3}}^{(t)} = \mathbf{c}_{\text{s3},t}^{(r_{t-1})}$ and $\Delta\mathbf{w}_{\text{shared}}^{(t)} = -(\mathbf{G}_{\text{shared},t} + \lambda_{\text{shared}}\mathbf{I})^{-1}\mathbf{G}_{\text{shared},\text{s3},t}\mathbf{w}_{\text{s3}}^{(t)}$ wholly inside Tier $t$'s own `2,560 B` (`[5, 2, 64]`) slice, guaranteeing strict monotonic prefix improvement ($\text{bESR}_{\text{Ultra}} < \text{bESR}_{\text{Pro}} < \text{bESR}_{\text{Full}} < \text{bESR}_{\text{Lite}}$) and exact optimality under `REQ-5.6` tier stripping.

3. **`Theorem 5` — Simultaneous Psychoacoustic `EMR`-Weighted Selective `1/12`-Octave `MM-Cholesky`, `33`-Band `LTAS` KKT & Pre-Baked $\mathbf{c}_{s,\text{eff}}^{\text{MM-KKT}}$ in `STAT` (`ref[train.pipeline.stage-c-schur]`, `ref[qual.companion.ltas]`):**
   - Weights spectral residuals by the inverse simultaneous masking threshold $M_b^{-1/2}$ (`EMR`) and runs selective `1/12`-octave Majorization-Minimization (`MM-Cholesky`), keeping phase lock $\alpha(\omega) = 1.0$ wherever band error $\Delta L_b \le 0.25\text{ dB}$ and activating $\alpha(\omega) \in [0.5, 1.0)$ only in high-frequency bands above `1.5 kHz` where $\Delta L_b > 0.25\text{ dB}$.
   - Applies a **`33`-band `1/3`-octave `LTAS` dual Newton-KKT projection** to obtain $\mathbf{w}_s^{\text{MM-KKT}}$ ($\lvert\Delta\text{EQ}_{\text{dB}}(b)\rvert < 0.05\text{ dB}$) and pre-bakes $\mathbf{c}_{s,\text{eff}}^{\text{MM-KKT}} \triangleq (\mathbf{G}_s + \boldsymbol{\Lambda})\mathbf{w}_s^{\text{MM-KKT}}$ into `STAT`'s `C_audio_tier_f64` (`4,096 B`), enabling `< 5 ms` Fast-Path vault re-solve and `< 5 ms` live null-test auto-patching.

4. **`Theorem 6` — Rank-1 Static Kautz Zero Contraction ($\mathbf{P}_c$) in `STAT` (`uplo='U'`), `[STAB-ZERO-DC]` (`b_head = 0`), Stereo-Aware `24,000 B` / `48,000 B` `fp16` `INPC`/`STMC` Coresets & the 5 Solver Stages (`Stage A`–`Stage E`) (`ref[container.proj.preamble-and-directory]`, `ref[container.proj.stat-uplo-u]`, `ref[container.proj.stmc-capsule]`, `ref[train.pipeline.stage-a-poles]`):**
   - **`Theorem 6` (`STAT` Rank-1 Kautz Contraction, `[STAB-ZERO-DC]` & `INPC`/`STMC` `fp16` Coresets):** Locks unit readout zeros $(c_{\text{re},L,m}, c_{\text{im},L,m}, c_{\text{re},R,m}, c_{\text{im},R,m})$ inside each `128 B` 8-Vec SoA Kautz block and contracts modal state columns via $\tilde{\mathbf{G}}_s = \mathbf{P}_c^\top \mathbf{G}_s \mathbf{P}_c$ into the `256-D` `uplo='U'` `STAT` Gram matrix (`263,168 B` + `4,096 B` `C_audio_tier_f64`); hard-zeroes `b_head = 0` (`9,024..9,064` and `63,424..63,456`) to enforce `[STAB-ZERO-DC]` (`REQ-5.3`); and persists stereo-aware **`24,000 B` (`C_out = 1` mono) / `48,000 B` (`C_out = 2` stereo) `fp16` coreset slices** (`12,000` frames = `250 ms` at `48 kHz` inside `64 B`-aligned `INPC` `24,640 B`/`48,640 B` and `STMC` `28,800 B`/`52,800 B` capsules) for **`< 25 ms` Full Re-Pool without WAVs**.
   - **The 5 Analytical & `VarPro` Solver Stages (`Stage A`–`Stage E`):**
     - **`Stage A` (Transport, Pedestal-Sideband Ratio & Shared-Pole Kautz Lifting):** Sub-sample alignment $\hat{\tau}$, `Theorem 4` forward pedestal-sideband ratio, and Loewner / Sanathanan-Koerner (`SK`) + `2–3`-step `VarPro` Gauss-Newton (`VP-GN`) shared-pole Kautz lifting ($r_{\text{low},m} \in [0.85, 0.9995]$) into `128 B` 8-Vec SoA Stage III blocks.
     - **`Stage B` (Synchronized Stereo Feature Harvest, `MM-Cholesky` & `33`-Band `LTAS` KKT):** Steps `DnamEngine::step_frame_with_feature_bus` (`0.000000 ULP` parity) across stereo `Stage I`/`II`/`III`/`IV` to accumulate `256-D` `uplo='U'` `STAT` statistics and solves `Theorem 5` (`EMR`-weighted `1/12`-octave `MM-Cholesky` + `33`-band `LTAS` KKT + pre-baked $\mathbf{c}_{s,\text{eff}}^{\text{MM-KKT}}$).
     - **`Stage C` (Multi-Tier `64-D` Schur-Complement Re-Balancing & Reciprocal Gauge):** Applies `Theorem 1` intrinsic `128 B` 8-Vec SoA reciprocal gauge and `Theorem 2` intra-tier `64-D` Schur-complement prefix re-balancing inside each tier's `2,560 B` (`[5, 2, 64]`) slice across `Lite` $\to$ `Full` $\to$ `Pro` $\to$ `Ultra`.
     - **`Stage D` (`CP1` Arbitrary-$P$ Control Projections, Hadamard FM & 3-Phase Audit):** Solves `Theorem 3` node-anchored `eval_lobatto_basis`, native `Switch` `default_val` zero-baseline option contributors, `2` Hadamard diagonals (`ExactIntFactorAccumulator`), and `16 × 16` + `16 × 4` `W_ctrl` stage dispatch on disjoint `z_stage` (`Columns 4..8`), followed by the non-stop 3-Phase Control-Space Audit.
     - **`Stage E` (Rank-1 Kautz Contraction, `INPC`/`STMC` `fp16` Coreset, Watermarking & `v2.0` Emission):** Executes `Theorem 6` Rank-1 Kautz contraction ($\mathbf{P}_c$), `[STAB-ZERO-DC]` (`b_head = 0`), `INPC`/`STMC` `24,000 B` / `48,000 B` `fp16` coreset packing, Mechanism A $\to$ Mechanism B watermarking, and `64 B`-aligned canonical `v2.0` `.dnam` / `.dnam-proj` emission.

---

## 4. Verification & References

- **Canonical Specifications:** [`../specs/training/four_stage_pipeline.md`](../specs/training/four_stage_pipeline.md), [`../specs/training/iterative_resume_and_caching.md`](../specs/training/iterative_resume_and_caching.md), [`../specs/containers/project_container_format.md`](../specs/containers/project_container_format.md)
- **Requirements:** [`../requirements/system_goals_and_requirements.md`](../requirements/system_goals_and_requirements.md) (`REQ-1.5`, `REQ-3.1`–`REQ-3.3`, `REQ-4.1`–`REQ-4.3`)

