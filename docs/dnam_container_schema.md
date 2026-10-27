# `.dnam` & `.dnam-ckpt` Hybrid Binary Container Schema, Zero-Copy Hardware Loader & Continuous-Time ZOH Rescaling

Free from legacy NAM A2 text-JSON weight array constraints and Python `pickle` (`torch.save`) checkpoints, DNAM defines two companion **zero-copy, 64-byte cache-line and SIMD-aligned Hybrid Binary Container formats** engineered for **$< 50\text{ ns}$ `#[no_std]` hardware loading**:

1. **`.dnam` (Magic `b"DNAM"`, Version `1.0`):** The single-file real-time deployment container ($10.5\text{ KB}$ `Lite` / $21.8\text{ KB}$ `Full` / $35.4\text{ KB}$ `Ultra` binary arena) loaded via zero-copy `mmap` or `Float32Array` view by the Rust SIMD engine (`packages/engine`), audio plugin (`packages/plugin`), Python wheel (`packages/python`), and Node.js/WASM runtime (`packages/js`).
2. **`.dnam-ckpt` (Magic `b"DNCK"`, Version `1.0`):** The deduplicated, `pickle`-free single-file iterative training checkpoint. Its leading tensor arenas (`Arena 0..2`) are **bit-for-bit identical** to the exported `.dnam` model, followed by `Arena 3` containing the tier-ordered **Column-Major Upper-Triangular (`uplo='U'`) `float64` audio sufficient statistics** $\lbrace \mathrm{pack}_{\text{U}}\big(\mathbf{G}_{\text{audio}}^{(s)}\big), \mathbf{C}_{\text{audio}}^{(s)} \rbrace$ (**$17.25\text{ KB/stem}$ `Lite` / $66.50\text{ KB/stem}$ `Full` / $147.75\text{ KB/stem}$ `Ultra`**—nearly $2\times$ smaller than full matrix storage and 100% 64-byte aligned).
3. **`.dnam-seal` (Magic `b"DNSL"`, Version `1.0`):** The 100% opaque, immutable 64-byte cache-line aligned OEM hardware wrapper (`DnamSealEnvelope`, `DnamProtectionProfile::EnclaveSealed`) that wraps an inner `ENCRYPTED_ARENA = 1` `.dnam` container (`cek_arena`) inside an outer `ChaCha20` + `X25519` `HPKE` binary envelope (`cek_outer`) for factory-attested hardware TrustZone enclaves (`OemDeviceCertificate`, see [`digital_marketplace_and_provenance.md` §3](digital_marketplace_and_provenance.md#3-pillar-2-hardware-enclave--oem-drm-oemdevicecertificate-attestation--dnam-seal-bdnsl-wrapper)).

---

## 1. Hybrid Binary Container Wire Framing & `< 50 ns` Zero-JSON Hardware Fast-Path

Both `.dnam` and `.dnam-ckpt` share an identical three-region binary layout designed so **UI/host threads** can inspect human-readable JSON metadata in $< 0.05\text{ ms}$, while **real-time embedded DSP threads (`#[no_std]` Rust on Cortex-A76)** skip the JSON header entirely and cast the 64-byte aligned binary arena directly to `#[repr(C, align(64))]` structs in **$< 50\text{ ns}$ with zero heap allocations and zero string parsing**:

```text
+-----------------------------------------------------------------------+ 0x0000 (Byte 0)
| 1. Fixed 64-Byte Binary Preamble (#[repr(C, align(64))], Little-End.) |
|    • Magic: "DNAM" (0x4D414E44) or "DNCK" (0x4B434E44)                |
|    • Version (u16=1, u16=0), Flags (u16), P (u8), RingLog2 (u8)       |
|    • Train SR (u32) | JSON Byte Len (u32) | Arena Offset & Len (u32)  |
|    • Tier Prefix End Offsets & CRC32Cs (Lite/Full/Ultra) | Tail (f32) |
+-----------------------------------------------------------------------+ 0x0040 (Byte 64, 64B-aligned)
| 2. UTF-8 JSON Schema, Essential Metadata, Presets & Tensor Directory  | <- Read by UI / Host / Python only;
|    • Essential `metadata`, open `user_metadata`, `controls_schema`    |    skipped completely by #[no_std]
|    • Built-in `presets`, `ui_hints` & prefix `arena_crc32c`           |    real-time DSP engine!
|    • Space-padded (0x20) to 4 KB OS page boundary (in-place slack!)   |
+=======================================================================+ tensor_arena_offset (4096B page / 64B-aligned)
| 3A. Arena 0: Lite Base Tier (C=4, M=4, 10,752 B) [f32, align(64)]     | <- Lite host maps only up to here
|     • Binary ControlClampDescriptor[P] + Execution-Order Core1 Blocks |
+-----------------------------------------------------------------------+ arena_offset_lite_end (64B-aligned)
| 3B. Arena 1: Full Delta Tier (+4 ch, +4 FDN, +11,584 B) [align(64)]   | <- Full host maps only up to here
+-----------------------------------------------------------------------+ arena_offset_full_end (64B-aligned)
| 3C. Arena 2: Ultra Delta Tier (+4 ch, +0 FDN, +13,888 B) [align(64)]  | <- .dnam EOF / Ultra host ends here
+=======================================================================+ arena_offset_ultra_end (64B-aligned)
| 3D. Arena 3 (.dnam-ckpt only): Per-Stem Packed Upper-Triangular f64   |
|     Sufficient Statistics {pack_U(G_audio^(s)), C_audio^(s)}          |
|     • 17,664 B/stem (Lite) | 68,096 B/stem (Full) | 151,296 B (Ultra) |
+-----------------------------------------------------------------------+ EOF (64B-aligned)
```

### 1.1 Fixed 64-Byte Binary Preamble Layout (`#[repr(C, align(64))]`)

All fields are naturally aligned with **zero compiler-inserted padding holes** (`sizeof(DnamPreamble) == 64`, `alignof(DnamPreamble) == 64`) in **Little-Endian** byte order (native to ARM64 NEON, x86_64 AVX2, and WebAssembly SIMD128), making the 64-byte preamble **100% self-sufficient for `#[no_std]` hardware verification, scratch allocation, and execution without parsing a single byte of JSON**:

| Byte Offset | Type | Field Name | Description & Fail-Fast Invariant |
| :--- | :--- | :--- | :--- |
| `0x00..0x04` | `[u8; 4]` | `magic` | `b"DNAM"` (`0x4D414E44`) for `.dnam`; `b"DNCK"` (`0x4B434E44`) for `.dnam-ckpt` |
| `0x04..0x06` | `u16` | `version_major` | Schema major version (`1`); reject if `!= 1` |
| `0x06..0x08` | `u16` | `version_minor` | Schema minor version (`0`) |
| `0x08..0x0A` | `u16` | `flags` | **Auto-populated by trainer/exporter (except Bit `8`):** Bit `0`: `HAS_LITE` (`1`), Bit `1`: `HAS_FULL_DELTA` (`2`), Bit `2`: `HAS_ULTRA_DELTA` (`4`), Bit `3`: `HAS_CKPT_STEMS` (`8`), Bit `4`: `EXEC_ORDER_SOA` (`16`), Bit `5`: `DEFAULT_ANCHORED` (`32`), Bit `6`: `CORE2_ACTIVE` (`64`, auto-pruned if `0`), Bit `7`: `LFO_ENV_ACTIVE` (`128`, auto-pruned if `0`), Bit `8`: `STEREO_OUT` (`256`, the **only** user-configurable mode flag: `1 = stereo`, `0 = mono`), Bit `9`: `HAS_SIGNATURES` (`512`, auto-set when `metadata.signatures` is present), Bit `10`: `ENCRYPTED_ARENA` (`1024`, auto-set when `Arena 0..2` is `ChaCha20`-wrapped for `authorized_devices`) |
| `0x0A..0x0B` | `u8` | `num_controls` | Total user controls `P = P_range + P_switch ∈ [0, 255]` (`ControlClampDescriptor` count) |
| `0x0B..0x0C` | `u8` | `ring_buffer_pow2_log2` | Auto-populated by trainer: `0` when `CORE2_ACTIVE == 0`, or `⌈log₂(max_m D_m + ΔD_max + 8)⌉ ∈ [8, 15]` when Core 2 is active |
| `0x0C..0x10` | `u32` | `sample_rate_train_hz` | Reference training sample rate in Hz (e.g., `48000`) |
| `0x10..0x14` | `u32` | `json_header_byte_len` | Unpadded UTF-8 byte length of the JSON metadata object starting at fixed offset `0x40` (`64`) |
| `0x14..0x18` | `u32` | `tensor_arena_offset` | File byte offset of `Arena 0` (`% 64 == 0`, `≥ 64 + L_json`; defaults to `4,096 B` OS page boundary `0x1000` with `0x20` space slack for zero-TLB-split `mmap` and in-place preset/metadata edits) |
| `0x18..0x1C` | `u32` | `tensor_arena_byte_len` | Total byte length of all binary tensor arenas (`% 64 == 0`) |
| `0x1C..0x20` | `u32` | `arena_offset_lite_end` | Arena-relative byte end of `Arena 0 (Lite)` (`10,752`, `% 64 == 0`) |
| `0x20..0x24` | `u32` | `arena_offset_full_end` | Arena-relative byte end of `Arena 0 + Arena 1 (Full)` (`22,336`, `% 64 == 0`) |
| `0x24..0x28` | `u32` | `arena_offset_ultra_end` | Arena-relative byte end of `Arena 0..2 (Ultra)` (`36,224`, `% 64 == 0`; equals `tensor_arena_byte_len` in `.dnam`) |
| `0x28..0x2C` | `u32` | `arena_crc32c_lite` | Castagnoli `CRC32C` (`0x1EDC6F41`) over `Arena 0` (`arena[0..arena_offset_lite_end]`) |
| `0x2C..0x30` | `u32` | `arena_crc32c_full` | Castagnoli `CRC32C` over `Arena 0..1` (`arena[0..arena_offset_full_end]`) |
| `0x30..0x34` | `u32` | `arena_crc32c_ultra` | Castagnoli `CRC32C` over `Arena 0..2` (`arena[0..arena_offset_ultra_end]`) |
| `0x34..0x36` | `u16` | `num_ckpt_stems` | `0` for `.dnam`; `S ≥ 1` for `.dnam-ckpt` (number of stem records in `Arena 3`) |
| `0x36..0x37` | `u8` | `control_decimation` | Control-bus decimation stride `K_ctrl` at `f_{s,train}` (default `16`) |
| `0x37..0x38` | `u8` | `head_svd_rank` | Active SVD principal control subspace rank `D_h ∈ [0, 4]` (`CL-18`) |
| `0x38..0x3C` | `f32` | `tail_time_ms` | Auto-computed `T₆₀` decay tail time in milliseconds for DAW host auto-sleep |
| `0x3C..0x40` | `u32` | `header_crc32c` | Castagnoli `CRC32C` (`0x1EDC6F41`) over `bytes[0x00..tensor_arena_offset]` with `bytes[0x3C..0x40]` zeroed |

> [!TIP]
> **Zero-JSON Prefix-Verifiable `CRC32C` & 4 KB Page-Aligned In-Place Preset Saving:**
> 1. **100% Zero-JSON `#[no_std]` Integrity:** Storing `arena_crc32c_lite` (`0x28`), `arena_crc32c_full` (`0x2C`), and `arena_crc32c_ultra` (`0x30`) directly inside the 64-byte preamble allows an embedded `#[no_std]` `Lite`-only (`10.5 KB`) or `Full`-only (`21.8 KB`) loader—or a browser HTTP `Range` request—to verify hardware Castagnoli `CRC32C` (`__crc32cd` on ARMv8-A / SSE4.2) over **only the prefix bytes it loaded**, without parsing a single byte of JSON!
> 2. **4 KB OS Page Alignment & `< 0.1 ms` In-Place Preset Updates:** By aligning `tensor_arena_offset` (`0x14`) to a `4,096 B` (`0x1000`) OS virtual-memory page boundary and padding `[64 + json_header_byte_len .. tensor_arena_offset]` with `0x20` ASCII spaces, `Arena 0` maps onto an exact hardware page boundary (zero TLB split), while reserving $\sim 1.2\text{–}1.8\text{ KB}$ of slack so DAW plugins and platforms can add/edit `presets` and `user_metadata` **100% in-place** inside the first `4 KB` page without moving `Arena 0..3` or invalidating `arena_crc32c_*`.

---

## 2. Five Hardware-Loading Optimizations (`< 50 ns` Zero-Allocation Load)

### 2.1 Eliminating 33.33% Structural Zeros via Contiguous Tier-Delta Arenas

In a naive dense $C_{\text{max}} \times C_{\text{max}}$ ($12 \times 12$) lower-block-triangular convolution kernel, `Lite` (`0..3`) never reads channels `4..11` ($4 \times 8 = 32$ zeros/tap) and `Full` (`4..7`) never reads channels `8..11` ($4 \times 4 = 16$ zeros/tap)—wasting **48 / 144 = 33.33% of all convolution weights on structural zeros** and forcing strided cache-line jumps when executing `Lite` or `Full`.

DNAM eliminates 100% of structural zeros by partitioning the causal convolution matrix $\mathbf{W}_{\text{conv}}^{(\ell)} \in \mathbb{R}^{K \times 12 \times 12}$ ($K = 6$) and all layer parameters into **three contiguous, 64-byte aligned Tier-Delta Arenas**:

$$\mathbf{W}_{\text{conv}}^{(\ell)} = \Big[\text{Row}_{\text{Lite}}: \big(\mathbf{W}_{\text{Lite}}^{(\ell)} \in \mathbb{R}^{K \times 4 \times 4}, \mathbf{0}_{K \times 4 \times 8}\big) \Big| \text{Row}_{\text{Full}}: \big(\mathbf{W}_{\text{Full},\Delta}^{(\ell)} \in \mathbb{R}^{K \times 4 \times 8}, \mathbf{0}_{K \times 4 \times 4}\big) \Big| \text{Row}_{\text{Ultra}}: \big(\mathbf{W}_{\text{Ultra},\Delta}^{(\ell)} \in \mathbb{R}^{K \times 4 \times 12}\big)\Big]^\top$$

### 2.2 Execution-Order Layer Interleaving (`Core1BlockSlice`) for 100% Linear L1D Prefetching

Instead of storing all 8 `conv` layers in one array, all 8 `polyadaa` layers in a second array, and all 4 `rglru` layers in a third array (which forces the CPU to track 3 separate memory streams per block), each Tier Arena interleaves Core 1 weights into **4 contiguous `Core1BlockSlice` structs in exact $48\text{ kHz}$ forward execution order**:

$$\text{Core1BlockSlice}_b = \Big[\underbrace{\mathbf{W}_{\text{conv}}^{(2b)} \big| \text{PolyADAA}_{\text{SoA}}^{(2b)}}_{\text{Dilated Conv Layer } 2b} \Big| \underbrace{\mathbf{W}_{\text{conv}}^{(2b+1)} \big| \text{PolyADAA}_{\text{SoA}}^{(2b+1)}}_{\text{Dilated Conv Layer } 2b+1} \Big| \underbrace{\text{SelectiveRgLru}_{\text{SoA}}^{(b)}}_{\text{Complex RG-LRU } b}\Big], \quad b \in \lbrace 0, 1, 2, 3 \rbrace$$

During the $48\text{ kHz}$ audio loop, Core 1 walks a **single monotonic pointer (`ptr += 64`)** from the first byte of Block 0 to the last byte of Block 3—achieving a **100% sequential L1D hardware prefetcher hit rate**:

| Tier Arena | Channels / FDN Lines | `Core1BlockSlice` Size (per block × 4 blocks) | Total Arena Slice Size | Runtime Execution & Cache Behavior |
| :--- | :--- | :---: | :---: | :--- |
| **`Arena 0: Lite`** | `ch 0..3`, `FDN 0..3` | `1,024 B × 4` = **`4,096 B`** (`16 CL/blk`) | **`10,752 B` (`10.5 KB`)** | **100% self-contained `Lite` prefix** (`168 × 64 B` cache lines). |
| **`Arena 1: Full Delta`** | `ch 4..7`, `FDN 4..7` | `1,792 B × 4` = **`7,168 B`** (`28 CL/blk`) | **`+11,584 B` (`22,336 B` end)** | Appended to `Arena 0` to form **`Full`** (`21.8 KB`); zero `Ultra` bytes enter L1D cache. |
| **`Arena 2: Ultra Delta`** | `ch 8..11`, `FDN —` | `2,560 B × 4` = **`10,240 B`** (`40 CL/blk`) | **`+13,888 B` (`36,224 B` end)** | Appended to `Arena 0..1` to form **`Ultra`** (`35.4 KB`); maps 1:1 to Pass 2 of **`8 + 4` NEON loop**! |

### 2.3 Pre-Packed SIMD Structure-of-Arrays (`SoA`), Division-Free `ControlClampDescriptor` & Pop-Free 3-State DC Equilibrium

1. **Zero-JSON Binary Header (`tier_gain_trim_lin`) & 64-Byte `ControlClampDescriptor` (`0 FDIV`, `0` String Parsing at $3\text{ kHz}$):**
   - `Arena 0` begins with `core1_input_and_head_bias` (`128 B = 2 × 64 B` cache lines): Cache Line 0 holds the `4×4` stereo-to-4-channel input projection (`[x_mid, x_side, 0, 0] -> ch 0..3`); Cache Line 1 holds the stereo SVD head bias $\mathbf{b}_{\text{head}} \in \mathbb{R}^{(1+D_h) \times 2}$ (`10 × f32`), the **pre-computed linear loudness trims** `tier_gain_trim_lin: [f32; 3]` ($10^{\text{output-trim-db}/20}$ for `[Lite, Full, Ultra]`, enabling $0.00\text{ dB}$ LUFS tier switching in `#[no_std]` with `0 pow10f` calls), and `[mix_dry_gain, mix_wet_gain, r_ac_train]` (`3 × f32`).
   - `control_bus_and_clamp_table` begins with the binary `ControlClampDescriptor` array ($64\text{ B} = 1\text{ cache line}$ per control $p \in \lbrace 0 \dots P-1 \rbrace$), packing normalization and up to **3 binary switch-coupled conditional clamp rules** (`[CLAMP-STABILITY]`, `[CLAMP-DYNAMIC]`) into a single 64-byte L1D cache line:
     - `0x00..0x04`: `ctrl_kind: u8` (`0=range, 1=switch`), `lobatto_degree: u8` ($K_p \in \lbrace 0..3 \rbrace$), `active_options_mask: u16` (`[CLAMP-SWITCH]`).
     - `0x04..0x18`: `default_val: f32`, `safe_min_base: f32`, `safe_max_base: f32`, `scale_norm: f32` ($\frac{2}{u_{\text{decl-max}} - u_{\text{decl-min}}}$), `offset_norm: f32` ($-1 - u_{\text{decl-min}}\text{scale-norm}$).
     - `0x18..0x28`: `anchor_nodes_norm: [f32; 4]` ($[u_{p,1}, u_{p,2}, u_{p,3}, 0.0]$ in $[-1, +1]$).
     - `0x28..0x40`: `cond_rules: [ConditionalClampRule; 3]` (`3 × 8 B = 24 B`: `[switch_ctrl_idx: u8, switch_opt_idx: u8, rule_op: u8 (0=none, 1=max_ceil, 2=min_floor), _pad: u8, bound_val: f32]`).
     - Runtime normalization (`CL-12`) and conditional clamping execute in **branchless scalar/SIMD instructions with `0 FDIV` and `0` string parsing**:
       $$u_p = \mathrm{fma}\big(x_{p,\text{clamped}}, \text{scale-norm}_p, \text{offset-norm}_p\big)$$
2. **Exact 3-State `PolyADAA` DC Resting Equilibrium (`[b_conv_c, alpha_c, beta_c, delta_c]`, `64 B`/Layer) for Pop-Free Preset Switching:**
   - Each 4-channel `PolyADAA` `SoA` block (`[b_conv_c (4×f32), alpha_c (4×f32), beta_c (4×f32), delta_c (4×f32)]`, `16 floats = 64 B`) stores the per-channel convolution/grid operating-point bias $b_{\text{conv},c}$ alongside $(\alpha_c, \beta_c, \delta_c)$.
   - At $48\text{ kHz}$, `vld1q_f32(b_conv_c)` initializes the 6-tap convolution accumulator directly (`0` extra instructions). On model load or preset change, `reset_to_dc_equilibrium()` sets $u_{\text{rest},c} = \mathrm{clamp}(\frac{2}{3} b_{\text{conv},c}, -1.0, +1.0)$ and initializes **all three `PolyADAA` states** to their analytical silence fixed point:
     $$u_c[-1] = u_{\text{rest},c}, \qquad \mathcal{A}_c[-1] = g_{\text{nl},c}(u_{\text{rest},c}), \qquad m_{\text{dc},c}[-1] = g_{\text{nl},c}(u_{\text{rest},c})$$
     eliminating both the 2-sample B-spline step transient and the $30\text{ ms}$ AC-coupling settling transient (**exact `0.000000` DC output from Sample 0, Sample 1, and all subsequent samples**).
3. **Pre-Baked $f_{s,\text{train}}$ ($48\text{ kHz}$) Discrete Anchors in `SelectiveRgLru` (`128 B`) & Pre-Inverted Reciprocals in `core2_mufdn_soa` (`256 B`):**
   - `SelectiveRgLru` `SoA` (`32 floats = 128 B`) stores `[r_low_train, dr_span_train, theta0_rad_train, b_gamma_norm_train, w_sel, c_real, c_imag, u_rot_init_cos_sin]` where $\Delta r_c = r_{\text{high},c} - r_{\text{low},c}$ and $b_{\gamma,c} = b_c \sqrt{1 - \bar{r}_c^2}$.
   - `core2_mufdn_soa` (`64 floats = 256 B` per 4 FDN lines at `6784..7040` in `Arena 0` and `20544..20800` in `Arena 1`) pre-stores `inv_sigma_knee = 1.0 / sigma_m` (`CL-10`), `two_over_3v_ceil = 2.0 / (3.0 * V_ceil_m)` (`CL-11`), `d_ap_samples_train`, `[psi_givens_train, r_dc_train, tau_ref_samples_train, switch_xfade_samples_train]`, and at relative byte offsets `208..256` (`48 B`) the learned **`WT-UFDN` grain & freeze weights** (`grain_len_tr[4]`, `grain_warp_rate[4]`, and `[grain_window_mix, grain_spray_samples, grain_feedback_mix, head_tap_inject_gain]` in `Arena 0` / `[freeze_alpha_default, onset_sync_thresh, 0.0, 0.0]` in `Arena 1`)—while `control_bus_and_clamp_table` (`7040..10752` in `Arena 0`) stores the `8 × 32` (`1,024 B`) $3\text{ kHz}$ state-space projection matrix `W_ctrl_core2` at `9216..10240`—so the $48\text{ kHz}$ Core 2 loop executes **zero `FDIV` instructions** and **loading at $f_{s,\text{host}} == f_{s,\text{train}}$ requires $0$ flops**.
4. **Default-Anchored Base Weights ($\mathbf{W}_{\text{head},0}, \boldsymbol{\theta}_0$) for Instant Sample-0 Audio:**
   - Because each control's first Newton/Lobatto node $u_{p,1}$ is anchored at its `default` setting ($u_{p,1} = u_{p,\text{default}} \implies \tilde{\boldsymbol{\psi}}_p(u_{p,\text{default}}) \equiv \mathbf{0}$), the stored base head $\mathbf{W}_{\text{head},0}$ and base physical parameters $\boldsymbol{\theta}_0$ are **already the exact collapsed weights for the default preset at Sample 0**.
5. **100% Automatic Subsystem Pruning, Dynamic Ring-Buffer Sizing & DAW Tail Reporting (`Zero User Config`):**
   - **Only 1 User-Configurable Mode Flag (`output_mode`: `"stereo"` | `"mono"`):** The user only ever selects `stereo` or `mono` output (`STEREO_OUT` bit `8`; when `0`, the engine evaluates only channel 0 of the output head, saving 50% of head MACs on mono pedalboard blocks).
   - **Automatic `CORE2_ACTIVE` & `LFO_ENV_ACTIVE` Bypass:** At export, if Stage-3 Group-Lasso ($L_{2,1}$) pruning zeroes out Core 2 (`M-UFDN`) on a pure amp/preamp/drive/pickup model, the trainer automatically clears `CORE2_ACTIVE` (`bit 6 = 0`) and sets `ring_buffer_pow2_log2 = 0`—dropping `Full` compute from $\sim 4,625$ to **$\sim 3,150\text{ MACs/sample}$ ($3.8\times$ faster than NAM A2)** and Core 2 RAM from **$208\text{ KB} \to 0\text{ KB}$** with zero user configuration.
   - **Automatic `ring_buffer_pow2_log2` & `tail_time_ms`:** When Core 2 is active, the trainer automatically sets `ring_buffer_pow2_log2` from the Phase-3 `[CLAMP-DYNAMIC]` worst-case delay excursion (e.g., `10` $\to 1,024$ samples = `16 KB` for Chorus/Flanger vs. `15` $\to 32,768$ samples = `208 KB` for long delay) and computes `tail_time_ms` ($T_{60}$) and `latency_samples: 0` for VST3/CLAP host sleep/PDC management.

---

## 3. Essential Metadata, Provenance, Presets, `ui_hints` & Open `user_metadata`

DNAM keeps the core `metadata` schema strictly to the **minimal essentials**, includes first-class **`presets`** and intrinsic **`ui_hints`** for parametric controls, and leaves storefront/platform taxonomies or host/rig bindings (e.g., licensed marketplace categories, MIDI CC / OSC mappings, or hardware encoder slots) to the host or an open `user_metadata: dict[str, JsonValue]` dictionary:

1. **Minimal Standardized `metadata`:**
   - **Identity:** `name` (`str`), `author` (`str | null`), `description` (`str | null`), `license` (`str | null`), and `created_at` (`str`, ISO-8601 UTC timestamp).
   - **Audio Calibration:** `input_reference_dbfs` (`float`, e.g., `-20.5`), paired with per-tier `loudness_lufs` (`float`), `output_trim_db` (`float`), `validation_esr_db` (`float`), and `prefix_crc32c` (`u32`) inside `slimmable_tiers` (`lite`, `full`, `ultra`) so switching tiers at runtime maintains exact $0.00\text{ dB}$ loudness parity.
   - **Platform-Agnostic Content-Addressable `provenance`:**
     - `dataset_manifest_sha256` (`str`): SHA-256 over the sorted `(stem_id, input_pcm_sha256, target_pcm_sha256, controls_meta)` tuples, where `input_pcm_sha256` and `target_pcm_sha256` are computed over raw little-endian IEEE-754 `float32` PCM frames (independent of WAV/FLAC headers or OS paths).
     - `config_recipe_sha256` (`str | null`): Optional SHA-256 of the training configuration.
     - Enables self-contained $< 0.05\text{ ms}$ cache-freshness checks (with no sidecar `manifest.json`) and **zero-cost stem deduplication / in-place stem replacement** during `dnam-train --resume model.dnam-ckpt --add-stems ...`.
   - **Optional Split-Payload `Ed25519` `signatures` (`creator_signature` & `marketplace_receipt`):**
     - Supports offline-verifiable, hardware-agnostic creator attribution and commercial marketplace licensing (see [§7](#7-hardware-agnostic-marketplace-security-split-payload-ed25519-signatures--isometric-gauge-space-traitor-tracing)), while excluding `presets` and `user_metadata` from the signed `immutable_header_sha256` digest so buyers can save custom DAW presets in-place without invalidating either signature.
2. **Intrinsic Per-Control `ui_hints` (Auto-Layout, Widget Archetypes & Midpoint-Anchored Skew):**
   Each control in `controls_schema` includes an optional `ui_hints` object describing **only the intrinsic visual and interaction properties of the modeled control** (keeping host/rig bindings like MIDI CCs, OSC paths, or hardware encoder slots out of the container):
   - **Section & Layout Ordering:**
     - `section` (`str | null`, e.g., `"Preamp"`, `"Echo"`): Logical module grouping so plugins, `DnamPluginRack.tsx`, and multi-page hardware screens auto-group related controls cleanly.
     - `order` (`int`, default `0`): Sort index within the section.
     - `size` (`"sm"` | `"md"` | `"lg"`, default `"md"`): Relative visual prominence (e.g., `"lg"` for a primary `Drive` or `Time` knob, `"sm"` for a secondary trimpot).
     - `color_hex` (`str | null`, e.g., `"#F59E0B"`): Optional accent color for the control arc, cap, or switch indicator.
   - **`range` Control Interaction (`widget`, `polarity`, `midpoint`, `detents`, `step`, `decimals`, `scale_labels`):**
     - `widget`: `"knob"` (default rotary pot) | `"slider"` (linear fader / graphic EQ band) | `"trim"` (compact trimpot).
     - `polarity`: `"unipolar"` (fills from minimum to value, e.g., `Drive`) | `"bipolar"` (fills from center/zero detent, e.g., active EQ `-12..+12 dB` or `Pan`).
     - `midpoint` (`float | null`): Physical parameter value at 50% ($12\text{ o'clock}$) UI gesture position $t = 0.5$. Unlike logarithmic tapers that break when $x_{\min} \le 0$, anchoring $x(0.5) = x_{\text{mid}} \in (x_{\min}, x_{\max})$ defines a smooth, closed-form invertible power-skew curve across any $[x_{\min}, x_{\max}]$:
       $$\gamma_{\text{skew}} = \frac{\ln(0.5)}{\ln\left(\frac{x_{\text{mid}} - x_{\min}}{x_{\max} - x_{\min}}\right)}, \qquad x(t) = x_{\min} + (x_{\max} - x_{\min}) t^{1 / \gamma_{\text{skew}}}, \qquad t(x) = \left(\frac{x - x_{\min}}{x_{\max} - x_{\min}}\right)^{\gamma_{\text{skew}}}$$
       *(When `midpoint` is `null` or equals $\frac{1}{2}(x_{\min} + x_{\max})$, $\gamma_{\text{skew}} = 1.0$ and the mapping is linear.)*
     - `detents` (`list[float]`, default `[]`): Physical click/snap positions (e.g., `[0.0]` on a bipolar EQ knob or `[50.0]` on a Wet/Dry mix knob).
     - `step` (`float | null`, default `null` for continuous) & `decimals` (`int`, default `1`): UI drag increment and numeric readout precision.
     - `scale_labels` (`dict[str, str]`, default `{}`): Optional human-readable landmark labels at specific values (e.g., `{"0.0": "Dry", "50.0": "50/50", "100.0": "Wet"}`).
   - **`switch` Control Interaction (`widget`, `option_labels`):**
     - `widget`: `"toggle"` (2-way switch) | `"segmented"` (inline pill selector) | `"rotary_select"` (stepped rotary selector) | `"footswitch"` (stomp switch).
     - `option_labels` (`dict[str, str]`, default `{}`): Maps machine option IDs in `declared_options` to human-readable UI labels (e.g., `{"off": "Bypass", "on": "Vintage +6 dB", "ultra": "Hi-Gain +12 dB"}`).
3. **Per-Control Physical Calibration (`controls_schema[*].calibration`, Signed by `immutable_header_sha256`):**
   - `role`: `"delay_time_ms" | "lfo_rate_hz" | "feedback" | "mix" | "crossover_hz" | "stereo_mode" | "stereo_width" | "freeze_latch" | "grain_size_ms" | "grain_pitch_semitones" | "grain_spray"`.
   - `lobatto_physical_anchors`: Optional `[v_min, v_mid, v_max]` physical values at $u \in \lbrace -1, 0, +1 \rbrace$ for single-range controls.
   - `conditional_physical_anchors`: Map of `"switch_id=option_id" -> [v_min, v_mid, v_max]` for multi-range controls (e.g., Darkglass Anagram Delay's `Time (0..10)` paired with `Range ("0.01-0.1s", "0.1-1s", "1-2s")`).
   - `tempo_sync_default_subdivision`: One of the **12 canonical `TempoSubdivision` variants** from `1/2` to `1/16` with straight, dotted (`.`), and triplet (`T`) for each (`"1/2."`, `"1/2"`, `"1/2T"`, `"1/4."`, `"1/4"`, `"1/4T"`, `"1/8."`, `"1/8"`, `"1/8T"`, `"1/16."`, `"1/16"`, `"1/16T"`).
4. **Model-Level Capabilities & Auto Unity Leveling (`io_config.capabilities` & `io_config.leveling`, Signed by `immutable_header_sha256`):**
   - `io_config.capabilities`:
     - `blend_mode`: `"none" | "host_parallel" | "modeled_control"`.
     - `default_blend`: `float` in `[0.0, 1.0]` (default `1.0`).
     - `wet_polarity_inverted`: `bool` (when `true`, `DnamEngine::configure_host_mix_and_leveling` inverts `wet_g *= -1.0` with linear voltage mixing `dry = 1 - b, wet = b` so `0%` blend matches bypass polarity and `50%` blend sums constructively without phase cancellation).
     - `supports_trails`: `bool` (backed by `DnamEngine::set_bypass(bypassed, trails)` spillover execution).
     - `tempo_sync_capable`: `bool`.
     - `supports_freeze`: `bool` (backed by `DnamEngine::set_freeze(amount)` and `ControlCalibrationRole::FreezeLatch`).
     - `freeze_mutes_dry`: `bool` (`false` = default Sound-on-Sound mode where live dry + `Core 1` soloing continues over the frozen `Core 2` tank; `true` = mutes dry/Core-1 output when frozen).
   - `io_config.leveling`:
     - Anchored **strictly at the default control settings (`controls_schema[*].default`, normalized $\mathbf{u} = \mathbf{0}$ / $\mathbf{z}_{\text{head}} = \mathbf{0}$)**—never `presets[0]`.
     - `auto_unity_enabled`: `bool` (default `true`).
     - `leveling_metric`: `"rms_dbfs" | "k_weighted_lufs"` (defaults to gated unweighted broadband `"rms_dbfs"` for electrical `Vrms` pedal-to-pedal gain staging).
     - `reference_input_level_db`, `measured_default_output_level_db`, `unity_gain_offset_db`, `target_gain_db`, and `calibrated_trim_db`.
5. **Built-In `presets` Array (Parametric Control Snapshots):**
   - Stores named snapshots of `controls_schema` settings (`id`: `str`, `name`: `str`, `description`: `str | null`, `controls`: `dict[str, float | str]`, `user_metadata`: `dict[str, JsonValue]`) directly inside the `.dnam` header.
   - Every preset's `controls` map is validated against `controls_schema` (`safe_clamped_range`, `active_options`, and `conditional_max`) and preserved across `dnam-train --resume` runs.
6. **Open `user_metadata: dict[str, JsonValue]`:**
   - Top-level `user_metadata` (plus optional per-control `controls_schema[*].user_metadata`, per-preset `presets[*].user_metadata`, and per-stem `stems_registry[*].user_metadata`) accepts any valid JSON object.
   - Automatically preserved and merged across `dnam-train --resume` runs, and editable in $< 1\text{ ms}$ without retraining by updating only the JSON header at `0x40` and recomputing `header_crc32c`.

---

## 4. Continuous-Time Zero-Order Hold (ZOH) Sample-Rate Adaptability ($44.1\text{–}96\text{ kHz}$)

Because `.dnam` stores pre-baked discrete anchors at $f_{s,\text{train}}$ (e.g., $48,000\text{ Hz}$), **loading at $f_{s,\text{host}} == f_{s,\text{train}}$ requires $0$ flops**. When loaded by a DAW or host at a non-native sample rate $f_{s,\text{host}} \in [44,100, 96,000]\text{ Hz}$ ($\rho_{\text{sr}} = \frac{f_{s,\text{train}}}{f_{s,\text{host}}} \ne 1.0$), `set_sample_rate()` rescales the anchored parameters in closed form:

1. **Core 1B Complex Pole Radii & Angles (with `CL-03` & `CL-04` Safety Guards):**
   $$r_c(f_{s,\text{host}}) = \mathrm{clamp}\Big(\big(r_c(f_{s,\text{train}})\big)^{\rho_{\text{sr}}}, 0.25, 0.9995\Big), \qquad \theta_c(f_{s,\text{host}}) = \mathrm{clamp}\Big(\rho_{\text{sr}} \theta_c(f_{s,\text{train}}), 10^{-4}, \pi - 10^{-4}\Big)$$
2. **Core 1B $\mathcal{H}_2$-Norm Input Normalizer (`CL-04`):**
   $$\gamma_{\text{norm},c}(f_{s,\text{host}}) = \sqrt{1 - \bar{r}_c(f_{s,\text{host}})^2}$$
3. **Core 2 Fractional Delay Lines, All-Pass Diffusers, TZF Reference Tap & Switch Crossfade (`CL-06`, `CL-07`, `CL-08`):**
   $$\tau_{0,m}^{\text{samples}}(f_{s,\text{host}}) = \max\left(4.0, \frac{\tau_{0,m}^{\text{samples}}(f_{s,\text{train}})}{\rho_{\text{sr}}}\right), \qquad D_{\text{ap},m}(f_{s,\text{host}}) = \mathrm{round}\left(\frac{D_{\text{ap},m}(f_{s,\text{train}})}{\rho_{\text{sr}}}\right)$$
   $$\tau_{\text{ref}}^{\text{samples}}(f_{s,\text{host}}) = \frac{\tau_{\text{ref}}^{\text{samples}}(f_{s,\text{train}})}{\rho_{\text{sr}}}, \qquad L_{\text{xfade}}(f_{s,\text{host}}) = \mathrm{round}\left(\frac{L_{\text{xfade}}(f_{s,\text{train}})}{\rho_{\text{sr}}}\right)$$
4. **Control-Bus Decimation Stride, LFO/Envelope Rate Compensation, Core 1A AC Coupling & Core 2 DC Blocker (`CL-02`, `CL-09`):**
   $$K_{\text{ctrl}}(f_{s,\text{host}}) = \max\left(8, \mathrm{round}\left(\frac{f_{s,\text{host}}}{3000.0}\right)\right), \qquad \rho_{\text{ctrl}} = \frac{f_{s,\text{train}} \cdot K_{\text{ctrl}}(f_{s,\text{host}})}{f_{s,\text{host}} \cdot K_{\text{ctrl}}(f_{s,\text{train}})}$$
   $$\Delta\phi_{\text{lfo},j}(f_{s,\text{host}}) = \rho_{\text{ctrl}} \Delta\phi_{\text{lfo},j}(f_{s,\text{train}}), \qquad R_{\text{env},b}(f_{s,\text{host}}) = \big(R_{\text{env},b}(f_{s,\text{train}})\big)^{\rho_{\text{ctrl}}}$$
   $$R_{\text{ac}}(f_{s,\text{host}}) = \big(R_{\text{ac}}(f_{s,\text{train}})\big)^{\rho_{\text{sr}}}, \qquad R_{\text{dc}}(f_{s,\text{host}}) = \big(R_{\text{dc}}(f_{s,\text{train}})\big)^{\rho_{\text{sr}}}$$
   *(Rescaling $\Delta\phi_{\text{lfo},j}$ and $R_{\text{env},b}$ by the exact Control-Bus tick ratio $\rho_{\text{ctrl}}$ eliminates the 2% integer-decimation tempo shift at $44.1\text{ kHz}$ and $88.2\text{ kHz}$ where $K_{\text{ctrl}} = 15$ or $29$, keeping LFO frequencies in Hz 100% invariant.)*

---

## 5. Canonical `.dnam` JSON Header Schema

The UTF-8 JSON header at byte offset `0x40` contains the model configuration, essential `metadata`, open `user_metadata`, parametric `controls_schema` (with `ui_hints`), `presets`, and 64-byte aligned `tensor_directory`:

```json
{
  "version": "1.0.0",
  "container_kind": "dnam",
  "architecture": "DNAM_TriPhysics",
  "sample_rate_train_hz": 48000,
  "metadata": {
    "name": "Vintage Tape Echo & Tube Preamp",
    "author": "Studio User",
    "description": "True-stereo multi-head tape echo into a push-pull tube preamp.",
    "license": "CC-BY-4.0",
    "created_at": "2026-10-22T21:45:00Z",
    "input_reference_dbfs": -20.5,
    "provenance": {
      "dataset_manifest_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
      "config_recipe_sha256": "8f434346648f6b96df89dda901c5176b10a6d83961dd3c1ac88b59b2dc327aa4",
      "epochs_trained": 120
    },
    "signatures": {
      "immutable_header_sha256": "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
      "master_arena_sha256": "60303ae22b998861bce3b28f33eec1be758a213c86c93c076dbe9f558c11c752",
      "creator_signature": {
        "signer_id": "studio_user",
        "pubkey_ed25519": "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a",
        "sig_ed25519": "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e065224901555fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b"
      },
      "marketplace_receipt": null
    }
  },
  "user_metadata": {},
  "io_config": {
    "inputs": 2,
    "output_mode": "stereo",
    "latency_samples": 0,
    "tail_time_ms": 1850.0,
    "capabilities": {
      "blend_mode": "host_parallel",
      "default_blend": 1.0,
      "wet_polarity_inverted": false,
      "supports_trails": true,
      "tempo_sync_capable": true
    },
    "leveling": {
      "auto_unity_enabled": true,
      "leveling_metric": "rms_dbfs",
      "reference_input_level_db": -20.5,
      "measured_default_output_level_db": -20.5,
      "unity_gain_offset_db": 0.0,
      "target_gain_db": 0.0,
      "calibrated_trim_db": 0.0
    }
  },
  "slimmable_tiers": {
    "lite": {
      "channels": 4,
      "fdn_lines": 4,
      "early_taps": 4,
      "head_svd_rank": 4,
      "loudness_lufs": -20.5,
      "output_trim_db": 0.0,
      "validation_esr_db": -39.4,
      "arena_end_byte_offset": 10752,
      "prefix_crc32c": 2849103412
    },
    "full": {
      "channels": 8,
      "fdn_lines": 8,
      "early_taps": 8,
      "head_svd_rank": 4,
      "loudness_lufs": -20.5,
      "output_trim_db": 0.0,
      "validation_esr_db": -45.8,
      "arena_end_byte_offset": 22336,
      "prefix_crc32c": 1938475621
    },
    "ultra": {
      "channels": 12,
      "fdn_lines": 8,
      "early_taps": 8,
      "head_svd_rank": 4,
      "loudness_lufs": -20.5,
      "output_trim_db": 0.0,
      "validation_esr_db": -47.63,
      "arena_end_byte_offset": 36224,
      "prefix_crc32c": 4019283746
    }
  },
  "controls_schema": [
    {
      "id": "gain",
      "name": "Drive",
      "type": "range",
      "unit": "dB",
      "lobatto_degree": 2,
      "anchor_nodes_norm": [0.0, -0.6, 0.6],
      "declared_range": [0.0, 10.0],
      "safe_clamped_range": [2.0, 8.0],
      "default": 5.0,
      "conditional_min": {},
      "conditional_max": {},
      "clamp_warning": "CLAMP-RANGE: Clamped to observed stem coverage [2.0, 8.0]",
      "ui_hints": {
        "section": "Preamp",
        "order": 0,
        "widget": "knob",
        "size": "lg",
        "polarity": "unipolar",
        "midpoint": 4.0,
        "detents": [5.0],
        "step": 0.1,
        "decimals": 1,
        "scale_labels": { "0.0": "Clean", "10.0": "Saturated" },
        "color_hex": "#F59E0B"
      },
      "user_metadata": {}
    },
    {
      "id": "feedback",
      "name": "Repeats",
      "type": "range",
      "unit": "%",
      "lobatto_degree": 2,
      "anchor_nodes_norm": [0.0, -1.0, 1.0],
      "declared_range": [0.0, 10.0],
      "safe_clamped_range": [0.0, 10.0],
      "default": 5.0,
      "conditional_min": {},
      "conditional_max": { "boost=on": 9.37 },
      "clamp_warning": "CLAMP-STABILITY: Clamped <= 9.37 when boost='on' (g_lin <= 0.9990)",
      "ui_hints": {
        "section": "Tape Echo",
        "order": 0,
        "widget": "knob",
        "size": "md",
        "polarity": "unipolar",
        "midpoint": 5.0,
        "detents": [],
        "step": 0.05,
        "decimals": 1,
        "scale_labels": { "0.0": "Single", "7.5": "Swell" },
        "color_hex": "#10B981"
      },
      "user_metadata": {}
    },
    {
      "id": "boost",
      "name": "Boost Mode",
      "type": "switch",
      "declared_options": ["off", "on", "ultra"],
      "active_options": ["off", "on"],
      "default": "off",
      "clamp_warning": "CLAMP-SWITCH: Disabled unobserved option ['ultra']",
      "ui_hints": {
        "section": "Preamp",
        "order": 1,
        "widget": "segmented",
        "size": "md",
        "option_labels": {
          "off": "Bypass",
          "on": "Vintage +6 dB",
          "ultra": "Hi-Gain +12 dB"
        },
        "color_hex": "#EF4444"
      },
      "user_metadata": {}
    }
  ],
  "presets": [
    {
      "id": "warm_slapback",
      "name": "Warm Slapback",
      "description": "Subtle preamp warmth with short tape repeats.",
      "controls": { "gain": 3.5, "feedback": 2.5, "boost": "off" },
      "user_metadata": {}
    },
    {
      "id": "dub_swell",
      "name": "Edge-of-Oscillation Dub Swell",
      "description": "High-regen saturated repeats right at the coercive stability knee.",
      "controls": { "gain": 6.5, "feedback": 9.2, "boost": "on" },
      "user_metadata": {}
    }
  ],
  "control_bus": {
    "decimation": 16,
    "envelopes": 3,
    "lfo_oscillators": 2,
    "lfo_harmonics": 4,
    "fm_interaction_rank": 4,
    "head_trust_region_radius": 2.5
  },
  "core1_hybrid_ssm": {
    "blocks": 4,
    "conv_kernel": 6,
    "dilations": [1, 3, 7, 19],
    "adaa_mode": "1x_bspline_algebraic",
    "ac_coupling_pole": 0.9993,
    "weight_layout": "tier_delta_exec_order_soa_align64"
  },
  "core2_mufdn": {
    "active": true,
    "delay_storage": "fp16",
    "ring_buffer_pow2_log2": 15,
    "interp": "cubic_lagrange_4tap",
    "tzf_reference_tap_samples": 64.0,
    "switch_xfade_samples": 128,
    "allpass_delays_samples": [19, 31, 47, 73, 103, 139, 179, 223],
    "dc_blocker_pole": 0.995,
    "loop_ceiling": 0.985
  },
  "tensor_directory": {
    "arena0_lite.core1_input_and_head_bias": { "arena": 0, "dtype": "f32", "shape": [2, 4, 4], "byte_offset": 0, "byte_length": 128 },
    "arena0_lite.core1_exec_blocks_soa": { "arena": 0, "dtype": "f32", "shape": [4, 256], "byte_offset": 128, "byte_length": 4096 },
    "arena0_lite.core1_stereo_head_svd": { "arena": 0, "dtype": "f32", "shape": [5, 2, 16, 4], "byte_offset": 4224, "byte_length": 2560 },
    "arena0_lite.core2_mufdn_soa": { "arena": 0, "dtype": "f32", "shape": [16, 4], "byte_offset": 6784, "byte_length": 256 },
    "arena0_lite.control_bus_and_clamp_table": { "arena": 0, "dtype": "f32", "shape": [232, 4], "byte_offset": 7040, "byte_length": 3712 },
    "arena1_full_delta.core1_input_proj": { "arena": 1, "dtype": "f32", "shape": [1, 4, 4], "byte_offset": 10752, "byte_length": 64 },
    "arena1_full_delta.core1_exec_blocks_soa": { "arena": 1, "dtype": "f32", "shape": [4, 448], "byte_offset": 10816, "byte_length": 7168 },
    "arena1_full_delta.core1_stereo_head_svd": { "arena": 1, "dtype": "f32", "shape": [5, 2, 16, 4], "byte_offset": 17984, "byte_length": 2560 },
    "arena1_full_delta.core2_mufdn_soa": { "arena": 1, "dtype": "f32", "shape": [16, 4], "byte_offset": 20544, "byte_length": 256 },
    "arena1_full_delta.control_bus_proj": { "arena": 1, "dtype": "f32", "shape": [96, 4], "byte_offset": 20800, "byte_length": 1536 },
    "arena2_ultra_delta.core1_input_proj": { "arena": 2, "dtype": "f32", "shape": [1, 4, 4], "byte_offset": 22336, "byte_length": 64 },
    "arena2_ultra_delta.core1_exec_blocks_soa": { "arena": 2, "dtype": "f32", "shape": [4, 640], "byte_offset": 22400, "byte_length": 10240 },
    "arena2_ultra_delta.core1_stereo_head_svd": { "arena": 2, "dtype": "f32", "shape": [5, 2, 16, 4], "byte_offset": 32640, "byte_length": 2560 },
    "arena2_ultra_delta.control_bus_proj": { "arena": 2, "dtype": "f32", "shape": [64, 4], "byte_offset": 35200, "byte_length": 1024 }
  }
}
```

---

## 6. Deduplicated `.dnam-ckpt` Schema & Column-Major Upper-Triangular Gram Packing (`uplo='U'`)

### 6.1 Zero Weight Duplication & Zero `pickle` Vulnerability

Unlike legacy checkpoints that store both an exported model dictionary and a redundant PyTorch `state_dict` via `torch.save` (`pickle`), `.dnam-ckpt` (`magic = b"DNCK"`):
1. **Stores Model Weights Exactly Once (`Arena 0..2`):** The first `arena_offset_ultra_end` (`36,224`) bytes of the `.dnam-ckpt` tensor arena ARE the exact `Arena 0..2` runtime tensors of the `.dnam` model. Exporting a `.dnam` after a Tier-1 Kronecker stem addition updates the SVD head tensors in `Arena 0..2` and writes the prefix directly.
2. **Eliminates Python `pickle`:** `.dnam-ckpt` is a pure Hybrid Binary Container (`DNCK` preamble + JSON metadata + `f32`/`f64` IEEE-754 arrays) that is 100% safe against arbitrary code execution and directly readable from Rust, Python, and TypeScript/WASM.

### 6.2 Tier-Ordered Column-Major Upper-Triangular Packing (`uplo='U'`) for $\mathbf{G}_{\text{audio}}^{(s)}$

For each recorded training stem $s \in \lbrace 1, \dots, S \rbrace$, let $\mathbf{\Phi}_s \in \mathbb{R}^{L_s \times N}$ ($N = 16C \in \lbrace 64, 128, 192 \rbrace$) be the 16-tap unfolded Core 1 feature matrix, ordered by **Slimmable Tier**:
- Indices `0..63`: `Lite` channels (`ch 0..3` $\times$ 16 taps)
- Indices `64..127`: `Full` delta channels (`ch 4..7` $\times$ 16 taps)
- Indices `128..191`: `Ultra` delta channels (`ch 8..11` $\times$ 16 taps)

Because the per-stem audio Gram matrix $\mathbf{G}_{\text{audio}}^{(s)} = \mathbf{\Phi}_s^\top \mathbf{\Phi}_s \in \mathbb{R}^{N \times N}$ is **symmetric positive semi-definite** ($\mathbf{G}_{i,j}^{(s)} = \mathbf{G}_{j,i}^{(s)}$), storing the full $N \times N$ matrix wastes $\frac{N(N-1)}{2}$ redundant `float64` entries (49.7% of the matrix). Instead, `Arena 3` packs the upper triangle ($0 \le i \le j < N$) in **Column-Major Order (`LAPACK uplo='U'`)**:

$$\mathrm{pack}_{\text{U}}\big(\mathbf{G}_{\text{audio}}^{(s)}\big)\left[i + \frac{j(j + 1)}{2}\right] = \mathbf{G}_{\text{audio}}^{(s)}[i, j], \qquad 0 \le i \le j < N$$

> [!IMPORTANT]
> **The Nested Principal Submatrix Prefix Theorem & Exact 64-Byte Cache Alignment:**
> Because column-major upper-triangular index mapping $\text{idx}(i, j) = i + \frac{j(j+1)}{2}$ depends only on $i \le j$, the packed upper triangle of any leading principal submatrix of dimension $N_{\text{tier}} \in \lbrace 64, 128, 192 \rbrace$ occupies the **exact contiguous prefix** $\big[0 \dots \frac{N_{\text{tier}}(N_{\text{tier}} + 1)}{2} - 1\big]$ of the `Ultra` packed array! Furthermore, every tier's packed $\mathrm{pack}_{\text{U}}(\mathbf{G}_{\text{audio}}^{(s)})$ and tier-ordered $\mathbf{C}_{\text{audio}}^{(s)} \in \mathbb{R}^{N \times 2}$ are **exact integer multiples of 64 bytes** with zero padding overhead:

| Slimmable Tier | Feature Dim `N = 16C` | Unpacked `G_audio^(s)` (`f64`) | Packed `pack_U(G_audio^(s))` (`f64`) | Cross-Corr `C_audio^(s)` (`f64`) | Total Per-Stem `Arena 3` Record | 64-Byte Cache Lines |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Lite` (`C=4`)** | `N = 64` | `32,768 B` | `2,080 f64` = **`16,640 B`** | `128 f64` = **`1,024 B`** | **`17,664 B` (`17.25 KB/stem`)** | Exact `276 × 64 B` |
| **`Full` (`C=8`)** | `N = 128` | `131,072 B` | `8,256 f64` = **`66,048 B`** | `256 f64` = **`2,048 B`** | **`68,096 B` (`66.50 KB/stem`)** | Exact `1,064 × 64 B` |
| **`Ultra` (`C=12`)** | `N = 192` | `294,912 B` | `18,528 f64` = **`148,224 B`** | `384 f64` = **`3,072 B`** | **`151,296 B` (`147.75 KB/stem`)** | Exact `2,364 × 64 B` |

### 6.3 `.dnam-ckpt` JSON Header Extensions (`stems_registry` with Per-Stem PCM Provenance & `Arena 3` `CRC32C`)

In addition to all standard `.dnam` JSON fields (`metadata`, `user_metadata`, `controls_schema`, `presets`, `tensor_directory`), a `.dnam-ckpt` JSON header sets `"container_kind": "dnam-ckpt"` and appends `stems_registry` and `audit_report`, recording each stem's **platform-agnostic IEEE-754 PCM SHA-256 digests** (`input_pcm_sha256`, `target_pcm_sha256`), per-stem `stem_arena3_crc32c` (`u32`, Castagnoli `CRC32C` over that stem's $151,296\text{ B}$ `Arena 3` record so individual stem replacements only recompute 1 stem's checksum), optional `user_metadata`, and 64-byte aligned slice inside `Arena 3`:

```json
{
  "container_kind": "dnam-ckpt",
  "stems_registry": [
    {
      "stem_id": "stem_00_noon",
      "controls_meta": { "gain": 5.0, "feedback": 5.0, "boost": "off" },
      "provenance": {
        "input_pcm_sha256": "4b227777d4dd1fc61c6f884f48641d02b4d121d3fd328cb08b5531fcacdabf8a",
        "target_pcm_sha256": "ef2d127de37b942baad06145e54b0c619a1f22327b2ebbcfbec78f5564afe39d",
        "num_frames": 1440000
      },
      "stem_arena3_crc32c": 3194827105,
      "user_metadata": {},
      "extracted_physics": {
        "delays_ms": [12.4, 19.8, 31.2, 47.5, 73.1, 112.0, 168.4, 245.0],
        "lfo_freqs_hz": [0.85, 3.40],
        "decay_gains_lin": [0.82, 0.81, 0.79, 0.78, 0.76, 0.75, 0.73, 0.70],
        "detected_grain_size_ms": 0.0,
        "detected_pitch_shift_semitones": 0.0
      },
      "g_audio_uplo_u_f64": {
        "arena": 3,
        "dtype": "f64",
        "packing": "uplo_upper_col_major_tier_ordered",
        "feature_dim": 192,
        "packed_len": 18528,
        "byte_offset": 36224,
        "byte_length": 148224
      },
      "c_audio_tier_f64": {
        "arena": 3,
        "dtype": "f64",
        "shape": [3, 64, 2],
        "byte_offset": 184448,
        "byte_length": 3072
      }
    }
  ],
  "audit_report": {
    "phase1_clamps": ["CLAMP-RANGE", "CLAMP-SWITCH", "CLAMP-STABILITY"],
    "phase2_rail_warnings": [],
    "phase3_clamps": ["CLAMP-DYNAMIC"],
    "prescriptions": [
      "Add stems at gain=0.0 and gain=10.0 to lift [CLAMP-RANGE] on 'gain'",
      "Add >= 1 stem with boost='ultra' to lift [CLAMP-SWITCH] on 'boost'",
      "Add corner stem (feedback=10.0, boost='on') to lift [CLAMP-STABILITY] on 'feedback'"
    ]
  }
}
```

---

## 7. 4-Pillar Marketplace Security, Creator `DistributionPolicy`, `.dnam-seal` (`DNSL`) Hardware DRM & Isometric Gauge Traitor Tracing

> [!IMPORTANT]
> **Canonical Marketplace Specification:** See **[`digital_marketplace_and_provenance.md`](digital_marketplace_and_provenance.md)** for the complete 4-Pillar Security Architecture, 3 Container Protection Profiles (`Open`, `DeviceAllowlisted`, `EnclaveSealed`), creator-signed `DistributionPolicy`, `.dnam-seal` (`b"DNSL"`) 64-byte binary envelope, `OemDeviceCertificate` factory key attestation, Isometric Gauge-Space & Spread-Spectrum Tardos Traitor-Tracing proofs, and `TiDB Cloud Starter` schema.

To support commercial model distribution across open desktop DAWs (`Open`), device-bound desktop/mobile plugins (`DeviceAllowlisted`, `ENCRYPTED_ARENA = 1`), and factory-attested OEM hardware pedalboards (`EnclaveSealed`, `.dnam-seal` `b"DNSL"`), `.dnam` defines an optional creator-signed `metadata.distribution_policy` and `metadata.signatures` envelope (`HAS_SIGNATURES` bit `9` in preamble `flags`):

1. **Preset-Safe `immutable_header_sha256`, Creator `DistributionPolicy` & 2-Certificate `Ed25519` Delegation Chain (`< 80 µs` Offline UI Verification, See [`digital_marketplace_and_provenance.md` §2](digital_marketplace_and_provenance.md#2-pillar-1-cryptographic-provenance-ed25519-split-payload-chain--creator-signed-distributionpolicy)):**
   - `immutable_header_sha256` hashes the RFC 8785 canonical JSON header **including** `metadata.distribution_policy` (`allowed_protection_profiles`, `allowed_oem_vendors` supporting `"darkglass"` and `"darkglass:anagram_rk3582"`) while **excluding** `presets`, `user_metadata`, `metadata.signatures`, and `slimmable_tiers.*.prefix_crc32c`, allowing buyers to save custom DAW presets in-place (`0x40..0x1000`) on `.dnam` files without invalidating either signature.
   - **`creator_signature`:** Signed once offline by the creator's `Ed25519` private key over `immutable_header_sha256 || master_arena_sha256`.
   - **`marketplace_receipt`:** Counter-signed on download (`< 0.1 ms`) by the marketplace's `Ed25519` key binding `master_arena_sha256` $\to$ `buyer_arena_sha256`, `license_id`, `buyer_hash`, `license_tier`, `authorized_devices`, and optional `royalty_splits`:

```json
{
  "immutable_header_sha256": "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
  "master_arena_sha256": "60303ae22b998861bce3b28f33eec1be758a213c86c93c076dbe9f558c11c752",
  "creator_signature": {
    "signer_id": "did:web:studio-user.com",
    "pubkey_ed25519": "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a",
    "sig_ed25519": "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e065224901555fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b"
  },
  "marketplace_receipt": {
    "marketplace_id": "dnam.store",
    "license_id": "0192b6c0-7e4a-7000-8000-1a2b3c4d5e6f",
    "buyer_hash": "3a7bd3e2360a3d29eea436fcfb7e44c735d117c42d1c1835420b6b9942dd4f1b",
    "license_tier": "personal",
    "issued_at": "2026-10-22T22:45:00Z",
    "buyer_arena_sha256": "4e07408562bedb8b60ce05c1decfe3ad16b72230967de01f640b7e4729b49fce",
    "encryption_mode": "none",
    "arena_nonce_96": null,
    "ephemeral_pubkey_x25519": null,
    "authorized_devices": [
      {
        "oem_vendor": "darkglass",
        "model_code": "anagram_rk3582",
        "device_id_hash": "9c1185a5c5e9fc54612808977ee8f548",
        "wrapped_cek": null
      }
    ],
    "royalty_splits": [
      { "recipient_id": "did:web:studio-user.com", "role": "amp_capture", "share_bps": 8000 },
      { "recipient_id": "did:web:cab-labs.io", "role": "ir_cabinet", "share_bps": 2000 }
    ],
    "receipt_sig_ed25519": "8a1b2c3d4e5f60718293a4b5c6d7e8f90112233445566778899aabbccddeeff00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff00"
  }
}
```

2. **Dual-Mechanism Buyer Watermarking in `Arena 0..2` (`< 0.2 ms` Edge Personalization, See [`digital_marketplace_and_provenance.md` §4](digital_marketplace_and_provenance.md#4-pillar-3-forensic-traitor-tracing--02-ms-dual-mechanism-buyer-watermarking)):**
   - **Mechanism A (Exact Isometric Gauge Watermark, `0.0000 dB` Audio Distortion):** Applies 48-channel RG-LRU reciprocal scaling ($\kappa_{b,c}$ on $b_{\gamma,b,c}$ and $\kappa_{b,c}^{-1}$ on $(c_{\text{real},b,c}, c_{\text{imag},b,c})$) plus SVD control subspace orthogonal rotation ($\mathbf{Q}_{\text{buyer}} \in SO(D_h)$).
   - **Mechanism B (Coupled Spread-Spectrum Tardos Watermark, `+36.63 dB` Gain across `4,608` Conv Weights):** Survives JSON header stripping, RG-LRU gauge canonicalization, `f32 -> fp16` truncation, and multi-buyer weight-averaging collusion (`min(Z_colluder) >= 24.98 sigma` in `Test 8H`).
3. **3-Tier Protection Profiles & `.dnam-seal` (`b"DNSL"`) Hardware Enclave Wrapper (See [`digital_marketplace_and_provenance.md` §3](digital_marketplace_and_provenance.md#3-pillar-2-hardware-enclave--oem-drm-oemdevicecertificate-attestation--dnam-seal-bdnsl-wrapper) & [`§6`](digital_marketplace_and_provenance.md#6-three-container-protection-profiles-dnamprotectionprofile--loader-auto-detection)):**
   - **`Open` (`"open"`):** Standard `.dnam` (`ENCRYPTED_ARENA = 0`).
   - **`DeviceAllowlisted` (`"device_allowlisted"`):** `.dnam` (`ENCRYPTED_ARENA = 1`) with plaintext `4 KB` header for DAW/OS indexing and `ChaCha20`-encrypted `Arena 0..2` (`cek_arena`).
   - **`EnclaveSealed` (`"enclave_sealed"`):** Canonical `.dnam-seal` (`b"DNSL"`) 64-byte aligned binary wrapper (`DnamSealEnvelope`) encrypting 100% of an inner `ENCRYPTED_ARENA = 1` `.dnam` container (`cek_outer` + inner `cek_arena`) for factory-attested OEM hardware TrustZone enclaves (`OemDeviceCertificate`).


