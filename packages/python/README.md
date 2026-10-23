# `dnam-engine` — Python 3.14 Interface for Dynamic Neural Audio Modeler (DNAM)

`dnam-engine` provides a zero-copy `PyO3` + `NumPy` SIMD engine and an idiomatic, strictly typed Python 3.14 interface for the **Dynamic Neural Audio Modeler (DNAM / "Tri-Physics")** Rust core (`packages/engine`).

## Key Features

- **Callable `DnamEngine` (`pedalboard`-Compatible Duck-Typing):** Render 1D mono `(N,)` or 2D `(1, N)` / `(2, N)` stereo `float32` NumPy arrays with GIL release, Continuous-Time ZOH rescaling ($44.1\text{--}96\text{ kHz}$), SLIMMABLE tiers (`Lite`, `Full`, `Ultra`), explicit SIMD/Scalar backend selection, and zero-allocation hot-reloading (`engine.reload(...)`).
- **Live `ControlRack` (`engine.controls`):** `MutableMapping` proxy supporting smooth $3\text{ kHz}$ control slewing (`engine.controls["gain"] = 7.5`), immediate snapping (`engine.controls.snap(gain=7.5)`), preset loading (`engine.controls.apply_preset(...)`), and 3-Phase clamp diagnostics (`engine.controls.clamped`).
- **First-Class `DnamContainer` & `GramMatrix`:** Load, inspect, edit headers in-place within the 4 KB page boundary, pack/unpack `Arena 3` `uplo='U'` Kronecker matrices, and verify/watermark `.dnam` and `.dnam-ckpt` containers.
- **Callable Stateful Reference DSP (`dnam_engine.dsp`) & Control Bus (`dnam_engine.control`):** Validated Pydantic v2 parameter models (`PolyAdaaParams`, `SelectiveRgLruParams`, `MufdnParams`, `LobattoParams`) driving `PolyAdaa`, `SelectiveRgLru`, `ModulatedUfdn`, `LobattoBasis`, and `Rank4Fm`.
