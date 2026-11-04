# `dnam_trainer/cli/` — Headless `dnam-train` Worker & Streaming CLI Router

**`dnam_trainer.cli`** provides the `dnam-train` executable invoked by `packages/studio` (`dnam train` and Tauri Desktop Studio) or standalone training scripts.

## 1. Modules

| File | Responsibility | Canonical Spec |
| :--- | :--- | :--- |
| [`__init__.py`](__init__.py) | `cyclopts` `dnam-train` entrypoint (`main`) | [`four_stage_pipeline.md`](../../../../../docs/specs/training/four_stage_pipeline.md) |
| [`worker.py`](worker.py) | File-backed JSON job runner (`--request-json`, `--checkpoint-in`, `--output-dnam`, `--output-ckpt`, `--reports-dir`) & `--status-json` probe | [`iterative_resume_and_caching.md`](../../../../../docs/specs/training/iterative_resume_and_caching.md) |
| [`observers.py`](observers.py) | `JsonlTrainingObserver`, `SseTrainingObserver` & `RichTrainingObserver` | [`control_space_audit.md`](../../../../../docs/specs/training/control_space_audit.md) |
