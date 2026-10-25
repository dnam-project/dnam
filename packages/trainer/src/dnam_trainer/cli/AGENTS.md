# `dnam_trainer/cli/` — Headless `dnam-train` Worker & Streaming CLI Router

**`dnam_trainer.cli`** provides the `dnam-train` executable invoked by `packages/studio` (`dnam train` and Tauri Desktop Studio) or standalone training scripts.

## 1. Modules

| File | Responsibility | Canonical Spec |
| :--- | :--- | :--- |
| [`__init__.py`](__init__.py) | `cyclopts` `dnam-train` entrypoint (`main`) | [`training_and_iterative_resume.md`](../../../../docs/training_and_iterative_resume.md) |
| [`worker.py`](worker.py) | File-backed JSON job runner (`--request-json`, `--checkpoint-in`, `--output-dnam`, `--output-ckpt`, `--reports-dir`) & `--status-json` probe | [`training_and_iterative_resume.md`](../../../../docs/training_and_iterative_resume.md) |
| [`observers.py`](observers.py) | `JsonlTrainingObserver`, `SseTrainingObserver` & `RichTrainingObserver` | [`clamp_and_audit_registry.md`](../../../../docs/clamp_and_audit_registry.md) |
