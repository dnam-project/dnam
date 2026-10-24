# `dnam_trainer/project/` — Multi-Model Project Workspace, Planner & Runner

**`dnam_trainer.project`** manages DNAM multi-model project directories (`dnam.toml` and `models/<model_id>/model.toml`), OS-aware `Ed25519` creator keypair provisioning, Star + Switch + Corner + Prescription capture planning, and offline `.dnam` / `.dnam-ckpt` container inspection and rendering.

## 1. Module Routing Map

| Module | Key Exports & Responsibilities | Canonical Documentation (SSOT) |
| :--- | :--- | :--- |
| [`workspace.py`](workspace.py) | `DnamProject`, `default_user_config_dir`, `ensure_creator_ed25519_keypair`, `parse_compact_range_spec`, `parse_compact_switch_spec`, atomic TOML CRUD | [`dnam_container_schema.md`](../../../../docs/dnam_container_schema.md), [`digital_marketplace_and_provenance.md`](../../../../docs/digital_marketplace_and_provenance.md) |
| [`planner.py`](planner.py) | `PlannedStemCapture`, `plan_model_capture_schedule`, `compute_control_deltas` | [`parametric_control_bus.md`](../../../../docs/parametric_control_bus.md), [`clamp_and_audit_registry.md`](../../../../docs/clamp_and_audit_registry.md) |
| [`runner.py`](runner.py) | `ContainerInspectionSummary`, `inspect_dnam_container`, `render_with_dnam_engine` | [`dnam_container_schema.md`](../../../../docs/dnam_container_schema.md) |

## 2. Invariants

- **Zero `torch` Imports:** Keep `dnam_trainer.project` free of `torch` imports for `< 20 ms` CLI responsiveness.
- **Public Key Only in `dnam.toml`:** Store the creator's secret `Ed25519` key seed in the OS user config directory (`default_user_config_dir() / "keys" / "creator_ed25519.key"`) and persist only `author_pubkey_ed25519` in `dnam.toml`.
