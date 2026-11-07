"""Training telemetry row normalization, JSONL writer, and interactive Altair HTML report."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import altair as alt

from dnam_trainer.schema import TrainingTelemetryRow

TrainingStepRecord = TrainingTelemetryRow

_TIER_LABELS: tuple[tuple[str, str], ...] = (
    ("esr_lite_db", "Lite (C=4)"),
    ("esr_full_db", "Full (C=8)"),
    ("esr_pro_db", "Pro (C=12)"),
    ("esr_ultra_db", "Ultra (C=16)"),
)


def build_training_telemetry_rows(
    records: Sequence[TrainingTelemetryRow],
) -> list[TrainingTelemetryRow]:
    """Normalizes a sequence of [`TrainingTelemetryRow`] entries (filling `esr_pro_db` when 0)."""
    normalized: list[TrainingTelemetryRow] = []
    for r in records:
        pro_db = float(
            r.esr_pro_db if r.esr_pro_db != 0.0 else 0.5 * (r.esr_full_db + r.esr_ultra_db)
        )
        normalized.append(
            r.model_copy(
                update={
                    "esr_lite_db": float(r.esr_lite_db),
                    "esr_full_db": float(r.esr_full_db),
                    "esr_pro_db": pro_db,
                    "esr_ultra_db": float(r.esr_ultra_db),
                    "barrier_loss": float(r.barrier_loss),
                    "group_lasso_loss": float(r.group_lasso_loss),
                    "surrogate_sigma_samples": float(r.surrogate_sigma_samples),
                    "elapsed_sec": float(r.elapsed_sec),
                }
            )
        )
    return normalized


def save_telemetry_jsonl(
    rows: Sequence[TrainingTelemetryRow],
    path: str | Path,
) -> Path:
    """Writes normalized [`TrainingTelemetryRow`] records to a JSONL file at `path`."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    normalized = build_training_telemetry_rows(rows)
    payload = "\n".join(r.model_dump_json() for r in normalized)
    if payload:
        payload += "\n"
    out_path.write_text(payload, encoding="utf-8")
    return out_path


def plot_esr_convergence_chart(
    rows: Sequence[TrainingTelemetryRow],
    *,
    model_id: str = "dnam-model",
) -> alt.LayerChart | alt.Chart:
    """Creates an interactive Altair multi-tier ESR (`Lite`/`Full`/`Pro`/`Ultra`) chart."""
    normalized = build_training_telemetry_rows(rows)
    if not normalized:
        empty_rows: list[dict[str, Any]] = [
            {"epoch": 0, "tier": "Ultra (C=16)", "esr_db": 0.0, "stage": 1},
        ]
        return (
            alt.Chart(alt.Data(values=empty_rows))
            .mark_line(point=True)
            .encode(
                x=alt.X("epoch:Q", title="Training Epoch"),
                y=alt.Y("esr_db:Q", title="Validation bESR (dB)"),
                color=alt.Color("tier:N", title="Slimmable Tier"),
            )
            .properties(
                title=f"DNAM 4-Stage bESR Convergence — {model_id}",
                width=680,
                height=340,
            )
        )

    long_records: list[dict[str, Any]] = [
        {
            "stage": r.stage,
            "epoch": r.epoch,
            "elapsed_sec": float(r.elapsed_sec),
            "tier": tier_label,
            "esr_db": float(getattr(r, attr_name)),
        }
        for r in normalized
        for attr_name, tier_label in _TIER_LABELS
    ]
    return (
        alt.Chart(alt.Data(values=long_records))
        .mark_line(point=True)
        .encode(
            x=alt.X("epoch:Q", title="Training Epoch"),
            y=alt.Y("esr_db:Q", title="Validation bESR (dB)"),
            color=alt.Color("tier:N", title="Slimmable Tier"),
            tooltip=[
                alt.Tooltip("stage:Q", title="Stage"),
                alt.Tooltip("epoch:Q", title="Epoch"),
                alt.Tooltip("tier:N", title="Tier"),
                alt.Tooltip("esr_db:Q", title="bESR (dB)", format=".2f"),
                alt.Tooltip("elapsed_sec:Q", title="Elapsed (s)", format=".3f"),
            ],
        )
        .properties(
            title=f"DNAM 4-Stage bESR Convergence — {model_id}",
            width=680,
            height=340,
        )
    )


def save_telemetry_html_report(
    rows: Sequence[TrainingTelemetryRow],
    path: str | Path,
    *,
    model_id: str = "dnam-model",
) -> Path:
    """Saves an interactive Altair HTML telemetry report to `path`."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    chart = plot_esr_convergence_chart(rows, model_id=model_id)
    html_str = chart.to_html()
    out_path.write_text(html_str, encoding="utf-8")
    return out_path
