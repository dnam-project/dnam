"""Polars + Altair training telemetry dataframe builder, Parquet writer, and interactive HTML
report."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import altair as alt
import polars as pl

from dnam_trainer.schema import TrainingTelemetryRow

TrainingStepRecord = TrainingTelemetryRow

TELEMETRY_SCHEMA: dict[str, pl.DataType] = {
    "stage": pl.Int64(),
    "epoch": pl.Int64(),
    "esr_lite_db": pl.Float64(),
    "esr_full_db": pl.Float64(),
    "esr_ultra_db": pl.Float64(),
    "barrier_loss": pl.Float64(),
    "group_lasso_loss": pl.Float64(),
    "surrogate_sigma_samples": pl.Float64(),
    "elapsed_sec": pl.Float64(),
}


def build_training_telemetry_df(records: Sequence[TrainingTelemetryRow]) -> pl.DataFrame:
    """Builds a strongly typed Polars `DataFrame` from [`TrainingTelemetryRow`] entries."""
    if not records:
        return pl.DataFrame(schema=TELEMETRY_SCHEMA)
    rows = [
        {
            "stage": r.stage,
            "epoch": r.epoch,
            "esr_lite_db": float(r.esr_lite_db),
            "esr_full_db": float(r.esr_full_db),
            "esr_ultra_db": float(r.esr_ultra_db),
            "barrier_loss": float(r.barrier_loss),
            "group_lasso_loss": float(r.group_lasso_loss),
            "surrogate_sigma_samples": float(r.surrogate_sigma_samples),
            "elapsed_sec": float(r.elapsed_sec),
        }
        for r in records
    ]
    return pl.DataFrame(rows, schema=TELEMETRY_SCHEMA)


def save_telemetry_parquet(
    df: pl.DataFrame,
    path: str | Path,
) -> Path:
    """Writes the telemetry Polars `DataFrame` to a Parquet file at `path`."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path)
    return out_path


def plot_esr_convergence_chart(
    df: pl.DataFrame,
    *,
    model_id: str = "dnam-model",
) -> alt.LayerChart | alt.Chart:
    """Creates an interactive Altair multi-tier ESR (`Lite`, `Full`, `Ultra`) convergence chart."""
    if df.is_empty():
        empty_rows = [
            {"epoch": 0, "tier": "Ultra (C=12)", "esr_db": 0.0, "stage": 1},
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

    long_df = df.unpivot(
        index=["stage", "epoch", "elapsed_sec"],
        on=["esr_lite_db", "esr_full_db", "esr_ultra_db"],
        variable_name="tier",
        value_name="esr_db",
    ).with_columns(
        pl.col("tier").replace(
            {
                "esr_lite_db": "Lite (C=4)",
                "esr_full_db": "Full (C=8)",
                "esr_ultra_db": "Ultra (C=12)",
            }
        )
    )
    records = long_df.to_dicts()
    return (
        alt.Chart(alt.Data(values=records))
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
    df: pl.DataFrame,
    path: str | Path,
    *,
    model_id: str = "dnam-model",
) -> Path:
    """Saves an interactive Altair HTML telemetry report to `path`."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    chart = plot_esr_convergence_chart(df, model_id=model_id)
    html_str = chart.to_html()
    out_path.write_text(html_str, encoding="utf-8")
    return out_path
