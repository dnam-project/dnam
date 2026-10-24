"""Unit tests for `dnam_trainer.training.telemetry` (Polars + Altair Parquet and HTML reports)."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from dnam_trainer.schema import TrainingTelemetryRow
from dnam_trainer.training.telemetry import (
    build_training_telemetry_df,
    plot_esr_convergence_chart,
    save_telemetry_html_report,
    save_telemetry_parquet,
)


def test_telemetry_dataframe_parquet_and_altair_html(tmp_path: Path) -> None:
    # Empty dataframe handling
    empty_df = build_training_telemetry_df([])
    assert empty_df.is_empty()
    empty_chart = plot_esr_convergence_chart(empty_df, model_id="empty_model")
    assert empty_chart is not None

    rows = [
        TrainingTelemetryRow(
            stage=2,
            epoch=0,
            esr_lite_db=-21.0,
            esr_full_db=-24.0,
            esr_ultra_db=-25.5,
            barrier_loss=0.0,
            group_lasso_loss=0.0,
            surrogate_sigma_samples=35.0,
            elapsed_sec=0.1,
        ),
        TrainingTelemetryRow(
            stage=4,
            epoch=5,
            esr_lite_db=-32.0,
            esr_full_db=-38.0,
            esr_ultra_db=-44.0,
            barrier_loss=0.0,
            group_lasso_loss=0.0,
            surrogate_sigma_samples=0.0,
            elapsed_sec=0.5,
        ),
    ]
    df = build_training_telemetry_df(rows)
    assert df.height == 2

    pq_path = save_telemetry_parquet(df, tmp_path / "reports" / "telemetry.parquet")
    assert pq_path.is_file()
    reloaded = pl.read_parquet(pq_path)
    assert reloaded.height == 2

    html_path = save_telemetry_html_report(
        df,
        tmp_path / "reports" / "telemetry.html",
        model_id="b7k_drive",
    )
    assert html_path.is_file()
    assert "vega" in html_path.read_text(encoding="utf-8").lower()
