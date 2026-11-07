"""Unit tests for `dnam_trainer.training.telemetry` (JSONL and Altair HTML reports)."""

from __future__ import annotations

from pathlib import Path

from dnam_trainer.schema import TrainingTelemetryRow
from dnam_trainer.training.telemetry import (
    build_training_telemetry_rows,
    plot_esr_convergence_chart,
    save_telemetry_html_report,
    save_telemetry_jsonl,
)


def test_telemetry_rows_jsonl_and_altair_html(tmp_path: Path) -> None:
    # Empty rows handling
    empty_rows = build_training_telemetry_rows([])
    assert empty_rows == []
    empty_chart = plot_esr_convergence_chart(empty_rows, model_id="empty_model")
    assert empty_chart is not None
    empty_jsonl = save_telemetry_jsonl(empty_rows, tmp_path / "reports" / "empty.jsonl")
    assert empty_jsonl.read_text(encoding="utf-8") == ""

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
    normalized = build_training_telemetry_rows(rows)
    assert len(normalized) == 2
    assert normalized[0].esr_pro_db == -24.75

    jsonl_path = save_telemetry_jsonl(normalized, tmp_path / "reports" / "telemetry.jsonl")
    assert jsonl_path.is_file()
    reloaded = [
        TrainingTelemetryRow.model_validate_json(line)
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(reloaded) == 2
    assert reloaded[0].esr_pro_db == -24.75

    html_path = save_telemetry_html_report(
        normalized,
        tmp_path / "reports" / "telemetry.html",
        model_id="b7k_drive",
    )
    assert html_path.is_file()
    assert "vega" in html_path.read_text(encoding="utf-8").lower()
