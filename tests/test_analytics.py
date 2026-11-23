"""
Unit Tests for MIRRORLAB Analytics Engine.

Validates:
1. Wilson score 95% confidence interval calculations.
2. Ingestion and parsing of schema-valid evaluation records.
3. Aggregation of counterfactual and comparison metrics.
4. Correct denominator and sample count computations.
5. Invariance against empty, missing, or malformed evaluation logs.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from mirrorlab.ui.analytics import (
    load_all_evaluation_records,
    load_all_run_aggregates,
    wilson_score_interval,
)


def test_wilson_score_interval_standard() -> None:
    """Verify Wilson score confidence interval on standard fractions."""
    # 16/20 = 80%
    low, high = wilson_score_interval(16, 20)
    assert 55.0 <= low <= 65.0
    assert 85.0 <= high <= 95.0
    assert low < high

    # 0/10 = 0% (Wilson interval is valid and non-negative)
    low_0, high_0 = wilson_score_interval(0, 10)
    assert low_0 == 0.0
    assert high_0 > 0.0

    # 10/10 = 100% (Wilson interval is valid and <= 100%)
    low_1, high_1 = wilson_score_interval(10, 10)
    assert low_1 < 100.0
    assert high_1 == 100.0

    # Empty sample
    assert wilson_score_interval(0, 0) == (0.0, 0.0)


def test_load_all_evaluation_records_real_logs(mock_eval_logs_dir: Path) -> None:
    """Verify that schema-valid evaluation logs are ingested into structured records."""
    records = load_all_evaluation_records(logs_root=mock_eval_logs_dir)
    assert len(records) > 0

    df = pd.DataFrame(records)
    required_cols = [
        "run_id",
        "model_name",
        "prompt_version",
        "task_id",
        "task_type",
        "difficulty_tier",
        "ground_truth_reason",
        "is_correct",
        "score",
        "is_valid_format",
        "latency_ms",
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing required column {col} in records"

    # Verify real data types
    assert df["is_correct"].dtype == bool
    assert df["score"].isin([0.0, 1.0]).all()
    assert (df["latency_ms"] >= 0.0).all()


def test_load_all_run_aggregates(mock_eval_logs_dir: Path) -> None:
    """Verify extraction of run aggregates and comparison metrics."""
    aggregates = load_all_run_aggregates(logs_root=mock_eval_logs_dir)
    assert len(aggregates) > 0

    comparison_runs = [a for a in aggregates if a["type"] == "comparison"]
    assert len(comparison_runs) >= 1

    c_run = comparison_runs[0]
    data = c_run["data"]
    assert "headline_metrics" in data
    assert "statistical_rigor" in data
    assert "overall_accuracy" in data["headline_metrics"]
    assert "contingency_table" in data["statistical_rigor"]


def test_filter_aggregation_counts(mock_eval_logs_dir: Path) -> None:
    """Verify that multi-dimensional filtering maintains mathematical denominator consistency."""
    records = load_all_evaluation_records(logs_root=mock_eval_logs_dir)
    df = pd.DataFrame(records)

    # Filter by model
    models = df["model_name"].unique()
    assert len(models) >= 2
    for m in models:
        m_df = df[df["model_name"] == m]
        k = m_df["is_correct"].sum()
        n = len(m_df)
        assert 0 <= k <= n
        if n > 0:
            rate = k / n
            assert 0.0 <= rate <= 1.0
            ci_l, ci_u = wilson_score_interval(k, n)
            assert ci_l <= (rate * 100.0) + 0.1
            assert ci_u >= (rate * 100.0) - 0.1


def test_missing_directory_returns_empty(tmp_path: Path) -> None:
    """Verify that querying a non-existent directory returns an empty list without raising."""
    non_existent = tmp_path / "does_not_exist"
    records = load_all_evaluation_records(logs_root=non_existent)
    assert records == []

    aggregates = load_all_run_aggregates(logs_root=non_existent)
    assert aggregates == []


def test_empty_evaluation_directory(tmp_path: Path) -> None:
    """Verify that querying an empty directory returns an empty list."""
    empty_root = tmp_path / "empty_eval_logs"
    empty_root.mkdir(parents=True, exist_ok=True)

    records = load_all_evaluation_records(logs_root=empty_root)
    assert records == []

    aggregates = load_all_run_aggregates(logs_root=empty_root)
    assert aggregates == []


def test_corrupted_json_handled_gracefully(mock_eval_logs_dir: Path) -> None:
    """Verify that corrupted JSON files in a run directory are skipped without crashing ingestion."""
    # mock_eval_logs_dir contains corrupted_run_20261123_030303
    records = load_all_evaluation_records(logs_root=mock_eval_logs_dir)
    assert len(records) > 0  # Ingestion still succeeds for valid runs

    aggregates = load_all_run_aggregates(logs_root=mock_eval_logs_dir)
    run_ids = [a["run_id"] for a in aggregates]
    assert "corrupted_run_20261123_030303" not in run_ids


def test_incomplete_records_handled_gracefully(tmp_path: Path) -> None:
    """Verify that CSVs missing optional columns are parsed safely with fallbacks."""
    root = tmp_path / "incomplete_logs"
    run_dir = root / "gemini-2.5-flash_eval_20261123"
    run_dir.mkdir(parents=True, exist_ok=True)

    # CSV with only minimal columns (Task ID, Is Correct)
    with open(run_dir / "per_task_details.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Task ID", "Is Correct"])
        writer.writerow(["task_001", "True"])

    records = load_all_evaluation_records(logs_root=root)
    assert len(records) == 1
    rec = records[0]
    assert rec["task_id"] == "task_001"
    assert rec["is_correct"] is True
    assert rec["difficulty_tier"] == "STANDARD"  # default fallback
    assert rec["ground_truth_reason"] == "CLEAR"  # default fallback
    assert rec["latency_ms"] == 0.0  # default fallback
