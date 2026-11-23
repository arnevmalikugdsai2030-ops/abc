"""
Shared Pytest Fixtures for MIRRORLAB Testing.

Provides deterministic, schema-valid mock evaluation directories and test fixtures
completely isolated from developer local disk state (eval_logs/).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest


@pytest.fixture
def mock_eval_logs_dir(tmp_path: Path) -> Path:
    """
    Creates an isolated temporary directory structured as a schema-valid eval_logs/ tree.

    Contains:
    1. Dual-provider comparison run: comparison_20261123_010101
    2. Single model run: mock-vlm_dev_eval_20261123_020202
    3. Corrupted run (testing graceful JSONDecodeError handling)
    4. Empty run directory (testing directory traversal resilience)
    """
    root = tmp_path / "eval_logs"
    root.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # 1. Dual-Provider Comparison Run
    # -----------------------------------------------------------------------
    comp_dir = root / "comparison_20261123_010101"
    comp_dir.mkdir(parents=True, exist_ok=True)

    comp_meta = {
        "metadata": {
            "provider_1": "gemini",
            "provider_2": "openai",
            "tasks_evaluated": 10,
            "prompt_version": "1.0.0",
            "is_simulated": True,
            "evaluation_type": "mock",
            "provenance": "SIMULATED — NOT AN ACTUAL MODEL API RESULT",
        },
        "headline_metrics": {
            "overall_accuracy": {"gemini": 70.0, "openai": 60.0},
        },
        "statistical_rigor": {
            "contingency_table": {
                "both_correct": 5,
                "gemini_only": 2,
                "openai_only": 1,
                "both_wrong": 2,
            },
            "mcnemar_test": {"chi2_stat": 0.33, "p_value": 0.56},
        },
        "latency_ms": {
            "gemini": {"median": 310.0, "p95": 450.0},
            "openai": {"median": 820.0, "p95": 1100.0},
        },
        "financial_cost_usd": {
            "gemini": 0.005,
            "openai": 0.045,
        },
    }
    with open(comp_dir / "provider_comparison.json", "w", encoding="utf-8") as f:
        json.dump(comp_meta, f, indent=2)

    # Itemized CSVs
    task_specs = [
        ("task_vis_01", "TASK_A_VISIBILITY", "STANDARD", "CLEAR", True, 250.0),
        ("task_vis_02", "TASK_A_VISIBILITY", "STANDARD", "WRONG_SIDE", False, 310.0),
        (
            "task_obs_01",
            "TASK_B_OBSTRUCTION",
            "OCCLUDED",
            "BLOCKED_INCIDENT",
            True,
            290.0,
        ),
        ("task_obs_02", "TASK_B_OBSTRUCTION", "OCCLUDED", "CLEAR", False, 330.0),
        ("task_rep_01", "TASK_C_REPAIR", "BOUNDARY", "CLEAR", True, 410.0),
        ("task_rep_02", "TASK_C_REPAIR", "BOUNDARY", "MIRROR_MISS", False, 380.0),
        ("task_cf_01", "TASK_D_COUNTERFACTUAL", "STANDARD", "CLEAR", True, 260.0),
        (
            "task_cf_02",
            "TASK_D_COUNTERFACTUAL",
            "STANDARD",
            "OUTSIDE_FOV",
            False,
            280.0,
        ),
        ("task_ood_01", "TASK_E_OOD", "STRESS", "CLEAR", True, 520.0),
        ("task_ood_02", "TASK_E_OOD", "STRESS", "WRONG_SIDE", False, 490.0),
    ]

    for mod_key in ("gemini", "openai"):
        csv_file = comp_dir / f"{mod_key}_itemized_evaluation.csv"
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "Task ID",
                    "Task Type",
                    "Difficulty Tier",
                    "Reason Code",
                    "Model Correct",
                    "Format Valid",
                    "API Error",
                    "Latency (ms)",
                ]
            )
            for i, (tid, ttype, diff, rcode, corr, lat) in enumerate(task_specs):
                # Introduce slight variation between models
                actual_corr = (
                    corr if mod_key == "gemini" else (not corr if i == 1 else corr)
                )
                writer.writerow(
                    [tid, ttype, diff, rcode, str(actual_corr), "True", "False", lat]
                )

    # Metrics JSON files with failure traces
    gemini_evals = [
        {
            "task_id": tid,
            "task_type": ttype,
            "is_correct": corr,
            "ground_truth_reason": rcode,
        }
        for tid, ttype, diff, rcode, corr, lat in task_specs
    ]
    with open(comp_dir / "metrics_gemini.json", "w", encoding="utf-8") as f:
        json.dump(
            {"overall_accuracy": 0.70, "per_sample_evaluations": gemini_evals},
            f,
            indent=2,
        )

    with open(comp_dir / "metrics_openai.json", "w", encoding="utf-8") as f:
        json.dump(
            {"overall_accuracy": 0.60, "per_sample_evaluations": gemini_evals},
            f,
            indent=2,
        )

    # -----------------------------------------------------------------------
    # 2. Single Model Run Directory
    # -----------------------------------------------------------------------
    single_dir = root / "mock-vlm_dev_eval_20261123_020202"
    single_dir.mkdir(parents=True, exist_ok=True)
    single_data = {
        "run_id": "mock-vlm_dev_eval_20261123_020202",
        "model_name": "mock-vlm",
        "total_tasks_evaluated": 10,
        "overall_accuracy": {"rate": 0.60, "correct": 6, "total": 10},
        "is_mock": True,
        "evaluation_type": "mock",
        "latency": {"median_ms": 320.0},
        "per_sample_evaluations": gemini_evals,
    }
    with open(single_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(single_data, f, indent=2)

    # -----------------------------------------------------------------------
    # 3. Corrupted Run Directory (testing graceful exception handling)
    # -----------------------------------------------------------------------
    corrupt_dir = root / "corrupted_run_20261123_030303"
    corrupt_dir.mkdir(parents=True, exist_ok=True)
    with open(corrupt_dir / "provider_comparison.json", "w", encoding="utf-8") as f:
        f.write("{ INVALID JSON DATA FOR ERROR HANDLING TESTING")

    # -----------------------------------------------------------------------
    # 4. Empty Run Directory (testing empty folder handling)
    # -----------------------------------------------------------------------
    empty_dir = root / "empty_run_20261123_040404"
    empty_dir.mkdir(parents=True, exist_ok=True)

    return root
