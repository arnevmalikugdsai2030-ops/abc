"""
Benchmark Evaluation Runner for MIRRORLAB.

Executes blind inference, raw trace logging, structured parsing,
isolated ground-truth joining, and metrics reporting.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from mirrorlab.evaluation.metrics import BenchmarkMetricsCalculator, EvaluationMetrics
from mirrorlab.evaluation.model_adapter import BaseVLMAdapter, ModelResponse
from mirrorlab.evaluation.structured_parser import ParsedAnswer, StructuredParser


class RunConfig(BaseModel):
    """Configuration metadata for a benchmark evaluation run."""

    run_id: str
    model_name: str
    split: str
    timestamp: str
    temperature: float = 0.0
    limit: int | None = None
    is_mock: bool = False


class BenchmarkEvaluator:
    """
    Orchestrates end-to-end blind evaluation of Vision-Language Models.
    Guarantees strict separation between model inference and private ground truth.
    """

    def __init__(
        self,
        adapter: BaseVLMAdapter,
        split_name: str = "test",
        splits_root: Path | str = "data/benchmark_splits",
        output_dir: Path | str = "eval_logs",
        limit: int | None = None,
    ):
        self.adapter = adapter
        self.split_name = split_name
        self.splits_root = Path(splits_root)
        self.output_root = Path(output_dir)
        self.limit = limit

        self.timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        safe_model = adapter.model_name.replace("/", "_").replace(":", "_")
        self.run_id = f"{safe_model}_{self.split_name}_{self.timestamp}"
        self.run_dir = self.output_root / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> tuple[EvaluationMetrics, Path]:
        """
        Execute full evaluation pipeline:
        1. Load public tasks and images (Blind Input)
        2. Run VLM inference and log raw outputs
        3. Parse outputs into structured answers
        4. Join with private ground truth
        5. Compute metrics and render report
        """
        # 1. Locate public input paths
        if self.split_name == "test":
            pub_tasks_path = self.splits_root / "test" / "public" / "tasks.jsonl"
            pub_img_dir = self.splits_root / "test" / "public" / "images"
            priv_gt_path = self.splits_root / "test" / "private" / "ground_truth.jsonl"
            priv_pairs_path = self.splits_root / "test" / "private" / "pairs.jsonl"
        else:
            pub_tasks_path = self.splits_root / self.split_name / "tasks.jsonl"
            pub_img_dir = self.splits_root / self.split_name / "images"
            priv_gt_path = self.splits_root / self.split_name / "ground_truth.jsonl"
            priv_pairs_path = self.splits_root / self.split_name / "pairs.jsonl"

        if not pub_tasks_path.is_file():
            raise FileNotFoundError(f"Public tasks file missing: {pub_tasks_path}")

        # Read public tasks
        tasks: list[dict[str, Any]] = []
        with open(pub_tasks_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    tasks.append(json.loads(line))

        if self.limit and self.limit > 0:
            tasks = tasks[: self.limit]

        # 2. Execute inference and write raw traces
        raw_log_path = self.run_dir / "raw_responses.jsonl"
        predictions_path = self.run_dir / "predictions.jsonl"

        eval_records: list[dict[str, Any]] = []

        with open(raw_log_path, "w", encoding="utf-8") as f_raw, open(
            predictions_path, "w", encoding="utf-8"
        ) as f_pred:
            for task in tasks:
                tid = task["task_id"]
                ttype = task.get("task_type", "TASK_A_VISIBILITY")
                prompt = task.get("prompt", "")
                img_name = task.get("image_filename") or f"{tid}.png"
                img_path = pub_img_dir / img_name

                # Perform blind model inference
                response: ModelResponse = self.adapter.generate(
                    prompt=prompt,
                    image_path=img_path,
                    task_id=tid,
                    task_type=ttype,
                )

                # Save raw response
                raw_record = {
                    "task_id": tid,
                    "task_type": ttype,
                    "model_name": response.model_name,
                    "latency_sec": response.latency_sec,
                    "prompt_tokens": response.prompt_tokens,
                    "completion_tokens": response.completion_tokens,
                    "total_tokens": response.total_tokens,
                    "raw_text": response.raw_text,
                    "error": response.error,
                    "is_mock": response.is_mock,
                    "timestamp": response.timestamp,
                }
                f_raw.write(json.dumps(raw_record) + "\n")

                # Parse structured output
                parsed: ParsedAnswer = StructuredParser.parse(
                    raw_text=response.raw_text,
                    task_type=ttype,
                    task_id=tid,
                )

                parsed_dict = parsed.model_dump()
                f_pred.write(json.dumps(parsed_dict) + "\n")

                eval_records.append(
                    {
                        "task_id": tid,
                        "task_type": ttype,
                        "parsed_answer": parsed_dict,
                        "ground_truth": {},  # Populated in join step
                        "difficulty_tier": task.get("difficulty_tier", "STANDARD"),
                    }
                )

        # 3. Join with private ground truth
        gt_map: dict[str, dict[str, Any]] = {}
        if priv_gt_path.is_file():
            with open(priv_gt_path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        gt_obj = json.loads(line)
                        gt_map[gt_obj["task_id"]] = gt_obj

        for rec in eval_records:
            rec["ground_truth"] = gt_map.get(rec["task_id"], {})

        # Load pair relationships
        pairs_records: list[dict[str, Any]] = []
        if priv_pairs_path.is_file():
            with open(priv_pairs_path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        pairs_records.append(json.loads(line))

        # 4. Compute Metrics
        metrics = BenchmarkMetricsCalculator.compute(
            eval_records=eval_records,
            pairs_records=pairs_records,
        )

        # 5. Save metrics JSON and Markdown Report
        metrics_file = self.run_dir / "metrics.json"
        with open(metrics_file, "w", encoding="utf-8") as f:
            f.write(metrics.model_dump_json(indent=2))

        report_md = self._render_report_markdown(metrics)
        report_file = self.run_dir / "evaluation_report.md"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report_md)

        return metrics, report_file

    def _render_report_markdown(self, metrics: EvaluationMetrics) -> str:
        """Render publication-grade scientific Markdown evaluation report."""
        is_mock_str = "[MOCK RUNNER - NOT BENCHMARK DATA]" if getattr(self.adapter, "strategy", None) else "[LIVE API EVALUATION]"

        md = []
        md.append(f"# MIRRORLAB Benchmark Evaluation Report: `{self.adapter.model_name}`\n")
        md.append(f"**Run ID:** `{self.run_id}` | **Status:** {is_mock_str} | **Split:** `{self.split_name}`\n")

        md.append("## 1. Executive Summary & Headline Accuracy\n")
        md.append("| Metric | Result | Benchmark Baseline (Chance) | Description |")
        md.append("|---|---|---|---|")
        md.append(f"| **Overall Task Accuracy** | **{metrics.overall_accuracy * 100:.1f}%** | ~33.3% | Composite accuracy across all tasks |")
        md.append(f"| **Task A: Visibility** | **{metrics.task_a_accuracy * 100:.1f}%** | 50.0% | Specular sightline binary detection |")
        md.append(f"| **Task A: Reason Precision** | {metrics.task_a_reason_accuracy * 100:.1f}% | ~16.7% | Exact optical failure mode classification |")
        md.append(f"| **Task B: Obstruction (Binary)** | {metrics.task_b_blocked_accuracy * 100:.1f}% | 50.0% | Detection of blocking barrier |")
        md.append(f"| **Task B: 3-Field Precision** | **{metrics.task_b_full_accuracy * 100:.1f}%** | ~8.3% | Blocker label + path segment exact attribution |")
        md.append(f"| **Task C: Mirror Repair** | **{metrics.task_c_accuracy * 100:.1f}%** | 33.3% | Selection of the single valid mirror configuration |")
        md.append(f"| **Counterfactual Pair Consistency** | **{metrics.pair_consistency_score * 100:.1f}%** | 25.0% | **Joint Correctness** (Both A and B correct) |")
        md.append(f"| **Cosmetic Theme Invariance** | {metrics.cosmetic_invariance_rate * 100:.1f}% | 50.0% | Consistency across 4 CAD visual themes |")
        md.append(f"| **Format Compliance** | {metrics.format_compliance_rate * 100:.1f}% | 100.0% | Adherence to requested JSON response schema |")
        md.append("")

        md.append("## 2. Counterfactual Pair Consistency Breakdown\n")
        md.append("> **Note on Guessing Bias:** A model guessing 'YES' on all pairs scores **0.0%** on Flip and Obstacle Insertion pairs. Joint correctness completely eliminates single-answer bias.\n")
        md.append("| Counterfactual Pair Type | Consistency Rate | Pairs Evaluated | Target Relationship |")
        md.append("|---|---|---|---|")
        for ptype, rate in metrics.pair_consistency_by_type.items():
            md.append(f"| `{ptype}` | **{rate * 100:.1f}%** | Monitored | Outcome flips or invariant |")
        md.append("")

        md.append("## 3. Parser Extraction Pass Distribution\n")
        md.append("| Extraction Method | Count | Percentage |")
        md.append("|---|---|---|")
        tot = metrics.total_tasks_evaluated or 1
        for p_name, cnt in metrics.extraction_pass_counts.items():
            md.append(f"| `{p_name}` | {cnt} | {(cnt / tot) * 100:.1f}% |")
        md.append("")

        if metrics.confusion_by_reason_code:
            md.append("## 4. Failure Mode Diagnostic Breakdown\n")
            md.append("| True Ground-Truth Reason | Correct Predictions | Incorrect Predictions | Accuracy |")
            md.append("|---|---|---|---|")
            for rcode, data in metrics.confusion_by_reason_code.items():
                c_ok = data.get("correct", 0)
                c_err = data.get("incorrect", 0)
                tot_r = c_ok + c_err
                acc_r = (c_ok / tot_r * 100) if tot_r > 0 else 0.0
                md.append(f"| `{rcode}` | {c_ok} | {c_err} | {acc_r:.1f}% |")
            md.append("")

        md.append("## 5. Artifact Provenance\n")
        md.append(f"- Raw Model Responses: `{self.run_dir / 'raw_responses.jsonl'}`\n")
        md.append(f"- Structured Predictions: `{self.run_dir / 'predictions.jsonl'}`\n")
        md.append(f"- Metrics JSON: `{self.run_dir / 'metrics.json'}`\n")

        return "\n".join(md)
