"""
Benchmark Metrics and Scientific Scoring Engine for MIRRORLAB.

Implements rigorous evaluation metrics:
- Overall Task Accuracy
- Per-Task Breakdown (Task A, B, C, D)
- Counterfactual Pair Consistency (Joint Correctness: Both A and B must be correct)
- Breakdown by Counterfactual Type (Mirror Flip, Cosmetic, Obstacle Insertion, Irrelevant Obstacle)
- Cosmetic Invariance Rate
- Obstacle Attribution Precision (Task B 3-field exact match)
- Format Compliance Rate
- Diagnostic Confusion Matrix
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class EvaluationMetrics(BaseModel):
    """Scientific evaluation summary metrics across benchmark tasks."""

    total_tasks_evaluated: int = 0
    overall_accuracy: float = 0.0

    # Task A: Visibility
    task_a_total: int = 0
    task_a_accuracy: float = 0.0
    task_a_reason_accuracy: float = 0.0

    # Task B: Obstruction Identification
    task_b_total: int = 0
    task_b_blocked_accuracy: float = 0.0  # Binary blocked status match
    task_b_full_accuracy: float = 0.0  # Exact match on is_blocked + blocker_label + blocked_segment

    # Task C: Mirror Repair
    task_c_total: int = 0
    task_c_accuracy: float = 0.0  # Correct candidate selection (A/B/C)

    # Task D: Counterfactual Paired Scenes
    total_pairs_evaluated: int = 0
    pair_consistency_score: float = 0.0  # Joint Correctness: (pred_a == gt_a) and (pred_b == gt_b)
    pair_consistency_by_type: dict[str, float] = Field(default_factory=dict)
    cosmetic_invariance_rate: float = 0.0

    # Format Compliance
    format_compliance_rate: float = 0.0
    extraction_pass_counts: dict[str, int] = Field(default_factory=dict)

    # Diagnostic Breakdowns
    confusion_by_reason_code: dict[str, dict[str, int]] = Field(default_factory=dict)
    per_difficulty_accuracy: dict[str, float] = Field(default_factory=dict)


class BenchmarkMetricsCalculator:
    """Calculates all metrics from joined predictions and ground-truth records."""

    @classmethod
    def compute(
        cls,
        eval_records: list[dict[str, Any]],
        pairs_records: list[dict[str, Any]] | None = None,
    ) -> EvaluationMetrics:
        """
        Parameters
        ----------
        eval_records:
            List of merged records containing 'task_id', 'task_type', 'prediction',
            'is_valid_format', 'ground_truth', and task-specific extracted fields.
        pairs_records:
            Optional list of pair records from pairs.jsonl.
        """
        if not eval_records:
            return EvaluationMetrics()

        total = len(eval_records)
        correct_overall = 0
        valid_format_count = 0
        pass_counts: dict[str, int] = {"pass_1": 0, "pass_2": 0, "pass_3": 0, "pass_4": 0, "failed": 0}

        # Task A accumulators
        a_total = 0
        a_correct_ans = 0
        a_correct_reason = 0

        # Task B accumulators
        b_total = 0
        b_correct_blocked = 0
        b_correct_full = 0

        # Task C accumulators
        c_total = 0
        c_correct_cand = 0

        # Confusion by ground-truth reason code
        confusion: dict[str, dict[str, int]] = {}

        # Difficulty tracking
        diff_totals: dict[str, int] = {}
        diff_correct: dict[str, int] = {}

        task_map: dict[str, dict[str, Any]] = {}

        for rec in eval_records:
            tid = rec.get("task_id", "")
            ttype = rec.get("task_type", "TASK_A_VISIBILITY")
            gt = rec.get("ground_truth", {})
            pred = rec.get("parsed_answer", {})
            task_map[tid] = rec

            # Format tracking
            is_valid_fmt = pred.get("is_valid_format", False)
            if is_valid_fmt:
                valid_format_count += 1
            pass_num = pred.get("extraction_pass", 0)
            if pass_num in (1, 2, 3, 4):
                pass_counts[f"pass_{pass_num}"] += 1
            else:
                pass_counts["failed"] += 1

            # Difficulty tier
            diff = gt.get("difficulty_tier") or rec.get("difficulty_tier") or "STANDARD"
            diff_totals[diff] = diff_totals.get(diff, 0) + 1

            # Task A / D Visibility grading
            if ttype in ("TASK_A_VISIBILITY", "TASK_D_COUNTERFACTUAL"):
                a_total += 1
                gt_ans = str(gt.get("answer", "")).upper()
                p_ans = str(pred.get("prediction", "")).upper()
                ans_ok = (p_ans == gt_ans) and is_valid_fmt

                if ans_ok:
                    correct_overall += 1
                    a_correct_ans += 1
                    diff_correct[diff] = diff_correct.get(diff, 0) + 1

                gt_reason = str(gt.get("reason_code", "")).upper()
                p_reason = str(pred.get("reason_code", "")).upper()
                if p_reason and p_reason == gt_reason:
                    a_correct_reason += 1

                # Confusion matrix
                if gt_reason:
                    if gt_reason not in confusion:
                        confusion[gt_reason] = {"correct": 0, "incorrect": 0}
                    if ans_ok:
                        confusion[gt_reason]["correct"] += 1
                    else:
                        confusion[gt_reason]["incorrect"] += 1

            # Task B Obstruction grading
            elif ttype == "TASK_B_OBSTRUCTION":
                b_total += 1
                gt_blocked = gt.get("is_blocked")
                p_blocked = pred.get("is_blocked")
                raw_gt_blocker = gt.get("blocking_obstacle_label") or gt.get("blocker_label")
                gt_blocker = str(raw_gt_blocker).upper() if raw_gt_blocker else "NONE"
                raw_p_blocker = pred.get("blocker_label")
                p_blocker = str(raw_p_blocker).upper() if raw_p_blocker else "NONE"

                gt_seg = str(gt.get("blocked_segment") or "NONE").upper()
                p_seg = str(pred.get("blocked_segment") or "NONE").upper()

                blocked_ok = (p_blocked == gt_blocked) and is_valid_fmt
                if blocked_ok:
                    b_correct_blocked += 1

                full_ok = (
                    blocked_ok
                    and (p_blocker == gt_blocker)
                    and (p_seg == gt_seg)
                )
                if full_ok:
                    b_correct_full += 1
                    correct_overall += 1
                    diff_correct[diff] = diff_correct.get(diff, 0) + 1

            # Task C Repair grading
            elif ttype == "TASK_C_REPAIR":
                c_total += 1
                gt_winner = str(gt.get("correct_candidate", "")).upper()
                p_winner = str(pred.get("selected_candidate") or pred.get("prediction", "")).upper()

                cand_ok = (p_winner == gt_winner) and is_valid_fmt
                if cand_ok:
                    c_correct_cand += 1
                    correct_overall += 1
                    diff_correct[diff] = diff_correct.get(diff, 0) + 1

        # Counterfactual Pair Consistency Evaluation
        total_pairs = 0
        correct_pairs = 0
        pairs_by_type_total: dict[str, int] = {}
        pairs_by_type_correct: dict[str, int] = {}
        cosmetic_total = 0
        cosmetic_invariant = 0

        if pairs_records:
            for p in pairs_records:
                total_pairs += 1
                ptype = p.get("pair_type", "UNKNOWN")
                child_ids = p.get("child_scene_ids", [])
                pairs_by_type_total[ptype] = pairs_by_type_total.get(ptype, 0) + 1

                if len(child_ids) >= 2:
                    rec_a = task_map.get(child_ids[0])
                    rec_b = task_map.get(child_ids[1])

                    if rec_a and rec_b:
                        gt_a = str(rec_a["ground_truth"].get("answer", "")).upper()
                        gt_b = str(rec_b["ground_truth"].get("answer", "")).upper()

                        pred_a = str(rec_a["parsed_answer"].get("prediction", "")).upper()
                        pred_b = str(rec_b["parsed_answer"].get("prediction", "")).upper()

                        fmt_a = rec_a["parsed_answer"].get("is_valid_format", False)
                        fmt_b = rec_b["parsed_answer"].get("is_valid_format", False)

                        # JOINT CORRECTNESS RULE: Both Scene A and Scene B must match their respective ground truths
                        is_a_correct = fmt_a and (pred_a == gt_a)
                        is_b_correct = fmt_b and (pred_b == gt_b)
                        pair_ok = is_a_correct and is_b_correct

                        if pair_ok:
                            correct_pairs += 1
                            pairs_by_type_correct[ptype] = pairs_by_type_correct.get(ptype, 0) + 1

                        if ptype == "COSMETIC_CONTROL":
                            cosmetic_total += 1
                            if pair_ok and (pred_a == pred_b):
                                cosmetic_invariant += 1

        pair_consistency = (correct_pairs / total_pairs) if total_pairs > 0 else 0.0
        cosmetic_rate = (cosmetic_invariant / cosmetic_total) if cosmetic_total > 0 else 0.0

        pair_by_type_rates = {
            k: (pairs_by_type_correct.get(k, 0) / tot)
            for k, tot in pairs_by_type_total.items()
            if tot > 0
        }

        diff_rates = {
            k: (diff_correct.get(k, 0) / tot)
            for k, tot in diff_totals.items()
            if tot > 0
        }

        return EvaluationMetrics(
            total_tasks_evaluated=total,
            overall_accuracy=correct_overall / total if total > 0 else 0.0,
            task_a_total=a_total,
            task_a_accuracy=a_correct_ans / a_total if a_total > 0 else 0.0,
            task_a_reason_accuracy=a_correct_reason / a_total if a_total > 0 else 0.0,
            task_b_total=b_total,
            task_b_blocked_accuracy=b_correct_blocked / b_total if b_total > 0 else 0.0,
            task_b_full_accuracy=b_correct_full / b_total if b_total > 0 else 0.0,
            task_c_total=c_total,
            task_c_accuracy=c_correct_cand / c_total if c_total > 0 else 0.0,
            total_pairs_evaluated=total_pairs,
            pair_consistency_score=pair_consistency,
            pair_consistency_by_type=pair_by_type_rates,
            cosmetic_invariance_rate=cosmetic_rate,
            format_compliance_rate=valid_format_count / total if total > 0 else 0.0,
            extraction_pass_counts=pass_counts,
            confusion_by_reason_code=confusion,
            per_difficulty_accuracy=diff_rates,
        )

    @classmethod
    def compute_scoring_report(
        cls,
        eval_records: list[dict[str, Any]],
        pairs_records: list[dict[str, Any]] | None = None,
    ):
        """Compute the full comprehensive ScoringReport with exact denominators."""
        from mirrorlab.evaluation.scoring_engine import ScoringEngine

        return ScoringEngine.evaluate(eval_records=eval_records, pairs_records=pairs_records)
