"""
Complete MIRRORLAB Benchmark Scoring Engine.

Implements rigorous, mathematically verified evaluation across all task families:
- Overall Accuracy
- Task A: Visibility Accuracy (binary and reason code)
- Task B: Obstruction Accuracy (separate blocker ID, blocked leg, blocked status, and joint)
- Task C: Repair Accuracy (exact unique candidate setting)
- Task D: Counterfactual Pair Consistency (joint correctness)
- Specific Pair Breakdowns:
  - Mirror-Angle Flip Consistency
  - Cosmetic Invariance Rate
  - Obstacle Insertion Consistency
  - Irrelevant Obstacle Consistency
- Invalid Response Rate
- Per-Difficulty Accuracy Breakdown (Easy, Standard, Hard, Boundary)
- Per-Reason-Code Accuracy Breakdown (Clear, Missed Mirror, Outside FOV, Behind Mirror, Blocked Leg 1/2)
- Latency Profiling (Median, P90, Mean, Min, Max)
- Denominators reported for every single metric
- Robust JSON and CSV export facilities
- Strict Invariant: Zero tolerance for API failures or missing results (never scored correct)
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field


class MetricFraction(BaseModel):
    """
    Standard metric container reporting numerator, denominator, and computed rate.
    Guarantees denominators are never omitted or fabricated.
    """

    correct: int = Field(default=0, ge=0, description="Numerator / number of successful items")
    total: int = Field(default=0, ge=0, description="Denominator / total evaluated items")
    rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Normalized proportion [0.0, 1.0]")

    @property
    def numerator(self) -> int:
        return self.correct

    @property
    def denominator(self) -> int:
        return self.total

    @property
    def percentage(self) -> float:
        return round(self.rate * 100.0, 2)

    @classmethod
    def create(cls, correct: int, total: int) -> MetricFraction:
        """Create a validated MetricFraction with safe zero-division guard."""
        if total <= 0:
            return cls(correct=0, total=0, rate=0.0)
        rate_val = round(float(correct) / float(total), 6)
        # Numerical guard for float precision bounds [0.0, 1.0]
        rate_val = max(0.0, min(1.0, rate_val))
        return cls(correct=int(correct), total=int(total), rate=rate_val)


class ObstructionMetrics(BaseModel):
    """
    Granular scoring for Task B (Obstruction Identification).
    Scores blocker ID and blocked leg separately as well as jointly.
    """

    blocked_status: MetricFraction = Field(
        default_factory=MetricFraction,
        description="Accuracy of binary is_blocked prediction",
    )
    blocker_id: MetricFraction = Field(
        default_factory=MetricFraction,
        description="Accuracy of obstacle identifier prediction (e.g. O1, O2, NONE)",
    )
    blocked_leg: MetricFraction = Field(
        default_factory=MetricFraction,
        description="Accuracy of blocked path leg prediction (CAMERA_TO_MIRROR, MIRROR_TO_TARGET, NONE)",
    )
    joint: MetricFraction = Field(
        default_factory=MetricFraction,
        description="Joint exact match across status, blocker ID, and path leg simultaneously",
    )


class LatencyMetrics(BaseModel):
    """Latency distribution metrics in milliseconds."""

    count: int = Field(default=0, ge=0, description="Number of measured inferences")
    median_ms: float = Field(default=0.0, ge=0.0, description="50th percentile latency in ms")
    p90_ms: float = Field(default=0.0, ge=0.0, description="90th percentile latency in ms")
    mean_ms: float = Field(default=0.0, ge=0.0, description="Arithmetic mean latency in ms")
    min_ms: float = Field(default=0.0, ge=0.0, description="Minimum latency in ms")
    max_ms: float = Field(default=0.0, ge=0.0, description="Maximum latency in ms")

    @classmethod
    def compute(cls, latencies_ms: list[float]) -> LatencyMetrics:
        """Calculate distribution statistics from a list of latency samples."""
        valid_samples = [float(x) for x in latencies_ms if x >= 0.0]
        if not valid_samples:
            return cls()

        arr = np.array(valid_samples, dtype=np.float64)
        return cls(
            count=len(valid_samples),
            median_ms=round(float(np.median(arr)), 2),
            p90_ms=round(float(np.percentile(arr, 90)), 2),
            mean_ms=round(float(np.mean(arr)), 2),
            min_ms=round(float(np.min(arr)), 2),
            max_ms=round(float(np.max(arr)), 2),
        )


class SampleEvaluationResult(BaseModel):
    """Individual evaluated sample record with itemized scores and classification."""

    task_id: str
    task_type: str
    difficulty_tier: str
    ground_truth_reason: str
    is_valid_format: bool
    is_api_error: bool
    is_correct: bool
    latency_ms: float = 0.0
    details: dict[str, Any] = Field(default_factory=dict)


class ScoringReport(BaseModel):
    """
    Comprehensive scientific evaluation report produced by the Scoring Engine.
    Reports exact denominators for every metric with zero fabricated data.
    """

    total_tasks_evaluated: int = 0
    overall_accuracy: MetricFraction = Field(default_factory=MetricFraction)

    # Task A: Visibility
    visibility_accuracy: MetricFraction = Field(default_factory=MetricFraction)
    visibility_reason_accuracy: MetricFraction = Field(default_factory=MetricFraction)

    # Task B: Obstruction Identification (Separated & Joint)
    obstruction_accuracy: ObstructionMetrics = Field(default_factory=ObstructionMetrics)

    # Task C: Mirror Repair (Exact Unique Candidate Setting)
    repair_accuracy: MetricFraction = Field(default_factory=MetricFraction)

    # Task D: Counterfactual Paired Scenes (Joint Correctness)
    pair_accuracy: MetricFraction = Field(default_factory=MetricFraction)
    mirror_angle_flip_consistency: MetricFraction = Field(default_factory=MetricFraction)
    cosmetic_invariance: MetricFraction = Field(default_factory=MetricFraction)
    obstacle_insertion_consistency: MetricFraction = Field(default_factory=MetricFraction)
    irrelevant_obstacle_consistency: MetricFraction = Field(default_factory=MetricFraction)

    # Formatting & API Compliance
    invalid_response_rate: MetricFraction = Field(default_factory=MetricFraction)

    # Stratified Breakdowns
    per_difficulty_accuracy: dict[str, MetricFraction] = Field(default_factory=dict)
    per_reason_code_accuracy: dict[str, MetricFraction] = Field(default_factory=dict)

    # Latency Profiling
    latency: LatencyMetrics = Field(default_factory=LatencyMetrics)

    # Evaluated Item Records
    per_sample_evaluations: list[SampleEvaluationResult] = Field(default_factory=list)

    # -----------------------------------------------------------------------
    # Export Utilities
    # -----------------------------------------------------------------------
    def to_json(self, indent: int = 2) -> str:
        """Export report to a formatted JSON string."""
        return self.model_dump_json(indent=indent)

    def export_json(self, filepath: Path | str) -> Path:
        """Save report to a JSON file on disk."""
        target = Path(filepath)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as fp:
            fp.write(self.to_json(indent=2))
        return target

    def to_summary_csv(self) -> str:
        """Export high-level metric table to CSV string."""
        out = io.StringIO()
        writer = csv.writer(out)
        writer.writerow(["Metric Category", "Metric Name", "Correct / Count", "Total / Denominator", "Rate", "Percentage"])

        def _row(cat: str, name: str, frac: MetricFraction) -> None:
            writer.writerow([cat, name, frac.correct, frac.total, f"{frac.rate:.6f}", f"{frac.percentage:.2f}%"])

        _row("Headline", "Overall Accuracy", self.overall_accuracy)
        _row("Task A: Visibility", "Binary Visibility Accuracy", self.visibility_accuracy)
        _row("Task A: Visibility", "Reason Code Accuracy", self.visibility_reason_accuracy)

        _row("Task B: Obstruction", "Blocked Status Accuracy", self.obstruction_accuracy.blocked_status)
        _row("Task B: Obstruction", "Blocker ID Accuracy", self.obstruction_accuracy.blocker_id)
        _row("Task B: Obstruction", "Blocked Leg Accuracy", self.obstruction_accuracy.blocked_leg)
        _row("Task B: Obstruction", "Joint Obstruction Accuracy", self.obstruction_accuracy.joint)

        _row("Task C: Repair", "Mirror Repair Accuracy (Unique Setting)", self.repair_accuracy)

        _row("Task D: Pairs", "Overall Pair Accuracy (Joint)", self.pair_accuracy)
        _row("Task D: Pairs", "Mirror-Angle Flip Consistency", self.mirror_angle_flip_consistency)
        _row("Task D: Pairs", "Cosmetic Invariance Rate", self.cosmetic_invariance)
        _row("Task D: Pairs", "Obstacle Insertion Consistency", self.obstacle_insertion_consistency)
        _row("Task D: Pairs", "Irrelevant Obstacle Consistency", self.irrelevant_obstacle_consistency)

        _row("Compliance", "Invalid Response Rate", self.invalid_response_rate)

        # Per-Difficulty
        for diff_k, diff_frac in sorted(self.per_difficulty_accuracy.items()):
            _row("Difficulty Stratification", f"Difficulty: {diff_k}", diff_frac)

        # Per-Reason
        for r_k, r_frac in sorted(self.per_reason_code_accuracy.items()):
            _row("Optical Reason Code", f"Reason: {r_k}", r_frac)

        # Latency Summary rows
        writer.writerow(["Latency", "Median Latency (ms)", "", "", f"{self.latency.median_ms:.2f}", ""])
        writer.writerow(["Latency", "P90 Latency (ms)", "", "", f"{self.latency.p90_ms:.2f}", ""])
        writer.writerow(["Latency", "Mean Latency (ms)", "", "", f"{self.latency.mean_ms:.2f}", ""])

        return out.getvalue()

    def to_per_task_csv(self) -> str:
        """Export granular sample-level evaluation results to CSV string."""
        out = io.StringIO()
        writer = csv.writer(out)
        writer.writerow([
            "Task ID",
            "Task Type",
            "Difficulty Tier",
            "Reason Code",
            "Format Valid",
            "API Error",
            "Is Correct",
            "Latency (ms)",
            "Details",
        ])
        for s in self.per_sample_evaluations:
            writer.writerow([
                s.task_id,
                s.task_type,
                s.difficulty_tier,
                s.ground_truth_reason,
                s.is_valid_format,
                s.is_api_error,
                s.is_correct,
                f"{s.latency_ms:.2f}",
                json.dumps(s.details),
            ])
        return out.getvalue()

    def export_csv(
        self,
        summary_path: Path | str,
        per_task_path: Path | str | None = None,
    ) -> tuple[Path, Path | None]:
        """Save metric summary and optional per-sample breakdown CSV files to disk."""
        s_target = Path(summary_path)
        s_target.parent.mkdir(parents=True, exist_ok=True)
        with open(s_target, "w", newline="", encoding="utf-8") as fp:
            fp.write(self.to_summary_csv())

        t_target = None
        if per_task_path is not None:
            t_target = Path(per_task_path)
            t_target.parent.mkdir(parents=True, exist_ok=True)
            with open(t_target, "w", newline="", encoding="utf-8") as fp:
                fp.write(self.to_per_task_csv())

        return s_target, t_target


class ScoringEngine:
    """
    Master Evaluator and Scoring Engine for MIRRORLAB.

    Accepts evaluation records (raw or parsed) paired with ground truth keys
    and counterfactual pair mappings, calculating all required benchmark metrics.
    """

    @classmethod
    def evaluate(
        cls,
        eval_records: list[dict[str, Any]],
        pairs_records: list[dict[str, Any]] | None = None,
    ) -> ScoringReport:
        """
        Execute comprehensive scoring on evaluation records.

        Args:
            eval_records: List of dictionaries representing evaluated task items.
                Each record typically contains:
                - 'task_id': str
                - 'task_type': str (e.g. TASK_A_VISIBILITY, TASK_B_OBSTRUCTION, etc.)
                - 'ground_truth': dict with expected answer fields
                - 'parsed_answer' or 'prediction': dict with model's extracted fields
                - 'is_valid_format': bool (or within parsed_answer)
                - 'is_api_error': bool (optional)
                - 'error': str (optional)
                - 'latency_sec' or 'latency_ms': float (optional)
            pairs_records: Optional list of counterfactual pair metadata dictionaries.

        Returns:
            ScoringReport containing all required metrics with exact denominators.
        """
        if not eval_records:
            return ScoringReport()

        total_tasks = len(eval_records)
        overall_correct = 0
        invalid_format_count = 0

        # Task A
        vis_total = 0
        vis_correct = 0
        vis_reason_correct = 0

        # Task B (Separated & Joint)
        obs_total = 0
        obs_blocked_correct = 0
        obs_blocker_id_correct = 0
        obs_leg_correct = 0
        obs_joint_correct = 0

        # Task C
        repair_total = 0
        repair_correct = 0

        # Latency
        latencies_ms: list[float] = []

        # Stratifications
        diff_counts: dict[str, int] = {}
        diff_correct: dict[str, int] = {}

        reason_counts: dict[str, int] = {}
        reason_correct: dict[str, int] = {}

        # Task lookup map for paired evaluation
        task_map: dict[str, dict[str, Any]] = {}
        sample_results: list[SampleEvaluationResult] = []

        for rec in eval_records:
            tid = str(rec.get("task_id", ""))
            ttype = str(rec.get("task_type", "TASK_A_VISIBILITY")).upper()
            gt = rec.get("ground_truth", {})
            pred = rec.get("parsed_answer") or rec.get("prediction") or {}

            task_map[tid] = rec

            # Latency capture (support seconds or ms)
            lat_ms = 0.0
            if "latency_ms" in rec:
                lat_ms = float(rec["latency_ms"])
            elif "latency_sec" in rec:
                lat_ms = float(rec["latency_sec"]) * 1000.0
            elif isinstance(pred, dict) and "latency_sec" in pred:
                lat_ms = float(pred["latency_sec"]) * 1000.0
            latencies_ms.append(lat_ms)

            # API Error & Format Validity Check
            # Rule: Never treat failed API requests as correct answers
            is_api_error = bool(rec.get("is_api_error") or rec.get("error"))

            # Determine format validity
            is_valid_fmt = False
            if not is_api_error:
                if "is_valid_format" in rec:
                    is_valid_fmt = bool(rec["is_valid_format"])
                elif isinstance(pred, dict) and "is_valid_format" in pred:
                    is_valid_fmt = bool(pred["is_valid_format"])
                elif isinstance(pred, dict) and pred:
                    # If parsed_answer is a non-empty dictionary without error
                    is_valid_fmt = True

            if not is_valid_fmt or is_api_error:
                invalid_format_count += 1

            # Stratification labels
            diff_tier = str(gt.get("difficulty_tier") or rec.get("difficulty_tier") or "STANDARD").upper()
            diff_counts[diff_tier] = diff_counts.get(diff_tier, 0) + 1

            gt_reason = str(gt.get("reason_code") or gt.get("ground_truth_reason") or "CLEAR").upper()
            reason_counts[gt_reason] = reason_counts.get(gt_reason, 0) + 1

            # Evaluate Task Item
            is_sample_correct = False
            item_details: dict[str, Any] = {}

            if is_api_error or not is_valid_fmt:
                # Immediate fail: zero credit for invalid format or API faults
                is_sample_correct = False
                item_details["error"] = rec.get("error") or "FORMAT_INVALID"
            elif ttype in ("TASK_A_VISIBILITY", "VISIBILITY_PREDICTION", "TASK_D_COUNTERFACTUAL"):
                vis_total += 1

                # Normalize ground-truth and prediction boolean / string
                gt_ans_bool = cls._extract_bool_visibility(gt)
                p_ans_bool = cls._extract_bool_visibility(pred)

                ans_match = (gt_ans_bool is not None) and (gt_ans_bool == p_ans_bool)
                if ans_match:
                    vis_correct += 1
                    is_sample_correct = True

                # Reason code match
                gt_r = str(gt.get("reason_code") or "").upper()
                p_r = str(pred.get("reason_code") or "").upper()
                if p_r and p_r == gt_r:
                    vis_reason_correct += 1
                item_details["visibility_match"] = ans_match
                item_details["reason_match"] = (p_r == gt_r)

            elif ttype in ("TASK_B_OBSTRUCTION", "OBSTRUCTION_REASONING"):
                obs_total += 1

                # 1. Blocked status
                gt_is_blocked = cls._extract_bool_blocked(gt)
                p_is_blocked = cls._extract_bool_blocked(pred)
                status_ok = (gt_is_blocked is not None) and (gt_is_blocked == p_is_blocked)
                if status_ok:
                    obs_blocked_correct += 1

                # 2. Blocker ID
                gt_blocker = cls._normalize_str(
                    gt.get("blocking_obstacle_label") or gt.get("blocker_label") or gt.get("obstacle_id") or "NONE"
                )
                p_blocker = cls._normalize_str(
                    pred.get("blocker_label") or pred.get("blocking_obstacle_label") or pred.get("obstacle_id") or "NONE"
                )
                blocker_ok = (p_blocker == gt_blocker)
                if blocker_ok:
                    obs_blocker_id_correct += 1

                # 3. Blocked Leg
                gt_leg = cls._normalize_leg(gt.get("blocked_segment") or gt.get("blocked_path_leg") or "NONE")
                p_leg = cls._normalize_leg(pred.get("blocked_segment") or pred.get("blocked_path_leg") or "NONE")
                leg_ok = (p_leg == gt_leg)
                if leg_ok:
                    obs_leg_correct += 1

                # 4. Joint Match (All 3 correct simultaneously)
                joint_ok = status_ok and blocker_ok and leg_ok
                if joint_ok:
                    obs_joint_correct += 1
                    is_sample_correct = True

                item_details["status_match"] = status_ok
                item_details["blocker_id_match"] = blocker_ok
                item_details["leg_match"] = leg_ok
                item_details["joint_match"] = joint_ok

            elif ttype in ("TASK_C_REPAIR", "MIRROR_REPAIR"):
                repair_total += 1

                # Exact unique candidate setting (e.g. "A", "B", or "C")
                gt_cand = cls._normalize_candidate(gt.get("correct_candidate") or gt.get("selected_candidate") or gt.get("answer"))
                p_cand = cls._normalize_candidate(pred.get("selected_candidate") or pred.get("setting") or pred.get("prediction") or pred.get("answer"))

                # Strict requirement: candidate must be valid ("A", "B", "C") and exactly match
                cand_ok = bool(gt_cand and p_cand and (p_cand == gt_cand) and p_cand in ("A", "B", "C"))
                if cand_ok:
                    repair_correct += 1
                    is_sample_correct = True

                item_details["candidate_match"] = cand_ok
                item_details["predicted_candidate"] = p_cand
                item_details["expected_candidate"] = gt_cand

            if is_sample_correct:
                overall_correct += 1
                diff_correct[diff_tier] = diff_correct.get(diff_tier, 0) + 1
                reason_correct[gt_reason] = reason_correct.get(gt_reason, 0) + 1

            sample_results.append(
                SampleEvaluationResult(
                    task_id=tid,
                    task_type=ttype,
                    difficulty_tier=diff_tier,
                    ground_truth_reason=gt_reason,
                    is_valid_format=is_valid_fmt,
                    is_api_error=is_api_error,
                    is_correct=is_sample_correct,
                    latency_ms=lat_ms,
                    details=item_details,
                )
            )

        # -------------------------------------------------------------------
        # Counterfactual Pair Consistency Evaluation
        # -------------------------------------------------------------------
        pair_total = 0
        pair_correct = 0

        flip_total = 0
        flip_correct = 0

        cosm_total = 0
        cosm_invariant = 0

        obs_ins_total = 0
        obs_ins_correct = 0

        irrel_obs_total = 0
        irrel_obs_correct = 0

        if pairs_records:
            for p in pairs_records:
                pair_total += 1
                ptype = str(p.get("pair_type", "UNKNOWN")).upper()
                child_ids = p.get("child_scene_ids", [])

                if ptype in ("MIRROR_ANGLE_FLIP", "PAIR_TYPE_1_MIRROR_FLIP"):
                    flip_total += 1
                elif ptype in ("COSMETIC_CONTROL", "PAIR_TYPE_2_COSMETIC"):
                    cosm_total += 1
                elif ptype in ("OBSTACLE_INSERTION", "PAIR_TYPE_3_OBSTACLE_INSERTION"):
                    obs_ins_total += 1
                elif ptype in ("IRRELEVANT_OBSTACLE", "PAIR_TYPE_4_IRRELEVANT_OBSTACLE"):
                    irrel_obs_total += 1

                if len(child_ids) >= 2:
                    rec_a = task_map.get(child_ids[0])
                    rec_b = task_map.get(child_ids[1])

                    if rec_a and rec_b:
                        res_a = cls._is_sample_accurate(rec_a)
                        res_b = cls._is_sample_accurate(rec_b)

                        # Joint Correctness Invariant: Both Scene A and Scene B must be accurate
                        is_joint_pair_correct = res_a and res_b

                        if is_joint_pair_correct:
                            pair_correct += 1

                        if (
                            ptype in ("MIRROR_ANGLE_FLIP", "PAIR_TYPE_1_MIRROR_FLIP")
                            and is_joint_pair_correct
                        ):
                            flip_correct += 1

                        elif ptype in ("COSMETIC_CONTROL", "PAIR_TYPE_2_COSMETIC"):
                            # Cosmetic Invariance: both must be correct AND predictions identical
                            pred_a_ans = cls._extract_pred_raw(rec_a)
                            pred_b_ans = cls._extract_pred_raw(rec_b)
                            if is_joint_pair_correct and (pred_a_ans is not None) and (pred_a_ans == pred_b_ans):
                                cosm_invariant += 1

                        elif (
                            ptype in ("OBSTACLE_INSERTION", "PAIR_TYPE_3_OBSTACLE_INSERTION")
                            and is_joint_pair_correct
                        ):
                            obs_ins_correct += 1

                        elif (
                            ptype in ("IRRELEVANT_OBSTACLE", "PAIR_TYPE_4_IRRELEVANT_OBSTACLE")
                            and is_joint_pair_correct
                        ):
                            irrel_obs_correct += 1

        # Calculate MetricFractions with strict denominators
        overall_frac = MetricFraction.create(overall_correct, total_tasks)
        vis_frac = MetricFraction.create(vis_correct, vis_total)
        vis_reason_frac = MetricFraction.create(vis_reason_correct, vis_total)

        obs_metrics = ObstructionMetrics(
            blocked_status=MetricFraction.create(obs_blocked_correct, obs_total),
            blocker_id=MetricFraction.create(obs_blocker_id_correct, obs_total),
            blocked_leg=MetricFraction.create(obs_leg_correct, obs_total),
            joint=MetricFraction.create(obs_joint_correct, obs_total),
        )

        repair_frac = MetricFraction.create(repair_correct, repair_total)

        pair_frac = MetricFraction.create(pair_correct, pair_total)
        flip_frac = MetricFraction.create(flip_correct, flip_total)
        cosm_frac = MetricFraction.create(cosm_invariant, cosm_total)
        obs_ins_frac = MetricFraction.create(obs_ins_correct, obs_ins_total)
        irrel_obs_frac = MetricFraction.create(irrel_obs_correct, irrel_obs_total)

        invalid_frac = MetricFraction.create(invalid_format_count, total_tasks)

        # Per-Difficulty
        per_diff = {
            k: MetricFraction.create(diff_correct.get(k, 0), tot)
            for k, tot in sorted(diff_counts.items())
        }

        # Per-Reason Code
        per_reason = {
            k: MetricFraction.create(reason_correct.get(k, 0), tot)
            for k, tot in sorted(reason_counts.items())
        }

        latency_metrics = LatencyMetrics.compute(latencies_ms)

        return ScoringReport(
            total_tasks_evaluated=total_tasks,
            overall_accuracy=overall_frac,
            visibility_accuracy=vis_frac,
            visibility_reason_accuracy=vis_reason_frac,
            obstruction_accuracy=obs_metrics,
            repair_accuracy=repair_frac,
            pair_accuracy=pair_frac,
            mirror_angle_flip_consistency=flip_frac,
            cosmetic_invariance=cosm_frac,
            obstacle_insertion_consistency=obs_ins_frac,
            irrelevant_obstacle_consistency=irrel_obs_frac,
            invalid_response_rate=invalid_frac,
            per_difficulty_accuracy=per_diff,
            per_reason_code_accuracy=per_reason,
            latency=latency_metrics,
            per_sample_evaluations=sample_results,
        )

    # -----------------------------------------------------------------------
    # Helper Normalizers
    # -----------------------------------------------------------------------
    @classmethod
    def _is_sample_accurate(cls, rec: dict[str, Any]) -> bool:
        """Evaluate whether a single sample prediction matches its ground truth."""
        if bool(rec.get("is_api_error") or rec.get("error")):
            return False

        pred = rec.get("parsed_answer") or rec.get("prediction") or {}
        gt = rec.get("ground_truth", {})

        is_valid_fmt = rec.get("is_valid_format")
        if is_valid_fmt is None and isinstance(pred, dict):
            is_valid_fmt = pred.get("is_valid_format", bool(pred))
        if not is_valid_fmt:
            return False

        ttype = str(rec.get("task_type", "TASK_A_VISIBILITY")).upper()

        if ttype in ("TASK_A_VISIBILITY", "VISIBILITY_PREDICTION", "TASK_D_COUNTERFACTUAL"):
            gt_b = cls._extract_bool_visibility(gt)
            p_b = cls._extract_bool_visibility(pred)
            return (gt_b is not None) and (gt_b == p_b)

        elif ttype in ("TASK_B_OBSTRUCTION", "OBSTRUCTION_REASONING"):
            gt_bl = cls._extract_bool_blocked(gt)
            p_bl = cls._extract_bool_blocked(pred)
            if gt_bl != p_bl:
                return False

            gt_id = cls._normalize_str(gt.get("blocking_obstacle_label") or gt.get("blocker_label") or "NONE")
            p_id = cls._normalize_str(pred.get("blocker_label") or pred.get("blocking_obstacle_label") or "NONE")
            if gt_id != p_id:
                return False

            gt_leg = cls._normalize_leg(gt.get("blocked_segment") or gt.get("blocked_path_leg") or "NONE")
            p_leg = cls._normalize_leg(pred.get("blocked_segment") or pred.get("blocked_path_leg") or "NONE")
            return gt_leg == p_leg

        elif ttype in ("TASK_C_REPAIR", "MIRROR_REPAIR"):
            gt_cand = cls._normalize_candidate(gt.get("correct_candidate") or gt.get("selected_candidate") or gt.get("answer"))
            p_cand = cls._normalize_candidate(pred.get("selected_candidate") or pred.get("setting") or pred.get("prediction") or pred.get("answer"))
            return bool(gt_cand and p_cand and gt_cand == p_cand and p_cand in ("A", "B", "C"))

        return False

    @staticmethod
    def _extract_bool_visibility(d: dict[str, Any]) -> bool | None:
        """Extract boolean visibility from diverse key representations."""
        if not isinstance(d, dict):
            return None
        if "can_see" in d and isinstance(d["can_see"], bool):
            return d["can_see"]
        if "visible" in d and isinstance(d["visible"], bool):
            return d["visible"]
        if "is_visible" in d and isinstance(d["is_visible"], bool):
            return d["is_visible"]
        ans = str(d.get("answer") or d.get("prediction") or "").strip().upper()
        if ans in ("YES", "TRUE", "VISIBLE"):
            return True
        if ans in ("NO", "FALSE", "NOT_VISIBLE", "BLOCKED"):
            return False
        return None

    @staticmethod
    def _extract_bool_blocked(d: dict[str, Any]) -> bool | None:
        """Extract boolean blocked status from diverse key representations."""
        if not isinstance(d, dict):
            return None
        if "is_blocked" in d and isinstance(d["is_blocked"], bool):
            return d["is_blocked"]
        if "blocked" in d and isinstance(d["blocked"], bool):
            return d["blocked"]
        ans = str(d.get("answer") or d.get("prediction") or "").strip().upper()
        if ans in ("YES", "TRUE", "BLOCKED"):
            return True
        if ans in ("NO", "FALSE", "CLEAR", "UNBLOCKED"):
            return False
        return None

    @staticmethod
    def _extract_pred_raw(rec: dict[str, Any]) -> str | None:
        """Extract standardized string answer for invariance testing."""
        pred = rec.get("parsed_answer") or rec.get("prediction") or {}
        if not isinstance(pred, dict):
            return str(pred).strip().upper()
        ans = pred.get("answer") or pred.get("prediction") or pred.get("selected_candidate") or pred.get("setting")
        if ans is not None:
            return str(ans).strip().upper()
        if "visible" in pred:
            return str(pred["visible"]).upper()
        if "can_see" in pred:
            return str(pred["can_see"]).upper()
        return None

    @staticmethod
    def _normalize_str(val: Any) -> str:
        """Normalize obstacle labels or general strings."""
        if val is None:
            return "NONE"
        s = str(val).strip().upper()
        return s if s else "NONE"

    @staticmethod
    def _normalize_leg(val: Any) -> str:
        """Normalize path leg descriptions to canonical enum representation."""
        if val is None:
            return "NONE"
        s = str(val).strip().upper().replace(" ", "_").replace("-", "_")
        if "CAMERA_TO_MIRROR" in s or s in ("LEG_1", "LEG1", "INCIDENT"):
            return "CAMERA_TO_MIRROR"
        if "MIRROR_TO_TARGET" in s or s in ("LEG_2", "LEG2", "REFLECTED"):
            return "MIRROR_TO_TARGET"
        if s in ("NONE", "CLEAR", "UNBLOCKED"):
            return "NONE"
        return s

    @staticmethod
    def _normalize_candidate(val: Any) -> str | None:
        """Normalize mirror repair candidates to canonical 'A', 'B', 'C'."""
        if val is None:
            return None
        s = str(val).strip().upper()
        # Handle formats like "SETTING A", "CANDIDATE B", or "C"
        for cand in ("A", "B", "C"):
            if s == cand or s.endswith(f" {cand}") or s.startswith(f"SETTING_{cand}"):
                return cand
        return s if s in ("A", "B", "C") else None
