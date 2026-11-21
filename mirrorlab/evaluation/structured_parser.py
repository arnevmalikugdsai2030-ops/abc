"""
Resilient Multi-Pass Structured Response Parser for MIRRORLAB Evaluation.

Extracts structured predictions from VLM text responses across:
- Task A: Visibility (YES/NO, reason code)
- Task B: Obstruction (is_blocked, blocker_label, blocked_segment)
- Task C: Mirror Repair (selected_candidate A/B/C)
- Task D: Counterfactual Paired Scenes

Extraction Passes:
1. Strict Direct JSON Parsing
2. Markdown Fenced Code Blocks (```json ... ```)
3. Outermost Balanced Curly-Brace Extraction ({ ... })
4. Robust Heuristic Keyword & Regex Fallback
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel


class ParsedAnswer(BaseModel):
    """Structured extraction of a model's prediction."""

    task_id: str
    task_type: str
    is_valid_format: bool
    format_error: str | None = None
    extraction_pass: int = 0  # 1 to 4, or 0 if failed
    prediction: str | None = None  # Normalized primary prediction (YES, NO, A, B, C)
    can_see: bool | None = None  # Task A / D
    is_blocked: bool | None = None  # Task B
    reason_code: str | None = None  # Task A / D
    blocker_label: str | None = None  # Task B (O1, O2, O3, NONE)
    blocked_segment: str | None = None  # Task B (CAMERA_TO_MIRROR, MIRROR_TO_TARGET, NONE)
    selected_candidate: str | None = None  # Task C (A, B, C)
    explanation: str | None = None
    raw_data: dict[str, Any] | None = None


class StructuredParser:
    """Multi-pass parser tolerant to conversational wrappers, markdown fences, and verbose commentary."""

    @classmethod
    def parse(
        cls,
        raw_text: str,
        task_type: str = "TASK_A_VISIBILITY",
        task_id: str = "unknown",
    ) -> ParsedAnswer:
        if not raw_text or not raw_text.strip():
            return ParsedAnswer(
                task_id=task_id,
                task_type=task_type,
                is_valid_format=False,
                format_error="Empty or whitespace response from model.",
                extraction_pass=0,
            )

        text = raw_text.strip()

        # Pass 1: Strict JSON Parse
        parsed_dict, _err = cls._try_parse_json(text)
        if parsed_dict is not None:
            return cls._build_from_dict(
                parsed_dict, task_type, task_id, extraction_pass=1
            )

        # Pass 2: Markdown Code Block Extraction
        fenced_match = re.findall(
            r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE
        )
        for block in fenced_match:
            parsed_dict, _ = cls._try_parse_json(block.strip())
            if parsed_dict is not None:
                return cls._build_from_dict(
                    parsed_dict, task_type, task_id, extraction_pass=2
                )

        # Pass 3: Outermost Balanced Curly-Brace Extraction
        brace_candidate = cls._extract_outermost_json_object(text)
        if brace_candidate:
            parsed_dict, _ = cls._try_parse_json(brace_candidate)
            if parsed_dict is not None:
                return cls._build_from_dict(
                    parsed_dict, task_type, task_id, extraction_pass=3
                )

        # Pass 4: Robust Heuristic Regex Fallback
        heuristic_res = cls._extract_heuristic(text, task_type, task_id)
        if heuristic_res is not None:
            return heuristic_res

        # All passes failed
        return ParsedAnswer(
            task_id=task_id,
            task_type=task_type,
            is_valid_format=False,
            format_error="Failed to extract structured answer via JSON or regex heuristics.",
            extraction_pass=0,
            explanation=text[:300] if len(text) > 300 else text,
        )

    @staticmethod
    def _try_parse_json(content: str) -> tuple[dict[str, Any] | None, str | None]:
        try:
            val = json.loads(content)
            if isinstance(val, dict):
                return val, None
            return None, "JSON root is not a dictionary"
        except Exception as e:  # noqa: BLE001
            return None, str(e)

    @staticmethod
    def _extract_outermost_json_object(text: str) -> str | None:
        """Finds the first '{' and its matching closing '}' respecting nesting."""
        start = text.find("{")
        if start == -1:
            return None

        depth = 0
        in_string = False
        escape = False

        for idx in range(start, len(text)):
            c = text[idx]
            if escape:
                escape = False
                continue
            if c == "\\":
                escape = True
                continue
            if c == '"':
                in_string = not in_string
                continue
            if not in_string:
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        return text[start : idx + 1]
        return None

    @classmethod
    def _build_from_dict(
        cls,
        data: dict[str, Any],
        task_type: str,
        task_id: str,
        extraction_pass: int,
    ) -> ParsedAnswer:
        # Case-insensitive key lookup helper
        norm_keys = {k.lower(): v for k, v in data.items()}

        explanation = str(
            norm_keys.get("explanation")
            or norm_keys.get("reasoning")
            or norm_keys.get("thought")
            or ""
        )

        if task_type in ("TASK_A_VISIBILITY", "TASK_D_COUNTERFACTUAL"):
            raw_ans = norm_keys.get("answer") or norm_keys.get("prediction")
            can_see = norm_keys.get("can_see")

            # Determine primary YES/NO prediction
            norm_ans: str | None = None
            if isinstance(can_see, bool):
                norm_ans = "YES" if can_see else "NO"
            elif isinstance(raw_ans, bool):
                can_see = raw_ans
                norm_ans = "YES" if raw_ans else "NO"
            elif isinstance(raw_ans, str):
                u = raw_ans.strip().upper()
                if u in ("YES", "TRUE", "VISIBLE", "CLEAR"):
                    norm_ans = "YES"
                    can_see = True
                elif u in ("NO", "FALSE", "NOT_VISIBLE", "BLOCKED", "HIDDEN"):
                    norm_ans = "NO"
                    can_see = False

            raw_reason = str(norm_keys.get("reason") or norm_keys.get("reason_code") or "").upper()
            valid_reasons = {
                "CLEAR", "MIRROR_MISS", "OUTSIDE_FOV", "WRONG_SIDE",
                "BLOCKED_CAMERA_TO_MIRROR", "BLOCKED_MIRROR_TO_TARGET"
            }
            norm_reason = raw_reason if raw_reason in valid_reasons else None

            is_valid = norm_ans is not None
            return ParsedAnswer(
                task_id=task_id,
                task_type=task_type,
                is_valid_format=is_valid,
                format_error=None if is_valid else "Missing 'answer' or 'can_see' in dictionary.",
                extraction_pass=extraction_pass,
                prediction=norm_ans,
                can_see=can_see,
                reason_code=norm_reason,
                explanation=explanation,
                raw_data=data,
            )

        elif task_type == "TASK_B_OBSTRUCTION":
            raw_blocked = norm_keys.get("is_blocked")
            raw_ans = norm_keys.get("answer")

            is_blocked: bool | None = None
            norm_ans: str | None = None

            if isinstance(raw_blocked, bool):
                is_blocked = raw_blocked
                norm_ans = "YES" if is_blocked else "NO"
            elif isinstance(raw_ans, bool):
                is_blocked = raw_ans
                norm_ans = "YES" if raw_ans else "NO"
            elif isinstance(raw_ans, str):
                u = raw_ans.strip().upper()
                if u in ("YES", "TRUE", "BLOCKED", "OBSTRUCTED"):
                    is_blocked = True
                    norm_ans = "YES"
                elif u in ("NO", "FALSE", "CLEAR", "UNOBSTRUCTED"):
                    is_blocked = False
                    norm_ans = "NO"

            raw_blocker = str(norm_keys.get("blocker_label") or norm_keys.get("blocker") or "").upper().strip()
            blocker_match = re.search(r"\b(O[1-3]|NONE)\b", raw_blocker)
            norm_blocker = blocker_match.group(1) if blocker_match else None

            raw_segment = str(norm_keys.get("blocked_segment") or norm_keys.get("segment") or "").upper().strip()
            if "CAMERA" in raw_segment or "INCIDENT" in raw_segment or "FIRST" in raw_segment:
                norm_seg = "CAMERA_TO_MIRROR"
            elif "TARGET" in raw_segment or "REFLECTED" in raw_segment or "SECOND" in raw_segment:
                norm_seg = "MIRROR_TO_TARGET"
            elif "NONE" in raw_segment or raw_segment == "":
                norm_seg = "NONE" if (is_blocked is False or norm_blocker == "NONE") else None
            else:
                norm_seg = None

            if is_blocked is False:
                if not norm_blocker:
                    norm_blocker = "NONE"
                if not norm_seg:
                    norm_seg = "NONE"

            is_valid = norm_ans is not None
            return ParsedAnswer(
                task_id=task_id,
                task_type=task_type,
                is_valid_format=is_valid,
                format_error=None if is_valid else "Missing 'is_blocked' or 'answer' in dictionary.",
                extraction_pass=extraction_pass,
                prediction=norm_ans,
                is_blocked=is_blocked,
                blocker_label=norm_blocker,
                blocked_segment=norm_seg,
                explanation=explanation,
                raw_data=data,
            )

        elif task_type == "TASK_C_REPAIR":
            cand = norm_keys.get("selected_candidate") or norm_keys.get("candidate") or norm_keys.get("answer")
            norm_cand: str | None = None
            if isinstance(cand, str):
                cand_match = re.search(r"\b([ABC])\b", cand.upper())
                if cand_match:
                    norm_cand = cand_match.group(1)

            is_valid = norm_cand is not None
            return ParsedAnswer(
                task_id=task_id,
                task_type=task_type,
                is_valid_format=is_valid,
                format_error=None if is_valid else "Missing valid candidate (A, B, or C) in dictionary.",
                extraction_pass=extraction_pass,
                prediction=norm_cand,
                selected_candidate=norm_cand,
                explanation=explanation,
                raw_data=data,
            )

        # Generic default
        return ParsedAnswer(
            task_id=task_id,
            task_type=task_type,
            is_valid_format=True,
            extraction_pass=extraction_pass,
            raw_data=data,
        )

    @classmethod
    def _extract_heuristic(
        cls,
        text: str,
        task_type: str,
        task_id: str,
    ) -> ParsedAnswer | None:
        """Regex-based fallback extraction from plain conversational text."""
        upper_text = text.upper()

        if task_type in ("TASK_A_VISIBILITY", "TASK_D_COUNTERFACTUAL"):
            # Look for answer patterns: "Answer: YES", "can_see: true", "The answer is NO"
            ans_match = re.search(
                r"(?:answer|prediction|can_see)\s*[:=]?\s*[\"']?(YES|NO|TRUE|FALSE|VISIBLE|NOT_VISIBLE)[\"']?",
                upper_text,
            )
            if not ans_match:
                # Standalone declaration e.g. "Therefore, the camera can see the scratch: YES"
                ans_match = re.search(r"\b(?:IS\s+)?(VISIBLE|NOT\s+VISIBLE)\b", upper_text)

            norm_ans: str | None = None
            can_see: bool | None = None
            if ans_match:
                token = ans_match.group(1).replace(" ", "_")
                if token in ("YES", "TRUE", "VISIBLE"):
                    norm_ans = "YES"
                    can_see = True
                elif token in ("NO", "FALSE", "NOT_VISIBLE"):
                    norm_ans = "NO"
                    can_see = False

            if norm_ans is not None:
                # Extract reason code if present
                reason_match = re.search(
                    r"\b(CLEAR|MIRROR_MISS|OUTSIDE_FOV|WRONG_SIDE|BLOCKED_CAMERA_TO_MIRROR|BLOCKED_MIRROR_TO_TARGET)\b",
                    upper_text,
                )
                reason_code = reason_match.group(1) if reason_match else None

                return ParsedAnswer(
                    task_id=task_id,
                    task_type=task_type,
                    is_valid_format=True,
                    extraction_pass=4,
                    prediction=norm_ans,
                    can_see=can_see,
                    reason_code=reason_code,
                    explanation=text[:300],
                )

        elif task_type == "TASK_B_OBSTRUCTION":
            # Search for blocked status
            block_match = re.search(
                r"(?:is_blocked|blocked|obstructed)\s*[:=]?\s*[\"']?(YES|NO|TRUE|FALSE)[\"']?",
                upper_text,
            )
            norm_ans: str | None = None
            is_blocked: bool | None = None
            if block_match:
                val = block_match.group(1)
                is_blocked = val in ("YES", "TRUE")
                norm_ans = "YES" if is_blocked else "NO"
            else:
                if re.search(r"\b(IS\s+BLOCKED|PATH\s+IS\s+BLOCKED)\b", upper_text):
                    norm_ans = "YES"
                    is_blocked = True
                elif re.search(r"\b(NOT\s+BLOCKED|PATH\s+IS\s+CLEAR|UNOBSTRUCTED)\b", upper_text):
                    norm_ans = "NO"
                    is_blocked = False

            if norm_ans is not None:
                blocker_match = re.search(r"\b(O[1-3]|NONE)\b", upper_text)
                blocker_label = blocker_match.group(1) if blocker_match else None

                seg_match = re.search(
                    r"\b(CAMERA_TO_MIRROR|MIRROR_TO_TARGET|INCIDENT|REFLECTED)\b", upper_text
                )
                if seg_match:
                    s = seg_match.group(1)
                    segment = "CAMERA_TO_MIRROR" if s in ("CAMERA_TO_MIRROR", "INCIDENT") else "MIRROR_TO_TARGET"
                else:
                    segment = "NONE" if norm_ans == "NO" else None

                return ParsedAnswer(
                    task_id=task_id,
                    task_type=task_type,
                    is_valid_format=True,
                    extraction_pass=4,
                    prediction=norm_ans,
                    is_blocked=is_blocked,
                    blocker_label=blocker_label,
                    blocked_segment=segment,
                    explanation=text[:300],
                )

        elif task_type == "TASK_C_REPAIR":
            cand_match = re.search(
                r"(?:selected_candidate|candidate|mirror|answer)\s*[:=]?\s*[\"']?([ABC])[\"']?",
                upper_text,
            )
            if not cand_match:
                cand_match = re.search(r"\b(?:CANDIDATE|MIRROR)\s+([ABC])\b", upper_text)

            if cand_match:
                cand = cand_match.group(1)
                return ParsedAnswer(
                    task_id=task_id,
                    task_type=task_type,
                    is_valid_format=True,
                    extraction_pass=4,
                    prediction=cand,
                    selected_candidate=cand,
                    explanation=text[:300],
                )

        return None
