"""
Test Script for MIRRORLAB Provider-Independent Vision Model Adapter.

Demonstrates:
1. Loading secrets securely from environment variables / .env.
2. Initializing the configured Vision Model Adapter (Google Gemini).
3. Sending a 2D optical inspection diagram and text question.
4. Capturing the verbatim raw API response.
5. Inspecting response metadata: latency, tokens, retries, model version, finish reason.
6. Graceful error handling and retry logic.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from mirrorlab.evaluation.model_adapter import (
    GeminiVisionAdapter,
    MockStrategy,
    MockVisionAdapter,
    ModelResponse,
    ModelSettings,
)

# Load secrets from .env
load_dotenv()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="MIRRORLAB Provider-Independent Vision Model Adapter Test"
    )
    parser.add_argument(
        "--image",
        type=str,
        default="data/benchmark_tasks/task_vis_10000_valid_visible/diagram_blind.png",
        help="Path to the sample diagram image PNG",
    )
    parser.add_argument(
        "--question",
        type=str,
        default=(
            "You are an industrial machine-vision inspector analyzing a 2D optical diagram.\n"
            "Can the camera see the center of the red X scratch defect through the finite mirror?\n"
            "Respond in JSON format with:\n"
            "{\n"
            '  "can_see": true or false,\n'
            '  "answer": "YES" or "NO",\n'
            '  "reason": "CLEAR" | "MIRROR_MISS" | "OUTSIDE_FOV" | "WRONG_SIDE" | "BLOCKED_CAMERA_TO_MIRROR" | "BLOCKED_MIRROR_TO_TARGET",\n'
            '  "explanation": "<your step-by-step reasoning>"\n'
            "}"
        ),
        help="Inspection question or prompt text",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.5-flash",
        help="Model name (e.g. gemini-2.5-flash, gemini-2.5-pro)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0.0 for deterministic benchmark reproduction)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retry attempts on transient failures",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Request timeout in seconds",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Force offline mock runner simulation",
    )

    args = parser.parse_args()

    print("=" * 75)
    print("  MIRRORLAB: Provider-Independent Vision Model Adapter Verification")
    print("=" * 75)
    print(f"Image Path:    {args.image}")
    print(f"Model:         {args.model}")
    print(f"Temperature:   {args.temperature}")
    print(f"Max Retries:   {args.max_retries}")
    print(f"Timeout:       {args.timeout}s")
    print("-" * 75)

    img_path = Path(args.image)
    if not img_path.is_file():
        print(f"[ERROR] Image file does not exist: {img_path}")
        return 1

    settings = ModelSettings(
        temperature=args.temperature,
        max_retries=args.max_retries,
        timeout_seconds=args.timeout,
    )

    # Check for GEMINI_API_KEY
    api_key = os.environ.get("GEMINI_API_KEY")
    use_mock = args.mock or not api_key

    if use_mock and not args.mock:
        print("[INFO] No GEMINI_API_KEY found in environment or .env.")
        print("[INFO] Falling back to MockVisionAdapter (Offline Oracle Simulation).")
        print("[TIP] To test live Google Gemini API, set GEMINI_API_KEY in your .env file.\n")
        adapter = MockVisionAdapter(
            strategy=MockStrategy.NOISY_WRAPPER,
            model_name=args.model,
            settings=settings,
            ground_truth_map={
                "sample": {
                    "can_see": True,
                    "answer": "YES",
                    "reason_code": "CLEAR",
                    "detailed_explanation": "Analytical proof: Unobstructed specular reflection at mirror surface.",
                }
            },
        )
    elif args.mock:
        print("[INFO] Running in explicit --mock simulation mode.\n")
        adapter = MockVisionAdapter(
            strategy=MockStrategy.NOISY_WRAPPER,
            model_name=args.model,
            settings=settings,
        )
    else:
        masked_key = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "***"
        print(f"[INFO] GEMINI_API_KEY detected: {masked_key}")
        print(f"[INFO] Initializing GeminiVisionAdapter with model '{args.model}'...\n")
        adapter = GeminiVisionAdapter(
            model_name=args.model,
            api_key=api_key,
            settings=settings,
        )

    # Execute adapter query via the standard provider-independent interface
    print("Sending diagram image and inspection question to vision model...")
    response: ModelResponse = adapter.ask(
        image_path=img_path,
        question=args.question,
        task_id="sample",
        task_type="TASK_A_VISIBILITY",
    )

    print("\n" + "=" * 75)
    print("  RAW MODEL API RESPONSE")
    print("=" * 75)
    if response.error:
        print(f"[API ERROR ENCOUNTERED]\n{response.error}")
    else:
        print(response.raw_text)

    print("\n" + "=" * 75)
    print("  RESPONSE METADATA & TELEMETRY")
    print("=" * 75)
    meta_summary = {
        "model_name": response.model_name,
        "model_version": response.model_version,
        "latency_seconds": round(response.latency_sec, 4),
        "retry_count": response.retry_count,
        "prompt_tokens": response.prompt_tokens,
        "completion_tokens": response.completion_tokens,
        "total_tokens": response.total_tokens,
        "finish_reason": response.finish_reason,
        "is_mock": response.is_mock,
        "timestamp": response.timestamp,
        "error": response.error,
        "raw_metadata": response.raw_metadata,
    }
    print(json.dumps(meta_summary, indent=2))
    print("=" * 75)

    print("\n[EXPLANATION OF OPERATION]")
    print(
        "1. Image Validation: Checked image existence and read binary bytes without tampering.\n"
        "2. Provider Interface: Called adapter.ask(image_path, question) through BaseVisionAdapter.\n"
        "3. Controlled Retries: Configured exponential backoff handles transient 429/503 faults.\n"
        "4. Telemetry Capture: Recorded precise latency, token counts, version, and finish reason.\n"
        "5. Extensibility: New providers (Claude, GPT-4o, Local VLMs) implement the identical\n"
        "   BaseVisionAdapter contract without requiring changes to scoring or parsing."
    )

    return 0 if not response.error else 1


if __name__ == "__main__":
    sys.exit(main())
