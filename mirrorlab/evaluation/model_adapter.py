"""
Provider-Independent Vision Model Adapter for MIRRORLAB.

Provides a unified, resilient interface to interact with Vision-Language Models:
- Accepts an image path and a text prompt/question.
- Communicates with configured vision providers (starting with Google Gemini).
- Manages credentials loaded strictly from environment variables or .env.
- Supports configurable hyperparameters (temperature, max tokens, timeout).
- Implements exponential backoff retry logic for transient API faults.
- Captures granular response metadata: latency, token counts, model version,
  finish reasons, retry counts, and API errors.
"""

from __future__ import annotations

import base64
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from enum import Enum
import json
import os
from pathlib import Path
import time
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Ensure environment variables are loaded
load_dotenv()


class ModelSettings(BaseModel):
    """Configurable model hyperparameters and execution settings."""

    temperature: float = Field(default=0.0, ge=0.0, le=2.0, description="Sampling temperature (0.0 for deterministic benchmark)")
    max_output_tokens: int = Field(default=2048, ge=1, description="Maximum completion tokens")
    timeout_seconds: float = Field(default=30.0, gt=0.0, description="HTTP/API call timeout in seconds")
    max_retries: int = Field(default=3, ge=0, description="Maximum retry attempts on transient failures")
    initial_backoff_seconds: float = Field(default=1.0, gt=0.0, description="Initial backoff delay in seconds")
    backoff_multiplier: float = Field(default=2.0, ge=1.0, description="Multiplier for exponential backoff")
    system_instruction: str | None = Field(default=None, description="Optional system prompt")


class ModelResponse(BaseModel):
    """Standardized response container capturing raw text, tokens, timing, and errors."""

    raw_text: str
    model_name: str
    model_version: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    latency_sec: float = 0.0
    retry_count: int = 0
    finish_reason: str | None = None
    timestamp: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )
    error: str | None = None
    is_mock: bool = False
    raw_metadata: dict[str, Any] = Field(default_factory=dict)


class MockStrategy(str, Enum):
    """Supported simulation strategies for offline mock runner."""

    ORACLE = "oracle"  # Simulates 100% correct vision reasoning using reference truth
    ALL_YES = "all_yes"  # Predicts YES/visible for all tasks (50% guessing baseline)
    ALL_NO = "all_no"  # Predicts NO/blocked for all tasks
    RANDOM = "random"  # Seeded pseudo-random guesser
    NOISY_WRAPPER = "noisy_wrapper"  # Surrounds correct answer with conversational commentary & markdown
    FORMAT_ERROR = "format_error"  # Returns invalid/non-JSON text to test parser resilience


class BaseVisionAdapter(ABC):
    """
    Common provider-independent abstract base class for all vision models.
    Enables adding future providers (Claude, GPT, local VLMs) without altering the scoring engine.
    """

    def __init__(
        self,
        model_name: str,
        settings: ModelSettings | None = None,
    ):
        self.model_name = model_name
        self.settings = settings or ModelSettings()

    @property
    def temperature(self) -> float:
        return self.settings.temperature

    def validate_image(self, image_path: Path | str) -> Path:
        """Validates that the image file exists and is a readable file."""
        img_p = Path(image_path)
        if not img_p.is_file():
            raise FileNotFoundError(f"Diagram image not found: {img_p}")
        return img_p

    @abstractmethod
    def ask(
        self,
        image_path: Path | str,
        question: str,
        **kwargs: Any,
    ) -> ModelResponse:
        """
        Accept an image path and text question, query the vision model, and return ModelResponse.

        Parameters
        ----------
        image_path:
            Path to the diagram image (PNG, JPEG, etc.).
        question:
            Text prompt or question describing the inspection task.

        Returns
        -------
        ModelResponse:
            Standardized container with verbatim output, metadata, latency, and error state.
        """

    def generate(
        self,
        prompt: str,
        image_path: Path | str,
        **kwargs: Any,
    ) -> ModelResponse:
        """
        Backward-compatible alias for the benchmark evaluation runner.
        Maps (prompt, image_path) -> ask(image_path, question=prompt).
        """
        return self.ask(image_path=image_path, question=prompt, **kwargs)


# Backward compatibility alias
BaseVLMAdapter = BaseVisionAdapter


class GeminiVisionAdapter(BaseVisionAdapter):
    """
    Vision model adapter for Google Gemini via the official google-genai SDK.
    Supports gemini-2.5-flash, gemini-2.5-pro, gemini-1.5-pro, etc.
    """

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        api_key: str | None = None,
        settings: ModelSettings | None = None,
        temperature: float | None = None,
    ):
        if settings is None:
            settings = ModelSettings()
        if temperature is not None:
            settings.temperature = temperature

        super().__init__(model_name=model_name, settings=settings)
        # Secrets loaded strictly from environment or explicit parameter
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            if not self.api_key:
                raise ValueError(
                    "GEMINI_API_KEY is not set. Please set the GEMINI_API_KEY environment variable "
                    "or pass api_key to the adapter. See .env.example."
                )
            from google import genai

            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def _is_retryable_error(self, err: Exception) -> bool:
        """Identifies transient network and server errors that warrant a retry."""
        err_msg = str(err).lower()
        retryable_keywords = [
            "429", "rate limit", "quota", "resource exhausted",
            "503", "unavailable", "500", "internal server error",
            "timeout", "deadline exceeded", "connection reset", "econnreset"
        ]
        return any(kw in err_msg for kw in retryable_keywords)

    def ask(
        self,
        image_path: Path | str,
        question: str,
        **kwargs: Any,
    ) -> ModelResponse:
        img_p = self.validate_image(image_path)

        with open(img_p, "rb") as f:
            img_bytes = f.read()

        t0 = time.perf_counter()
        retries = 0
        backoff = self.settings.initial_backoff_seconds

        # Check credentials early
        if not self.api_key:
            latency = time.perf_counter() - t0
            return ModelResponse(
                raw_text="",
                model_name=self.model_name,
                latency_sec=latency,
                error="GEMINI_API_KEY is not set. Provide api_key or set environment variable.",
                is_mock=False,
            )

        try:
            client = self._get_client()
            from google.genai import types

            image_part = types.Part.from_bytes(
                data=img_bytes,
                mime_type="image/png",
            )

            config = types.GenerateContentConfig(
                temperature=self.settings.temperature,
                max_output_tokens=self.settings.max_output_tokens,
                system_instruction=self.settings.system_instruction,
            )
        except Exception as e:  # noqa: BLE001
            latency = time.perf_counter() - t0
            return ModelResponse(
                raw_text="",
                model_name=self.model_name,
                latency_sec=latency,
                error=f"Client initialization error: {e}",
                is_mock=False,
            )

        # Retry loop with exponential backoff
        last_error_msg: str | None = None
        for attempt in range(self.settings.max_retries + 1):
            try:
                response = client.models.generate_content(
                    model=self.model_name,
                    contents=[image_part, question],
                    config=config,
                )
                latency = time.perf_counter() - t0

                raw_text = response.text or ""
                prompt_tokens = None
                completion_tokens = None
                total_tokens = None
                raw_meta: dict[str, Any] = {}

                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    prompt_tokens = getattr(
                        response.usage_metadata, "prompt_token_count", None
                    )
                    completion_tokens = getattr(
                        response.usage_metadata, "candidates_token_count", None
                    )
                    total_tokens = getattr(
                        response.usage_metadata, "total_token_count", None
                    )
                    raw_meta["prompt_tokens"] = prompt_tokens
                    raw_meta["completion_tokens"] = completion_tokens
                    raw_meta["total_tokens"] = total_tokens

                finish_reason = None
                if hasattr(response, "candidates") and response.candidates:
                    first_cand = response.candidates[0]
                    finish_reason = str(getattr(first_cand, "finish_reason", "STOP"))
                    raw_meta["finish_reason"] = finish_reason

                model_version = getattr(response, "model_version", None) or self.model_name

                return ModelResponse(
                    raw_text=raw_text,
                    model_name=self.model_name,
                    model_version=model_version,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    latency_sec=latency,
                    retry_count=retries,
                    finish_reason=finish_reason,
                    is_mock=False,
                    raw_metadata=raw_meta,
                )

            except Exception as e:  # noqa: BLE001
                last_error_msg = str(e)
                if attempt < self.settings.max_retries and self._is_retryable_error(e):
                    retries += 1
                    time.sleep(backoff)
                    backoff *= self.settings.backoff_multiplier
                    continue
                break

        latency = time.perf_counter() - t0
        return ModelResponse(
            raw_text="",
            model_name=self.model_name,
            latency_sec=latency,
            retry_count=retries,
            error=last_error_msg or "Unknown generation error",
            is_mock=False,
        )


# Backward compatibility alias
GeminiModelAdapter = GeminiVisionAdapter


class AnthropicVisionAdapter(BaseVisionAdapter):
    """
    Adapter for Anthropic Claude models (e.g. claude-3-7-sonnet, claude-3-5-sonnet).
    """

    def __init__(
        self,
        model_name: str = "claude-3-7-sonnet-20250219",
        api_key: str | None = None,
        settings: ModelSettings | None = None,
        temperature: float | None = None,
    ):
        if settings is None:
            settings = ModelSettings()
        if temperature is not None:
            settings.temperature = temperature
        super().__init__(model_name=model_name, settings=settings)
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            if not self.api_key:
                raise ValueError("ANTHROPIC_API_KEY is not set.")
            try:
                import anthropic

                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "The anthropic package is required for AnthropicVisionAdapter. Install with 'pip install anthropic'."
                )
        return self._client

    def ask(
        self,
        image_path: Path | str,
        question: str,
        **kwargs: Any,
    ) -> ModelResponse:
        img_p = self.validate_image(image_path)
        with open(img_p, "rb") as f:
            b64_data = base64.b64encode(f.read()).decode("utf-8")

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": b64_data,
                        },
                    },
                    {"type": "text", "text": question},
                ],
            }
        ]

        t0 = time.perf_counter()
        try:
            client = self._get_client()
            res = client.messages.create(
                model=self.model_name,
                messages=messages,
                temperature=self.settings.temperature,
                max_tokens=self.settings.max_output_tokens,
            )
            latency = time.perf_counter() - t0
            raw_text = res.content[0].text if res.content else ""
            return ModelResponse(
                raw_text=raw_text,
                model_name=self.model_name,
                prompt_tokens=getattr(res.usage, "input_tokens", None),
                completion_tokens=getattr(res.usage, "output_tokens", None),
                total_tokens=(
                    getattr(res.usage, "input_tokens", 0)
                    + getattr(res.usage, "output_tokens", 0)
                ),
                latency_sec=latency,
                is_mock=False,
            )
        except Exception as e:  # noqa: BLE001
            latency = time.perf_counter() - t0
            return ModelResponse(
                raw_text="",
                model_name=self.model_name,
                latency_sec=latency,
                error=str(e),
                is_mock=False,
            )


# Backward compatibility alias
AnthropicModelAdapter = AnthropicVisionAdapter


class OpenAIVisionAdapter(BaseVisionAdapter):
    """
    Adapter for OpenAI GPT models (e.g. gpt-4o, gpt-4o-mini).
    """

    def __init__(
        self,
        model_name: str = "gpt-4o",
        api_key: str | None = None,
        settings: ModelSettings | None = None,
        temperature: float | None = None,
    ):
        if settings is None:
            settings = ModelSettings()
        if temperature is not None:
            settings.temperature = temperature
        super().__init__(model_name=model_name, settings=settings)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            if not self.api_key:
                raise ValueError("OPENAI_API_KEY is not set.")
            try:
                import openai

                self._client = openai.OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "The openai package is required for OpenAIVisionAdapter. Install with 'pip install openai'."
                )
        return self._client

    def ask(
        self,
        image_path: Path | str,
        question: str,
        **kwargs: Any,
    ) -> ModelResponse:
        img_p = self.validate_image(image_path)
        with open(img_p, "rb") as f:
            b64_data = base64.b64encode(f.read()).decode("utf-8")

        data_url = f"data:image/png;base64,{b64_data}"
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ]

        t0 = time.perf_counter()
        try:
            client = self._get_client()
            res = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.settings.temperature,
                max_tokens=self.settings.max_output_tokens,
            )
            latency = time.perf_counter() - t0
            raw_text = res.choices[0].message.content or ""
            return ModelResponse(
                raw_text=raw_text,
                model_name=self.model_name,
                prompt_tokens=getattr(res.usage, "prompt_tokens", None),
                completion_tokens=getattr(res.usage, "completion_tokens", None),
                total_tokens=getattr(res.usage, "total_tokens", None),
                latency_sec=latency,
                is_mock=False,
            )
        except Exception as e:  # noqa: BLE001
            latency = time.perf_counter() - t0
            return ModelResponse(
                raw_text="",
                model_name=self.model_name,
                latency_sec=latency,
                error=str(e),
                is_mock=False,
            )


# Backward compatibility alias
OpenAIModelAdapter = OpenAIVisionAdapter


class MockVisionAdapter(BaseVisionAdapter):
    """
    Offline Mock Vision Runner for testing pipelines and benchmark verification.
    Provides controllable simulation strategies without making external network calls.
    """

    def __init__(
        self,
        strategy: MockStrategy = MockStrategy.ORACLE,
        ground_truth_map: dict[str, dict[str, Any]] | None = None,
        seed: int = 42,
        model_name: str | None = None,
        settings: ModelSettings | None = None,
    ):
        name = model_name or f"mock-{strategy.value}"
        super().__init__(model_name=name, settings=settings)
        self.strategy = strategy
        self.ground_truth_map = ground_truth_map or {}
        self.seed = seed
        self._counter = 0

    def ask(
        self,
        image_path: Path | str,
        question: str,
        **kwargs: Any,
    ) -> ModelResponse:
        self.validate_image(image_path)
        t0 = time.perf_counter()
        task_id = kwargs.get("task_id", "")
        task_type = kwargs.get("task_type", "TASK_A_VISIBILITY")

        gt = self.ground_truth_map.get(task_id, {})

        if self.strategy == MockStrategy.ORACLE:
            text = self._generate_oracle(task_type, gt)
        elif self.strategy == MockStrategy.ALL_YES:
            text = self._generate_all_yes(task_type)
        elif self.strategy == MockStrategy.ALL_NO:
            text = self._generate_all_no(task_type)
        elif self.strategy == MockStrategy.RANDOM:
            text = self._generate_random(task_type)
        elif self.strategy == MockStrategy.NOISY_WRAPPER:
            base_json = self._generate_oracle(task_type, gt)
            text = (
                f"Hello! I am an AI assistant analyzing this diagram carefully.\n\n"
                f"Here is my step-by-step physical analysis:\n"
                f"```json\n{base_json}\n```\n\n"
                f"I hope this clear derivation is helpful for the industrial inspection team!"
            )
        elif self.strategy == MockStrategy.FORMAT_ERROR:
            text = (
                "Sorry, I cannot determine the exact specular reflection angle. "
                "The diagram has too many obstacles and I cannot answer."
            )
        else:
            text = self._generate_oracle(task_type, gt)

        self._counter += 1
        latency = time.perf_counter() - t0

        return ModelResponse(
            raw_text=text,
            model_name=self.model_name,
            model_version=f"{self.model_name}-v1",
            prompt_tokens=256,
            completion_tokens=64,
            total_tokens=320,
            latency_sec=latency,
            retry_count=0,
            finish_reason="STOP",
            is_mock=True,
            raw_metadata={"strategy": self.strategy.value, "task_id": task_id},
        )

    def _generate_oracle(self, task_type: str, gt: dict[str, Any]) -> str:
        if task_type in ("TASK_A_VISIBILITY", "TASK_D_COUNTERFACTUAL"):
            can_see = gt.get("can_see", True)
            ans = gt.get("answer", "YES" if can_see else "NO")
            reason = gt.get("reason_code", "CLEAR" if can_see else "OUTSIDE_FOV")
            data = {
                "can_see": can_see,
                "answer": ans,
                "reason": reason,
                "explanation": gt.get(
                    "detailed_explanation", "Analytical geometric proof."
                ),
            }
        elif task_type == "TASK_B_OBSTRUCTION":
            is_blocked = gt.get("is_blocked", True)
            ans = "YES" if is_blocked else "NO"
            blocker = gt.get("blocking_obstacle_label") or gt.get("blocker_label") or ("O1" if is_blocked else "NONE")
            segment = gt.get("blocked_segment") or ("CAMERA_TO_MIRROR" if is_blocked else "NONE")
            data = {
                "is_blocked": is_blocked,
                "answer": ans,
                "blocker_label": blocker,
                "blocked_segment": segment,
                "explanation": gt.get(
                    "detailed_explanation", "Obstacle intersection proof."
                ),
            }
        elif task_type == "TASK_C_REPAIR":
            winner = gt.get("correct_candidate", "A")
            data = {
                "selected_candidate": winner,
                "answer": winner,
                "explanation": gt.get("selection_reason", "Candidate restores view."),
            }
        else:
            data = {"answer": "YES", "explanation": "Default response"}

        return json.dumps(data, indent=2)

    def _generate_all_yes(self, task_type: str) -> str:
        if task_type in ("TASK_A_VISIBILITY", "TASK_D_COUNTERFACTUAL"):
            data = {
                "can_see": True,
                "answer": "YES",
                "reason": "CLEAR",
                "explanation": "Specular path is clear.",
            }
        elif task_type == "TASK_B_OBSTRUCTION":
            data = {
                "is_blocked": True,
                "answer": "YES",
                "blocker_label": "O1",
                "blocked_segment": "CAMERA_TO_MIRROR",
                "explanation": "Path is blocked by barrier.",
            }
        elif task_type == "TASK_C_REPAIR":
            data = {
                "selected_candidate": "A",
                "answer": "A",
                "explanation": "Mirror A works.",
            }
        else:
            data = {"answer": "YES"}
        return json.dumps(data, indent=2)

    def _generate_all_no(self, task_type: str) -> str:
        if task_type in ("TASK_A_VISIBILITY", "TASK_D_COUNTERFACTUAL"):
            data = {
                "can_see": False,
                "answer": "NO",
                "reason": "OUTSIDE_FOV",
                "explanation": "Outside field of view.",
            }
        elif task_type == "TASK_B_OBSTRUCTION":
            data = {
                "is_blocked": False,
                "answer": "NO",
                "blocker_label": "NONE",
                "blocked_segment": "NONE",
                "explanation": "No obstacle blocks path.",
            }
        elif task_type == "TASK_C_REPAIR":
            data = {
                "selected_candidate": "B",
                "answer": "B",
                "explanation": "Candidate B chosen.",
            }
        else:
            data = {"answer": "NO"}
        return json.dumps(data, indent=2)

    def _generate_random(self, task_type: str) -> str:
        is_yes = (self._counter % 2) == 0
        if task_type in ("TASK_A_VISIBILITY", "TASK_D_COUNTERFACTUAL"):
            data = {
                "can_see": is_yes,
                "answer": "YES" if is_yes else "NO",
                "reason": "CLEAR" if is_yes else "MIRROR_MISS",
                "explanation": "Random baseline guess.",
            }
        elif task_type == "TASK_B_OBSTRUCTION":
            data = {
                "is_blocked": is_yes,
                "answer": "YES" if is_yes else "NO",
                "blocker_label": "O1" if is_yes else "NONE",
                "blocked_segment": "CAMERA_TO_MIRROR" if is_yes else "NONE",
                "explanation": "Random baseline guess.",
            }
        elif task_type == "TASK_C_REPAIR":
            cand = ["A", "B", "C"][self._counter % 3]
            data = {
                "selected_candidate": cand,
                "answer": cand,
                "explanation": "Random baseline guess.",
            }
        else:
            data = {"answer": "YES"}
        return json.dumps(data, indent=2)


# Backward compatibility alias
MockVLMAdapter = MockVisionAdapter


def get_model_adapter(
    model_name: str,
    api_key: str | None = None,
    temperature: float = 0.0,
    ground_truth_map: dict[str, dict[str, Any]] | None = None,
    mock_strategy: MockStrategy | str | None = None,
    settings: ModelSettings | None = None,
) -> BaseVisionAdapter:
    """
    Factory function to instantiate a provider-independent vision model adapter.

    Examples:
    - get_model_adapter("gemini-2.5-flash")
    - get_model_adapter("claude-3-7-sonnet")
    - get_model_adapter("gpt-4o")
    - get_model_adapter("mock-oracle", ground_truth_map=gt_dict)
    - get_model_adapter("mock-all-yes")
    """
    model_lower = model_name.lower()

    if settings is None:
        settings = ModelSettings(temperature=temperature)

    if model_lower.startswith("mock"):
        strat = MockStrategy.ORACLE
        if mock_strategy:
            strat = (
                MockStrategy(mock_strategy)
                if isinstance(mock_strategy, str)
                else mock_strategy
            )
        elif "noisy" in model_lower:
            strat = MockStrategy.NOISY_WRAPPER
        elif "all_yes" in model_lower or "all-yes" in model_lower:
            strat = MockStrategy.ALL_YES
        elif "all_no" in model_lower or "all-no" in model_lower:
            strat = MockStrategy.ALL_NO
        elif "random" in model_lower:
            strat = MockStrategy.RANDOM
        elif "format" in model_lower or "error" in model_lower:
            strat = MockStrategy.FORMAT_ERROR

        return MockVisionAdapter(
            strategy=strat,
            ground_truth_map=ground_truth_map,
            model_name=model_name,
            settings=settings,
        )

    if "gemini" in model_lower:
        return GeminiVisionAdapter(
            model_name=model_name,
            api_key=api_key,
            settings=settings,
        )

    if "claude" in model_lower or "anthropic" in model_lower:
        return AnthropicVisionAdapter(
            model_name=model_name,
            api_key=api_key,
            settings=settings,
        )

    if "gpt" in model_lower or "openai" in model_lower:
        return OpenAIVisionAdapter(
            model_name=model_name,
            api_key=api_key,
            settings=settings,
        )

    # Fallback to Mock
    return MockVisionAdapter(
        strategy=MockStrategy.ORACLE,
        ground_truth_map=ground_truth_map,
        model_name=model_name,
        settings=settings,
    )
