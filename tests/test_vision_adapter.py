"""
Unit and Integration Tests for Provider-Independent Vision Model Adapter.

Tests:
1. ModelSettings validation and defaults.
2. BaseVisionAdapter interface and image validation.
3. GeminiVisionAdapter initialization and secret loading.
4. Retry logic and exponential backoff on transient faults.
5. Error recording and latency telemetry.
6. MockVisionAdapter contract compliance.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image
from pydantic import ValidationError

from mirrorlab.evaluation.model_adapter import (
    PROVIDER_PRICING,
    GeminiVisionAdapter,
    MockStrategy,
    MockVisionAdapter,
    ModelSettings,
    OpenAIVisionAdapter,
    get_model_adapter,
)


@pytest.fixture
def sample_image(tmp_path: Path) -> Path:
    """Creates a temporary sample PNG diagram for testing."""
    img_path = tmp_path / "sample_diagram.png"
    img = Image.new("RGB", (100, 100), color=(20, 30, 40))
    img.save(img_path)
    return img_path


class TestModelSettings:
    """Tests configuration and hyperparameter validation."""

    def test_default_settings(self):
        settings = ModelSettings()
        assert settings.temperature == 0.0
        assert settings.max_output_tokens == 2048
        assert settings.timeout_seconds == 30.0
        assert settings.max_retries == 3
        assert settings.initial_backoff_seconds == 1.0
        assert settings.backoff_multiplier == 2.0
        assert settings.system_instruction is None

    def test_custom_valid_settings(self):
        settings = ModelSettings(
            temperature=0.5,
            max_output_tokens=1000,
            timeout_seconds=45.0,
            max_retries=5,
        )
        assert settings.temperature == 0.5
        assert settings.max_retries == 5

    def test_invalid_temperature_raises_validation_error(self):
        with pytest.raises(ValidationError):
            ModelSettings(temperature=-0.1)

    def test_invalid_retries_raises_validation_error(self):
        with pytest.raises(ValidationError):
            ModelSettings(max_retries=-1)


class TestBaseVisionAdapterContract:
    """Verifies interface contract and input validation."""

    def test_validate_image_missing_file_raises_not_found(self):
        adapter = MockVisionAdapter(strategy=MockStrategy.ORACLE)
        with pytest.raises(FileNotFoundError):
            adapter.validate_image("non_existent_file.png")

    def test_validate_image_existing_file_succeeds(self, sample_image: Path):
        adapter = MockVisionAdapter(strategy=MockStrategy.ORACLE)
        valid_path = adapter.validate_image(sample_image)
        assert valid_path == sample_image

    def test_generate_is_alias_for_ask(self, sample_image: Path):
        adapter = MockVisionAdapter(strategy=MockStrategy.ALL_YES)
        res_ask = adapter.ask(image_path=sample_image, question="Can camera see?")
        res_gen = adapter.generate(prompt="Can camera see?", image_path=sample_image)
        assert res_ask.raw_text == res_gen.raw_text
        assert res_ask.model_name == res_gen.model_name


class TestGeminiVisionAdapterLifecycle:
    """Tests Gemini adapter initialization, secret loading, and error handling."""

    def test_missing_api_key_returns_structured_error_response(self, sample_image: Path):
        # Explicitly pass api_key="" and patch os.environ
        with patch.dict("os.environ", {}, clear=True):
            adapter = GeminiVisionAdapter(model_name="gemini-2.5-flash", api_key="")
            response = adapter.ask(sample_image, "Is target visible?")
            assert response.error is not None
            assert "GEMINI_API_KEY is not set" in response.error
            assert response.raw_text == ""
            assert response.latency_sec >= 0.0

    def test_secret_loaded_from_environment(self):
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test_env_key_123"}):
            adapter = GeminiVisionAdapter(model_name="gemini-2.5-flash")
            assert adapter.api_key == "test_env_key_123"

    def test_explicit_api_key_overrides_environment(self):
        with patch.dict("os.environ", {"GEMINI_API_KEY": "env_key"}):
            adapter = GeminiVisionAdapter(
                model_name="gemini-2.5-flash", api_key="explicit_key"
            )
            assert adapter.api_key == "explicit_key"

    def test_is_retryable_error_identification(self):
        adapter = GeminiVisionAdapter(api_key="dummy")
        assert adapter._is_retryable_error(Exception("429 Too Many Requests")) is True
        assert adapter._is_retryable_error(Exception("503 Service Unavailable")) is True
        assert adapter._is_retryable_error(Exception("Resource Exhausted")) is True
        assert adapter._is_retryable_error(Exception("Request Timeout")) is True
        assert adapter._is_retryable_error(Exception("401 Unauthorized: Invalid API Key")) is False
        assert adapter._is_retryable_error(Exception("Invalid argument format")) is False


class TestRetryMechanismWithMocks:
    """Tests controlled retry loop and exponential backoff using mock clients."""

    def test_successful_retry_after_transient_failure(self, sample_image: Path):
        settings = ModelSettings(
            max_retries=2,
            initial_backoff_seconds=0.01,
            backoff_multiplier=1.5,
        )
        adapter = GeminiVisionAdapter(api_key="mock_key", settings=settings)

        # Mock client to fail once with 503 then succeed
        mock_response = MagicMock()
        mock_response.text = '{"answer": "YES"}'
        mock_response.model_version = "gemini-2.5-flash-v1"
        mock_response.usage_metadata = MagicMock(
            prompt_token_count=100,
            candidates_token_count=50,
            total_token_count=150,
        )
        mock_response.candidates = [MagicMock(finish_reason="STOP")]

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = [
            Exception("503 Service Unavailable"),
            mock_response,
        ]

        with patch.object(adapter, "_get_client", return_value=mock_client):
            res = adapter.ask(sample_image, "Is target visible?")

            assert res.raw_text == '{"answer": "YES"}'
            assert res.error is None
            assert res.retry_count == 1
            assert res.prompt_tokens == 100
            assert res.completion_tokens == 50
            assert res.finish_reason == "STOP"
            assert mock_client.models.generate_content.call_count == 2

    def test_non_retryable_error_aborts_immediately(self, sample_image: Path):
        settings = ModelSettings(max_retries=3, initial_backoff_seconds=0.01)
        adapter = GeminiVisionAdapter(api_key="mock_key", settings=settings)

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception(
            "400 Invalid argument: prompt too long"
        )

        with patch.object(adapter, "_get_client", return_value=mock_client):
            res = adapter.ask(sample_image, "Is target visible?")

            assert res.error is not None
            assert "Invalid argument" in res.error
            assert res.retry_count == 0
            assert mock_client.models.generate_content.call_count == 1

    def test_persistent_failure_captures_error_and_retries(self, sample_image: Path):
        settings = ModelSettings(max_retries=2, initial_backoff_seconds=0.01)
        adapter = GeminiVisionAdapter(api_key="mock_key", settings=settings)

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception(
            "429 Quota Exceeded"
        )

        with patch.object(adapter, "_get_client", return_value=mock_client):
            res = adapter.ask(sample_image, "Is target visible?")

            assert res.error is not None
            assert "429 Quota Exceeded" in res.error
            assert res.retry_count == 2
            assert mock_client.models.generate_content.call_count == 3


class TestOpenAIVisionAdapterLifecycle:
    """Tests OpenAI adapter initialization, secret loading, retry loop, and errors."""

    def test_missing_api_key_returns_structured_error_response(self, sample_image: Path):
        with patch.dict("os.environ", {}, clear=True):
            adapter = OpenAIVisionAdapter(model_name="gpt-4o", api_key="")
            response = adapter.ask(sample_image, "Is target visible?")
            assert response.error is not None
            assert "OPENAI_API_KEY is not set" in response.error
            assert response.raw_text == ""
            assert response.latency_sec >= 0.0

    def test_secret_loaded_from_environment(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test-key-456"}):
            adapter = OpenAIVisionAdapter(model_name="gpt-4o")
            assert adapter.api_key == "sk-test-key-456"

    def test_explicit_api_key_overrides_environment(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "env_key"}):
            adapter = OpenAIVisionAdapter(
                model_name="gpt-4o", api_key="explicit_sk_key"
            )
            assert adapter.api_key == "explicit_sk_key"

    def test_is_retryable_error_identification(self):
        adapter = OpenAIVisionAdapter(api_key="dummy")
        assert adapter._is_retryable_error(Exception("429 Too Many Requests")) is True
        assert adapter._is_retryable_error(Exception("503 Service Unavailable")) is True
        assert adapter._is_retryable_error(Exception("Request timed out")) is True
        assert adapter._is_retryable_error(Exception("401 Unauthorized")) is False

    def test_successful_retry_after_transient_failure(self, sample_image: Path):
        settings = ModelSettings(
            max_retries=2,
            initial_backoff_seconds=0.01,
            backoff_multiplier=1.5,
        )
        adapter = OpenAIVisionAdapter(api_key="mock_key", settings=settings)

        # Mock OpenAI response
        mock_choice = MagicMock()
        mock_choice.message.content = '{"answer": "YES"}'
        mock_choice.finish_reason = "stop"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.model = "gpt-4o-2024-08-06"
        mock_response.usage = MagicMock(
            prompt_tokens=300,
            completion_tokens=60,
            total_tokens=360,
        )

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            Exception("503 Service Unavailable"),
            mock_response,
        ]

        with patch.object(adapter, "_get_client", return_value=mock_client):
            res = adapter.ask(sample_image, "Is target visible?")

            assert res.raw_text == '{"answer": "YES"}'
            assert res.error is None
            assert res.retry_count == 1
            assert res.prompt_tokens == 300
            assert res.completion_tokens == 60
            assert res.finish_reason == "stop"
            assert mock_client.chat.completions.create.call_count == 2


class TestRealisticBaselinesAndPricing:
    """Tests realistic baseline simulations and pricing registries."""

    def test_provider_pricing_registry(self):
        assert "gemini-2.5-flash" in PROVIDER_PRICING
        assert "gpt-4o" in PROVIDER_PRICING
        assert PROVIDER_PRICING["gemini-2.5-flash"]["input_price_per_m"] == 0.075
        assert PROVIDER_PRICING["gpt-4o"]["input_price_per_m"] == 2.50

    def test_realistic_gpt4o_baseline_simulation(self, sample_image: Path):
        adapter = MockVisionAdapter(
            strategy=MockStrategy.REALISTIC_GPT4O_BASELINE,
            model_name="gpt-4o (simulated)",
        )
        gt = {
            "can_see": True,
            "reason_code": "CLEAR",
            "is_blocked": False,
        }
        adapter.set_ground_truth(gt, task_id="task_vis_001")
        resp = adapter.ask(
            sample_image,
            "Is target visible?",
            task_id="task_vis_001",
            task_type="TASK_A_VISIBILITY",
        )

        assert resp.error is None
        assert resp.is_mock is True
        assert resp.prompt_tokens == 340
        assert resp.completion_tokens == 58
        assert "can_see" in resp.raw_text

    def test_factory_resolves_mock_gpt4o(self):
        adapter = get_model_adapter("mock-gpt4o")
        assert isinstance(adapter, MockVisionAdapter)
        assert adapter.strategy == MockStrategy.REALISTIC_GPT4O_BASELINE

