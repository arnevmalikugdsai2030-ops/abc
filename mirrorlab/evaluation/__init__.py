"""
Evaluation and Auditing Package for MIRRORLAB.
"""

from mirrorlab.evaluation.dataset_auditor import (
    AuditCheckResult,
    ComprehensiveAuditReport,
    DatasetAuditor,
    DatasetSplitter,
    render_audit_report_markdown,
)
from mirrorlab.evaluation.evaluator import BenchmarkEvaluator, RunConfig
from mirrorlab.evaluation.metrics import BenchmarkMetricsCalculator, EvaluationMetrics
from mirrorlab.evaluation.model_adapter import (
    AnthropicModelAdapter,
    AnthropicVisionAdapter,
    BaseVisionAdapter,
    BaseVLMAdapter,
    GeminiModelAdapter,
    GeminiVisionAdapter,
    MockStrategy,
    MockVisionAdapter,
    MockVLMAdapter,
    ModelResponse,
    ModelSettings,
    OpenAIModelAdapter,
    OpenAIVisionAdapter,
    get_model_adapter,
)
from mirrorlab.evaluation.structured_parser import ParsedAnswer, StructuredParser

__all__ = [
    "AnthropicModelAdapter",
    "AnthropicVisionAdapter",
    "AuditCheckResult",
    "BaseVLMAdapter",
    "BaseVisionAdapter",
    "BenchmarkEvaluator",
    "BenchmarkMetricsCalculator",
    "ComprehensiveAuditReport",
    "DatasetAuditor",
    "DatasetSplitter",
    "EvaluationMetrics",
    "GeminiModelAdapter",
    "GeminiVisionAdapter",
    "MockStrategy",
    "MockVLMAdapter",
    "MockVisionAdapter",
    "ModelResponse",
    "ModelSettings",
    "OpenAIModelAdapter",
    "OpenAIVisionAdapter",
    "ParsedAnswer",
    "RunConfig",
    "StructuredParser",
    "get_model_adapter",
    "render_audit_report_markdown",
]
