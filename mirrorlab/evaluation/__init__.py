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
    BaseVLMAdapter,
    GeminiModelAdapter,
    MockStrategy,
    MockVLMAdapter,
    ModelResponse,
    OpenAIModelAdapter,
    get_model_adapter,
)
from mirrorlab.evaluation.structured_parser import ParsedAnswer, StructuredParser

__all__ = [
    "AnthropicModelAdapter",
    "AuditCheckResult",
    "BaseVLMAdapter",
    "BenchmarkEvaluator",
    "BenchmarkMetricsCalculator",
    "ComprehensiveAuditReport",
    "DatasetAuditor",
    "DatasetSplitter",
    "EvaluationMetrics",
    "GeminiModelAdapter",
    "MockStrategy",
    "MockVLMAdapter",
    "ModelResponse",
    "OpenAIModelAdapter",
    "ParsedAnswer",
    "RunConfig",
    "StructuredParser",
    "get_model_adapter",
    "render_audit_report_markdown",
]
