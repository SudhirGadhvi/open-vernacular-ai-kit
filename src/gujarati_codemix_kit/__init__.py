"""Gujarati CodeMix Kit.

Public API is intentionally small. Prefer using `CodeMixPipeline` + `CodeMixConfig` for SDK usage.
"""

from .app_flows import (
    BatchProcessSummary,
    WhatsAppMessage,
    clean_whatsapp_chat_text,
    parse_whatsapp_export,
    process_csv_batch,
    process_jsonl_batch,
)
from .codemix_render import (
    analyze_codemix,
    analyze_codemix_with_config,
    render_codemix,
    render_codemix_with_config,
)
from .codeswitch import CodeSwitchMetrics, compute_code_switch_metrics
from .config import CodeMixConfig
from .dialects import (
    DialectDetection,
    DialectNormalizationResult,
    GujaratiDialect,
    detect_dialect,
    detect_dialect_from_tagged_tokens,
    detect_dialect_from_tokens,
    normalize_dialect_tokens,
)
from .normalize import normalize_text
from .pipeline import CodeMixPipeline, CodeMixPipelineResult

__all__ = [
    "CodeMixConfig",
    "CodeMixPipeline",
    "CodeMixPipelineResult",
    "analyze_codemix",
    "analyze_codemix_with_config",
    "CodeSwitchMetrics",
    "compute_code_switch_metrics",
    "DialectDetection",
    "DialectNormalizationResult",
    "GujaratiDialect",
    "detect_dialect",
    "detect_dialect_from_tokens",
    "detect_dialect_from_tagged_tokens",
    "normalize_dialect_tokens",
    "normalize_text",
    "render_codemix",
    "render_codemix_with_config",
    "BatchProcessSummary",
    "WhatsAppMessage",
    "clean_whatsapp_chat_text",
    "parse_whatsapp_export",
    "process_csv_batch",
    "process_jsonl_batch",
]

__version__ = "0.4.0"
 
