"""Gujarati CodeMix Kit.

Public API is intentionally small. Prefer using `normalize_text()` and `render_codemix()`.
"""

from .codemix_render import analyze_codemix, render_codemix
from .normalize import normalize_text

__all__ = ["analyze_codemix", "normalize_text", "render_codemix"]

__version__ = "0.1.0"
 
