from __future__ import annotations


class GckError(Exception):
    """
    Base error class for gujarati-codemix-kit.

    This is intentionally small: callers can catch GckError for SDK-level issues while still
    preserving backward-compat with built-in exception types via multiple inheritance.
    """


class InvalidConfigError(ValueError, GckError):
    """
    Raised when a user-provided config/argument is invalid.

    Subclasses ValueError for backward compatibility.
    """


class OptionalDependencyError(ImportError, GckError):
    """
    Raised when an optional dependency is required but not installed.
    """


class OfflinePolicyError(RuntimeError, GckError):
    """
    Raised when an operation would require network/remote downloads but offline-first policy blocks it.

    Subclasses RuntimeError for backward compatibility.
    """


class DownloadError(RuntimeError, GckError):
    """
    Raised when a dataset/model download fails.
    """


class IntegrationError(RuntimeError, GckError):
    """
    Raised for external service / integration failures (e.g., missing API keys).
    """

