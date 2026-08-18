from __future__ import annotations

import logging
import time
from typing import Callable, Iterable, Type

LOGGER = logging.getLogger(__name__)


class AppError(Exception):
    """Base class for app-specific errors."""


class OllamaNotRunningError(AppError):
    """Raised when the local Ollama service is unavailable."""


class CheapSharkError(AppError):
    """Raised when CheapShark API requests fail."""


class VectorStoreError(AppError):
    """Raised when vector store operations fail."""


class GameDataError(AppError):
    """Raised when local game data cannot be loaded."""


def retry_with_backoff(
    func: Callable[[], object],
    *,
    exceptions: Iterable[Type[BaseException]],
    max_retries: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 4.0,
) -> object:
    """Retry a callable with exponential backoff."""
    last_error: BaseException | None = None
    for attempt in range(max_retries):
        try:
            return func()
        except tuple(exceptions) as exc:  # type: ignore[arg-type]
            last_error = exc
            if attempt >= max_retries - 1:
                break
            delay = min(max_delay, base_delay * (2**attempt))
            LOGGER.warning("Retrying after error", extra={"error": str(exc), "delay": delay})
            time.sleep(delay)
    if last_error:
        raise last_error
    raise RuntimeError("Retry failed without captured error")


def friendly_error_message(exc: BaseException) -> str:
    if isinstance(exc, OllamaNotRunningError):
        return "Ollama is not running. Start Ollama with: ollama serve"
    if isinstance(exc, VectorStoreError):
        return "Vector search is unavailable. Falling back to keyword search."
    if isinstance(exc, CheapSharkError):
        return "CheapShark is unavailable. Please try again in a moment."
    if isinstance(exc, GameDataError):
        return "Game database is missing. Run the dataset script to generate it."
    return f"Unexpected error: {exc}"
