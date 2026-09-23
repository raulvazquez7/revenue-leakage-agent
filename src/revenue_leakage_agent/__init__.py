"""Stateful LangGraph agent that detects and corrects revenue leakage."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("revenue-leakage-agent")
except PackageNotFoundError:  # pragma: no cover - running from an uninstalled tree
    __version__ = "0.0.0"

__all__ = ["__version__"]
