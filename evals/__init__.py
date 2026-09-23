"""Live-model trajectory evals for the revenue leakage agent.

Unlike ``tests/``, everything under ``evals/`` calls the real OpenAI models
configured in ``.env`` (no scripted chat models). These are trajectory
evals, not unit tests: they assert on tool-call order, forbidden tools, and
regexes over the model's own prose, which makes them flaky by nature and
unsuitable for CI. See ``evals/README.md`` for how and when to run them.
"""

from __future__ import annotations
