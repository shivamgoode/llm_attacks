"""Model selection for focus shifting runs."""
from __future__ import annotations

import random

from focus_shifting.models.base import BaseLLM, Conversation, PromptModelAdapter


class MockFocusModel(BaseLLM):
    """Dry-run model that alternates between preserving and losing constraints."""

    def __init__(self, name: str, seed: int, preserve_probability: float) -> None:
        self.name = name
        self._rng = random.Random(seed)
        self._preserve_probability = preserve_probability

    def generate(self, conversation: Conversation) -> tuple[str, float]:
        latency = round(self._rng.uniform(0.1, 1.5), 3)
        if self._rng.random() < self._preserve_probability:
            return "this review mentions alpha and ends with thank you", latency
        return "Hello", latency


def load_models(dry_run: bool, seed: int) -> list[BaseLLM]:
    """Load dry-run mocks or wrap the repo's configured API models."""
    if dry_run:
        return [
            MockFocusModel("Mock-Robust", seed, 0.8),
            MockFocusModel("Mock-Mixed", seed + 1, 0.5),
            MockFocusModel("Mock-Fragile", seed + 2, 0.2),
        ]

    from models import MODELS

    return [PromptModelAdapter(model) for model in MODELS]


def select_models(models: list[BaseLLM], requested: list[str], skipped: list[str]) -> list[BaseLLM]:
    """Select configured models by display name."""
    available = {model.name.lower(): model for model in models}
    requested_names = {name.lower() for name in requested}
    selected = list(models)

    if "all" not in requested_names:
        missing = sorted(name for name in requested if name.lower() not in available)
        if missing:
            available_names = ", ".join(model.name for model in models)
            raise ValueError(f"Unknown model(s): {', '.join(missing)}. Available models: {available_names}")
        selected = [available[name.lower()] for name in requested]

    if skipped:
        skip_set = {model.lower() for model in skipped}
        selected = [model for model in selected if model.name.lower() not in skip_set]

    return selected

