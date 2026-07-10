"""Unified model interface for conversation-level benchmarks."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol


Conversation = list[dict[str, str]]


class BaseLLM(ABC):
    """Base interface for LLMs that accept full conversations."""

    name: str

    @abstractmethod
    def generate(self, conversation: Conversation) -> tuple[str, float]:
        """Generate a response for the given conversation."""


class PromptModel(Protocol):
    name: str

    def generate(self, prompt: str) -> tuple[str, float]:
        """Generate from a single rendered prompt."""


class PromptModelAdapter(BaseLLM):
    """Adapt the repo's existing single-prompt models to the conversation interface."""

    def __init__(self, model: PromptModel) -> None:
        self._model = model
        self.name = model.name

    def generate(self, conversation: Conversation) -> tuple[str, float]:
        return self._model.generate(render_conversation(conversation))


def render_conversation(conversation: Conversation) -> str:
    """Render role-tagged messages for prompt-only adapters."""
    return "\n\n".join(
        f"{message.get('role', 'user').upper()}:\n{message.get('content', '')}"
        for message in conversation
    )

