"""Attack interfaces for focus shifting experiments."""
from __future__ import annotations

from abc import ABC, abstractmethod


class PromptAttack(ABC):
    """Base class for attacks that try to shift focus away from earlier constraints."""

    name: str = "base"

    @abstractmethod
    def generate_attack(self, strength: str = "medium") -> str:
        """Return the attack text for a configured strength."""


class StaticPromptAttack(PromptAttack):
    """Simple attack implementation backed by strength-specific prompt text."""

    def __init__(self, name: str, prompts: dict[str, str]) -> None:
        self.name = name
        self._prompts = prompts

    def generate_attack(self, strength: str = "medium") -> str:
        return self._prompts.get(strength, self._prompts.get("medium", next(iter(self._prompts.values()))))

