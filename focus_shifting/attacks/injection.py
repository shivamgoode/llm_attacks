"""Conversation attack injection strategies."""
from __future__ import annotations

import random
from copy import deepcopy


def inject_attack(
    conversation: list[dict[str, str]],
    attack_text: str,
    strategy: str,
    mid_position: int | None = None,
    rng: random.Random | None = None,
) -> list[dict[str, str]]:
    """Inject attack text into a conversation using the requested strategy."""
    messages = deepcopy(conversation)
    attack_message = {"role": "user", "content": attack_text}

    if strategy == "prefix":
        messages.insert(_first_user_index(messages), attack_message)
    elif strategy == "suffix":
        messages.append(attack_message)
    elif strategy == "mid_conversation":
        messages.insert(_mid_insertion_index(messages, mid_position), attack_message)
    elif strategy == "last_message":
        index = _last_user_index(messages)
        separator = "\n\n"
        messages[index]["content"] = f"{messages[index]['content']}{separator}{attack_text}"
    elif strategy == "random_position":
        generator = rng or random.Random()
        messages.insert(generator.randint(0, len(messages)), attack_message)
    else:
        raise ValueError(f"Unsupported injection strategy: {strategy}")

    return messages


def _first_user_index(messages: list[dict[str, str]]) -> int:
    for index, message in enumerate(messages):
        if message.get("role") == "user":
            return index
    return 0


def _last_user_index(messages: list[dict[str, str]]) -> int:
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == "user":
            return index
    return len(messages) - 1


def _mid_insertion_index(messages: list[dict[str, str]], mid_position: int | None) -> int:
    if len(messages) <= 1:
        return len(messages)
    if mid_position is not None:
        return max(1, min(int(mid_position), len(messages)))
    return max(1, len(messages) // 2)

