"""Prompt construction. Pure string functions: testable without a GPU."""

from __future__ import annotations

from typing import Sequence

from .config import ASSIST_HEADER, EOT, USER_HEADER
from .types import Turn


def build_conversation_text(history: Sequence[Turn], question: str) -> str:
    """Text that follows the image embeddings.

    Layout in embedding space is: [BOS + user header] [image embeddings] [this text].
    The first user header lives in the prefix, so the first turn's question carries no
    header of its own; every later turn does. This mirrors the training format.
    """
    text = ""
    for i, turn in enumerate(history):
        text += turn.question if i == 0 else USER_HEADER + turn.question
        text += ASSIST_HEADER + turn.answer + EOT
    text += (question if not history else USER_HEADER + question) + ASSIST_HEADER
    return text


def build_text_prompt(prompt: str) -> str:
    """Plain (image-free) N-ATLaS chat prompt."""
    return USER_HEADER + prompt + ASSIST_HEADER


def build_translation_prompt(text: str, target_name: str, source_name: str) -> str:
    return f"Translate this {source_name} text to {target_name}. Reply with only the translation.\n\n{text}"
