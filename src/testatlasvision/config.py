"""Constants, supported languages and generation settings."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Optional

from .exceptions import UnsupportedLanguageError

DEFAULT_REPO = "Modularcomputing/AtlasVision"  # adapters + projector weights
DEFAULT_LLM = "NCAIR1/N-ATLaS"  # base language model (GATED)
DEFAULT_VISION = "google/siglip2-base-patch16-224"  # vision encoder

# Llama-3 style chat markers used by N-ATLaS. Do not change without retraining.
USER_HEADER = "<|start_header_id|>user<|end_header_id|>\n\n"
ASSIST_HEADER = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
EOT = "<|eot_id|>"


class Stage(str, Enum):
    """Which trained checkpoint to load from the AtlasVision repo."""

    CAPTIONER = "1"  # projector only, no LoRA
    VQA = "2"  # older VQA checkpoint
    VQA_V2B = "2b"  # recommended


DEFAULT_STAGE = Stage.VQA_V2B

LANGUAGES = {"en": "English", "ig": "Igbo", "yo": "Yoruba", "ha": "Hausa"}
_NAME_TO_CODE = {name.lower(): code for code, name in LANGUAGES.items()}


def normalize_language(value) -> str:
    """Accept 'ig', 'IG', 'Igbo', 'igbo' and return the 2-letter code."""
    key = str(value).strip().lower()
    if key in LANGUAGES:
        return key
    if key in _NAME_TO_CODE:
        return _NAME_TO_CODE[key]
    raise UnsupportedLanguageError(
        f"Unsupported language {value!r}. Supported: " + ", ".join(f"{c} ({n})" for c, n in LANGUAGES.items())
    )


@dataclass(frozen=True)
class GenerationConfig:
    """Decoding settings. `max_new_tokens=None` means 'use the method's default'."""

    max_new_tokens: Optional[int] = None
    temperature: float = 0.0  # 0 => greedy
    top_p: float = 0.9  # only used when temperature > 0
    repetition_penalty: float = 1.1

    def __post_init__(self):
        if self.max_new_tokens is not None and self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be > 0")

    def resolved(self, default_max_new_tokens: int) -> "GenerationConfig":
        """Return a copy with max_new_tokens filled in."""
        if self.max_new_tokens is not None:
            return self
        return replace(self, max_new_tokens=default_max_new_tokens)
