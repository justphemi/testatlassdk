"""Plain data types. No torch imports, so the API layer can import these on any machine."""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass(frozen=True)
class Turn:
    """One finished question/answer exchange. Always stored in English."""

    question: str
    answer: str


@dataclass
class Session:
    """Conversation memory for follow-up questions about one image.

    The client itself is stateless; pass a Session to `ask`/`chat` to get follow-ups.
    Sessions serialise to plain dicts so the API can keep them in Redis/DB.
    """

    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    history: List[Turn] = field(default_factory=list)

    def reset(self) -> None:
        self.history.clear()

    def to_dict(self) -> Dict[str, Any]:
        return {"session_id": self.session_id, "history": [asdict(t) for t in self.history]}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Session":
        return cls(
            session_id=data.get("session_id") or uuid.uuid4().hex,
            history=[Turn(**t) for t in data.get("history", [])],
        )


@dataclass(frozen=True)
class ChatResult:
    """Result of a (possibly translated) question about an image."""

    answer: str  # in `language`
    language: str  # code of the language the user asked in, e.g. "ig"
    english_question: str  # what the vision model actually saw
    english_answer: str  # raw vision-model answer before translating back

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModelInfo:
    """Runtime facts about a loaded client. Handy for a /health endpoint."""

    stage: str
    llm: str
    vision: str
    repo: str
    device: str
    dtype: str
    load_in_4bit: bool
    supported_languages: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
