"""Provider-neutral LLM interface used by ``LLMAgent``."""

from __future__ import annotations

import contextvars
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence, runtime_checkable

NO_TARGET = "none"


class LLMError(RuntimeError):
    """Raised by clients for transport, auth, budget, or schema failures."""


class LLMBudgetExhausted(LLMError):
    """Raised by a client that refused a call before sending it."""


@runtime_checkable
class LLMClient(Protocol):
    """Anything that can return a JSON object matching a JSON Schema.

    Implementations must never log or echo their API key.
    """

    name: str
    model: str

    async def complete_json(
        self,
        *,
        system: str,
        prompt: str,
        schema: Dict[str, Any],
        max_tokens: int = 256,
    ) -> Dict[str, Any]:
        ...


@dataclass
class LLMBudget:
    """Shared, thread-safe cap on calls (and a tally of reported tokens)."""

    max_calls: int = 200
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def try_acquire(self) -> bool:
        with self._lock:
            if self.calls >= self.max_calls:
                return False
            self.calls += 1
            return True

    def release(self) -> None:
        """Refund one acquired call that was not actually made."""
        with self._lock:
            self.calls = max(0, self.calls - 1)

    def add_usage(self, input_tokens: int = 0, output_tokens: int = 0) -> None:
        with self._lock:
            self.input_tokens += int(input_tokens or 0)
            self.output_tokens += int(output_tokens or 0)


# Extra budget that HTTP clients also report token usage to: LLMAgent sets it
# to the calling simulation's budget around each call, so per-simulation
# token tallies work with a client shared across simulations.
usage_sink: "contextvars.ContextVar[Optional[LLMBudget]]" = contextvars.ContextVar(
    "sandfish_llm_usage_sink", default=None
)


def record_usage(own: Optional[LLMBudget], input_tokens: int, output_tokens: int) -> None:
    """Add token usage to a client's own budget and the active usage sink."""
    if own is not None:
        own.add_usage(input_tokens, output_tokens)
    sink = usage_sink.get()
    if sink is not None and sink is not own:
        sink.add_usage(input_tokens, output_tokens)


def action_schema(actions: Sequence[str], peers: Sequence[str]) -> Dict[str, Any]:
    """Strict JSON Schema for one agent decision.

    ``target`` is an enum of the visible peers plus ``"none"`` so the model
    can't invent agents; every field is required and no extras are allowed
    (the intersection that Anthropic tool input, Gemini responseJsonSchema,
    and OpenAI/OpenRouter strict json_schema all accept).
    """
    targets: List[str] = list(dict.fromkeys([*peers, NO_TARGET]))
    return {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": list(actions)},
            "target": {"type": "string", "enum": targets},
            "content": {"type": "string", "description": "Post/comment text, <=280 chars; empty otherwise."},
            "reason": {"type": "string", "description": "One short sentence."},
        },
        "required": ["action", "target", "content", "reason"],
        "additionalProperties": False,
    }


def redact(text: str, secret: Optional[str]) -> str:
    """Remove a secret from error text before it can reach logs."""
    if secret and text:
        return text.replace(secret, "<redacted>")
    return text
