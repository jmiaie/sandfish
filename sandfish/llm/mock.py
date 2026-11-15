"""Deterministic, offline LLM client for tests and dry runs."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from .base import NO_TARGET


class MockLLMClient:
    """Returns schema-valid decisions without any network access.

    Pass ``responder(system, prompt, schema) -> dict`` for custom behaviour;
    the default picks the first non-idle action and the first real target.
    Every call is recorded in ``calls`` for assertions.
    """

    name = "mock"

    def __init__(
        self,
        responder: Optional[Callable[[str, str, Dict[str, Any]], Dict[str, Any]]] = None,
        model: str = "mock-1",
        fail_with: Optional[BaseException] = None,
    ):
        self.model = model
        self._responder = responder
        self._fail_with = fail_with
        self.calls: List[Dict[str, Any]] = []

    async def complete_json(
        self,
        *,
        system: str,
        prompt: str,
        schema: Dict[str, Any],
        max_tokens: int = 256,
    ) -> Dict[str, Any]:
        self.calls.append({"system": system, "prompt": prompt, "schema": schema})
        if self._fail_with is not None:
            raise self._fail_with
        if self._responder is not None:
            return self._responder(system, prompt, schema)
        props = schema["properties"]
        actions = [a for a in props["action"]["enum"] if a != "do_nothing"] or props["action"]["enum"]
        targets = [t for t in props["target"]["enum"] if t != NO_TARGET]
        return {
            "action": actions[0],
            "target": targets[0] if targets else NO_TARGET,
            "content": "mock thought",
            "reason": "deterministic mock",
        }
