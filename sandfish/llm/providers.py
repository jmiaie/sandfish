"""HTTP clients for Anthropic, Gemini, and OpenRouter (structured output).

Each client forces schema-conforming JSON:

- Anthropic Messages API: a single tool whose ``input_schema`` is the
  decision schema, with ``tool_choice`` forcing it.
- Gemini ``generateContent``: ``responseMimeType=application/json`` plus
  ``responseJsonSchema``.
- OpenRouter (OpenAI-compatible): ``response_format`` ``json_schema`` with
  ``strict: true``.

Keys come from the environment (or the constructor) and are never logged:
error messages are passed through ``redact``. Requires ``httpx``
(``pip install 'sandfish[llm]'``).
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Dict, Optional

from .base import LLMBudget, LLMBudgetExhausted, LLMError, record_usage, redact

DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-5-5",
    "gemini": "gemini-3.5-flash-lite",
    "openrouter": "anthropic/claude-sonnet-5.5",
}

_KEY_ENV = {
    "anthropic": ("ANTHROPIC_API_KEY", "LLM_API_KEY"),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY", "LLM_API_KEY"),
    "openrouter": ("OPENROUTER_API_KEY", "LLM_API_KEY"),
}


def _import_httpx() -> Any:
    try:
        import httpx  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on extras
        raise LLMError("LLM providers need httpx: pip install 'sandfish[llm]'") from exc
    return httpx


class _HTTPClient:
    name = "base"
    base_url = ""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        *,
        timeout: float = 30.0,
        max_concurrency: int = 8,
        budget: Optional[LLMBudget] = None,
        transport: Any = None,
        base_url: Optional[str] = None,
        max_retries: int = 2,
    ):
        self._api_key = api_key or _key_from_env(self.name)
        if not self._api_key:
            raise LLMError(f"No API key for {self.name}; set {' or '.join(_KEY_ENV[self.name])}.")
        self.model = model or os.environ.get("SANDFISH_LLM_MODEL") or DEFAULT_MODELS[self.name]
        self.timeout = timeout
        self.budget = budget
        self._sem = asyncio.Semaphore(max(1, max_concurrency))
        self._transport = transport  # httpx.MockTransport in tests
        self.max_retries = max(0, int(max_retries))
        if base_url:
            self.base_url = base_url

    def __repr__(self) -> str:  # never include the key
        return f"{type(self).__name__}(model={self.model!r})"

    async def _post(self, url: str, headers: Dict[str, str], body: Dict[str, Any]) -> Dict[str, Any]:
        httpx = _import_httpx()
        if self.budget is not None and not self.budget.try_acquire():
            raise LLMBudgetExhausted("LLM call budget exhausted")
        resp = None
        async with self._sem:
            for attempt in range(self.max_retries + 1):
                try:
                    async with httpx.AsyncClient(timeout=self.timeout, transport=self._transport) as client:
                        resp = await client.post(url, headers=headers, json=body)
                except Exception as exc:
                    if attempt < self.max_retries:
                        await asyncio.sleep(0.5 * 2 ** attempt)
                        continue
                    raise LLMError(redact(f"{self.name} request failed: {type(exc).__name__}: {exc}", self._api_key)) from None
                if resp.status_code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                    await asyncio.sleep(0.5 * 2 ** attempt)
                    continue
                break
        assert resp is not None
        if resp.status_code >= 400:
            raise LLMError(redact(f"{self.name} HTTP {resp.status_code}: {resp.text[:300]}", self._api_key))
        try:
            data: Dict[str, Any] = resp.json()
            return data
        except ValueError:
            raise LLMError(f"{self.name} returned non-JSON body") from None

    @staticmethod
    def _parse_json_text(text: str, provider: str) -> Dict[str, Any]:
        try:
            value = json.loads(text)
        except (TypeError, ValueError):
            raise LLMError(f"{provider} did not return JSON") from None
        if not isinstance(value, dict):
            raise LLMError(f"{provider} returned non-object JSON")
        return value


def _key_from_env(provider: str) -> Optional[str]:
    for var in _KEY_ENV[provider]:
        value = os.environ.get(var)
        if value and value.strip():
            return value.strip()
    return None


class AnthropicClient(_HTTPClient):
    name = "anthropic"
    base_url = "https://api.anthropic.com/v1"

    async def complete_json(self, *, system: str, prompt: str, schema: Dict[str, Any],
                            max_tokens: int = 256) -> Dict[str, Any]:
        body = {
            "model": self.model,
            "max_tokens": max_tokens,
            # Cacheable persona/instructions prefix.
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": prompt}],
            "tools": [{
                "name": "choose_action",
                "description": "Choose this agent's action for the current round.",
                "input_schema": schema,
            }],
            "tool_choice": {"type": "tool", "name": "choose_action"},
        }
        headers = {
            "x-api-key": self._api_key or "",
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        data = await self._post(f"{self.base_url}/messages", headers, body)
        usage = data.get("usage") or {}
        record_usage(self.budget, usage.get("input_tokens", 0), usage.get("output_tokens", 0))
        for block in data.get("content") or []:
            if block.get("type") == "tool_use" and isinstance(block.get("input"), dict):
                decision: Dict[str, Any] = block["input"]
                return decision
        raise LLMError("anthropic response had no tool_use block")


class GeminiClient(_HTTPClient):
    name = "gemini"
    base_url = "https://generativelanguage.googleapis.com/v1beta"

    async def complete_json(self, *, system: str, prompt: str, schema: Dict[str, Any],
                            max_tokens: int = 256) -> Dict[str, Any]:
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseJsonSchema": schema,
                "maxOutputTokens": max_tokens,
            },
        }
        headers = {"x-goog-api-key": self._api_key or "", "content-type": "application/json"}
        data = await self._post(f"{self.base_url}/models/{self.model}:generateContent", headers, body)
        usage = data.get("usageMetadata") or {}
        record_usage(self.budget, usage.get("promptTokenCount", 0), usage.get("candidatesTokenCount", 0))
        try:
            parts = data["candidates"][0]["content"]["parts"]
            text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        except (KeyError, IndexError, TypeError):
            raise LLMError("gemini response had no candidate text") from None
        return self._parse_json_text(text, "gemini")


class OpenRouterClient(_HTTPClient):
    name = "openrouter"
    base_url = "https://openrouter.ai/api/v1"

    async def complete_json(self, *, system: str, prompt: str, schema: Dict[str, Any],
                            max_tokens: int = 256) -> Dict[str, Any]:
        body = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "choose_action", "strict": True, "schema": schema},
            },
            "provider": {"require_parameters": True},
        }
        headers = {
            "authorization": f"Bearer {self._api_key}",
            "content-type": "application/json",
            "x-title": "SandFish",
        }
        data = await self._post(f"{self.base_url}/chat/completions", headers, body)
        usage = data.get("usage") or {}
        record_usage(self.budget, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise LLMError("openrouter response had no message content") from None
        return self._parse_json_text(text, "openrouter")


_PROVIDERS = {"anthropic": AnthropicClient, "gemini": GeminiClient, "openrouter": OpenRouterClient}


def client_from_env(provider: Optional[str] = None, **kwargs: Any) -> Any:
    """Build a client from ``SANDFISH_LLM_PROVIDER`` (or ``provider``).

    Returns None when no provider is configured, which keeps LLM agents off.
    ``"mock"`` returns the offline ``MockLLMClient``.
    """
    name = (provider or os.environ.get("SANDFISH_LLM_PROVIDER") or "").strip().lower()
    if not name:
        return None
    if name == "mock":
        from .mock import MockLLMClient

        return MockLLMClient()
    if name not in _PROVIDERS:
        raise LLMError(f"Unknown LLM provider {name!r}; choose from {sorted(_PROVIDERS) + ['mock']}")
    return _PROVIDERS[name](**kwargs)
