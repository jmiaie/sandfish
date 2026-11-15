"""Optional, off-by-default LLM backends for SandFish agents.

Nothing here runs unless you opt in: bundled agents stay heuristic and make
no network calls. See ``sandfish.llm.agent.enable_llm_agents``.
"""

from .base import LLMBudget, LLMClient, LLMError, action_schema
from .mock import MockLLMClient
from .providers import (
    AnthropicClient,
    GeminiClient,
    OpenRouterClient,
    client_from_env,
)

__all__ = [
    "AnthropicClient",
    "GeminiClient",
    "LLMBudget",
    "LLMClient",
    "LLMError",
    "MockLLMClient",
    "OpenRouterClient",
    "action_schema",
    "client_from_env",
]
