"""
Agent / workflow registry.

Anything that handles a chat request is registered here by name.
The name must match the `model` field sent by LibreChat.

Supported target types:
  - agents.Agent          → run via Runner (streaming supported)
  - async generator fn    → async def fn(messages: list) that yields str chunks
  - regular coroutine fn  → async def fn(messages: list) -> str  (non-streaming fallback)
"""

from typing import Callable, Union

from agents import Agent

AgentOrWorkflow = Union[Agent, Callable]

_registry: dict[str, AgentOrWorkflow] = {}


def register(name: str, target: AgentOrWorkflow) -> None:
    _registry[name] = target


def get(name: str) -> AgentOrWorkflow | None:
    return _registry.get(name)


def list_models() -> list[str]:
    return list(_registry.keys())
