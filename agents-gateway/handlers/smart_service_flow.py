"""
SmartServiceFlow — main orchestration workflow.

Entry point registered as model "smart_service_flow".

Flow:
  1. Guardrail   — LLM-based scope check (no moderation API).
  2. Classifier  — query rewrite + deterministic route selection.
  3. Router      — dispatches to the correct specialised agent:
                     doc_kb        → Document Knowledge Base Agent
                     structured_qa → Structured Data QA Agent
                     booking       → Booking Agent
                     out_of_scope  → polite refusal (fallback)
"""

import time
from dataclasses import dataclass, field
from typing import AsyncGenerator

from agents import Agent, ModelSettings, Runner
from openai.types.shared.reasoning import Reasoning
from logger import get_logger, timed
from registry import register
from handlers.agents.guardrail import check_scope
from handlers.agents.classifier import classify
from handlers.agents.doc_kb import run_doc_kb
from handlers.agents.structured_qa import run_structured_qa
from handlers.agents.booking import run_booking
from handlers.agents.conversational import run_conversational

log = get_logger("smart_service_flow")

_title_agent = Agent(
    name="Title Generator",
    model="gpt-5-mini",
    instructions="Generate a short conversation title. Follow the user's instructions exactly.",
    model_settings=ModelSettings(
        store=True,
        reasoning=Reasoning(effort="minimal", summary="auto"),
    ),
)

_OUT_OF_SCOPE_MSG = (
    "Mi dispiace, ma posso supportarti solo per richieste relative alla "
    "knowledge base interna, ai dati strutturati disponibili o alla gestione "
    "delle prenotazioni."
)


def _last_user_text(messages: list) -> str:
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "input_text":
                    return block.get("text", "")
    return ""


_TITLE_PROMPT_MARKERS = (
    "provide a concise",
    "5-word-or-less title",
    "using title case",
    "only return the title",
)


def _is_title_request(text: str) -> bool:
    lower = text.lower()
    return sum(1 for m in _TITLE_PROMPT_MARKERS if m in lower) >= 2


@dataclass
class RouteResult:
    """Result of the guardrail + classifier pipeline."""
    route: str          # doc_kb | structured_qa | booking | out_of_scope | blocked | title
    rewritten_query: str = ""
    message: str = ""   # rejection text when route is blocked or out_of_scope


async def get_route_and_query(messages: list) -> RouteResult:
    """
    Runs guardrail + classifier and returns a routing decision.
    Exported for use by main.py in the two-round tool_calls protocol.
    """
    user_text = _last_user_text(messages)

    if _is_title_request(user_text):
        return RouteResult(route="title", rewritten_query=user_text, message=user_text)

    log.info("workflow.start", msg_count=len(messages), user_text_preview=user_text[:120])

    with timed(log, "guardrail", user_text_preview=user_text[:80]):
        guardrail = await check_scope(user_text)

    log.info("guardrail.result", allowed=guardrail.allowed, rejection_message=guardrail.rejection_message or None)

    if not guardrail.allowed:
        log.warning("workflow.blocked", reason="guardrail_rejected")
        return RouteResult(route="blocked", rewritten_query=user_text, message=guardrail.rejection_message)

    with timed(log, "classifier"):
        classification = await classify(messages)

    log.info("classifier.result", route=classification.route, rewritten_query=classification.rewritten_query)

    if classification.route == "out_of_scope":
        log.warning("workflow.blocked", reason="out_of_scope")
        return RouteResult(route="out_of_scope", rewritten_query=classification.rewritten_query, message=_OUT_OF_SCOPE_MSG)

    return RouteResult(route=classification.route, rewritten_query=classification.rewritten_query)


async def smart_service_flow(messages: list) -> AsyncGenerator[str, None]:
    t0 = time.perf_counter()

    if _is_title_request(_last_user_text(messages)):
        log.debug("workflow.title_request", generating=True)
        result = await Runner.run(_title_agent, messages)
        yield str(result.final_output)
        return

    result = await get_route_and_query(messages)

    if result.route in ("blocked", "out_of_scope"):
        yield result.message
        return

    log.info("workflow.dispatch", route=result.route)

    if result.route == "doc_kb":
        async for chunk in run_doc_kb(messages, result.rewritten_query):
            yield chunk
    elif result.route == "structured_qa":
        async for chunk in run_structured_qa(messages, result.rewritten_query):
            yield chunk
    elif result.route == "booking":
        async for chunk in run_booking(messages, result.rewritten_query):
            yield chunk
    elif result.route == "conversational":
        async for chunk in run_conversational(messages, result.rewritten_query):
            yield chunk

    log.info("workflow.done", elapsed_ms=int((time.perf_counter() - t0) * 1000), route=result.route)


register("smart_service_flow", smart_service_flow)
