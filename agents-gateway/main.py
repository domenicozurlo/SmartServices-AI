"""
Agent Gateway — generic OpenAI-compatible chat completions endpoint.

Routes requests to the correct Agent or workflow based on the `model` field.
Add new agents/workflows by creating a module in handlers/ and calling register().
"""

import inspect
import json
import os
import time
import uuid
from typing import AsyncGenerator, List, Optional, Union

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from agents import Agent, Runner
from agents.stream_events import RawResponsesStreamEvent

import handlers  # noqa: F401 — side-effect: registers all agents and workflows
from content import normalize_messages
from handlers.agents.doc_kb import run_doc_kb
from handlers.agents.conversational import run_conversational
from handlers.agents.structured_qa import run_structured_qa
from handlers.agents.booking import run_booking
from handlers.smart_service_flow import RouteResult, get_route_and_query
from handlers.multimodal import (
    parse_tool_results,
    build_images_by_id,
    select_contextual_images,
    build_sources,
)
from logger import get_logger
from registry import get, list_models

log = get_logger("gateway")

app = FastAPI(title="Agent Gateway")

VALID_API_KEY = "123"


class Message(BaseModel):
    role: str
    content: Union[str, list, None] = None
    tool_calls: Optional[list] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: Optional[bool] = False
    tools: Optional[List[dict]] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    stop: Optional[Union[str, list]] = None
    user: Optional[str] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------

def _sse_chunk(delta: dict, model: str, chunk_id: str, finish_reason=None) -> str:
    payload = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(payload)}\n\n"


def _include_reasoning() -> bool:
    return os.getenv("AGENTS_GATEWAY_INCLUDE_REASONING", "false").lower() in ("true", "1", "yes")


def _stream_status_enabled() -> bool:
    return os.getenv("AGENTS_GATEWAY_STREAM_STATUS", "true").lower() in ("true", "1", "yes")


def _thinking_open(text: str) -> str:
    return f":::thinking\n{text.strip()}\n"


def _thinking_close() -> str:
    return "\n:::\n\n"


_TITLE_PROMPT_MARKERS = (
    "provide a concise",
    "5-word-or-less title",
    "using title case",
    "only return the title",
)


def _last_user_text(raw_messages: List[Message]) -> str:
    for msg in reversed(raw_messages):
        if msg.role != "user":
            continue
        if isinstance(msg.content, str):
            return msg.content
        if isinstance(msg.content, list):
            for block in msg.content:
                if isinstance(block, dict) and block.get("type") in ("text", "input_text"):
                    return block.get("text", "")
    return ""


def _is_title_request(raw_messages: List[Message]) -> bool:
    lower = _last_user_text(raw_messages).lower()
    return sum(1 for marker in _TITLE_PROMPT_MARKERS if marker in lower) >= 2


def _status_for_route(route: str) -> Optional[str]:
    return {
        "doc_kb": "Cerco nei documenti rilevanti...",
        "booking": "Sto verificando i dettagli della prenotazione...",
        "structured_qa": "Sto interrogando i dati strutturati...",
        "conversational": "Sto preparando la risposta...",
    }.get(route)


async def _stream_agent(agent: Agent, messages: list, model: str) -> AsyncGenerator[str, None]:
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    yield _sse_chunk({"role": "assistant", "content": ""}, model, chunk_id)

    include_reasoning = _include_reasoning()
    reasoning_open = False
    answer_started = False
    streamed = Runner.run_streamed(agent, messages)
    async for event in streamed.stream_events():
        if not isinstance(event, RawResponsesStreamEvent):
            continue
        raw = event.data
        event_type = getattr(raw, "type", None)
        if event_type == "response.reasoning_summary_text.delta":
            delta = getattr(raw, "delta", "")
            if delta and include_reasoning:
                if not reasoning_open:
                    yield _sse_chunk({"content": ":::thinking\n"}, model, chunk_id)
                    reasoning_open = True
                yield _sse_chunk({"content": delta}, model, chunk_id)
        elif event_type == "response.output_text.delta":
            delta = getattr(raw, "delta", "")
            if delta:
                if reasoning_open and not answer_started:
                    yield _sse_chunk({"content": "\n:::\n\n"}, model, chunk_id)
                answer_started = True
                yield _sse_chunk({"content": delta}, model, chunk_id)

    if reasoning_open and not answer_started:
        yield _sse_chunk({"content": "\n:::\n\n"}, model, chunk_id)
    yield _sse_chunk({}, model, chunk_id, finish_reason="stop")
    yield "data: [DONE]\n\n"


async def _stream_workflow(workflow, messages: list, model: str) -> AsyncGenerator[str, None]:
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    yield _sse_chunk({"role": "assistant", "content": ""}, model, chunk_id)

    status_open = _stream_status_enabled()
    status_closed = False
    if status_open:
        yield _sse_chunk({"content": _thinking_open("Sto analizzando la richiesta...")}, model, chunk_id)

    async for text in workflow(messages):
        if text:
            if status_open and not status_closed:
                yield _sse_chunk({"content": _thinking_close()}, model, chunk_id)
                status_closed = True
            yield _sse_chunk({"content": text}, model, chunk_id)

    if status_open and not status_closed:
        yield _sse_chunk({"content": _thinking_close()}, model, chunk_id)
    yield _sse_chunk({}, model, chunk_id, finish_reason="stop")
    yield "data: [DONE]\n\n"


async def _stream_static(text: str, model: str) -> AsyncGenerator[str, None]:
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    yield _sse_chunk({"role": "assistant", "content": text}, model, chunk_id)
    yield _sse_chunk({}, model, chunk_id, finish_reason="stop")
    yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Tool-calls protocol helpers
# ---------------------------------------------------------------------------

def _has_tool_message(raw_messages: List[Message]) -> bool:
    """Round 2 detection: the *last* message is a tool result from file_search.
    Checking only the tail avoids false positives when a previous doc_kb exchange
    is still in the conversation history.
    """
    return bool(raw_messages) and raw_messages[-1].role == "tool"


def _file_search_available(tools: Optional[List[dict]]) -> bool:
    return bool(tools) and any(
        t.get("type") == "function" and t.get("function", {}).get("name") == "file_search"
        for t in tools
    )


def _make_tool_call_response(model: str, query: str) -> dict:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": f"call_{uuid.uuid4().hex[:12]}",
                    "type": "function",
                    "function": {
                        "name": "file_search",
                        "arguments": json.dumps({"query": query}),
                    },
                }],
            },
            "finish_reason": "tool_calls",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def _build_round2_input(raw_messages: List[Message]) -> tuple[list, str, dict]:
    """Extract the last user query and tool content from Round 2 messages.

    Returns (agent_messages, user_query, multimodal_ctx) where:
    - agent_messages: ready for legacy run_doc_kb call (no context_groups)
    - user_query: the rewritten query from the last tool_calls assistant message
      (falls back to the last user message if not found)
    - multimodal_ctx: dict with keys context_groups, images_by_id, sources — or empty dict
      when the tool message is plain text (legacy format).
    """
    user_query = ""
    tool_content = ""

    # Prefer the rewritten query stored in the last assistant tool_call arguments.
    # This is the enriched query the classifier produced in Round 1, which is
    # much more specific than the raw user message and yields better answers.
    for m in reversed(raw_messages):
        if m.role != "assistant" or not m.tool_calls:
            continue
        for tc in m.tool_calls:
            fn = tc.get("function", {}) if isinstance(tc, dict) else getattr(tc, "function", None)
            if fn is None:
                continue
            args_raw = fn.get("arguments", "") if isinstance(fn, dict) else getattr(fn, "arguments", "")
            try:
                args = json.loads(args_raw)
                q = args.get("query", "")
                if q:
                    user_query = q
                    break
            except (json.JSONDecodeError, TypeError):
                pass
        if user_query:
            break

    # Fall back to the last raw user message if no tool_call query was found.
    if not user_query:
        for m in reversed(raw_messages):
            if m.role != "user":
                continue
            if isinstance(m.content, str):
                user_query = m.content
            elif isinstance(m.content, list):
                for block in m.content:
                    if isinstance(block, dict) and block.get("type") in ("text", "input_text"):
                        user_query = block.get("text", "")
                        break
            if user_query:
                break

    # Last tool result (file_search output for the current turn, not a previous one)
    for m in reversed(raw_messages):
        if m.role != "tool":
            continue
        if isinstance(m.content, str):
            tool_content = m.content
        elif isinstance(m.content, list):
            parts = [b.get("text", b.get("content", "")) for b in m.content if isinstance(b, dict)]
            tool_content = "\n".join(parts)
        break

    log.debug(
        "round2._build_round2_input",
        user_query_preview=user_query[:80],
        tool_content_len=len(tool_content),
    )
    log.info(
        "round2.tool_content_preview",
        preview=tool_content[:300],
    )

    # Try multimodal format first
    multimodal_data = parse_tool_results(tool_content)
    log.info("round2.parse_tool_results", is_multimodal=multimodal_data is not None)
    if multimodal_data:
        context_groups = multimodal_data.get("context_groups", [])
        images_by_id = build_images_by_id(context_groups)
        sources = build_sources(context_groups)
        log.info(
            "round2.multimodal_tool_message",
            groups=len(context_groups),
            images=len(images_by_id),
            sources=len(sources),
        )
        # Return empty agent_messages for multimodal path (context is passed separately)
        return [], user_query, {
            "context_groups": context_groups,
            "images_by_id": images_by_id,
            "sources": sources,
        }

    # Legacy plain-text tool message
    log.info("round2.legacy_tool_message", tool_content_len=len(tool_content))
    agent_messages = [
        {"role": "user", "content": [{"type": "input_text", "text": user_query}]},
    ]
    if tool_content:
        agent_messages.append({
            "role": "user",
            "content": [{"type": "input_text", "text": f"[Retrieved document context]\n{tool_content}"}],
        })
    return agent_messages, user_query, {}


async def _stream_doc_kb_synthesis(
    messages: list,
    query: str,
    model: str,
    multimodal_ctx: dict = None,
) -> AsyncGenerator[str, None]:
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    yield _sse_chunk({"role": "assistant", "content": ""}, model, chunk_id)
    status_open = _stream_status_enabled()
    status_closed = False
    if status_open:
        yield _sse_chunk(
            {"content": _thinking_open("Ho trovato i documenti rilevanti. Sto preparando la risposta...")},
            model,
            chunk_id,
        )

    ctx = multimodal_ctx or {}
    context_groups = ctx.get("context_groups")
    has_multimodal_context = bool(context_groups)
    images_by_id = ctx.get("images_by_id") or {}
    sources = ctx.get("sources") or []

    # Accumulate the full answer so we can build rag_context after streaming ends.
    full_answer_parts: list = []
    async for text in run_doc_kb(
        messages,
        query,
        context_groups=context_groups if has_multimodal_context else None,
        images_by_id=images_by_id if has_multimodal_context else None,
        sources=sources if has_multimodal_context else None,
    ):
        if text:
            if status_open and not status_closed:
                yield _sse_chunk({"content": _thinking_close()}, model, chunk_id)
                status_closed = True
            full_answer_parts.append(text)
            yield _sse_chunk({"content": text}, model, chunk_id)

    if status_open and not status_closed:
        yield _sse_chunk({"content": _thinking_close()}, model, chunk_id)

    # Build rag_context from the multimodal context available in this turn.
    # This follows the shape expected by the LibreChat RAG panel components:
    #   data_points.text      — raw chunk texts for SupportingContent
    #   data_points.images    — image URLs for inline/panel display
    #   data_points.citations — source file + page references
    #   thoughts              — query step shown in ThoughtProcess panel
    rag_context: Optional[dict] = None
    if has_multimodal_context:
        text_snippets: list = []
        citation_strs: list = []
        seen_citations: set = set()

        for group in context_groups:
            for chunk in group.get("chunks", []):
                if chunk.get("score", 0) > 0:
                    text_snippets.append(chunk.get("text", "").strip())
                m = chunk.get("metadata", {})
                sf = m.get("source_file", "")
                pg = m.get("page")
                key = f"{sf}:{pg}"
                if sf and key not in seen_citations:
                    seen_citations.add(key)
                    label = f"{sf}#page={pg}" if pg is not None else sf
                    citation_strs.append(label)

        image_urls = [
            img["url"]
            for img in images_by_id.values()
            if img.get("url")
        ]

        rag_context = {
            "data_points": {
                "text": text_snippets[:10],
                "images": image_urls[:8],
                "citations": citation_strs,
            },
            "thoughts": [
                {"title": "Query documento", "description": query}
            ],
            "followup_questions": None,
        }

    # Emit the stop chunk; if rag_context is available, include it as a
    # top-level field so the frontend can extract it without breaking
    # standard OpenAI SSE consumers (unknown fields are ignored).
    stop_payload = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    if rag_context is not None:
        stop_payload["rag_context"] = rag_context

    yield f"data: {json.dumps(stop_payload)}\n\n"
    yield "data: [DONE]\n\n"


async def _stream_direct(
    messages: list, route: str, rewritten_query: str, model: str
) -> AsyncGenerator[str, None]:
    """Stream a response directly from the appropriate handler, skipping workflow re-entry."""
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    yield _sse_chunk({"role": "assistant", "content": ""}, model, chunk_id)
    status_by_route = {
        "booking": "Sto verificando i dettagli della prenotazione...",
        "structured_qa": "Sto interrogando i dati strutturati...",
    }
    status = status_by_route.get(route)
    status_open = bool(status) and _stream_status_enabled()
    status_closed = False
    if status_open:
        yield _sse_chunk({"content": _thinking_open(status)}, model, chunk_id)
    if route == "conversational":
        handler = run_conversational(messages, rewritten_query)
    elif route == "structured_qa":
        handler = run_structured_qa(messages, rewritten_query)
    elif route == "booking":
        handler = run_booking(messages, rewritten_query)
    else:
        yield _sse_chunk({"content": "Route non supportata."}, model, chunk_id)
        yield _sse_chunk({}, model, chunk_id, finish_reason="stop")
        yield "data: [DONE]\n\n"
        return
    async for text in handler:
        if text:
            if status_open and not status_closed:
                yield _sse_chunk({"content": _thinking_close()}, model, chunk_id)
                status_closed = True
            yield _sse_chunk({"content": text}, model, chunk_id)
    if status_open and not status_closed:
        yield _sse_chunk({"content": _thinking_close()}, model, chunk_id)
    yield _sse_chunk({}, model, chunk_id, finish_reason="stop")
    yield "data: [DONE]\n\n"


async def _stream_round1_with_status(raw_messages: List[Message], model: str) -> AsyncGenerator[str, None]:
    """Start Round 1 streaming before guardrail/classifier latency is paid."""
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    yield _sse_chunk({"role": "assistant", "content": ""}, model, chunk_id)

    title_request = _is_title_request(raw_messages)
    status_open = _stream_status_enabled() and not title_request
    status_closed = False
    if status_open:
        yield _sse_chunk({"content": _thinking_open("Sto analizzando la richiesta...")}, model, chunk_id)

    norm_messages = normalize_messages(
        [{"role": m.role, "content": m.content} for m in raw_messages]
    )
    route_result = await get_route_and_query(norm_messages)
    log.info(
        "round1.route_resolved",
        route=route_result.route,
        rewritten_query=route_result.rewritten_query[:120] if route_result.rewritten_query else "",
    )

    if route_result.route == "title":
        from agents import Runner
        from handlers.smart_service_flow import _title_agent

        title_run = await Runner.run(_title_agent, norm_messages)
        static_text = str(title_run.final_output).strip()
        yield _sse_chunk({"content": static_text}, model, chunk_id)
        yield _sse_chunk({}, model, chunk_id, finish_reason="stop")
        yield "data: [DONE]\n\n"
        return

    if route_result.route in ("blocked", "out_of_scope"):
        if status_open and not status_closed:
            yield _sse_chunk({"content": _thinking_close()}, model, chunk_id)
            status_closed = True
        yield _sse_chunk({"content": route_result.message}, model, chunk_id)
        yield _sse_chunk({}, model, chunk_id, finish_reason="stop")
        yield "data: [DONE]\n\n"
        return

    route_status = _status_for_route(route_result.route)
    if status_open and route_status:
        yield _sse_chunk({"content": f"{route_status}\n"}, model, chunk_id)

    if route_result.route == "doc_kb":
        rag_query = route_result.search_query or route_result.rewritten_query
        log.info(
            "round1.emitting_tool_call",
            rag_query_preview=rag_query[:120],
            rewritten_preview=route_result.rewritten_query[:80],
            stream=True,
        )
        if status_open and not status_closed:
            yield _sse_chunk({"content": _thinking_close()}, model, chunk_id)
            status_closed = True
        call_id = f"call_{uuid.uuid4().hex[:12]}"
        yield _sse_chunk(
            {
                "tool_calls": [{
                    "index": 0,
                    "id": call_id,
                    "type": "function",
                    "function": {"name": "file_search", "arguments": ""},
                }]
            },
            model,
            chunk_id,
        )
        yield _sse_chunk(
            {"tool_calls": [{"index": 0, "function": {"arguments": json.dumps({"query": rag_query})}}]},
            model,
            chunk_id,
        )
        yield _sse_chunk({}, model, chunk_id, finish_reason="tool_calls")
        yield "data: [DONE]\n\n"
        return

    log.info("round1.direct_dispatch", route=route_result.route)
    if route_result.route == "conversational":
        handler = run_conversational(norm_messages, route_result.rewritten_query)
    elif route_result.route == "structured_qa":
        handler = run_structured_qa(norm_messages, route_result.rewritten_query)
    elif route_result.route == "booking":
        handler = run_booking(norm_messages, route_result.rewritten_query)
    else:
        handler = None

    if handler is None:
        if status_open and not status_closed:
            yield _sse_chunk({"content": _thinking_close()}, model, chunk_id)
            status_closed = True
        yield _sse_chunk({"content": "Route non supportata."}, model, chunk_id)
    else:
        async for text in handler:
            if text:
                if status_open and not status_closed:
                    yield _sse_chunk({"content": _thinking_close()}, model, chunk_id)
                    status_closed = True
                yield _sse_chunk({"content": text}, model, chunk_id)

    if status_open and not status_closed:
        yield _sse_chunk({"content": _thinking_close()}, model, chunk_id)
    yield _sse_chunk({}, model, chunk_id, finish_reason="stop")
    yield "data: [DONE]\n\n"


async def _run_direct(messages: list, route: str, rewritten_query: str, model: str) -> dict:
    """Non-streaming direct dispatch (same routes as _stream_direct)."""
    if route == "conversational":
        chunks = [c async for c in run_conversational(messages, rewritten_query)]
    elif route == "structured_qa":
        chunks = [c async for c in run_structured_qa(messages, rewritten_query)]
    elif route == "booking":
        chunks = [c async for c in run_booking(messages, rewritten_query)]
    else:
        chunks = ["Route non supportata."]
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "".join(chunks)}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


async def _stream_tool_call_response(model: str, query: str) -> AsyncGenerator[str, None]:
    """Emit a tool_calls finish in SSE format (for stream=true requests)."""
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    call_id = f"call_{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    def _chunk(delta: dict, finish_reason=None) -> str:
        payload = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        return f"data: {json.dumps(payload)}\n\n"

    yield _chunk({"role": "assistant", "content": None})
    yield _chunk({"tool_calls": [{"index": 0, "id": call_id, "type": "function",
                                   "function": {"name": "file_search", "arguments": ""}}]})
    yield _chunk({"tool_calls": [{"index": 0, "function": {"arguments": json.dumps({"query": query})}}]})
    yield _chunk({}, finish_reason="tool_calls")
    yield "data: [DONE]\n\n"



def _require_auth(request: Request) -> None:
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer ") or auth[len("Bearer "):] != VALID_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


@app.get("/v1/models")
async def list_available_models(request: Request):
    _require_auth(request)
    return {
        "object": "list",
        "data": [
            {"id": m, "object": "model", "created": 0, "owned_by": "agent-gateway"}
            for m in list_models()
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, body: ChatRequest):
    _require_auth(request)

    target = get(body.model)
    if target is None:
        raise HTTPException(status_code=404, detail=f"Model '{body.model}' not found")

    roles = [m.role for m in body.messages]
    has_tools_in_req = _file_search_available(body.tools)
    has_tool_result = _has_tool_message(body.messages)
    log.info(
        "request.received",
        model=body.model,
        stream=body.stream,
        msg_count=len(body.messages),
        roles=roles,
        file_search_in_tools=has_tools_in_req,
        has_tool_result=has_tool_result,
    )

    # ── Two-round tool_calls protocol ─────────────────────────────────────
    # Applies to smart_service_flow when LibreChat has file_search configured.
    if body.model == "smart_service_flow":

        # Round 2: LibreChat executed file_search and sent back results.
        if has_tool_result:
            agent_msgs, user_query, multimodal_ctx = _build_round2_input(body.messages)
            tool_chars = sum(
                len(m.content) if isinstance(m.content, str) else 0
                for m in body.messages if m.role == "tool"
            )
            log.info(
                "round2.synthesis",
                user_query_preview=user_query[:120],
                tool_content_chars=tool_chars,
                multimodal=bool(multimodal_ctx),
                stream=body.stream,
            )
            if body.stream:
                return StreamingResponse(
                    _stream_doc_kb_synthesis(
                        agent_msgs, user_query, body.model, multimodal_ctx=multimodal_ctx
                    ),
                    media_type="text/event-stream",
                    headers=_SSE_HEADERS,
                )
            chunks = [
                c
                async for c in run_doc_kb(
                    agent_msgs,
                    user_query,
                    context_groups=multimodal_ctx.get("context_groups"),
                    images_by_id=multimodal_ctx.get("images_by_id"),
                    sources=multimodal_ctx.get("sources"),
                )
            ]
            final = "".join(chunks)
            return {
                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": body.model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": final}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }

        # Round 1: file_search tool available → classify, then emit tool_call for doc_kb.
        if has_tools_in_req:
            log.info("round1.classifying", msg_count=len(body.messages))
            if body.stream:
                return StreamingResponse(
                    _stream_round1_with_status(body.messages, body.model),
                    media_type="text/event-stream",
                    headers=_SSE_HEADERS,
                )
            norm_messages = normalize_messages(
                [{"role": m.role, "content": m.content} for m in body.messages]
            )
            route_result = await get_route_and_query(norm_messages)
            log.info(
                "round1.route_resolved",
                route=route_result.route,
                rewritten_query=route_result.rewritten_query[:120] if route_result.rewritten_query else "",
            )
            if route_result.route == "title":
                # Let the title agent generate a real title from the conversation
                from agents import Runner
                from handlers.smart_service_flow import _title_agent
                title_run = await Runner.run(_title_agent, norm_messages)
                static_text = str(title_run.final_output).strip()
            elif route_result.route in ("blocked", "out_of_scope"):
                static_text = route_result.message
            else:
                static_text = None

            if static_text is not None:
                log.info("round1.static_response", route=route_result.route)
                if body.stream:
                    return StreamingResponse(
                        _stream_static(static_text, body.model),
                        media_type="text/event-stream",
                        headers=_SSE_HEADERS,
                    )
                return {
                    "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": body.model,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": static_text}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                }
            elif route_result.route == "doc_kb":
                # Use the short search_query for RAG retrieval (better vector search),
                # fall back to rewritten_query when search_query is not set.
                rag_query = route_result.search_query or route_result.rewritten_query
                log.info(
                    "round1.emitting_tool_call",
                    rag_query_preview=rag_query[:120],
                    rewritten_preview=route_result.rewritten_query[:80],
                    stream=body.stream,
                )
                if body.stream:
                    return StreamingResponse(
                        _stream_tool_call_response(body.model, rag_query),
                        media_type="text/event-stream",
                        headers=_SSE_HEADERS,
                    )
                return JSONResponse(_make_tool_call_response(body.model, rag_query))
            else:
                # conversational / booking / structured_qa: route already known, dispatch directly
                log.info("round1.direct_dispatch", route=route_result.route)
                if body.stream:
                    return StreamingResponse(
                        _stream_direct(norm_messages, route_result.route, route_result.rewritten_query, body.model),
                        media_type="text/event-stream",
                        headers=_SSE_HEADERS,
                    )
                return await _run_direct(norm_messages, route_result.route, route_result.rewritten_query, body.model)
        else:
            log.info("no_tools.direct_workflow", model=body.model)

    messages = normalize_messages(
        [{"role": m.role, "content": m.content} for m in body.messages]
    )

    if body.stream:
        if isinstance(target, Agent):
            generator = _stream_agent(target, messages, body.model)
        elif inspect.isasyncgenfunction(target):
            generator = _stream_workflow(target, messages, body.model)
        else:
            # coroutine workflow — run to completion, then emit as a single chunk
            result = await target(messages)
            generator = _stream_static(str(result), body.model)

        return StreamingResponse(generator, media_type="text/event-stream", headers=_SSE_HEADERS)

    if isinstance(target, Agent):
        run_result = await Runner.run(target, messages)
        final = str(run_result.final_output)
    elif inspect.isasyncgenfunction(target):
        chunks = [chunk async for chunk in target(messages)]
        final = "".join(chunks)
    else:
        final = str(await target(messages))

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": final},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


@app.get("/health")
async def health():
    return {"status": "ok"}
