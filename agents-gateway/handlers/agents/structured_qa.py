"""Structured Data QA Agent."""

import os
from typing import AsyncGenerator

from logger import get_logger
from handlers.structured_qa.adapter import StructuredQaAdapterError, build_adapter
from handlers.structured_qa.config import load_config
from handlers.structured_qa.formatter import (
    adapter_error_response,
    clarification_response,
    success_response,
    to_chat_text,
    to_json,
    unsupported_response,
    validation_error_response,
)
from handlers.structured_qa.planner import plan_structured_qa
from handlers.structured_qa.validator import (
    StructuredQaToolNotAllowedError,
    StructuredQaValidationError,
    validate_tool_call,
)

log = get_logger("structured_qa")


def _render_response(response) -> str:
    if os.getenv("STRUCTURED_QA_OUTPUT_FORMAT", "text").strip().lower() == "json":
        return to_json(response)
    return to_chat_text(response)


async def run_structured_qa(
    conversation: list,
    rewritten_query: str,
) -> AsyncGenerator[str, None]:
    config = load_config()
    adapter = build_adapter(config)
    log.info(
        "structured_qa.request.received",
        mode=config.mode,
        data_source=adapter.data_source,
        conversation_messages=len(conversation),
    )

    intent = await plan_structured_qa(rewritten_query)
    log.info(
        "structured_qa.intent.detected",
        tool_name=intent.tool_name.value if intent.tool_name else None,
        requires_clarification=intent.requires_clarification,
        unsupported_request=intent.unsupported_request,
    )

    if intent.unsupported_request:
        response = unsupported_response(adapter.data_source)
        log.info("structured_qa.response.generated", status="unsupported")
        yield _render_response(response)
        return

    if intent.requires_clarification:
        response = clarification_response(
            adapter.data_source,
            intent.clarification_message,
            intent.parameters,
        )
        log.info("structured_qa.response.generated", status="clarification")
        yield _render_response(response)
        return

    try:
        call = validate_tool_call(intent, config)
    except StructuredQaToolNotAllowedError:
        response = unsupported_response(adapter.data_source)
        log.warning("structured_qa.tool_selected", allowed=False)
        log.info("structured_qa.response.generated", status="tool_not_allowed")
        yield _render_response(response)
        return
    except StructuredQaValidationError as exc:
        response = validation_error_response(adapter.data_source, str(exc), intent.parameters)
        log.warning("structured_qa.tool_selected", allowed=False, reason=str(exc))
        log.info("structured_qa.response.generated", status="validation_error")
        yield _render_response(response)
        return

    log.info(
        "structured_qa.tool_selected",
        tool_name=call.name.value,
        allowed=True,
        parameter_keys=list(call.parameters.model_dump(by_alias=True, exclude_none=True).keys()),
    )

    try:
        log.info(
            "structured_qa.adapter.called",
            tool_name=call.name.value,
            data_source=adapter.data_source,
            mock=adapter.data_source == "mock_db",
        )
        result = await adapter.call_tool(call)
        log.info(
            "structured_qa.adapter.completed",
            tool_name=call.name.value,
            data_source=adapter.data_source,
            row_count=len(result.rows),
            mock=result.metadata.get("mock") is True,
        )
    except StructuredQaAdapterError as exc:
        log.error(
            "structured_qa.adapter.failed",
            tool_name=call.name.value,
            data_source=adapter.data_source,
            error_type=type(exc).__name__,
        )
        response = adapter_error_response(adapter.data_source, call)
        log.info("structured_qa.response.generated", status="adapter_error")
        yield _render_response(response)
        return

    response = success_response(adapter.data_source, call, result, rewritten_query)
    log.info(
        "structured_qa.response.generated",
        status="empty" if not result.rows else "success",
        data_source=adapter.data_source,
    )
    yield _render_response(response)
