import json

from pydantic import BaseModel

from handlers.structured_qa.analysis import build_structured_qa_answer
from handlers.structured_qa.types import (
    StructuredQaAdapterResult,
    StructuredQaParameters,
    StructuredQaResponse,
    StructuredQaToolCall,
)


EMPTY_RESULT_MESSAGE = "Non risultano dati per i filtri selezionati nel periodo indicato."
UNSUPPORTED_MESSAGE = "La richiesta non e' supportata dai tool structured QA attualmente disponibili."


def _dump_model(model: BaseModel) -> dict[str, object]:
    return model.model_dump(by_alias=True, exclude_none=True)


def _filters(parameters: StructuredQaParameters) -> dict[str, object]:
    return _dump_model(parameters)


def _used_tool(call: StructuredQaToolCall) -> dict[str, object]:
    return {"name": call.name.value, "parameters": _dump_model(call.parameters)}


def unsupported_response(data_source: str) -> StructuredQaResponse:
    return StructuredQaResponse(
        answer=UNSUPPORTED_MESSAGE,
        data_source=data_source,
        used_tools=[],
        filters={},
        requires_clarification=False,
        unsupported_request=True,
    )


def clarification_response(data_source: str, message: str, parameters: StructuredQaParameters) -> StructuredQaResponse:
    return StructuredQaResponse(
        answer=message,
        data_source=data_source,
        used_tools=[],
        filters=_filters(parameters),
        requires_clarification=True,
        unsupported_request=False,
    )


def validation_error_response(data_source: str, message: str, parameters: StructuredQaParameters) -> StructuredQaResponse:
    return StructuredQaResponse(
        answer=message,
        data_source=data_source,
        used_tools=[],
        filters=_filters(parameters),
        requires_clarification=True,
        unsupported_request=False,
    )


def adapter_error_response(data_source: str, call: StructuredQaToolCall) -> StructuredQaResponse:
    return StructuredQaResponse(
        answer="Non riesco a recuperare i dati structured QA in questo momento.",
        data_source=data_source,
        used_tools=[_used_tool(call)],
        filters=_filters(call.parameters),
        requires_clarification=False,
        unsupported_request=False,
    )


def success_response(
    data_source: str,
    call: StructuredQaToolCall,
    result: StructuredQaAdapterResult,
    question: str = "",
) -> StructuredQaResponse:
    if not result.rows:
        answer = EMPTY_RESULT_MESSAGE
    else:
        answer = build_structured_qa_answer(question, call, result)

    return StructuredQaResponse(
        answer=answer,
        data_source=data_source,
        used_tools=[_used_tool(call)],
        filters=_filters(call.parameters),
        requires_clarification=False,
        unsupported_request=False,
    )


def to_json(response: StructuredQaResponse) -> str:
    return json.dumps(response.model_dump(), ensure_ascii=False)


def to_chat_text(response: StructuredQaResponse) -> str:
    lines = [response.answer]
    if response.used_tools:
        tool_names = ", ".join(str(tool["name"]) for tool in response.used_tools)
        lines.append(f"\nFonte dati: {response.data_source}")
        lines.append(f"Tool usato: {tool_names}")
    elif response.data_source:
        lines.append(f"\nFonte dati: {response.data_source}")
    if response.filters:
        filters = ", ".join(f"{key}={value}" for key, value in response.filters.items())
        lines.append(f"Filtri: {filters}")
    return "\n".join(lines)
