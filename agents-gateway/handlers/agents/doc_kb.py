"""
Document Knowledge Base Agent.

Synthesises answers from document context already injected into the conversation
by LibreChat's native file_search / rag_api pipeline.

This agent does NOT query rag_api directly — context enrichment and citation
tracking are handled upstream by LibreChat before the request reaches this gateway.
"""

from typing import AsyncGenerator

from agents import Agent, ModelSettings, Runner
from agents.stream_events import RawResponsesStreamEvent
from openai.types.shared.reasoning import Reasoning
from logger import get_logger

log = get_logger("doc_kb")


_INSTRUCTIONS = """
You are a Document Knowledge Base Agent.

LibreChat has already retrieved relevant document chunks from the knowledge base
and injected them into the conversation context before this request reached you.
Your job is to give a short, direct answer based solely on that context.

Rules:
- Answer in the same language the user used.
- Be concise: one to three sentences maximum. No lengthy explanations.
- Ground every claim in the provided context; do not invent information.
- If the context does not contain the answer, say only: "Le informazioni richieste non sono presenti nei documenti forniti."
- Do NOT ask follow-up questions.
- Do NOT suggest uploading more documents or taking further actions.
- Do NOT offer alternatives or additional options.
- No markdown headings. Use bullet points only if listing two or more distinct items.
"""

_doc_agent = Agent(
    name="Document KB Agent",
    model="gpt-5-mini",
    instructions=_INSTRUCTIONS,
    model_settings=ModelSettings(
        store=True,
        reasoning=Reasoning(effort="low", summary="auto"),
    ),
)


async def run_doc_kb(
    conversation: list,
    rewritten_query: str,
) -> AsyncGenerator[str, None]:
    log.info("doc_kb.start", rewritten_query=rewritten_query)
    # LibreChat has enriched `conversation` with retrieved chunks.
    # Append the rewritten query so the agent has the clearest possible question.
    input_messages = [
        *conversation,
        {
            "role": "user",
            "content": [{"type": "input_text", "text": rewritten_query}],
        },
    ]
    streamed = Runner.run_streamed(_doc_agent, input_messages)
    async for event in streamed.stream_events():
        if not isinstance(event, RawResponsesStreamEvent):
            continue
        raw = event.data
        if getattr(raw, "type", None) == "response.output_text.delta":
            delta = getattr(raw, "delta", "")
            if delta:
                yield delta


