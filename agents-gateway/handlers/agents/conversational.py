"""
Conversational agent — handles greetings, small talk, and questions
about the assistant itself (name, capabilities, how it works).

Uses a fast, lightweight model without reasoning overhead.
"""

from typing import AsyncGenerator

from agents import Agent, ModelSettings, Runner
from agents.stream_events import RawResponsesStreamEvent
from logger import get_logger

log = get_logger("conversational")

_INSTRUCTIONS = """
You are a helpful enterprise assistant called SmartServiceFlow.

You specialise in three areas:
  1. Searching the internal document knowledge base (manuals, guides, procedures, FAQs).
  2. Answering questions about structured business data (sales, orders, headcount, KPIs).
  3. Managing calendar bookings and appointments.

For greetings and small talk, respond warmly and briefly, then invite the user
to ask a question in one of your three specialisation areas.

For questions about your name or capabilities, answer honestly and concisely.

Always respond in the same language the user used.
"""

_agent = Agent(
    name="Conversational Agent",
    model="gpt-4o-mini",
    instructions=_INSTRUCTIONS,
    model_settings=ModelSettings(temperature=0.7),
)


async def run_conversational(
    conversation: list,
    rewritten_query: str,
) -> AsyncGenerator[str, None]:
    log.info("conversational.start", rewritten_query=rewritten_query)
    input_messages = [
        *conversation,
        {"role": "user", "content": [{"type": "input_text", "text": rewritten_query}]},
    ]
    streamed = Runner.run_streamed(_agent, input_messages)
    async for event in streamed.stream_events():
        if not isinstance(event, RawResponsesStreamEvent):
            continue
        raw = event.data
        if getattr(raw, "type", None) == "response.output_text.delta":
            delta = getattr(raw, "delta", "")
            if delta:
                yield delta
