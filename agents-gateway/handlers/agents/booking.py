"""
Booking Agent.

Conversational agent that:
  1. Collects required booking information (date, time, service type, contact name).
  2. Presents a structured summary and asks for explicit user confirmation.
  3. On confirmation, executes the booking (stub — replace with real calendar API).

User approval is enforced conversationally: the agent will not finalise
a booking until the user replies "yes" / "confirm" / "sì" to the summary.
"""

from typing import AsyncGenerator

from agents import Agent, ModelSettings, Runner
from openai.types.shared.reasoning import Reasoning
from agents.stream_events import RawResponsesStreamEvent
from logger import get_logger

log = get_logger("booking")


_INSTRUCTIONS = """
You are a Booking Agent for an enterprise calendar system.

Your goal is to help users create, view, modify, or cancel appointments.

## Booking creation flow
1. Greet and collect ALL required fields if missing:
   - Service type / reason for the appointment
   - Preferred date (DD/MM/YYYY or natural language such as "next Monday")
   - Preferred time slot (e.g. "morning", "10:30")
   - Contact name (first + last name)
   - Optional: notes or special requests

2. Once you have all fields, present a clear confirmation summary:
   ```
   📅 Riepilogo prenotazione
   • Servizio: <service>
   • Data: <date>
   • Orario: <time>
   • Nominativo: <name>
   • Note: <notes or "nessuna">

   Confermando, la prenotazione verrà creata nel sistema.
   Rispondi **"sì"** o **"conferma"** per procedere, oppure indicami le modifiche da apportare.
   ```

3. Only when the user explicitly confirms (replies "sì", "si", "yes", "conferma", "ok"), 
   generate the confirmation response with a booking reference.

4. For modifications or cancellations, follow a similar collect → confirm → execute pattern.

## Important rules
- Never create a booking without explicit user confirmation.
- Always use the user's language (Italian or English based on their messages).
- If availability cannot be verified, inform the user and ask for an alternative slot.
- Booking reference format: BK-<YYYYMMDD>-<HHMM>-<INITIALS> (e.g. BK-20260601-1030-MR).

## Stub note
The actual calendar integration is not yet configured. After confirmation,
generate a plausible booking reference and inform the user that the booking
has been registered and they will receive a confirmation by email.
"""

_booking_agent = Agent(
    name="Booking Agent",
    model="gpt-5-mini",
    instructions=_INSTRUCTIONS,
    model_settings=ModelSettings(
        store=True,
        reasoning=Reasoning(effort="low", summary="auto"),
    ),
)


async def run_booking(
    conversation: list,
    rewritten_query: str,
) -> AsyncGenerator[str, None]:
    log.info("booking.start", rewritten_query=rewritten_query)
    # Build input: add the rewritten query as the latest context
    input_messages = [
        *conversation,
        {
            "role": "user",
            "content": [{"type": "input_text", "text": rewritten_query}],
        },
    ]

    streamed = Runner.run_streamed(_booking_agent, input_messages)
    async for event in streamed.stream_events():
        if not isinstance(event, RawResponsesStreamEvent):
            continue
        raw = event.data
        if getattr(raw, "type", None) == "response.output_text.delta":
            delta = getattr(raw, "delta", "")
            if delta:
                yield delta
