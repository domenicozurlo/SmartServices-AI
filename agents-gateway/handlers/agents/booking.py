"""
Booking Agent.

Conversational agent that:
  1. Collects required booking information (date, time, service type, contact name).
  2. Presents a structured summary and asks for explicit user confirmation.
  3. On confirmation, creates the event via Google Calendar REST API.

User approval is enforced conversationally: the agent will not finalise
a booking until the user replies "yes" / "confirm" / "sì" to the summary.
"""

import os
from typing import AsyncGenerator

import httpx
from agents import Agent, ModelSettings, Runner, function_tool
from openai.types.shared.reasoning import Reasoning
from agents.stream_events import RawResponsesStreamEvent
from logger import get_logger

_MCP_URL = os.environ.get("GOOGLE_CALENDAR_MCP_URL", "")

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

## Calendar integration
You have access to Google Calendar tools (create_event, list_events, update_event, delete_event).
Use them to check availability and create/modify events after user confirmation.
When creating an event always include title, start datetime, end datetime (default 1 hour), description.
"""

_MODEL_SETTINGS = ModelSettings(
    store=True,
    reasoning=Reasoning(effort="low", summary="auto"),
)


def _make_calendar_tools(base_url: str) -> list:
    @function_tool
    async def list_upcoming_events(max_results: int = 10) -> str:
        """List upcoming events from Google Calendar.

        Args:
            max_results: Maximum number of events to return (default 10).
        """
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                f"{base_url}/mcp/events/upcoming",
                params={"maxResults": max_results},
            )
            resp.raise_for_status()
            return resp.text

    @function_tool
    async def create_calendar_event(
        title: str,
        start_datetime: str,
        end_datetime: str,
        description: str = "",
        calendar_id: str = "primary",
    ) -> str:
        """Create a new event in Google Calendar.

        Args:
            title: Event title/summary.
            start_datetime: Start datetime in ISO 8601 format (e.g. 2026-06-10T10:30:00+02:00).
            end_datetime: End datetime in ISO 8601 format (e.g. 2026-06-10T11:30:00+02:00).
            description: Optional event description or notes.
            calendar_id: Calendar ID to create the event in (default "primary").
        """
        payload = {
            "eventData": {
                "calendarId": calendar_id,
                "summary": title,
                "description": description,
                "start": {"dateTime": start_datetime},
                "end": {"dateTime": end_datetime},
            }
        }
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(f"{base_url}/mcp/events/create", json=payload)
            resp.raise_for_status()
            return resp.text

    @function_tool
    async def list_calendars() -> str:
        """List all available Google Calendars."""
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(f"{base_url}/mcp/calendars")
            resp.raise_for_status()
            return resp.text

    return [list_upcoming_events, create_calendar_event, list_calendars]


async def run_booking(
    conversation: list,
    rewritten_query: str,
) -> AsyncGenerator[str, None]:
    log.info("booking.start", rewritten_query=rewritten_query, mcp_url=_MCP_URL or "none")
    input_messages = [
        *conversation,
        {
            "role": "user",
            "content": [{"type": "input_text", "text": rewritten_query}],
        },
    ]

    tools = _make_calendar_tools(_MCP_URL) if _MCP_URL else []
    agent = Agent(
        name="Booking Agent",
        model="gpt-5-mini",
        instructions=_INSTRUCTIONS,
        tools=tools,
        model_settings=_MODEL_SETTINGS,
    )

    streamed = Runner.run_streamed(agent, input_messages)
    async for event in streamed.stream_events():
        if not isinstance(event, RawResponsesStreamEvent):
            continue
        raw = event.data
        if getattr(raw, "type", None) == "response.output_text.delta":
            delta = getattr(raw, "delta", "")
            if delta:
                yield delta
