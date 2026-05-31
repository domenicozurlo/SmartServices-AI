"""
Structured Data QA Agent.

Follows the pattern from agents_workflow_examples/structured_data_qa.py:
  1. Domain selector → which data domain the question belongs to.
  2. Query creator   → translates NL question to a structured query.
  3. Query executor  → runs / simulates the query and returns results.
  4. Synthesiser     → formats the result for the end user.

Extend the domain list and ontologies to match your actual data warehouse.
"""

from typing import AsyncGenerator, Literal

from pydantic import BaseModel

from agents import Agent, ModelSettings, Runner
from agents.stream_events import RawResponsesStreamEvent
from openai.types.shared.reasoning import Reasoning
from logger import get_logger

log = get_logger("structured_qa")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class DomainSelection(BaseModel):
    domain: Literal["commerce", "personnel", "other"]


class QueryPlan(BaseModel):
    query: str
    explanation: str


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

_DOMAIN_SELECTOR_INSTRUCTIONS = """
You classify a user's natural language question into a data domain.

Domains:
• commerce   — sales, orders, GMV, revenue, customers, merchants, payment methods,
               refunds, chargebacks, channels, AOV, conversion.
• personnel  — headcount, employees, departments, payroll, benefits, overtime,
               hiring, terminations, leave, compensation.
• other      — does not fit either domain.

Respond with exactly one of: "commerce", "personnel", "other".
"""

_domain_selector = Agent(
    name="Domain Selector",
    model="gpt-5-mini",
    instructions=_DOMAIN_SELECTOR_INSTRUCTIONS,
    output_type=DomainSelection,
    model_settings=ModelSettings(
        store=True,
        reasoning=Reasoning(effort="minimal", summary="auto"),
    ),
)

_QUERY_CREATOR_INSTRUCTIONS = """
You are a SQL query planner.
Given a natural-language analytics question and the available data domain, produce:
  - query: a readable, executable SQL query (use standard SQL syntax).
  - explanation: a one-sentence plain-English description of what the query does.

Use only tables and columns that are plausible for the domain.
Do not execute the query — just produce the plan.
"""

_query_creator = Agent(
    name="Query Creator",
    model="gpt-5-mini",
    instructions=_QUERY_CREATOR_INSTRUCTIONS,
    output_type=QueryPlan,
    model_settings=ModelSettings(
        store=True,
        reasoning=Reasoning(effort="low", summary="auto"),
    ),
)

_EXECUTOR_INSTRUCTIONS = """
You are a query result simulator.
Given a SQL query and its description, return realistic-looking dummy data as a markdown table.
The data must be consistent with an active business (not null or empty).
Include at least 3–5 rows. Add a brief summary line above the table.
"""

_query_executor = Agent(
    name="Query Executor",
    model="gpt-5-mini",
    instructions=_EXECUTOR_INSTRUCTIONS,
    model_settings=ModelSettings(
        store=True,
        reasoning=Reasoning(effort="low", summary="auto"),
    ),
)

_SYNTHESISER_INSTRUCTIONS = """
You are a business analyst reporting results to a stakeholder.
Given the query results (a markdown table), write a concise, insight-driven summary:
- Start with the key finding in one sentence.
- Use bullet points to highlight notable numbers or trends.
- End with a short actionable observation if appropriate.
Answer in the same language the user used.
"""

_synthesiser = Agent(
    name="Data Synthesiser",
    model="gpt-5-mini",
    instructions=_SYNTHESISER_INSTRUCTIONS,
    model_settings=ModelSettings(
        store=True,
        reasoning=Reasoning(effort="low", summary="auto"),
    ),
)


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------

async def run_structured_qa(
    conversation: list,
    rewritten_query: str,
) -> AsyncGenerator[str, None]:
    log.info("structured_qa.start", rewritten_query=rewritten_query)
    # 1. Domain selection
    domain_result = await Runner.run(
        _domain_selector,
        [*conversation, {"role": "user", "content": [{"type": "input_text", "text": rewritten_query}]}],
    )
    domain = domain_result.final_output.domain
    log.info("structured_qa.domain_selected", domain=domain)

    if domain == "other":
        log.warning("structured_qa.domain_unknown", rewritten_query=rewritten_query)
        yield (
            "La domanda non sembra riguardare i dati strutturati disponibili (commerce o personnel). "
            "Prova a riformulare la richiesta specificando il tipo di dati che ti interessa."
        )
        return

    # 2. Query creation
    query_input = (
        f"Domain: {domain}\n"
        f"Question: {rewritten_query}"
    )
    query_result = await Runner.run(
        _query_creator,
        [{"role": "user", "content": [{"type": "input_text", "text": query_input}]}],
    )
    plan = query_result.final_output
    log.info("structured_qa.query_plan", sql=plan.query, explanation=plan.explanation)

    # 3. Query execution (simulation)
    exec_input = (
        f"SQL query:\n```sql\n{plan.query}\n```\n\n"
        f"Description: {plan.explanation}"
    )
    exec_result = await Runner.run(
        _query_executor,
        [{"role": "user", "content": [{"type": "input_text", "text": exec_input}]}],
    )
    raw_data = str(exec_result.final_output)
    log.debug("structured_qa.raw_data_preview", preview=raw_data[:200])

    # 4. Synthesis — streamed
    synth_input = (
        f"Original question: {rewritten_query}\n\n"
        f"Query: {plan.query}\n\n"
        f"Results:\n{raw_data}"
    )
    log.info("structured_qa.synthesising")
    streamed = Runner.run_streamed(
        _synthesiser,
        [*conversation, {"role": "user", "content": [{"type": "input_text", "text": synth_input}]}],
    )
    async for event in streamed.stream_events():
        if not isinstance(event, RawResponsesStreamEvent):
            continue
        raw = event.data
        if getattr(raw, "type", None) == "response.output_text.delta":
            delta = getattr(raw, "delta", "")
            if delta:
                yield delta
