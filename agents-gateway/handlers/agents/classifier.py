"""
Classification agent — query rewrite + intent routing.

Produces a structured result with:
  - route: which specialised agent to invoke
  - rewritten_query: a clearer, context-enriched version of the user's request
"""

from typing import Literal

from pydantic import BaseModel

from agents import Agent, ModelSettings, Runner
from openai.types.shared.reasoning import Reasoning
from logger import get_logger

log = get_logger("classifier")


class ClassificationResult(BaseModel):
    route: Literal["doc_kb", "structured_qa", "booking", "conversational", "out_of_scope"]
    rewritten_query: str


_INSTRUCTIONS = """
You are a classification and query-rewrite agent for an enterprise assistant.

The system supports three operational domains:

  • doc_kb        — ANY question that may be answered from documents, manuals, guides,
                    procedures, FAQs, policies, reports, or files attached by the user.
                    When in doubt, prefer this route — it is better to search and find
                    nothing than to refuse a legitimate question.
  • structured_qa — questions that explicitly ask for numbers, statistics, KPIs, sales
                    figures, customer counts, orders, headcount, payroll, or similar
                    quantitative data from a database.
  • booking       — explicit intent to create, view, modify or cancel a calendar appointment
                    or reservation.
  • out_of_scope  — use ONLY when the request is clearly unrelated to any enterprise context
                    (e.g. personal life advice, creative writing, coding help unrelated to the
                    company). If there is any chance the question relates to internal knowledge,
                    choose doc_kb instead.

Steps:
1. Analyse the full conversation history.
2. Rewrite the user's last request into a clear, self-contained query that retains all
   relevant context (expand pronouns, include time ranges mentioned earlier, etc.).
3. Choose the single best route. Default to doc_kb when uncertain.

Set rewritten_query to the improved query string (never empty).
"""

_classifier = Agent(
    name="Classifier",
    model="gpt-5-mini",
    instructions=_INSTRUCTIONS,
    output_type=ClassificationResult,
    model_settings=ModelSettings(
        store=True,
        reasoning=Reasoning(effort="low", summary="auto"),
    ),
)


async def classify(conversation: list) -> ClassificationResult:
    log.debug("classifier.run", msg_count=len(conversation))
    result = await Runner.run(_classifier, conversation)
    out = result.final_output
    log.info("classifier.output", route=out.route, rewritten_query=out.rewritten_query)
    return out
