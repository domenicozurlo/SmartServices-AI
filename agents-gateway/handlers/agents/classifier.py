"""
Classification agent - query rewrite + intent routing.

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
    search_query: str = ""
    """Short, focused RAG search query (1-2 sentences max). Falls back to rewritten_query if empty."""


_INSTRUCTIONS = """
You are a classification and query-rewrite agent for an enterprise assistant.

The system supports these operational domains:

  - doc_kb        - ANY question that may be answered from documents, manuals, guides,
                    procedures, FAQs, policies, reports, or files attached by the user.
                    When in doubt, prefer this route; it is better to search and find
                    nothing than to refuse a legitimate question.

  - structured_qa - questions that require checking, retrieving, counting, aggregating,
                    validating or summarising structured records from the database.
                    In this project the supported database domain is
                    irrifarm_ai.senshistory.

                    Route to structured_qa when the user asks to verify something in the DB,
                    asks for "dal db", "nel database", "letture", "ultimi record", "controlla",
                    "verifica", "quanti", "media", "trend", "stato", "anomalia", or "qualita"
                    and the subject is compatible with sensor/agronomic data.

                    Also route to structured_qa when the user asks about:
                    - centraline, motherboard, MBO_SN, seriale, slave or sensor readings;
                    - senshistory records, latest readings, readings by date range, counts,
                      averages, trends, summaries, status, anomalies or data quality;
                    - temperature, air humidity, soil humidity levels, battery level,
                      RF/radio signal, pH, EC, pressure, wind, solar radiation, leaf wetness,
                      rain/precipitation, evapotranspiration, liters, power or irrigation data;
                    - operational checks such as low battery, low RF signal, records without
                      event date, duplicated readings, inactive devices or out-of-range values.

                    Typical structured_qa examples:
                    - "Mostrami le letture della centralina IRR-001 dal 2026-06-01 al 2026-06-10"
                    - "Qual e' la media giornaliera di temperatura e umidita per MBO_SN IRR-001?"
                    - "Controlla batteria e segnale RF della centralina IRR-002"
                    - "Dammi pH ed EC per la centralina IRR-001 nel periodo indicato"
                    - "Quanti record senshistory sono senza data evento?"

                    Do not route to structured_qa just because a document describes the table.
                    If the user asks for explanations of fields, schema documentation, DDL,
                    or how a query works in an uploaded document, use doc_kb.

  - booking       - explicit intent to create, view, modify or cancel a calendar appointment
                    or reservation.

  - conversational - use when the user is:
                    - asking for confirmation or clarification of the assistant's PREVIOUS
                      answer (e.g. "sei sicuro?", "ne sei certo?", "davvero?", "perche?",
                      "puoi spiegarmi meglio?", "cosa intendi?", "quindi?", "ok grazie",
                      "are you sure?", "really?", "why?");
                    - making small talk or a purely social remark;
                    - asking a question entirely answered by the previous assistant turn
                      that does NOT require searching new documents.

  - out_of_scope  - use ONLY when the request is clearly unrelated to any enterprise context
                    (e.g. personal life advice, creative writing, coding help unrelated to the
                    company). If there is any chance the question relates to internal knowledge,
                    choose doc_kb instead.

Steps:
1. Analyse the full conversation history.
2. Choose the single best route. Default to structured_qa when the question asks for
   sensor/agronomic values, aggregates, status or anomalies from senshistory. Default to
   doc_kb when the question is about documents or schema explanations. Default to
   conversational when the question is meta / social / a follow-up on the assistant's
   previous answer.
3. Set `rewritten_query` to a clear, self-contained version of the user's question that
   retains all relevant context. For structured_qa, preserve exact centralina serials
   such as MBO_SN/IRR-001, date ranges, thresholds, requested metrics and limits.
   For conversational routes this can be a brief summary.
4. Set `search_query` to a SHORT, keyword-focused query (1 sentence, max 10-12 words)
   optimised for semantic vector search. It must capture only the core concept needed to
   find the right document chunks; no elaborations, no requirements lists, no fillers.
   Example: "come resettare IQOS 3 DUO" not "Come resettare il dispositivo IQOS? Fornire
   la procedura passo-passo ...". For non-doc_kb routes, set search_query = rewritten_query.

Set rewritten_query and search_query (never empty).
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
