import os
import re
from datetime import date

from pydantic import ValidationError

from handlers.structured_qa.types import (
    StructuredQaIntent,
    StructuredQaParameters,
    StructuredQaTool,
)

try:
    from agents import Agent, ModelSettings, Runner
    from openai.types.shared.reasoning import Reasoning

    _AGENTS_AVAILABLE = True
except ImportError:
    Agent = None
    ModelSettings = None
    Runner = None
    Reasoning = None
    _AGENTS_AVAILABLE = False


_UNSUPPORTED_TERMS = (
    "grafico",
    "grafici",
    "chart",
    "dashboard",
    "diagramma",
    "plot",
    "sql",
    "query",
)

_TOOL_PATTERNS: tuple[tuple[StructuredQaTool, tuple[str, ...]], ...] = (
    (StructuredQaTool.DATA_QUALITY_SUMMARY, ("qualita", "quality", "senza data", "duplic", "campi a zero", "data quality")),
    (StructuredQaTool.DAILY_RAIN_ET, ("pioggia", "precipit", "evapotraspir", "eto", "etd", "etl")),
    (StructuredQaTool.PH_EC_TREND, ("ph", "ec", "conducibilita", "fertirrig")),
    (StructuredQaTool.RF_SIGNAL_STATUS, ("rf", "radio", "segnale")),
    (StructuredQaTool.BATTERY_STATUS, ("batteria", "battery")),
    (StructuredQaTool.SOIL_MOISTURE_SUMMARY, ("umidita terreno", "umidita suolo", "livelli")),
    (StructuredQaTool.DAILY_TEMPERATURE_HUMIDITY, ("temperatura", "umidita ambiente", "temp")),
    (StructuredQaTool.SENSOR_READINGS_BY_PERIOD, ("letture", "storico", "misure", "sensori")),
)

_SERIALE_PATTERN = re.compile(
    r"\b(?:seriale|centralina|motherboard|mbo_sn|mbo)\s*(?:[:#=-]\s*)?([a-zA-Z0-9][a-zA-Z0-9_-]{1,15})\b",
    re.IGNORECASE,
)
_DATE_PATTERN = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")
_LIMIT_PATTERN = re.compile(r"\b(?:top|limit|limite|primi|prime|ultime|ultimi)\s+(\d{1,2})\b", re.IGNORECASE)
_THRESHOLD_PATTERN = re.compile(r"\b(?:soglia|threshold|inferiore a|sotto|<)\s*(\d+(?:[.,]\d+)?)\b", re.IGNORECASE)
_UPPER_TOKEN_PATTERN = re.compile(r"\b[A-Z][A-Z0-9_-]{2,15}\b")
_LATEST_PATTERN = re.compile(r"\b(ultime|ultimi|recenti|ultima\s+lettura|ultimi\s+valori)\b", re.IGNORECASE)
_READINGS_PATTERN = re.compile(r"\b(letture|storico|misure|sensori)\b", re.IGNORECASE)
_SERIALE_STOPWORDS = {
    "SQL",
    "DB",
    "MCP",
    "QA",
    "RF",
    "EC",
    "PH",
    "SNS",
    "MBO",
    "MBO_SN",
    "CON",
    "DAL",
    "DEL",
    "NEL",
    "PER",
    "TRA",
    "FRA",
}
_MONTHS = {
    "gennaio": 1,
    "febbraio": 2,
    "marzo": 3,
    "aprile": 4,
    "maggio": 5,
    "giugno": 6,
    "luglio": 7,
    "agosto": 8,
    "settembre": 9,
    "ottobre": 10,
    "novembre": 11,
    "dicembre": 12,
}
_MONTH_RANGE_PATTERN = re.compile(r"\bda\s+([a-z]+)\s+a\s+([a-z]+)\s+(20\d{2})\b", re.IGNORECASE)
_SINGLE_MONTH_PATTERN = re.compile(
    r"\b(gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|settembre|ottobre|novembre|dicembre)\s+(20\d{2})\b",
    re.IGNORECASE,
)


_PLANNER_INSTRUCTIONS = """
You are the Structured QA Planner Agent for irrifarm_ai.senshistory.

Your only job is to map the user's request to one allowed read-only tool and typed
parameters. You must never generate SQL, table joins, column lists, schemas, or
free-form database queries.

Allowed tools:
- get_latest_sensor_readings: latest senshistory readings, optionally filtered by seriale. Required: none. Optional: seriale, limit.
- get_sensor_readings_by_period: raw readings by date range, optionally filtered by seriale. Required: fromDate, toDate. Optional: seriale, limit.
- get_daily_temperature_humidity: daily avg temperature and air humidity. Required: seriale, fromDate, toDate.
- get_soil_moisture_summary: daily avg soil humidity levels 1-6. Required: seriale, fromDate, toDate.
- get_battery_status: latest battery status. Required: seriale. Optional: threshold, limit. Default threshold: 20.
- get_rf_signal_status: latest RF/radio signal status. Required: seriale. Optional: threshold, limit. Default threshold: 20.
- get_ph_ec_trend: pH and EC readings. Required: seriale, fromDate, toDate. Optional: limit.
- get_daily_rain_et: daily rain and evapotranspiration. Required: seriale, fromDate, toDate.
- get_data_quality_summary: overall data quality summary. Required: none.

Database domain:
- Table: senshistory.
- Main entity: motherboard/centralina seriale MBO_SN.
- Sensor fields include temperature, air humidity, soil humidity levels, battery, RF, pH, EC,
  pressure, wind, leaf wetness, solar radiation, rain, evapotranspiration, liters, energy,
  event timestamp and insert timestamp.

Planning rules:
- If the user asks for latest/recent readings, use get_latest_sensor_readings.
- If the user asks for generic readings in a period, use get_sensor_readings_by_period.
- If the user asks "da maggio a giugno 2026", output fromDate="2026-05-01" and toDate="2026-07-01".
  toDate is exclusive.
- If the user asks "giugno 2026", output fromDate="2026-06-01" and toDate="2026-07-01".
- Preserve exact serials like IRR-001. Do not use generic words such as MBO_SN, DB, CON, DAL as seriale.
- If a required parameter is missing, set requires_clarification=true and explain exactly which
  parameters are needed. Do not guess seriale or dates.
- If the request asks for charts/dashboard/SQL/query text/schema explanation, mark unsupported_request=true.
- If no allowed tool can answer the request, mark unsupported_request=true.
- Return only the structured output matching the schema.
"""

_planner_agent = None


def _get_planner_agent():
    global _planner_agent
    if not _AGENTS_AVAILABLE:
        return None
    if _planner_agent is None:
        _planner_agent = Agent(
            name="Structured QA Planner",
            model=os.getenv("STRUCTURED_QA_PLANNER_MODEL", "gpt-5-mini"),
            instructions=_PLANNER_INSTRUCTIONS,
            output_type=StructuredQaIntent,
            model_settings=ModelSettings(
                store=True,
                reasoning=Reasoning(effort="low", summary="auto"),
            ),
        )
    return _planner_agent


async def plan_structured_qa(question: str) -> StructuredQaIntent:
    if os.getenv("STRUCTURED_QA_PLANNER_MODE", "llm").strip().lower() == "deterministic":
        return plan_structured_qa_deterministic(question)

    agent = _get_planner_agent()
    if agent is None:
        return plan_structured_qa_deterministic(question)

    try:
        result = await Runner.run(
            agent,
            [{"role": "user", "content": [{"type": "input_text", "text": question}]}],
        )
    except Exception:
        return plan_structured_qa_deterministic(question)

    return _normalize_intent(result.final_output, question)


def _normalize_intent(intent: StructuredQaIntent, question: str) -> StructuredQaIntent:
    if intent.unsupported_request or intent.requires_clarification:
        return intent

    fallback = plan_structured_qa_deterministic(question)
    params = intent.parameters
    if params.limit == 10:
        params.limit = fallback.parameters.limit
    if params.threshold is None:
        params.threshold = fallback.parameters.threshold
    if params.seriale and params.seriale.upper() in _SERIALE_STOPWORDS:
        params.seriale = None

    if intent.tool_name in {StructuredQaTool.BATTERY_STATUS, StructuredQaTool.RF_SIGNAL_STATUS} and params.threshold is None:
        params.threshold = 20
    return intent


def plan_structured_qa_deterministic(question: str) -> StructuredQaIntent:
    normalized_question = _normalize_text(question)
    tool_name = _select_tool(normalized_question)
    if tool_name is None:
        return StructuredQaIntent(
            unsupported_request=True,
            clarification_message="La richiesta non e' supportata dai tool structured QA attualmente disponibili.",
        )

    dates = _DATE_PATTERN.findall(question)
    period = _period_dates(normalized_question)
    threshold = _extract_threshold(question)
    if threshold is None and tool_name in {StructuredQaTool.BATTERY_STATUS, StructuredQaTool.RF_SIGNAL_STATUS}:
        threshold = 20
    try:
        parameters = StructuredQaParameters(
            seriale=_extract_seriale(question),
            fromDate=dates[0] if dates else period[0] if period else None,
            toDate=dates[1] if len(dates) > 1 else period[1] if period else None,
            limit=_extract_limit(question),
            threshold=threshold,
        )
    except ValidationError:
        return StructuredQaIntent(
            tool_name=tool_name,
            requires_clarification=True,
            clarification_message="Le date devono essere in formato ISO YYYY-MM-DD.",
        )
    values = {
        "seriale": parameters.seriale,
        "fromDate": parameters.from_date,
        "toDate": parameters.to_date,
    }
    missing = [label for label in _required_parameters(tool_name) if values[label] is None]
    if missing:
        return StructuredQaIntent(
            tool_name=tool_name,
            parameters=parameters,
            requires_clarification=True,
            clarification_message=f"Per procedere servono questi parametri: {', '.join(missing)}.",
        )

    return StructuredQaIntent(tool_name=tool_name, parameters=parameters)


def _select_tool(text: str) -> StructuredQaTool | None:
    lower = text.lower()
    normalized = _normalize_text(lower)
    if any(term in normalized for term in _UNSUPPORTED_TERMS):
        return None
    if _LATEST_PATTERN.search(normalized):
        return StructuredQaTool.LATEST_SENSOR_READINGS
    if _READINGS_PATTERN.search(normalized) and not _DATE_PATTERN.search(text) and not _period_dates(normalized):
        return StructuredQaTool.LATEST_SENSOR_READINGS
    for tool, patterns in _TOOL_PATTERNS:
        if any(_contains_pattern(normalized, pattern) for pattern in patterns):
            return tool
    if _DATE_PATTERN.search(text) or _period_dates(normalized):
        return StructuredQaTool.SENSOR_READINGS_BY_PERIOD
    return None


def _normalize_text(text: str) -> str:
    return (
        text.replace("à", "a")
        .replace("è", "e")
        .replace("é", "e")
        .replace("ì", "i")
        .replace("ò", "o")
        .replace("ù", "u")
        .replace("Ã ", "a")
        .replace("Ã¨", "e")
        .replace("Ã©", "e")
        .replace("Ã¬", "i")
        .replace("Ã²", "o")
        .replace("Ã¹", "u")
    )


def _contains_pattern(text: str, pattern: str) -> bool:
    if pattern in {"ph", "ec", "rf", "eto", "etd", "etl"}:
        return re.search(rf"\b{re.escape(pattern)}\b", text) is not None
    return pattern in text


def _extract_seriale(text: str) -> str | None:
    for match in _SERIALE_PATTERN.finditer(text):
        seriale = match.group(1).upper()
        if seriale not in _SERIALE_STOPWORDS:
            return seriale

    for token in _UPPER_TOKEN_PATTERN.findall(text):
        if token not in _SERIALE_STOPWORDS and any(char.isdigit() for char in token):
            return token
    return None


def _extract_limit(text: str) -> int:
    match = _LIMIT_PATTERN.search(text)
    if not match:
        return 10
    return max(1, min(int(match.group(1)), 50))


def _extract_threshold(text: str) -> float | None:
    match = _THRESHOLD_PATTERN.search(text)
    if not match:
        return None
    return float(match.group(1).replace(",", "."))


def _period_dates(text: str) -> tuple[str, str] | None:
    return _month_range_dates(text) or _single_month_dates(text)


def _month_range_dates(text: str) -> tuple[str, str] | None:
    match = _MONTH_RANGE_PATTERN.search(text)
    if not match:
        return None

    from_month = _MONTHS.get(match.group(1).lower())
    to_month = _MONTHS.get(match.group(2).lower())
    year = int(match.group(3))
    if from_month is None or to_month is None:
        return None

    to_year = year + 1 if to_month == 12 else year
    exclusive_to_month = 1 if to_month == 12 else to_month + 1
    return (
        date(year, from_month, 1).isoformat(),
        date(to_year, exclusive_to_month, 1).isoformat(),
    )


def _single_month_dates(text: str) -> tuple[str, str] | None:
    match = _SINGLE_MONTH_PATTERN.search(text)
    if not match:
        return None

    month = _MONTHS.get(match.group(1).lower())
    year = int(match.group(2))
    if month is None:
        return None

    to_year = year + 1 if month == 12 else year
    exclusive_to_month = 1 if month == 12 else month + 1
    return (
        date(year, month, 1).isoformat(),
        date(to_year, exclusive_to_month, 1).isoformat(),
    )


def _required_parameters(tool_name: StructuredQaTool) -> tuple[str, ...]:
    if tool_name == StructuredQaTool.SENSOR_READINGS_BY_PERIOD:
        return ("fromDate", "toDate")
    if tool_name in {
        StructuredQaTool.DAILY_TEMPERATURE_HUMIDITY,
        StructuredQaTool.SOIL_MOISTURE_SUMMARY,
        StructuredQaTool.PH_EC_TREND,
        StructuredQaTool.DAILY_RAIN_ET,
    }:
        return ("seriale", "fromDate", "toDate")
    if tool_name in {StructuredQaTool.BATTERY_STATUS, StructuredQaTool.RF_SIGNAL_STATUS}:
        return ("seriale",)
    return ()
