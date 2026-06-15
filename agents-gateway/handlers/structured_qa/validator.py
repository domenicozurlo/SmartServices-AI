from datetime import date

from handlers.structured_qa.config import StructuredQaConfig
from handlers.structured_qa.types import StructuredQaIntent, StructuredQaTool, StructuredQaToolCall


class StructuredQaValidationError(ValueError):
    pass


class StructuredQaToolNotAllowedError(StructuredQaValidationError):
    pass


def validate_tool_call(intent: StructuredQaIntent, config: StructuredQaConfig) -> StructuredQaToolCall:
    if intent.tool_name is None:
        raise StructuredQaValidationError("unsupported_tool")
    if intent.tool_name.value not in config.allowed_tools:
        raise StructuredQaToolNotAllowedError("tool_not_allowed")

    params = intent.parameters
    required = _required_parameters(intent.tool_name)
    values = {
        "seriale": params.seriale,
        "fromDate": params.from_date,
        "toDate": params.to_date,
    }
    missing = [name for name in required if values[name] is None]
    if missing:
        raise StructuredQaValidationError(f"missing_parameters:{','.join(missing)}")

    if params.from_date and params.to_date and date.fromisoformat(params.to_date) <= date.fromisoformat(params.from_date):
        raise StructuredQaValidationError("invalid_date_range")

    return StructuredQaToolCall(name=intent.tool_name, parameters=params)


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
