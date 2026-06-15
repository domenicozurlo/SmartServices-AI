from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StructuredQaTool(str, Enum):
    LATEST_SENSOR_READINGS = "get_latest_sensor_readings"
    SENSOR_READINGS_BY_PERIOD = "get_sensor_readings_by_period"
    DAILY_TEMPERATURE_HUMIDITY = "get_daily_temperature_humidity"
    SOIL_MOISTURE_SUMMARY = "get_soil_moisture_summary"
    BATTERY_STATUS = "get_battery_status"
    RF_SIGNAL_STATUS = "get_rf_signal_status"
    PH_EC_TREND = "get_ph_ec_trend"
    DAILY_RAIN_ET = "get_daily_rain_et"
    DATA_QUALITY_SUMMARY = "get_data_quality_summary"


class StructuredQaParameters(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    seriale: Optional[str] = Field(default=None, alias="seriale")
    from_date: Optional[str] = Field(default=None, alias="fromDate")
    to_date: Optional[str] = Field(default=None, alias="toDate")
    limit: int = Field(default=10, ge=1, le=50)
    threshold: Optional[float] = Field(default=None, ge=0)

    @field_validator("seriale")
    @classmethod
    def normalize_seriale(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = value.strip().upper()
        return normalized or None

    @field_validator("from_date", "to_date")
    @classmethod
    def validate_iso_date(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        from datetime import date

        date.fromisoformat(value)
        return value


class StructuredQaIntent(BaseModel):
    tool_name: Optional[StructuredQaTool] = None
    parameters: StructuredQaParameters = Field(default_factory=StructuredQaParameters)
    requires_clarification: bool = False
    unsupported_request: bool = False
    clarification_message: str = ""


class StructuredQaToolCall(BaseModel):
    name: StructuredQaTool
    parameters: StructuredQaParameters


class StructuredQaAdapterResult(BaseModel):
    tool_name: StructuredQaTool
    rows: list[dict[str, object]]
    metadata: dict[str, object] = Field(default_factory=dict)


class StructuredQaResponse(BaseModel):
    answer: str
    data_source: str
    used_tools: list[dict[str, object]]
    filters: dict[str, object]
    requires_clarification: bool
    unsupported_request: bool
