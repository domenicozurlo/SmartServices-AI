import asyncio
import json
from abc import ABC, abstractmethod
from datetime import datetime
from urllib import request

from handlers.structured_qa.config import StructuredQaConfig
from handlers.structured_qa.types import (
    StructuredQaAdapterResult,
    StructuredQaTool,
    StructuredQaToolCall,
)


class StructuredQaAdapterError(RuntimeError):
    pass


class StructuredQaConfigurationError(StructuredQaAdapterError):
    pass


class StructuredQaDataAdapter(ABC):
    @property
    @abstractmethod
    def data_source(self) -> str:
        raise NotImplementedError

    @abstractmethod
    async def call_tool(self, call: StructuredQaToolCall) -> StructuredQaAdapterResult:
        raise NotImplementedError


class MockStructuredQaDbAdapter(StructuredQaDataAdapter):
    def __init__(self) -> None:
        self._rows = _mock_senshistory_rows()

    @property
    def data_source(self) -> str:
        return "mock_db"

    async def call_tool(self, call: StructuredQaToolCall) -> StructuredQaAdapterResult:
        params = call.parameters
        if params.seriale == "ERROR":
            raise StructuredQaAdapterError("mock_adapter_failure")
        if params.seriale in {"EMPTY", "NO_DATA"}:
            return StructuredQaAdapterResult(tool_name=call.name, rows=[], metadata={"mock": True})

        builders = {
            StructuredQaTool.LATEST_SENSOR_READINGS: self._latest_sensor_readings,
            StructuredQaTool.SENSOR_READINGS_BY_PERIOD: self._sensor_readings_by_period,
            StructuredQaTool.DAILY_TEMPERATURE_HUMIDITY: self._daily_temperature_humidity,
            StructuredQaTool.SOIL_MOISTURE_SUMMARY: self._soil_moisture_summary,
            StructuredQaTool.BATTERY_STATUS: self._battery_status,
            StructuredQaTool.RF_SIGNAL_STATUS: self._rf_signal_status,
            StructuredQaTool.PH_EC_TREND: self._ph_ec_trend,
            StructuredQaTool.DAILY_RAIN_ET: self._daily_rain_et,
            StructuredQaTool.DATA_QUALITY_SUMMARY: self._data_quality_summary,
        }
        rows = builders[call.name](call)
        return StructuredQaAdapterResult(
            tool_name=call.name,
            rows=rows,
            metadata={"mock": True, "dataset": "synthetic_irrifarm_ai_senshistory"},
        )

    def _latest_sensor_readings(self, call: StructuredQaToolCall) -> list[dict[str, object]]:
        rows = self._for_seriale(call.parameters.seriale)
        return sorted(rows, key=lambda row: row["SNS_DtEvent"] or "", reverse=True)[: call.parameters.limit]

    def _sensor_readings_by_period(self, call: StructuredQaToolCall) -> list[dict[str, object]]:
        return self._period_rows(call)

    def _daily_temperature_humidity(self, call: StructuredQaToolCall) -> list[dict[str, object]]:
        return _group_daily(
            self._period_rows(call),
            ("SNS_TempAmb", "SNS_UmdAmb"),
            ("avg_temp", "avg_air_humidity"),
        )

    def _soil_moisture_summary(self, call: StructuredQaToolCall) -> list[dict[str, object]]:
        return _group_daily(
            self._period_rows(call),
            ("SNS_Umd1Liv", "SNS_Umd2Liv", "SNS_Umd3Liv", "SNS_Umd4Liv", "SNS_Umd5Liv", "SNS_Umd6Liv"),
            (
                "avg_soil_humidity_1",
                "avg_soil_humidity_2",
                "avg_soil_humidity_3",
                "avg_soil_humidity_4",
                "avg_soil_humidity_5",
                "avg_soil_humidity_6",
            ),
        )

    def _battery_status(self, call: StructuredQaToolCall) -> list[dict[str, object]]:
        threshold = call.parameters.threshold if call.parameters.threshold is not None else 20
        rows = self._latest_sensor_readings(call)
        return [
            {
                "MBO_SN": row["MBO_SN"],
                "SNS_DtEvent": row["SNS_DtEvent"],
                "SNS_BattLiv": row["SNS_BattLiv"],
                "status": "LOW" if float(row["SNS_BattLiv"]) < threshold else "OK",
            }
            for row in rows
        ]

    def _rf_signal_status(self, call: StructuredQaToolCall) -> list[dict[str, object]]:
        threshold = call.parameters.threshold if call.parameters.threshold is not None else 20
        rows = self._latest_sensor_readings(call)
        return [
            {
                "MBO_SN": row["MBO_SN"],
                "SNS_DtEvent": row["SNS_DtEvent"],
                "SNS_RFLiv": row["SNS_RFLiv"],
                "status": "LOW" if int(row["SNS_RFLiv"]) < threshold else "OK",
            }
            for row in rows
        ]

    def _ph_ec_trend(self, call: StructuredQaToolCall) -> list[dict[str, object]]:
        return [
            {
                "MBO_SN": row["MBO_SN"],
                "SNS_DtEvent": row["SNS_DtEvent"],
                "SNS_PH": row["SNS_PH"],
                "SNS_EC": row["SNS_EC"],
            }
            for row in self._period_rows(call)
        ][: call.parameters.limit]

    def _daily_rain_et(self, call: StructuredQaToolCall) -> list[dict[str, object]]:
        grouped = _group_daily(
            self._period_rows(call),
            ("SNS_ARK", "SNS_ETL", "SNS_ETD"),
            ("total_rain", "avg_eto_hour", "avg_eto_24h"),
            sums={"SNS_ARK"},
        )
        return grouped[: call.parameters.limit]

    def _data_quality_summary(self, call: StructuredQaToolCall) -> list[dict[str, object]]:
        total = len(self._rows)
        missing_event = sum(1 for row in self._rows if row["SNS_DtEvent"] is None)
        zero_core = sum(
            1
            for row in self._rows
            if row["SNS_TempAmb"] == 0
            and row["SNS_UmdAmb"] == 0
            and row["SNS_Umd1Liv"] == 0
            and row["SNS_BattLiv"] == 0
        )
        return [
            {
                "total_records": total,
                "records_without_event_date": missing_event,
                "percentage_without_event_date": round(missing_event * 100 / total, 2),
                "records_with_core_fields_zero": zero_core,
                "distinct_motherboards": len({row["MBO_SN"] for row in self._rows}),
            }
        ]

    def _period_rows(self, call: StructuredQaToolCall) -> list[dict[str, object]]:
        params = call.parameters
        rows = self._for_seriale(params.seriale)
        return [
            row
            for row in rows
            if row["SNS_DtEvent"] is not None
            and params.from_date <= str(row["SNS_DtEvent"])[:10] < params.to_date
        ][: params.limit]

    def _for_seriale(self, seriale: str | None) -> list[dict[str, object]]:
        if seriale is None:
            return self._rows
        return [row for row in self._rows if row["MBO_SN"] == seriale]


class McpStructuredQaAdapter(StructuredQaDataAdapter):
    def __init__(self, config: StructuredQaConfig):
        self._config = config

    @property
    def data_source(self) -> str:
        return "mcp_toolbox"

    async def call_tool(self, call: StructuredQaToolCall) -> StructuredQaAdapterResult:
        if not self._config.mcp_enabled:
            raise StructuredQaConfigurationError("mcp_disabled")
        if not self._config.mcp_url:
            raise StructuredQaConfigurationError("mcp_url_missing")
        if not self._config.mcp_toolset:
            raise StructuredQaConfigurationError("mcp_toolset_missing")

        payload = {
            "jsonrpc": "2.0",
            "id": f"structured_qa_{call.name.value}",
            "method": "tools/call",
            "params": {
                "name": call.name.value,
                "arguments": _tool_arguments(call),
            },
        }
        response = await asyncio.to_thread(
            _post_json_rpc,
            self._config.mcp_url,
            payload,
            self._config.mcp_timeout_seconds,
        )
        rows = _rows_from_mcp_response(response)
        return StructuredQaAdapterResult(
            tool_name=call.name,
            rows=rows,
            metadata={"toolset": self._config.mcp_toolset},
        )


def build_adapter(config: StructuredQaConfig) -> StructuredQaDataAdapter:
    if config.mode == "mcp":
        return McpStructuredQaAdapter(config)
    return MockStructuredQaDbAdapter()


def _tool_arguments(call: StructuredQaToolCall) -> dict[str, object]:
    arguments = call.parameters.model_dump(by_alias=True, exclude_none=True)
    if call.name in {StructuredQaTool.LATEST_SENSOR_READINGS, StructuredQaTool.SENSOR_READINGS_BY_PERIOD}:
        arguments["seriale"] = arguments.get("seriale", "")
    return arguments


def _post_json_rpc(url: str, payload: dict[str, object], timeout_seconds: float) -> dict[str, object]:
    body = json.dumps(payload).encode("utf-8")
    http_request = request.Request(
        url=url,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(http_request, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except OSError as exc:
        raise StructuredQaAdapterError("mcp_toolbox_request_failed") from exc


def _rows_from_mcp_response(response: dict[str, object]) -> list[dict[str, object]]:
    if "error" in response:
        raise StructuredQaAdapterError("mcp_toolbox_error")
    result = response.get("result")
    if isinstance(result, dict):
        rows = result.get("rows") or result.get("data")
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)]
    if isinstance(result, list):
        return [row for row in result if isinstance(row, dict)]
    raise StructuredQaAdapterError("mcp_toolbox_malformed_response")


def _group_daily(
    rows: list[dict[str, object]],
    source_fields: tuple[str, ...],
    output_fields: tuple[str, ...],
    sums: set[str] | None = None,
) -> list[dict[str, object]]:
    sums = sums or set()
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        day = str(row["SNS_DtEvent"])[:10]
        grouped.setdefault(day, []).append(row)

    summaries = []
    for day, day_rows in sorted(grouped.items()):
        summary: dict[str, object] = {
            "day": day,
            "readings_count": len(day_rows),
            "MBO_SN": day_rows[0]["MBO_SN"],
        }
        for source_field, output_field in zip(source_fields, output_fields):
            values = [float(row[source_field]) for row in day_rows]
            value = sum(values) if source_field in sums else sum(values) / len(values)
            summary[output_field] = round(value, 2)
        summaries.append(summary)
    return summaries


def _mock_senshistory_rows() -> list[dict[str, object]]:
    base_rows = [
        ("IRR-001", "2026-06-01 06:00:00", 21.4, 67.2, 42.5, 39.8, 36.1, 35.0, 33.4, 31.9, 88.0, 74, 6.4, 1260, 0.0, 0.21, 2.4, 18.0),
        ("IRR-001", "2026-06-01 12:00:00", 27.9, 52.6, 39.1, 37.2, 34.4, 33.8, 32.6, 31.1, 87.5, 72, 6.5, 1285, 0.0, 0.34, 2.5, 24.0),
        ("IRR-001", "2026-06-02 06:00:00", 20.9, 69.1, 44.2, 40.5, 37.3, 35.9, 33.8, 32.0, 87.2, 71, 6.4, 1274, 1.8, 0.18, 2.3, 17.5),
        ("IRR-001", "2026-06-02 12:00:00", 28.6, 49.4, 40.6, 38.1, 35.2, 34.0, 32.9, 31.7, 86.9, 69, 6.6, 1302, 0.0, 0.39, 2.6, 26.0),
        ("IRR-001", "2026-06-03 06:00:00", 22.1, 65.0, 41.7, 39.0, 36.6, 34.7, 32.8, 31.0, 86.4, 65, 6.5, 1290, 0.4, 0.24, 2.4, 19.2),
        ("IRR-002", "2026-06-01 06:00:00", 19.8, 71.4, 51.2, 48.0, 45.5, 43.2, 41.0, 39.5, 24.0, 28, 6.8, 1180, 0.0, 0.16, 1.9, 10.0),
        ("IRR-002", "2026-06-01 12:00:00", 26.1, 55.1, 48.4, 46.2, 43.7, 41.8, 39.9, 38.1, 19.5, 17, 6.9, 1194, 0.0, 0.31, 2.1, 12.4),
        ("IRR-002", "2026-06-02 06:00:00", 20.2, 70.2, 52.3, 49.1, 46.8, 44.0, 42.1, 40.2, 18.8, 16, 6.8, 1178, 4.6, 0.14, 1.8, 8.8),
        ("IRR-002", "2026-06-02 12:00:00", 27.0, 53.9, 49.0, 46.9, 44.1, 42.2, 40.2, 38.6, 18.2, 15, 6.9, 1201, 0.0, 0.33, 2.2, 13.1),
        ("IRR-003", "2026-06-01 06:00:00", 18.9, 74.3, 29.1, 27.8, 26.0, 25.4, 24.8, 23.9, 76.3, 62, 5.9, 1450, 0.0, 0.15, 1.7, 5.1),
        ("IRR-003", "2026-06-01 12:00:00", 31.2, 43.6, 24.3, 23.1, 22.0, 21.5, 20.7, 20.0, 75.8, 59, 5.8, 1488, 0.0, 0.44, 2.9, 6.0),
    ]
    rows = []
    for index, row in enumerate(base_rows, start=1):
        rows.append(
            {
                "SNS_ID": index,
                "MBO_SN": row[0],
                "SNS_DtEvent": row[1],
                "SNS_IdSlave": 1,
                "SNS_TempAmb": row[2],
                "SNS_UmdAmb": row[3],
                "SNS_Umd1Liv": row[4],
                "SNS_Umd2Liv": row[5],
                "SNS_Umd3Liv": row[6],
                "SNS_Umd4Liv": row[7],
                "SNS_Umd5Liv": row[8],
                "SNS_Umd6Liv": row[9],
                "SNS_BattLiv": row[10],
                "SNS_RFLiv": row[11],
                "SNS_PH": row[12],
                "SNS_EC": row[13],
                "SNS_ARK": row[14],
                "SNS_ETL": row[15],
                "SNS_ETD": row[16],
                "SNS_LitersCount": row[17],
                "SNS_Type": "N",
                "SNS_SysDateIns": datetime.fromisoformat(row[1]).strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
    rows.append(
        {
            "SNS_ID": 999,
            "MBO_SN": "IRR-004",
            "SNS_DtEvent": None,
            "SNS_IdSlave": 1,
            "SNS_TempAmb": 0,
            "SNS_UmdAmb": 0,
            "SNS_Umd1Liv": 0,
            "SNS_Umd2Liv": 0,
            "SNS_Umd3Liv": 0,
            "SNS_Umd4Liv": 0,
            "SNS_Umd5Liv": 0,
            "SNS_Umd6Liv": 0,
            "SNS_BattLiv": 0,
            "SNS_RFLiv": 0,
            "SNS_PH": 0,
            "SNS_EC": 0,
            "SNS_ARK": 0,
            "SNS_ETL": 0,
            "SNS_ETD": 0,
            "SNS_LitersCount": 0,
            "SNS_Type": "N",
            "SNS_SysDateIns": "2026-06-03 15:00:00",
        }
    )
    return rows
