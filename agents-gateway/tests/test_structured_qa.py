import asyncio
import importlib.util
import json
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


GATEWAY_ROOT = Path(__file__).resolve().parents[1]

handlers_pkg = types.ModuleType("handlers")
handlers_pkg.__path__ = [str(GATEWAY_ROOT / "handlers")]
sys.modules["handlers"] = handlers_pkg

agents_pkg = types.ModuleType("handlers.agents")
agents_pkg.__path__ = [str(GATEWAY_ROOT / "handlers" / "agents")]
sys.modules["handlers.agents"] = agents_pkg


class _TestLogger:
    def info(self, *_args, **_kwargs) -> None:
        pass

    def warning(self, *_args, **_kwargs) -> None:
        pass

    def error(self, *_args, **_kwargs) -> None:
        pass


logger_module = types.ModuleType("logger")
logger_module.get_logger = lambda _name: _TestLogger()
sys.modules["logger"] = logger_module

structured_qa_spec = importlib.util.spec_from_file_location(
    "handlers.agents.structured_qa",
    GATEWAY_ROOT / "handlers" / "agents" / "structured_qa.py",
)
structured_qa_module = importlib.util.module_from_spec(structured_qa_spec)
sys.modules["handlers.agents.structured_qa"] = structured_qa_module
structured_qa_spec.loader.exec_module(structured_qa_module)
run_structured_qa = structured_qa_module.run_structured_qa

from handlers.structured_qa.adapter import McpStructuredQaAdapter, MockStructuredQaDbAdapter, build_adapter
from handlers.structured_qa.config import StructuredQaConfig, load_config
from handlers.structured_qa.formatter import EMPTY_RESULT_MESSAGE, UNSUPPORTED_MESSAGE
from handlers.structured_qa.types import StructuredQaParameters, StructuredQaTool, StructuredQaToolCall


async def _run(question: str) -> dict:
    env = {"STRUCTURED_QA_OUTPUT_FORMAT": "json", "STRUCTURED_QA_PLANNER_MODE": "deterministic"}
    with patch.dict(os.environ, env, clear=False):
        chunks = [chunk async for chunk in run_structured_qa([], question)]
        return json.loads("".join(chunks))


async def _run_text(question: str) -> str:
    env = {"STRUCTURED_QA_OUTPUT_FORMAT": "text", "STRUCTURED_QA_PLANNER_MODE": "deterministic"}
    with patch.dict(os.environ, env, clear=False):
        chunks = [chunk async for chunk in run_structured_qa([], question)]
        return "".join(chunks)


class StructuredQaTest(unittest.TestCase):
    def test_supported_question_uses_mock_db_adapter(self) -> None:
        with patch.dict(os.environ, {"STRUCTURED_QA_MODE": "mock_db"}, clear=False):
            result = asyncio.run(
                _run("Mostrami le letture della centralina IRR-001 dal 2026-06-01 al 2026-06-10")
            )

        self.assertEqual(result["data_source"], "mock_db")
        self.assertFalse(result["requires_clarification"])
        self.assertFalse(result["unsupported_request"])
        self.assertEqual(result["used_tools"][0]["name"], "get_sensor_readings_by_period")
        self.assertEqual(result["used_tools"][0]["parameters"]["seriale"], "IRR-001")
        self.assertNotIn("SELECT", json.dumps(result).upper())

    def test_unsupported_request(self) -> None:
        with patch.dict(os.environ, {"STRUCTURED_QA_MODE": "mock_db"}, clear=False):
            result = asyncio.run(_run("Crea una dashboard dei risultati per centralina IRR-001"))

        self.assertTrue(result["unsupported_request"])
        self.assertEqual(result["answer"], UNSUPPORTED_MESSAGE)
        self.assertEqual(result["used_tools"], [])

    def test_missing_parameters_requires_clarification(self) -> None:
        with patch.dict(os.environ, {"STRUCTURED_QA_MODE": "mock_db"}, clear=False):
            result = asyncio.run(_run("Mostrami le medie giornaliere di temperatura e umidita ambiente"))

        self.assertTrue(result["requires_clarification"])
        self.assertIn("seriale", result["answer"])
        self.assertEqual(result["used_tools"], [])

    def test_malformed_date_requires_clarification(self) -> None:
        with patch.dict(os.environ, {"STRUCTURED_QA_MODE": "mock_db"}, clear=False):
            result = asyncio.run(
                _run("Mostrami le letture della centralina IRR-001 dal 2026-99-01 al 2026-06-10")
            )

        self.assertTrue(result["requires_clarification"])
        self.assertIn("YYYY-MM-DD", result["answer"])

    def test_empty_result(self) -> None:
        with patch.dict(os.environ, {"STRUCTURED_QA_MODE": "mock_db"}, clear=False):
            result = asyncio.run(
                _run("Mostrami le letture della centralina EMPTY dal 2026-06-01 al 2026-06-10")
            )

        self.assertEqual(result["answer"], EMPTY_RESULT_MESSAGE)
        self.assertEqual(result["data_source"], "mock_db")

    def test_adapter_error(self) -> None:
        with patch.dict(os.environ, {"STRUCTURED_QA_MODE": "mock_db"}, clear=False):
            result = asyncio.run(
                _run("Mostrami le letture della centralina ERROR dal 2026-06-01 al 2026-06-10")
            )

        self.assertEqual(result["data_source"], "mock_db")
        self.assertIn("Non riesco", result["answer"])
        self.assertEqual(result["used_tools"][0]["name"], "get_sensor_readings_by_period")

    def test_tool_outside_allowlist(self) -> None:
        env = {
            "STRUCTURED_QA_MODE": "mock_db",
            "MCP_STRUCTURED_QA_ALLOWED_TOOLS": "get_battery_status",
        }
        with patch.dict(os.environ, env, clear=False):
            result = asyncio.run(
                _run("Mostrami le letture della centralina IRR-001 dal 2026-06-01 al 2026-06-10")
            )

        self.assertTrue(result["unsupported_request"])
        self.assertEqual(result["used_tools"], [])

    def test_battery_status_includes_default_threshold(self) -> None:
        with patch.dict(os.environ, {"STRUCTURED_QA_MODE": "mock_db"}, clear=False):
            result = asyncio.run(_run("Controlla la batteria della centralina IRR-002"))

        self.assertEqual(result["used_tools"][0]["name"], "get_battery_status")
        self.assertEqual(result["used_tools"][0]["parameters"]["threshold"], 20.0)

    def test_latest_readings_from_db_does_not_require_seriale_or_dates(self) -> None:
        with patch.dict(os.environ, {"STRUCTURED_QA_MODE": "mock_db"}, clear=False):
            result = asyncio.run(_run("dammi ultime 10 letture dal db"))

        self.assertFalse(result["requires_clarification"])
        self.assertEqual(result["used_tools"][0]["name"], "get_latest_sensor_readings")
        self.assertNotIn("seriale", result["filters"])

    def test_mbo_sn_is_not_used_as_fake_seriale(self) -> None:
        with patch.dict(os.environ, {"STRUCTURED_QA_MODE": "mock_db"}, clear=False):
            result = asyncio.run(_run("dammi ultime 10 letture dal db MBO_SN"))

        self.assertFalse(result["requires_clarification"])
        self.assertEqual(result["used_tools"][0]["name"], "get_latest_sensor_readings")
        self.assertNotIn("seriale", result["filters"])

    def test_month_range_defaults_to_period_readings_without_fake_seriale(self) -> None:
        with patch.dict(os.environ, {"STRUCTURED_QA_MODE": "mock_db"}, clear=False):
            result = asyncio.run(_run("dammi le letture dal db da maggio a giugno 2026"))

        self.assertFalse(result["requires_clarification"])
        self.assertEqual(result["used_tools"][0]["name"], "get_sensor_readings_by_period")
        self.assertEqual(result["filters"]["fromDate"], "2026-05-01")
        self.assertEqual(result["filters"]["toDate"], "2026-07-01")
        self.assertNotIn("seriale", result["filters"])

    def test_plain_month_range_does_not_select_ph_ec_or_fake_con_seriale(self) -> None:
        with patch.dict(os.environ, {"STRUCTURED_QA_MODE": "mock_db"}, clear=False):
            result = asyncio.run(_run("da maggio a giugno 2026"))

        self.assertEqual(result["used_tools"][0]["name"], "get_sensor_readings_by_period")
        self.assertNotEqual(result["used_tools"][0]["name"], "get_ph_ec_trend")
        self.assertNotIn("seriale", result["filters"])

    def test_default_chat_output_is_not_raw_json(self) -> None:
        with patch.dict(os.environ, {"STRUCTURED_QA_MODE": "mock_db"}, clear=False):
            result = asyncio.run(_run_text("dammi ultime 10 letture dal db"))

        self.assertIn("Fonte dati: mock_db", result)
        self.assertIn("Tool usato: get_latest_sensor_readings", result)
        self.assertFalse(result.strip().startswith("{"))

    def test_month_all_centraline_response_contains_analysis(self) -> None:
        with patch.dict(os.environ, {"STRUCTURED_QA_MODE": "mock_db"}, clear=False):
            result = asyncio.run(_run_text("giugno 2026, per tutte le centraline"))

        self.assertIn("Sintesi valori:", result)
        self.assertIn("Temperatura media:", result)
        self.assertIn("Umidita' ambiente media:", result)
        self.assertIn("Ultimi valori per centralina:", result)

    def test_values_request_contains_reading_table(self) -> None:
        with patch.dict(os.environ, {"STRUCTURED_QA_MODE": "mock_db"}, clear=False):
            result = asyncio.run(_run_text("dammi i valori delle letture da maggio a giugno 2026"))

        self.assertIn("Valori letture:", result)
        self.assertIn("| Data evento | Centralina | Temp |", result)
        self.assertIn("IRR-001", result)

    def test_trend_request_contains_deltas(self) -> None:
        with patch.dict(os.environ, {"STRUCTURED_QA_MODE": "mock_db"}, clear=False):
            result = asyncio.run(_run_text("mostrami il trend delle letture da maggio a giugno 2026"))

        self.assertIn("Trend per centralina:", result)
        self.assertIn("temp", result)

    def test_mock_mode_does_not_build_mcp_adapter(self) -> None:
        with patch.dict(os.environ, {"STRUCTURED_QA_MODE": "mock_db"}, clear=False):
            adapter = build_adapter(load_config())

        self.assertIsInstance(adapter, MockStructuredQaDbAdapter)

    def test_mcp_mode_skeleton_config(self) -> None:
        env = {
            "STRUCTURED_QA_MODE": "mcp",
            "MCP_STRUCTURED_QA_URL": "http://localhost:5000/mcp/senshistory_readonly",
            "MCP_STRUCTURED_QA_TOOLSET": "senshistory_readonly",
            "MCP_STRUCTURED_QA_TIMEOUT_SECONDS": "7",
            "MCP_STRUCTURED_QA_ENABLED": "true",
            "MCP_STRUCTURED_QA_ALLOWED_TOOLS": "get_sensor_readings_by_period,get_battery_status",
        }
        with patch.dict(os.environ, env, clear=False):
            config = load_config()
            adapter = build_adapter(config)

        self.assertEqual(config.mode, "mcp")
        self.assertEqual(config.data_source, "mcp_toolbox")
        self.assertEqual(config.mcp_timeout_seconds, 7)
        self.assertEqual(config.allowed_tools, ("get_sensor_readings_by_period", "get_battery_status"))
        self.assertIsInstance(adapter, McpStructuredQaAdapter)

    def test_normalized_output_shape(self) -> None:
        with patch.dict(os.environ, {"STRUCTURED_QA_MODE": "mock_db"}, clear=False):
            result = asyncio.run(
                _run("Dammi il pH ed EC della centralina IRR-001 dal 2026-06-01 al 2026-06-10")
            )

        self.assertEqual(
            set(result.keys()),
            {"answer", "data_source", "used_tools", "filters", "requires_clarification", "unsupported_request"},
        )
        self.assertEqual(result["used_tools"][0]["name"], "get_ph_ec_trend")

    def test_mcp_adapter_calls_structured_tool(self) -> None:
        config = StructuredQaConfig(
            mode="mcp",
            mcp_url="http://toolbox.example/mcp/senshistory_readonly",
            mcp_toolset="senshistory_readonly",
            mcp_timeout_seconds=3,
            mcp_enabled=True,
            allowed_tools=("get_sensor_readings_by_period",),
        )
        call = StructuredQaToolCall(
            name=StructuredQaTool.SENSOR_READINGS_BY_PERIOD,
            parameters=StructuredQaParameters(
                seriale="IRR-001",
                fromDate="2026-06-01",
                toDate="2026-06-10",
                limit=5,
            ),
        )

        captured = {}

        def fake_post(url, payload, timeout):
            captured["url"] = url
            captured["payload"] = payload
            captured["timeout"] = timeout
            return {"result": {"rows": [{"MBO_SN": "IRR-001", "SNS_DtEvent": "2026-06-01 06:00:00"}]}}

        with patch("handlers.structured_qa.adapter._post_json_rpc", fake_post):
            result = asyncio.run(McpStructuredQaAdapter(config).call_tool(call))

        self.assertEqual(captured["url"], "http://toolbox.example/mcp/senshistory_readonly")
        self.assertEqual(captured["payload"]["method"], "tools/call")
        self.assertEqual(captured["payload"]["params"]["name"], "get_sensor_readings_by_period")
        self.assertEqual(captured["payload"]["params"]["arguments"]["seriale"], "IRR-001")
        self.assertEqual(captured["timeout"], 3)
        self.assertEqual(result.rows[0]["MBO_SN"], "IRR-001")

    def test_toolbox_config_is_readonly_custom_tools(self) -> None:
        tools_yaml = GATEWAY_ROOT / "toolbox" / "senshistory_tools.yaml"
        content = tools_yaml.read_text(encoding="utf-8")

        self.assertIn("type: mysql", content)
        self.assertIn("type: mysql-sql", content)
        self.assertIn("name: senshistory_readonly", content)
        self.assertNotIn("mysql-execute-sql", content)
        self.assertNotIn("execute_sql", content)


if __name__ == "__main__":
    unittest.main()
