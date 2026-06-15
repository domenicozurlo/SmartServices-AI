# Structured QA MCP Toolbox

This folder contains the Google MCP Toolbox for Databases configuration for
`irrifarm_ai.senshistory`.

## Mock mode

Default mode keeps using the in-memory mock adapter:

```env
STRUCTURED_QA_MODE=mock_db
STRUCTURED_QA_OUTPUT_FORMAT=text
STRUCTURED_QA_PLANNER_MODE=llm
STRUCTURED_QA_PLANNER_MODEL=gpt-5-mini
```

## MCP mode

Set these variables in `.env`:

```env
STRUCTURED_QA_MODE=mcp
STRUCTURED_QA_OUTPUT_FORMAT=text
STRUCTURED_QA_PLANNER_MODE=llm
STRUCTURED_QA_PLANNER_MODEL=gpt-5-mini
MCP_STRUCTURED_QA_ENABLED=true
MCP_STRUCTURED_QA_URL=http://structured-qa-toolbox:5000/mcp/senshistory_readonly
MCP_STRUCTURED_QA_TOOLSET=senshistory_readonly
MCP_STRUCTURED_QA_TIMEOUT_SECONDS=15
MCP_STRUCTURED_QA_ALLOWED_TOOLS=get_latest_sensor_readings,get_sensor_readings_by_period,get_daily_temperature_humidity,get_soil_moisture_summary,get_battery_status,get_rf_signal_status,get_ph_ec_trend,get_daily_rain_et,get_data_quality_summary

MYSQL_HOST=host.docker.internal
MYSQL_PORT=3306
MYSQL_DATABASE=irrifarm_ai
MYSQL_USER=your_readonly_user
MYSQL_PASSWORD=your_readonly_password
```

Use a read-only MySQL user. Do not grant write privileges.

Start Toolbox with the Docker Compose profile:

```bash
docker compose --profile mcp up -d structured-qa-toolbox agents-gateway
```

From the host, the MCP endpoint is:

```text
http://localhost:5000/mcp/senshistory_readonly
```

From `agents-gateway`, the MCP endpoint is:

```text
http://structured-qa-toolbox:5000/mcp/senshistory_readonly
```

The Toolbox config only exposes custom `mysql-sql` tools from
`senshistory_tools.yaml`. It does not expose `mysql-execute-sql`.
