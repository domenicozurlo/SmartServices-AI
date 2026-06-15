from collections import defaultdict

from handlers.structured_qa.types import StructuredQaAdapterResult, StructuredQaTool, StructuredQaToolCall


_VALUE_TERMS = ("valori", "dettaglio", "dettagli", "lista", "elenco", "mostra", "dammi")
_TREND_TERMS = ("trend", "andamento", "evoluzione", "variazione", "crescita", "calo", "aumento")
_AGGREGATE_TERMS = ("media", "medie", "aggreg", "riepilogo", "sintesi", "tutte", "tutti")


def build_structured_qa_answer(question: str, call: StructuredQaToolCall, result: StructuredQaAdapterResult) -> str:
    if not result.rows:
        return "Non risultano dati per i filtri selezionati nel periodo indicato."

    mode = _analysis_mode(question, call.name)
    if _is_raw_readings(result.rows):
        return _raw_readings_answer(result.rows, mode)
    if call.name == StructuredQaTool.DAILY_TEMPERATURE_HUMIDITY:
        return _daily_temperature_humidity_answer(result.rows)
    if call.name == StructuredQaTool.SOIL_MOISTURE_SUMMARY:
        return _soil_moisture_answer(result.rows)
    if call.name == StructuredQaTool.BATTERY_STATUS:
        return _status_answer(result.rows, "SNS_BattLiv", "batteria")
    if call.name == StructuredQaTool.RF_SIGNAL_STATUS:
        return _status_answer(result.rows, "SNS_RFLiv", "segnale RF")
    if call.name == StructuredQaTool.PH_EC_TREND:
        return _ph_ec_answer(result.rows)
    if call.name == StructuredQaTool.DAILY_RAIN_ET:
        return _rain_et_answer(result.rows)
    if call.name == StructuredQaTool.DATA_QUALITY_SUMMARY:
        return _quality_answer(result.rows[0])
    return f"Sono stati recuperati {len(result.rows)} record."


def _analysis_mode(question: str, tool_name: StructuredQaTool) -> str:
    lower = question.lower()
    if any(term in lower for term in _TREND_TERMS):
        return "trend"
    if any(term in lower for term in _VALUE_TERMS):
        return "values"
    if any(term in lower for term in _AGGREGATE_TERMS):
        return "aggregate"
    if tool_name in {StructuredQaTool.SENSOR_READINGS_BY_PERIOD, StructuredQaTool.LATEST_SENSOR_READINGS}:
        return "aggregate"
    return "summary"


def _is_raw_readings(rows: list[dict[str, object]]) -> bool:
    first = rows[0]
    return {"MBO_SN", "SNS_DtEvent", "SNS_TempAmb", "SNS_UmdAmb"}.issubset(first)


def _raw_readings_answer(rows: list[dict[str, object]], mode: str) -> str:
    valid_rows = [row for row in rows if row.get("SNS_DtEvent")]
    if not valid_rows:
        return "Le letture recuperate non hanno una data evento valorizzata."

    serials = sorted({str(row["MBO_SN"]) for row in valid_rows})
    first_event = min(str(row["SNS_DtEvent"]) for row in valid_rows)
    last_event = max(str(row["SNS_DtEvent"]) for row in valid_rows)
    lines = [
        f"Ho analizzato {len(valid_rows)} letture su {len(serials)} centraline ({', '.join(serials)}) tra {first_event} e {last_event}.",
        "",
        "Sintesi valori:",
        f"- Temperatura media: {_avg(valid_rows, 'SNS_TempAmb')} C (min {_min(valid_rows, 'SNS_TempAmb')}, max {_max(valid_rows, 'SNS_TempAmb')})",
        f"- Umidita' ambiente media: {_avg(valid_rows, 'SNS_UmdAmb')}% (min {_min(valid_rows, 'SNS_UmdAmb')}, max {_max(valid_rows, 'SNS_UmdAmb')})",
        f"- Umidita' terreno livello 1 media: {_avg(valid_rows, 'SNS_Umd1Liv')}%",
        f"- Batteria minima: {_min(valid_rows, 'SNS_BattLiv')} | RF minimo: {_min(valid_rows, 'SNS_RFLiv')}",
        f"- Pioggia totale: {_sum(valid_rows, 'SNS_ARK')} mm | Litri totali: {_sum(valid_rows, 'SNS_LitersCount')}",
    ]

    low_battery = [row for row in valid_rows if _float(row, "SNS_BattLiv") < 20]
    low_rf = [row for row in valid_rows if _float(row, "SNS_RFLiv") < 20]
    if low_battery or low_rf:
        lines.extend(
            [
                "",
                "Alert:",
                f"- Letture con batteria sotto 20: {len(low_battery)}",
                f"- Letture con RF sotto 20: {len(low_rf)}",
            ]
        )

    if mode == "trend":
        lines.extend(["", "Trend per centralina:", *_trend_lines(valid_rows)])
    else:
        lines.extend(["", "Ultimi valori per centralina:", *_latest_by_serial_lines(valid_rows)])

    if mode == "values":
        lines.extend(["", "Valori letture:", *_value_table_lines(valid_rows[:10])])

    return "\n".join(lines)


def _daily_temperature_humidity_answer(rows: list[dict[str, object]]) -> str:
    lines = ["Medie giornaliere temperatura/umidita':"]
    for row in rows[:10]:
        lines.append(
            f"- {row['day']} {row['MBO_SN']}: {row['avg_temp']} C, umidita' ambiente {row['avg_air_humidity']}% ({row['readings_count']} letture)"
        )
    return "\n".join(lines)


def _soil_moisture_answer(rows: list[dict[str, object]]) -> str:
    lines = ["Umidita' media del terreno per livelli:"]
    for row in rows[:10]:
        lines.append(
            f"- {row['day']} {row['MBO_SN']}: L1 {row['avg_soil_humidity_1']}%, L2 {row['avg_soil_humidity_2']}%, L3 {row['avg_soil_humidity_3']}%"
        )
    return "\n".join(lines)


def _status_answer(rows: list[dict[str, object]], value_field: str, label: str) -> str:
    low_rows = [row for row in rows if row.get("status") == "LOW"]
    lines = [f"Stato {label}: {len(low_rows)} letture sotto soglia su {len(rows)} controllate."]
    for row in rows[:10]:
        lines.append(f"- {row['SNS_DtEvent']} {row['MBO_SN']}: {row[value_field]} ({row['status']})")
    return "\n".join(lines)


def _ph_ec_answer(rows: list[dict[str, object]]) -> str:
    lines = [
        f"pH medio: {_avg(rows, 'SNS_PH')} (min {_min(rows, 'SNS_PH')}, max {_max(rows, 'SNS_PH')})",
        f"EC media: {_avg(rows, 'SNS_EC')} (min {_min(rows, 'SNS_EC')}, max {_max(rows, 'SNS_EC')})",
        "",
        "Valori pH/EC:",
    ]
    for row in rows[:10]:
        lines.append(f"- {row['SNS_DtEvent']} {row['MBO_SN']}: pH {row['SNS_PH']}, EC {row['SNS_EC']}")
    return "\n".join(lines)


def _rain_et_answer(rows: list[dict[str, object]]) -> str:
    total_rain = round(sum(float(row.get("total_rain", 0)) for row in rows), 2)
    lines = [f"Pioggia totale nel periodo: {total_rain} mm.", "Dettaglio giornaliero:"]
    for row in rows[:10]:
        lines.append(
            f"- {row['day']} {row['MBO_SN']}: pioggia {row['total_rain']} mm, ET 24h media {row['avg_eto_24h']}"
        )
    return "\n".join(lines)


def _quality_answer(row: dict[str, object]) -> str:
    return "\n".join(
        [
            "Riepilogo qualita' dati senshistory:",
            f"- Record totali: {row['total_records']}",
            f"- Record senza data evento: {row['records_without_event_date']} ({row['percentage_without_event_date']}%)",
            f"- Record con campi principali a zero: {row['records_with_core_fields_zero']}",
            f"- Centraline distinte: {row['distinct_motherboards']}",
        ]
    )


def _latest_by_serial_lines(rows: list[dict[str, object]]) -> list[str]:
    latest: dict[str, dict[str, object]] = {}
    for row in sorted(rows, key=lambda item: str(item["SNS_DtEvent"]), reverse=True):
        latest.setdefault(str(row["MBO_SN"]), row)

    return [
        (
            f"- {serial}: {row['SNS_DtEvent']} | temp {row['SNS_TempAmb']} C, "
            f"umidita' {row['SNS_UmdAmb']}%, terreno L1 {row['SNS_Umd1Liv']}%, "
            f"batt {row['SNS_BattLiv']}, RF {row['SNS_RFLiv']}"
        )
        for serial, row in sorted(latest.items())
    ]


def _trend_lines(rows: list[dict[str, object]]) -> list[str]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["MBO_SN"])].append(row)

    lines = []
    for serial, serial_rows in sorted(grouped.items()):
        ordered = sorted(serial_rows, key=lambda item: str(item["SNS_DtEvent"]))
        first = ordered[0]
        last = ordered[-1]
        lines.append(
            f"- {serial}: temp {_delta(first, last, 'SNS_TempAmb')} C, "
            f"umidita' ambiente {_delta(first, last, 'SNS_UmdAmb')} punti, "
            f"terreno L1 {_delta(first, last, 'SNS_Umd1Liv')} punti"
        )
    return lines


def _value_table_lines(rows: list[dict[str, object]]) -> list[str]:
    lines = ["| Data evento | Centralina | Temp | Umd amb | Umd L1 | Batt | RF | pH | EC |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(
            f"| {row['SNS_DtEvent']} | {row['MBO_SN']} | {row['SNS_TempAmb']} | {row['SNS_UmdAmb']} | "
            f"{row['SNS_Umd1Liv']} | {row['SNS_BattLiv']} | {row['SNS_RFLiv']} | {row['SNS_PH']} | {row['SNS_EC']} |"
        )
    return lines


def _avg(rows: list[dict[str, object]], field: str) -> float:
    values = [_float(row, field) for row in rows]
    return round(sum(values) / len(values), 2)


def _sum(rows: list[dict[str, object]], field: str) -> float:
    return round(sum(_float(row, field) for row in rows), 2)


def _min(rows: list[dict[str, object]], field: str) -> float:
    return round(min(_float(row, field) for row in rows), 2)


def _max(rows: list[dict[str, object]], field: str) -> float:
    return round(max(_float(row, field) for row in rows), 2)


def _delta(first: dict[str, object], last: dict[str, object], field: str) -> float:
    return round(_float(last, field) - _float(first, field), 2)


def _float(row: dict[str, object], field: str) -> float:
    return float(row.get(field, 0) or 0)
