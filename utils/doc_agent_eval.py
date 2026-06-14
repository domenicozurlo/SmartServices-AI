#!/usr/bin/env python3
"""
Evaluate agents-gateway doc_kb synthesis using rag_api multimodal context.

This simulates LibreChat round 2:
1. Query rag_api /query-multimodal for context_groups.
2. Send those context_groups to agents-gateway as a file_search tool result.
3. Evaluate final answer text, inline images, and source pages.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

import httpx
import jwt


DEFAULT_FILE_ID = "717f712b-b3b2-4a9a-9408-4ff676d4a80c"
DEFAULT_ENTITY_ID = "agent_jIb9PzVgCiQOZ6rG9E0kT"

CASES = [
    {
        "id": "reset",
        "query": "reset dispositivo",
        "expected_pages": {8},
        "must": ["premere", "pulsante", "caricatore tascabile", "luci"],
        "forbid": ["pulizia", "stick di tabacco"],
    },
    {
        "id": "reset_typo",
        "query": "reset dispositvo",
        "expected_pages": {8},
        "must": ["premere", "pulsante", "caricatore tascabile", "luci"],
        "forbid": ["pulizia", "stick di tabacco"],
    },
    {
        "id": "power_on",
        "query": "come accendere IQOS?",
        "expected_pages": {4},
        "must": ["tenere premuto", "4 secondi", "caricatore tascabile"],
        "forbid": ["pulizia", "servizio clienti"],
    },
    {
        "id": "holder_status",
        "query": "come controllare lo stato dell'holder",
        "expected_pages": {5},
        "must": ["premere", "holder", "luci", "batteria"],
        "forbid": ["luce rossa", "pulizia"],
    },
    {
        "id": "holder_status_typo",
        "query": "come conotrllare lo stato dell'holder",
        "expected_pages": {5},
        "must": ["premere", "holder", "luci", "batteria"],
        "forbid": ["luce rossa", "pulizia"],
    },
    {
        "id": "holder_charge",
        "query": "come caricare l'holder",
        "expected_pages": {4},
        "must": ["inserire", "holder", "caricatore tascabile", "chiudere"],
        "forbid": ["pulizia", "luce rossa"],
    },
    {
        "id": "use_iqos",
        "query": "come usare IQOS 3 DUO",
        "expected_pages": {6, 7},
        "must": ["stick", ["riscaldamento", "tenere premuto"], "holder"],
        "forbid": ["servizio clienti"],
    },
    {
        "id": "clean_holder",
        "query": "come pulire l'holder",
        "expected_pages": {7},
        "must": ["accessorio per la pulizia", "bastoncino", "holder"],
        "forbid": ["luce rossa", "4 secondi"],
    },
    {
        "id": "red_light",
        "query": "luce rossa lampeggiante cosa fare",
        "expected_pages": {8},
        "must": ["luce rossa", "resettare", "servizio clienti"],
        "forbid": ["pulizia", "stick"],
    },
    {
        "id": "white_lights",
        "query": "luci bianche lampeggiano due volte",
        "expected_pages": {8},
        "must": ["temperatura", ["ricaricato", "ricarica completa", "ricarica", "holder"]],
        "forbid": ["pulizia", "servizio clienti"],
    },
    {
        "id": "pocket_status",
        "query": "controllare stato caricatore tascabile batteria",
        "expected_pages": {8},
        "must": ["caricatore tascabile", "batteria", "luci"],
        "forbid": ["holder non carico", "pulizia"],
    },
    {
        "id": "remove_stick",
        "query": "come rimuovere stick tabacco usato",
        "expected_pages": {7},
        "must": ["cappuccio", "rimuovere", "stick di tabacco usato"],
        "forbid": ["luce rossa", "servizio clienti"],
    },
]


def load_jwt_secret() -> str:
    if os.getenv("JWT_SECRET"):
        return os.environ["JWT_SECRET"]

    env_path = Path(".env")
    if not env_path.exists():
        raise SystemExit("JWT_SECRET missing and .env not found")

    for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("JWT_SECRET="):
            return line.split("=", 1)[1].strip()

    raise SystemExit("JWT_SECRET missing")


def norm(text: str) -> str:
    normalized = (text or "").lower()
    normalized = re.sub(r":::thinking.*?:::", " ", normalized, flags=re.DOTALL)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def source_pages(answer: str) -> Set[int]:
    pages = set()
    for match in re.finditer(r"(?:page|pagina)\s+(\d+)", answer or "", flags=re.IGNORECASE):
        pages.add(int(match.group(1)))
    return pages


def image_count(answer: str) -> int:
    return len(re.findall(r"!\[[^\]]*]\([^)]+\)", answer or ""))


def criterion_hit(criterion: Any, answer_norm: str) -> bool:
    if isinstance(criterion, list):
        return any(norm(str(term)) in answer_norm for term in criterion)
    return norm(str(criterion)) in answer_norm


def source_section(answer: str) -> str:
    match = re.search(r"##\s*Sources\s*(.*)$", answer or "", flags=re.IGNORECASE | re.DOTALL)
    return match.group(1) if match else ""


def verdict(row: Dict[str, Any]) -> str:
    if row["status"] != 200:
        return "ERROR"
    if row["context_groups"] == 0:
        return "FAIL"
    if not row["source_page_hit"] or row["source_noise_pages"]:
        return "FAIL"
    if row["must_hits"] < row["must_total"]:
        return "PARZIALE"
    if row["forbidden_hits"]:
        return "PARZIALE"
    if row["images"] <= 0:
        return "PARZIALE"
    return "OK"


def build_gateway_messages(query: str, context_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    tool_call_id = "call_eval_doc_kb"
    return [
        {"role": "user", "content": query},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": "file_search",
                        "arguments": json.dumps({"query": query}, ensure_ascii=False),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": "file_search",
            "content": json.dumps(context_payload, ensure_ascii=False),
        },
    ]


def run_eval(args: argparse.Namespace) -> Dict[str, Any]:
    token = jwt.encode(
        {"sub": "doc-agent-eval", "role": "admin"},
        load_jwt_secret(),
        algorithm="HS256",
    )
    rag_headers = {"Authorization": f"Bearer {token}"}
    gateway_headers = {
        "Authorization": f"Bearer {args.gateway_api_key}",
        "Content-Type": "application/json",
    }

    rows = []
    with httpx.Client(timeout=args.timeout) as client:
        for case in CASES:
            rag_payload = {
                "query": case["query"],
                "file_id": args.file_id,
                "entity_id": args.entity_id,
                "k": args.k,
            }
            rag_response = client.post(
                f"{args.rag_url}/query-multimodal",
                json=rag_payload,
                headers=rag_headers,
            )
            row = {
                "id": case["id"],
                "query": case["query"],
                "status": rag_response.status_code,
                "context_groups": 0,
            }
            if rag_response.status_code != 200:
                row["error"] = rag_response.text[:500]
                row["verdict"] = "ERROR"
                rows.append(row)
                continue

            context_payload = rag_response.json()
            context_groups = context_payload.get("context_groups", [])
            row["context_groups"] = len(context_groups)

            gateway_payload = {
                "model": args.model,
                "stream": False,
                "messages": build_gateway_messages(case["query"], context_payload),
            }
            gateway_response = client.post(
                f"{args.gateway_url}/v1/chat/completions",
                json=gateway_payload,
                headers=gateway_headers,
            )
            row["status"] = gateway_response.status_code
            if gateway_response.status_code != 200:
                row["error"] = gateway_response.text[:500]
                row["verdict"] = "ERROR"
                rows.append(row)
                continue

            data = gateway_response.json()
            answer = data["choices"][0]["message"].get("content") or ""
            answer_norm = norm(answer)
            sources_text = source_section(answer)
            pages = source_pages(sources_text)
            source_noise = sorted(pages - case["expected_pages"])
            must_hits = sum(1 for criterion in case["must"] if criterion_hit(criterion, answer_norm))
            forbidden_hits = [term for term in case["forbid"] if norm(term) in answer_norm]
            row.update(
                {
                    "answer": re.sub(r"\s+", " ", answer).strip()[:500],
                    "source_pages": sorted(pages),
                    "source_page_hit": bool(pages & case["expected_pages"]),
                    "source_noise_pages": source_noise,
                    "must_hits": must_hits,
                    "must_total": len(case["must"]),
                    "forbidden_hits": forbidden_hits,
                    "images": image_count(answer),
                }
            )
            row["verdict"] = verdict(row)
            rows.append(row)

    return {
        "file_id": args.file_id,
        "entity_id": args.entity_id,
        "rag_url": args.rag_url,
        "gateway_url": args.gateway_url,
        "model": args.model,
        "rows": rows,
        "summary": {
            "ok": sum(1 for row in rows if row["verdict"] == "OK"),
            "partial": sum(1 for row in rows if row["verdict"] == "PARZIALE"),
            "fail": sum(1 for row in rows if row["verdict"] == "FAIL"),
            "error": sum(1 for row in rows if row["verdict"] == "ERROR"),
            "total": len(rows),
        },
    }


def print_markdown(result: Dict[str, Any]) -> None:
    summary = result["summary"]
    print(
        f"Summary: OK={summary['ok']} PARZIALE={summary['partial']} "
        f"FAIL={summary['fail']} ERROR={summary['error']}"
    )
    print()
    print("| Query | Esito | Fonti | Rumore fonti | Img | Must | Risposta |")
    print("|---|---:|---|---|---:|---:|---|")
    for row in result["rows"]:
        answer = (row.get("answer") or row.get("error") or "").replace("|", "\\|")
        must = f"{row.get('must_hits', 0)}/{row.get('must_total', 0)}"
        print(
            f"| {row['query']} | {row['verdict']} | {row.get('source_pages', [])} | "
            f"{row.get('source_noise_pages', [])} | {row.get('images', 0)} | "
            f"{must} | {answer[:180]} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate doc_kb agent synthesis")
    parser.add_argument("--rag-url", default=os.getenv("RAG_API_URL", "http://localhost:8000"))
    parser.add_argument("--gateway-url", default=os.getenv("AGENTS_GATEWAY_URL", "http://localhost:8001"))
    parser.add_argument("--gateway-api-key", default=os.getenv("AGENTS_GATEWAY_API_KEY", "123"))
    parser.add_argument("--file-id", default=DEFAULT_FILE_ID)
    parser.add_argument("--entity-id", default=DEFAULT_ENTITY_ID)
    parser.add_argument("--model", default="smart_service_flow")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--format", choices=["json", "markdown"], default="markdown")
    args = parser.parse_args()

    result = run_eval(args)
    if args.format == "json":
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    print_markdown(result)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
