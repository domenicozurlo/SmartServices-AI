#!/usr/bin/env python3
"""
Evaluate rag_api multimodal retrieval against known document questions.

Usage:
  python utils/rag_eval.py --file-id <file_id> --entity-id <entity_id>
  python utils/rag_eval.py --format markdown
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


DEFAULT_FILE_ID = "8096fe3e-042a-456b-a7ee-f601aa792fd7"
DEFAULT_ENTITY_ID = "agent_jIb9PzVgCiQOZ6rG9E0kT"

CASES = [
    {
        "id": "reset",
        "query": "reset dispositivo",
        "expected_pages": {8},
        "must": ["resettare", "pulsante", "luci"],
    },
    {
        "id": "reset_typo",
        "query": "reset dispositvo",
        "expected_pages": {8},
        "must": ["resettare", "pulsante", "luci"],
    },
    {
        "id": "power_on",
        "query": "come accendere IQOS?",
        "expected_pages": {4},
        "must": ["accensione", "4 secondi", "caricatore"],
    },
    {
        "id": "holder_status",
        "query": "come controllare lo stato dell'holder",
        "expected_pages": {5},
        "must": ["controllo", "holder", "pulsante"],
    },
    {
        "id": "holder_status_typo",
        "query": "come conotrllare lo stato dell'holder",
        "expected_pages": {5},
        "must": ["controllo", "holder", "pulsante"],
    },
    {
        "id": "holder_charge",
        "query": "come caricare l'holder",
        "expected_pages": {4},
        "must": ["caricamento", "holder", "chiudere"],
    },
    {
        "id": "use_iqos",
        "query": "come usare IQOS 3 DUO",
        "expected_pages": {6, 7},
        "must": ["stick", "riscaldamento", "holder"],
    },
    {
        "id": "clean_holder",
        "query": "come pulire l'holder",
        "expected_pages": {7},
        "must": ["pulizia", "holder", "bastoncino"],
    },
    {
        "id": "red_light",
        "query": "luce rossa lampeggiante cosa fare",
        "expected_pages": {8},
        "must": ["luce rossa", "resettare", "servizio clienti"],
    },
    {
        "id": "white_lights",
        "query": "luci bianche lampeggiano due volte",
        "expected_pages": {8},
        "must": ["luci bianche", "temperatura", "ricaricato"],
    },
    {
        "id": "pocket_status",
        "query": "controllare stato caricatore tascabile batteria",
        "expected_pages": {8},
        "must": ["controllo", "caricatore tascabile", "batteria"],
    },
    {
        "id": "remove_stick",
        "query": "come rimuovere stick tabacco usato",
        "expected_pages": {7},
        "must": ["rimozione", "stick", "cappuccio"],
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
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def group_text(group: Dict[str, Any]) -> str:
    return "\n".join(chunk.get("text") or "" for chunk in group.get("chunks", []))


def group_pages(group: Dict[str, Any]) -> List[int]:
    pages = []
    for chunk in group.get("chunks", []):
        page = (chunk.get("metadata") or {}).get("page")
        if page is None:
            continue
        try:
            pages.append(int(page))
        except (TypeError, ValueError):
            continue
    return pages


def group_image_count(group: Dict[str, Any]) -> int:
    return sum(
        len((chunk.get("metadata") or {}).get("image_ids") or [])
        for chunk in group.get("chunks", [])
    )


def eval_group(group: Dict[str, Any], expected_pages: Set[int], must: List[str]) -> Dict[str, Any]:
    text = norm(group_text(group))
    pages = group_pages(group)
    return {
        "score": group.get("score"),
        "distance": group.get("distance"),
        "pages": pages,
        "page_hit": bool(set(pages) & expected_pages),
        "must_hits": sum(1 for term in must if norm(term) in text),
        "images": group_image_count(group),
        "text": re.sub(r"\s+", " ", group_text(group)).strip()[:260],
    }


def verdict(top: List[Dict[str, Any]], relevant_top5: int) -> str:
    if not top:
        return "FAIL"
    top1 = top[0]
    top1_good = top1["page_hit"] and top1["must_hits"] > 0 and top1["images"] > 0
    if top1_good and relevant_top5 == len(top):
        return "OK"
    if top1_good or relevant_top5 >= 1:
        return "PARZIALE"
    return "FAIL"


def run_eval(args: argparse.Namespace) -> Dict[str, Any]:
    token = jwt.encode(
        {"sub": "rag-eval", "role": "admin"},
        load_jwt_secret(),
        algorithm="HS256",
    )
    headers = {"Authorization": f"Bearer {token}"}

    rows = []
    with httpx.Client(timeout=args.timeout) as client:
        for case in CASES:
            payload = {
                "query": case["query"],
                "file_id": args.file_id,
                "entity_id": args.entity_id,
                "k": args.k,
            }
            response = client.post(f"{args.rag_url}/query-multimodal", json=payload, headers=headers)
            row = {
                "id": case["id"],
                "query": case["query"],
                "status": response.status_code,
            }
            if response.status_code != 200:
                row["verdict"] = "ERROR"
                row["error"] = response.text[:500]
                rows.append(row)
                continue

            groups = response.json().get("context_groups", [])
            top = [
                eval_group(group, case["expected_pages"], case["must"])
                for group in groups[:5]
            ]
            relevant_top5 = sum(1 for item in top if item["page_hit"])
            row.update({
                "groups": len(groups),
                "relevant_top5": relevant_top5,
                "noise_top5": len(top) - relevant_top5,
                "image_top5": sum(1 for item in top if item["images"] > 0),
                "verdict": verdict(top, relevant_top5),
                "top": top,
            })
            rows.append(row)

    return {
        "file_id": args.file_id,
        "entity_id": args.entity_id,
        "rag_url": args.rag_url,
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
    print(f"Summary: OK={summary['ok']} PARZIALE={summary['partial']} FAIL={summary['fail']} ERROR={summary['error']}")
    print()
    print("| Query | Esito | Rumore top5 | Top1 pagine | Top1 immagini | Top1 testo |")
    print("|---|---:|---:|---|---:|---|")
    for row in result["rows"]:
        top1 = row.get("top", [{}])[0] if row.get("top") else {}
        text = (top1.get("text") or "").replace("|", "\\|")
        print(
            f"| {row['query']} | {row['verdict']} | {row.get('noise_top5', 0)} | {top1.get('pages', [])} | "
            f"{top1.get('images', 0)} | {text[:140]} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate rag_api multimodal retrieval")
    parser.add_argument("--rag-url", default=os.getenv("RAG_API_URL", "http://localhost:8000"))
    parser.add_argument("--file-id", default=DEFAULT_FILE_ID)
    parser.add_argument("--entity-id", default=DEFAULT_ENTITY_ID)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=40)
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
