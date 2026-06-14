#!/usr/bin/env python3
"""
Report pgvector multimodal ingest quality for a file.

Usage:
  python utils/rag_ingest_report.py
  python utils/rag_ingest_report.py --file-id <file_id>
  python utils/rag_ingest_report.py --format json
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional


def get_conn():
    try:
        import psycopg2
        import psycopg2.extras
    except ImportError:
        raise SystemExit("psycopg2 not installed - run: pip install psycopg2-binary")

    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        dbname=os.getenv("POSTGRES_DB", "mydatabase"),
        user=os.getenv("POSTGRES_USER", "myuser"),
        password=os.getenv("POSTGRES_PASSWORD", "mypassword"),
    )


def latest_file_id(conn) -> str:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT cmetadata->>'file_id'
        FROM langchain_pg_embedding
        WHERE cmetadata->>'file_id' IS NOT NULL
        GROUP BY 1
        ORDER BY MAX(uuid::text) DESC
        LIMIT 1
        """
    )
    row = cur.fetchone()
    if not row:
        raise SystemExit("No file_id found in pgvector")
    return row[0]


def report_for_file(conn, file_id: str) -> Dict[str, Any]:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT
          COUNT(*) AS chunks,
          COUNT(*) FILTER (WHERE cmetadata ? 'chunk_id') AS multimodal_chunks,
          COUNT(*) FILTER (WHERE NOT (cmetadata ? 'chunk_id')) AS legacy_chunks,
          COUNT(*) FILTER (
            WHERE cmetadata ? 'chunk_id'
              AND jsonb_array_length(COALESCE(cmetadata->'image_ids','[]'::jsonb)) > 0
          ) AS chunks_with_images,
          COUNT(*) FILTER (WHERE cmetadata ? 'chunk_id' AND cmetadata ? 'domain_hints') AS chunks_with_domain_hints,
          COUNT(*) FILTER (WHERE cmetadata ? 'chunk_id' AND cmetadata ? 'chunk_keywords') AS chunks_with_keywords,
          COUNT(*) FILTER (WHERE cmetadata ? 'chunk_id' AND cmetadata ? 'chunk_headings') AS chunks_with_headings,
          COUNT(*) FILTER (
            WHERE cmetadata ? 'chunk_id'
              AND length(trim(COALESCE(cmetadata->>'text',''))) < 80
          ) AS short_chunks,
          COUNT(*) FILTER (
            WHERE cmetadata ? 'chunk_id'
              AND length(trim(COALESCE(cmetadata->>'text',''))) < 80
              AND jsonb_array_length(COALESCE(cmetadata->'image_ids','[]'::jsonb)) > 0
          ) AS short_chunks_with_images,
          COUNT(*) FILTER (
            WHERE cmetadata ? 'chunk_id'
              AND jsonb_array_length(COALESCE(cmetadata->'image_ids','[]'::jsonb)) > 0
              AND NOT EXISTS (
                SELECT 1 FROM jsonb_array_elements(COALESCE(cmetadata->'images','[]'::jsonb)) img
                WHERE COALESCE(img->>'url', '') <> ''
              )
          ) AS image_chunks_without_url,
          MIN((cmetadata->>'page')::int) AS page_min,
          MAX((cmetadata->>'page')::int) AS page_max,
          MAX(cmetadata->>'source_file') AS source_file
        FROM langchain_pg_embedding
        WHERE cmetadata->>'file_id' = %s
        """,
        (file_id,),
    )
    row = cur.fetchone()
    if not row or row[0] == 0:
        raise SystemExit(f"No chunks found for file_id={file_id}")

    keys = [
        "chunks",
        "multimodal_chunks",
        "legacy_chunks",
        "chunks_with_images",
        "chunks_with_domain_hints",
        "chunks_with_keywords",
        "chunks_with_headings",
        "short_chunks",
        "short_chunks_with_images",
        "image_chunks_without_url",
        "page_min",
        "page_max",
        "source_file",
    ]
    data = dict(zip(keys, row))

    cur.execute(
        """
        SELECT cmetadata->'domain_hints'
        FROM langchain_pg_embedding
        WHERE cmetadata->>'file_id' = %s
          AND cmetadata ? 'domain_hints'
          AND cmetadata->'domain_hints' <> '{}'::jsonb
        LIMIT 1
        """,
        (file_id,),
    )
    hint_row = cur.fetchone()
    domain_hints = hint_row[0] if hint_row else {}

    return {
        "file_id": file_id,
        **data,
        "domain_hints": domain_hints,
        "warnings": warnings(data, domain_hints),
    }


def warnings(data: Dict[str, Any], domain_hints: Optional[Dict[str, Any]]) -> List[str]:
    result = []
    chunks = data["multimodal_chunks"] or 1
    if data["legacy_chunks"] and data["multimodal_chunks"]:
        result.append("This file_id contains legacy non-multimodal records mixed with multimodal chunks.")
    elif data["legacy_chunks"]:
        result.append("This file_id contains only legacy non-multimodal records.")
    if data["chunks_with_keywords"] < chunks:
        result.append("Some chunks do not have chunk_keywords metadata.")
    if data["chunks_with_domain_hints"] < chunks:
        result.append("Some chunks do not have domain_hints metadata.")
    if data["image_chunks_without_url"]:
        result.append("Some image chunks do not have a resolvable image URL.")
    if data["short_chunks"] / chunks > 0.25:
        result.append("More than 25% of chunks are short; check heading/image splitting.")
    if not domain_hints:
        result.append("OpenAI domain hints are empty or were not generated.")
    return result


def print_markdown(report: Dict[str, Any]) -> None:
    print(f"File: {report['file_id']}")
    print(f"Source: {report.get('source_file') or ''}")
    print(f"Pages: {report.get('page_min')}-{report.get('page_max')}")
    print()
    print("| Metric | Value |")
    print("|---|---:|")
    for key in (
        "chunks",
        "multimodal_chunks",
        "legacy_chunks",
        "chunks_with_images",
        "chunks_with_domain_hints",
        "chunks_with_keywords",
        "chunks_with_headings",
        "short_chunks",
        "short_chunks_with_images",
        "image_chunks_without_url",
    ):
        print(f"| {key} | {report[key]} |")
    hints = report.get("domain_hints") or {}
    if hints:
        print()
        print(f"Domain: {hints.get('domain', '')}")
        print(f"Key terms: {', '.join((hints.get('key_terms') or [])[:12])}")
    if report["warnings"]:
        print()
        print("Warnings:")
        for warning in report["warnings"]:
            print(f"- {warning}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Report pgvector multimodal ingest quality")
    parser.add_argument("--file-id")
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown")
    args = parser.parse_args()

    conn = get_conn()
    try:
        file_id = args.file_id or latest_file_id(conn)
        report = report_for_file(conn, file_id)
    finally:
        conn.close()

    if args.format == "json":
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    print_markdown(report)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
