#!/usr/bin/env python3
"""
Inspect pgvector chunks stored by rag_api.

Usage (from project root):
  # List all files and chunk counts:
  docker exec vectordb psql -U myuser -d mydatabase -c "SELECT cmetadata->>'file_id', COUNT(*) FROM langchain_pg_embedding GROUP BY 1;"

  # Run this script inside the vectordb container or from a machine with psycopg2:
  python utils/inspect_pgvector.py                         # list all files
  python utils/inspect_pgvector.py <file_id>               # all chunks for that file
  python utils/inspect_pgvector.py <file_id> --search foo  # full-text filter on chunk text
  python utils/inspect_pgvector.py <file_id> --page 8      # only page 8
  python utils/inspect_pgvector.py <file_id> --json        # dump raw JSON rows

Environment variables (defaults match docker-compose):
  POSTGRES_HOST      (default: localhost)
  POSTGRES_PORT      (default: 5432)
  POSTGRES_DB        (default: mydatabase)
  POSTGRES_USER      (default: myuser)
  POSTGRES_PASSWORD  (default: mypassword)
"""

import argparse
import json
import os
import sys
import textwrap


def _get_conn():
    try:
        import psycopg2
        import psycopg2.extras
    except ImportError:
        sys.exit("psycopg2 not installed — run: pip install psycopg2-binary")

    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        dbname=os.getenv("POSTGRES_DB", "mydatabase"),
        user=os.getenv("POSTGRES_USER", "myuser"),
        password=os.getenv("POSTGRES_PASSWORD", "mypassword"),
    )


def cmd_list(args):
    """List all files in the collection with chunk counts and metadata."""
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT
            cmetadata->>'file_id'                           AS file_id,
            COUNT(*)                                        AS chunks,
            MIN((cmetadata->>'page')::int)                  AS page_min,
            MAX((cmetadata->>'page')::int)                  AS page_max,
            MAX(cmetadata->>'source_file')                  AS source_file
        FROM langchain_pg_embedding
        WHERE cmetadata->>'file_id' IS NOT NULL
        GROUP BY 1
        ORDER BY chunks DESC
    """)
    rows = cur.fetchall()
    if not rows:
        print("No documents found in the collection.")
        return
    print(f"\n{'FILE_ID':<38} {'CHUNKS':>6}  {'PAGES':>10}  SOURCE FILE")
    print("-" * 110)
    for file_id, chunks, page_min, page_max, source_file in rows:
        pages = f"{page_min}-{page_max}" if page_min is not None else "?"
        sf = (source_file or "")[:60]
        print(f"{file_id:<38} {chunks:>6}  {pages:>10}  {sf}")
    conn.close()


def cmd_inspect(args):
    """Inspect all chunks for a specific file_id."""
    conn = _get_conn()
    cur = conn.cursor()

    where_clauses = ["cmetadata->>'file_id' = %s"]
    params = [args.file_id]

    if args.page is not None:
        where_clauses.append("(cmetadata->>'page')::int = %s")
        params.append(args.page)

    if args.search:
        where_clauses.append("document ILIKE %s")
        params.append(f"%{args.search}%")

    where_sql = " AND ".join(where_clauses)

    cur.execute(f"""
        SELECT
            uuid,
            custom_id,
            cmetadata,
            document,
            vector_dims(embedding) AS dims
        FROM langchain_pg_embedding
        WHERE {where_sql}
        ORDER BY (cmetadata->>'page')::int NULLS LAST, custom_id
    """, params)

    rows = cur.fetchall()
    conn.close()

    if not rows:
        print(f"No chunks found for file_id={args.file_id!r} with the given filters.")
        return

    if args.json:
        output = []
        for uuid, custom_id, meta, doc, dims in rows:
            output.append({
                "uuid": str(uuid),
                "custom_id": custom_id,
                "metadata": meta,
                "document": doc,
                "embedding_dims": dims,
            })
        print(json.dumps(output, indent=2, ensure_ascii=False))
        return

    # Pretty-print
    print(f"\n{'='*80}")
    print(f"FILE_ID : {args.file_id}")
    print(f"CHUNKS  : {len(rows)}")
    print(f"FILTER  : page={args.page!r}  search={args.search!r}")
    print(f"{'='*80}\n")

    for i, (uuid, custom_id, meta, doc, dims) in enumerate(rows, 1):
        page        = meta.get("page", "?")
        source_file = meta.get("source_file", "?")
        source_url  = meta.get("source_url", "")
        image_ids   = meta.get("image_ids", [])
        images_meta = meta.get("images", [])
        chunk_type  = meta.get("chunk_type", "text")
        extra_keys  = sorted(k for k in meta if k not in {
            "page", "source_file", "source_url", "file_id",
            "image_ids", "images", "chunk_type",
        })

        print(f"── CHUNK {i}/{len(rows)} ──────────────────────────────────────────────────────────")
        print(f"  uuid         : {uuid}")
        print(f"  custom_id    : {custom_id}")
        print(f"  source_file  : {source_file}")
        print(f"  page         : {page}")
        print(f"  chunk_type   : {chunk_type}")
        print(f"  embedding    : {dims}d vector")
        if source_url:
            print(f"  source_url   : {source_url}")
        if image_ids:
            print(f"  image_ids    : {image_ids}")
        if images_meta:
            print(f"  images       : {len(images_meta)} record(s)")
            for img in images_meta[:3]:
                iid = img.get("image_id") if isinstance(img, dict) else img
                url = img.get("url", "") if isinstance(img, dict) else ""
                print(f"    • {iid}  →  {url or '(no url)'}")
        if extra_keys:
            print(f"  other meta   : { {k: meta[k] for k in extra_keys} }")
        print()
        wrapped = textwrap.fill(doc or "", width=100, initial_indent="  ", subsequent_indent="  ")
        print(wrapped or "  (empty)")
        print()

    print(f"Total: {len(rows)} chunks displayed.\n")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect pgvector chunks stored by rag_api",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="cmd")

    # List subcommand
    sub.add_parser("list", help="List all files with chunk counts")

    # Inspect subcommand
    ins = sub.add_parser("inspect", help="Show chunks for a specific file_id")
    ins.add_argument("file_id", help="The file_id UUID to inspect")
    ins.add_argument("--page", type=int, default=None, help="Filter to a specific page number")
    ins.add_argument("--search", default=None, help="Filter chunks containing this text")
    ins.add_argument("--json", action="store_true", help="Output raw JSON instead of pretty-print")

    # Shortcut: python inspect_pgvector.py <file_id>  →  treat as inspect
    if len(sys.argv) >= 2 and sys.argv[1] not in ("list", "inspect", "-h", "--help"):
        sys.argv.insert(1, "inspect")

    args = parser.parse_args()

    if args.cmd == "list" or args.cmd is None:
        cmd_list(args)
    elif args.cmd == "inspect":
        cmd_inspect(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
