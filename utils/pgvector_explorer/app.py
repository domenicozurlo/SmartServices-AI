"""
pgvector Explorer — browse and semantically search chunks stored by rag_api.

Run:
    pip install fastapi uvicorn psycopg2-binary httpx
    python utils/pgvector_explorer/app.py

Then open http://localhost:8888
"""

import json
import os
import urllib.parse
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import httpx
import jwt
import psycopg2
import psycopg2.extras
import uvicorn
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PG_HOST = os.getenv("POSTGRES_HOST", "localhost")
PG_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
PG_DB = os.getenv("POSTGRES_DB", "mydatabase")
PG_USER = os.getenv("POSTGRES_USER", "myuser")
PG_PASS = os.getenv("POSTGRES_PASSWORD", "mypassword")

RAG_API_URL = os.getenv("RAG_API_URL", "http://localhost:8000")
JWT_SECRET = os.getenv("JWT_SECRET", "")

EXPLORER_PORT = int(os.getenv("EXPLORER_PORT", "8888"))


def _make_jwt() -> str:
    """Generate a non-expiring JWT for internal rag_api calls."""
    if not JWT_SECRET:
        return ""
    return jwt.encode({"sub": "pgvector-explorer", "role": "admin"}, JWT_SECRET, algorithm="HS256")


_RAG_TOKEN = _make_jwt()


def _conn():
    return psycopg2.connect(
        host=PG_HOST, port=PG_PORT, dbname=PG_DB, user=PG_USER, password=PG_PASS,
        cursor_factory=psycopg2.extras.RealDictCursor,
    )


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _doc_text(doc_info: Any) -> str:
    if not isinstance(doc_info, dict):
        return ""
    return (
        doc_info.get("page_content")
        or doc_info.get("document")
        or doc_info.get("text")
        or doc_info.get("content")
        or ""
    )


def _doc_metadata(doc_info: Any) -> Dict[str, Any]:
    if not isinstance(doc_info, dict):
        return {}
    metadata = doc_info.get("metadata") or doc_info.get("cmetadata") or {}
    return metadata if isinstance(metadata, dict) else {}


def _source_file(metadata: Dict[str, Any], fallback: str) -> str:
    source = metadata.get("source_file") or metadata.get("source") or fallback
    return os.path.basename(str(source)) or fallback


def _file_user_id(file_id: str) -> Optional[str]:
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT cmetadata->>'user_id' AS user_id
                FROM langchain_pg_embedding
                WHERE cmetadata->>'file_id' = %s
                  AND cmetadata->>'user_id' IS NOT NULL
                LIMIT 1
            """, (file_id,))
            row = cur.fetchone()
    return row["user_id"] if row else None


def _legacy_results_to_context_groups(data: Any, file_id: str) -> List[Dict[str, Any]]:
    if not isinstance(data, list):
        return []

    groups: List[Dict[str, Any]] = []
    for idx, item in enumerate(data):
        if not isinstance(item, list) or len(item) < 2:
            continue

        doc_info = item[0]
        distance = _as_float(item[1])
        score = max(0.0, 1.0 - distance)
        metadata = _doc_metadata(doc_info)
        text = _doc_text(doc_info)
        source_file = _source_file(metadata, file_id)
        page = metadata.get("page")
        result_file_id = metadata.get("file_id") or file_id

        groups.append({
            "group_id": f"legacy_{idx + 1}",
            "file_id": result_file_id,
            "source_file": source_file,
            "pages": [page] if page is not None else [],
            "score": score,
            "chunks": [{
                "chunk_id": "",
                "text": text,
                "score": score,
                "metadata": {
                    "file_id": result_file_id,
                    "source_file": source_file,
                    "page": page,
                    "source_url": metadata.get("source_url", ""),
                    "image_ids": metadata.get("image_ids", []),
                    "images": metadata.get("images", []),
                    "previous_chunk_id": metadata.get("previous_chunk_id"),
                    "next_chunk_id": metadata.get("next_chunk_id"),
                },
            }],
            "images": metadata.get("images", []),
            "sources": [{
                "source_file": source_file,
                "page": page,
                "source_url": metadata.get("source_url", ""),
            }],
        })

    return groups


def _normalize_search_response(data: Any, file_id: str, endpoint: str) -> Dict[str, Any]:
    if isinstance(data, dict) and isinstance(data.get("context_groups"), list):
        return {
            **data,
            "type": data.get("type", "multimodal_file_search_results"),
            "version": data.get("version", 1),
            "search_endpoint": endpoint,
        }

    return {
        "type": "multimodal_file_search_results",
        "version": 1,
        "context_groups": _legacy_results_to_context_groups(data, file_id),
        "search_endpoint": endpoint,
    }


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield

app = FastAPI(title="pgvector Explorer", lifespan=lifespan)


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

@app.get("/api/files")
def api_files():
    """List all files with chunk counts and page ranges."""
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    COALESCE(cmetadata->>'file_id', 'unknown')            AS file_id,
                    COUNT(*)                                              AS chunks,
                    COUNT(*) FILTER (WHERE cmetadata->>'source_file' <> '') AS structured_chunks,
                    MIN(CASE WHEN cmetadata->>'page' ~ '^[0-9]+$'
                             THEN (cmetadata->>'page')::int END)          AS page_min,
                    MAX(CASE WHEN cmetadata->>'page' ~ '^[0-9]+$'
                             THEN (cmetadata->>'page')::int END)          AS page_max,
                    MAX(NULLIF(cmetadata->>'source_file',''))             AS source_file,
                    SUM(LENGTH(document))                                 AS total_chars
                FROM langchain_pg_embedding
                GROUP BY 1
                ORDER BY chunks DESC
            """)
            rows = cur.fetchall()
    return [dict(r) for r in rows]


@app.get("/api/chunks")
def api_chunks(
    file_id: str,
    page: Optional[int] = Query(None),
    search: Optional[str] = Query(None),
    limit: int = Query(200),
):
    """Return chunks for a file_id, optionally filtered by page and text search."""
    where = ["cmetadata->>'file_id' = %(file_id)s"]
    params: Dict[str, Any] = {"file_id": file_id, "limit": limit}

    if page is not None:
        where.append("(cmetadata->>'page')::int = %(page)s")
        params["page"] = page

    if search:
        where.append("document ILIKE %(search)s")
        params["search"] = f"%{search}%"

    sql = f"""
        SELECT
            uuid::text,
            custom_id,
            cmetadata,
            document,
            vector_dims(embedding) AS embedding_dims
        FROM langchain_pg_embedding
        WHERE {' AND '.join(where)}
        ORDER BY (cmetadata->>'page')::int NULLS LAST, custom_id
        LIMIT %(limit)s
    """
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

    result = []
    for r in rows:
        row = dict(r)
        row["cmetadata"] = row["cmetadata"] or {}
        result.append(row)
    return result


@app.get("/api/pages")
def api_pages(file_id: str):
    """Return distinct page numbers for a file."""
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT DISTINCT (cmetadata->>'page')::int AS page
                FROM langchain_pg_embedding
                WHERE cmetadata->>'file_id' = %s
                  AND cmetadata->>'page' IS NOT NULL
                  AND cmetadata->>'page' ~ '^[0-9]+$'
                ORDER BY 1
            """, (file_id,))
            rows = cur.fetchall()
    return [r["page"] for r in rows]


@app.post("/api/search")
async def api_search(body: dict):
    """
    Semantic search: forwards query to rag_api and normalizes the response.
    Body: { "query": str, "file_id": str, "k": int }
    """
    query = body.get("query", "").strip()
    file_id = body.get("file_id", "")
    k = int(body.get("k", 5))

    if not query or not file_id:
        return JSONResponse({"error": "query and file_id are required"}, status_code=400)

    payload = {"query": query, "file_id": file_id, "k": k}
    entity_id = body.get("entity_id") or _file_user_id(file_id)
    if entity_id:
        payload["entity_id"] = entity_id
    headers = {"Authorization": f"Bearer {_RAG_TOKEN}"} if _RAG_TOKEN else {}
    errors: List[Dict[str, Any]] = []
    empty_response: Optional[Dict[str, Any]] = None

    async with httpx.AsyncClient(timeout=30) as client:
        for endpoint in ("query-multimodal", "query"):
            try:
                resp = await client.post(f"{RAG_API_URL}/{endpoint}", json=payload, headers=headers)
                resp.raise_for_status()
                data = _normalize_search_response(resp.json(), file_id, endpoint)
                if data["context_groups"]:
                    return data
                empty_response = data
            except httpx.HTTPStatusError as exc:
                errors.append({
                    "endpoint": endpoint,
                    "status": exc.response.status_code,
                    "body": exc.response.text[:500],
                })
            except httpx.HTTPError as exc:
                errors.append({"endpoint": endpoint, "error": str(exc)})

    if empty_response is not None:
        empty_response["fallback_errors"] = errors
        return empty_response

    return JSONResponse(
        {"error": "RAG search failed", "details": errors},
        status_code=502,
    )


@app.get("/api/stats")
def api_stats():
    """Overall collection stats."""
    try:
        with _conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT
                        COUNT(*)                              AS total_chunks,
                        COUNT(DISTINCT cmetadata->>'file_id') AS total_files,
                        SUM(LENGTH(document))                 AS total_chars,
                        AVG(LENGTH(document))::int            AS avg_chunk_chars
                    FROM langchain_pg_embedding
                """)
                row = dict(cur.fetchone())
                # dims in a separate query to avoid MIN(vector) issue
                cur.execute("SELECT vector_dims(embedding) FROM langchain_pg_embedding LIMIT 1")
                dim_row = cur.fetchone()
                row["embedding_dims"] = dim_row["vector_dims"] if dim_row else None
        return row
    except Exception as exc:
        return {"total_chunks": 0, "total_files": 0, "total_chars": 0, "avg_chunk_chars": 0, "embedding_dims": None, "error": str(exc)}


# ---------------------------------------------------------------------------
# HTML UI
# ---------------------------------------------------------------------------
_HTML = r"""<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>pgvector Explorer</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', system-ui, sans-serif; background: #0f1117; color: #e2e8f0; min-height: 100vh; }
  .header { background: #1a1d27; border-bottom: 1px solid #2d3148; padding: 16px 24px; display: flex; align-items: center; gap: 16px; }
  .header h1 { font-size: 1.1rem; font-weight: 600; color: #a78bfa; }
  .header .stats { font-size: 0.78rem; color: #64748b; margin-left: auto; }
  .layout { display: grid; grid-template-columns: 280px 1fr; height: calc(100vh - 57px); }
  .sidebar { background: #13151f; border-right: 1px solid #2d3148; overflow-y: auto; padding: 12px 0; }
  .sidebar-title { font-size: 0.7rem; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: .08em; padding: 0 16px 8px; }
  .file-item { padding: 8px 16px; cursor: pointer; border-left: 3px solid transparent; transition: all .15s; }
  .file-item:hover { background: #1e2235; }
  .file-item.active { background: #1e2235; border-left-color: #a78bfa; }
  .file-item .name { font-size: 0.8rem; color: #cbd5e1; word-break: break-all; line-height: 1.3; }
  .file-item .meta { font-size: 0.7rem; color: #475569; margin-top: 3px; }
  .main { display: flex; flex-direction: column; overflow: hidden; }
  .toolbar { background: #1a1d27; border-bottom: 1px solid #2d3148; padding: 12px 20px; display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }
  .toolbar input, .toolbar select { background: #0f1117; border: 1px solid #2d3148; color: #e2e8f0; border-radius: 6px; padding: 6px 10px; font-size: 0.82rem; outline: none; }
  .toolbar input:focus, .toolbar select:focus { border-color: #a78bfa; }
  .toolbar input.search-box { flex: 1; min-width: 160px; }
  .btn { padding: 6px 14px; border-radius: 6px; border: none; cursor: pointer; font-size: 0.82rem; font-weight: 500; transition: all .15s; }
  .btn-primary { background: #7c3aed; color: #fff; }
  .btn-primary:hover { background: #6d28d9; }
  .btn-sm { background: #1e2235; color: #94a3b8; border: 1px solid #2d3148; }
  .btn-sm:hover { background: #2d3148; color: #e2e8f0; }
  .tabs { display: flex; gap: 0; border-bottom: 1px solid #2d3148; background: #1a1d27; }
  .tab { padding: 10px 18px; font-size: 0.82rem; cursor: pointer; border-bottom: 2px solid transparent; color: #64748b; }
  .tab.active { color: #a78bfa; border-bottom-color: #a78bfa; }
  .content-area { flex: 1; overflow-y: auto; padding: 16px 20px; }
  /* Chunks grid */
  .chunks-grid { display: flex; flex-direction: column; gap: 10px; }
  .chunk-card { background: #1a1d27; border: 1px solid #2d3148; border-radius: 8px; padding: 14px; }
  .chunk-card:hover { border-color: #4338ca; }
  .chunk-header { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; margin-bottom: 8px; }
  .badge { display: inline-flex; align-items: center; gap: 4px; font-size: 0.7rem; padding: 2px 8px; border-radius: 999px; font-weight: 600; }
  .badge-page { background: #1e3a5f; color: #93c5fd; }
  .badge-score { background: #14532d; color: #86efac; }
  .badge-img { background: #431407; color: #fb923c; }
  .badge-chars { background: #1e1b4b; color: #a5b4fc; }
  .badge-nf { background: #450a0a; color: #fca5a5; }
  .chunk-file { font-size: 0.7rem; color: #475569; margin-bottom: 6px; font-family: monospace; }
  .chunk-text { font-size: 0.8rem; color: #cbd5e1; line-height: 1.6; white-space: pre-wrap; word-break: break-word; max-height: 300px; overflow-y: auto; background: #0f1117; border-radius: 4px; padding: 8px; }
  .chunk-text.expanded { max-height: none; }
  .expand-btn { font-size: 0.7rem; color: #7c3aed; cursor: pointer; margin-top: 4px; }
  .chunk-images { margin-top: 8px; display: flex; gap: 6px; flex-wrap: wrap; }
  .img-thumb { max-height: 80px; max-width: 100px; border-radius: 4px; border: 1px solid #2d3148; cursor: pointer; }
  /* Search results */
  .search-group { background: #1a1d27; border: 1px solid #2d3148; border-radius: 8px; padding: 14px; margin-bottom: 12px; }
  .search-group-header { font-size: 0.8rem; color: #a78bfa; font-weight: 600; margin-bottom: 8px; }
  .search-chunk { border-top: 1px solid #1e2235; padding-top: 8px; margin-top: 8px; }
  /* Empty / loading */
  .empty { text-align: center; color: #475569; padding: 60px 20px; font-size: 0.85rem; }
  .loading { display: flex; align-items: center; justify-content: center; padding: 40px; gap: 10px; color: #64748b; }
  .spinner { width: 20px; height: 20px; border: 2px solid #2d3148; border-top-color: #a78bfa; border-radius: 50%; animation: spin .6s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }
  /* Lightbox */
  .lightbox { display: none; position: fixed; inset: 0; background: rgba(0,0,0,.85); z-index: 1000; align-items: center; justify-content: center; }
  .lightbox.open { display: flex; }
  .lightbox img { max-width: 90vw; max-height: 90vh; border-radius: 8px; }
  .lightbox-close { position: absolute; top: 16px; right: 16px; background: none; border: none; color: #fff; font-size: 1.5rem; cursor: pointer; }
  .page-pills { display: flex; gap: 4px; flex-wrap: wrap; margin-bottom: 10px; }
  .pill { padding: 3px 10px; border-radius: 999px; font-size: 0.72rem; cursor: pointer; border: 1px solid #2d3148; color: #94a3b8; background: transparent; }
  .pill:hover, .pill.active { background: #4338ca; border-color: #4338ca; color: #fff; }
  .pill.all { border-color: #a78bfa; color: #a78bfa; }
  .pill.all.active { background: #a78bfa; color: #fff; }
  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: #2d3148; border-radius: 3px; }
  .no-file { display: flex; align-items: center; justify-content: center; height: 100%; color: #475569; font-size: 0.85rem; }
</style>
</head>
<body>

<div class="header">
  <h1>🔍 pgvector Explorer</h1>
  <div class="stats" id="global-stats">Caricamento...</div>
</div>

<div class="layout">
  <!-- Sidebar: file list -->
  <aside class="sidebar" id="sidebar">
    <div class="sidebar-title">Documenti</div>
    <div id="file-list"><div class="loading"><div class="spinner"></div> Caricamento...</div></div>
  </aside>

  <!-- Main area -->
  <main class="main">
    <div id="no-file-msg" class="no-file">← Seleziona un documento dalla sidebar</div>
    <div id="file-area" style="display:none; flex:1; display:none; flex-direction:column; overflow:hidden;">
      <div class="tabs">
        <div class="tab active" data-tab="browse" onclick="switchTab('browse')">Browse Chunk</div>
        <div class="tab" data-tab="search" onclick="switchTab('search')">Ricerca Semantica</div>
      </div>

      <!-- Browse tab -->
      <div id="tab-browse" style="flex:1; display:flex; flex-direction:column; overflow:hidden;">
        <div class="toolbar">
          <div class="page-pills" id="page-pills"></div>
          <input class="search-box" id="text-filter" placeholder="Filtra per testo..." oninput="debounceLoad()" />
          <span id="chunk-count" style="font-size:0.75rem;color:#475569;"></span>
        </div>
        <div class="content-area" id="chunk-list"><div class="empty">Seleziona un file</div></div>
      </div>

      <!-- Search tab -->
      <div id="tab-search" style="display:none; flex:1; flex-direction:column; overflow:hidden;">
        <div class="toolbar">
          <input class="search-box" id="sem-query" placeholder="Scrivi una domanda per la ricerca semantica..." style="flex:2" />
          <select id="sem-k">
            <option value="3">Top 3</option>
            <option value="5" selected>Top 5</option>
            <option value="10">Top 10</option>
          </select>
          <button class="btn btn-primary" onclick="doSemanticSearch()">Cerca</button>
        </div>
        <div class="content-area" id="search-results"><div class="empty">Inserisci una domanda e premi Cerca</div></div>
      </div>
    </div>
  </main>
</div>

<!-- Lightbox -->
<div class="lightbox" id="lightbox" onclick="closeLightbox()">
  <button class="lightbox-close" onclick="closeLightbox()">✕</button>
  <img id="lightbox-img" src="" alt="preview" onclick="event.stopPropagation()">
</div>

<script>
const RAG_API = 'http://localhost:8000';
let currentFileId = null;
let currentPage = null;
let debounceTimer = null;
let pages = [];

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
async function init() {
  await Promise.all([loadStats(), loadFiles()]);
}

async function loadStats() {
  try {
    const r = await fetch('/api/stats');
    const s = await r.json();
    document.getElementById('global-stats').textContent =
      `${s.total_files} file · ${s.total_chunks} chunk · embedding ${s.embedding_dims}d`;
  } catch(e) { document.getElementById('global-stats').textContent = 'Stats non disponibili'; }
}

async function loadFiles() {
  const list = document.getElementById('file-list');
  list.innerHTML = '<div class="loading"><div class="spinner"></div> Caricamento...</div>';
  try {
    const r = await fetch('/api/files');
    const files = await r.json();
    list.innerHTML = '';
    files.forEach(f => {
      const el = document.createElement('div');
      el.className = 'file-item';
      el.dataset.fileId = f.file_id;
      const name = f.source_file || f.file_id;
      const short = name.length > 35 ? '...' + name.slice(-32) : name;
      const pages = (f.page_min != null) ? `p.${f.page_min}–${f.page_max}` : '?';
      el.innerHTML = `
        <div class="name" title="${name}">${short}</div>
        <div class="meta">${f.chunks} chunk · ${pages} · ${Math.round(f.total_chars/1024)}KB</div>
      `;
      el.onclick = () => selectFile(f.file_id, f.source_file);
      list.appendChild(el);
    });
  } catch(e) {
    list.innerHTML = `<div class="empty">Errore: ${e.message}</div>`;
  }
}

// ---------------------------------------------------------------------------
// File selection
// ---------------------------------------------------------------------------
async function selectFile(fileId, label) {
  currentFileId = fileId;
  currentPage = null;
  document.getElementById('text-filter').value = '';
  document.querySelectorAll('.file-item').forEach(el => {
    el.classList.toggle('active', el.dataset.fileId === fileId);
  });
  document.getElementById('no-file-msg').style.display = 'none';
  const fa = document.getElementById('file-area');
  fa.style.display = 'flex';
  fa.style.flexDirection = 'column';
  await Promise.all([loadPages(fileId), loadChunks()]);
}

async function loadPages(fileId) {
  const pagesData = await fetch(`/api/pages?file_id=${encodeURIComponent(fileId)}`).then(r => r.json());
  pages = pagesData;
  const pills = document.getElementById('page-pills');
  pills.innerHTML = '';
  const allPill = document.createElement('button');
  allPill.className = 'pill all active';
  allPill.textContent = 'Tutte le pagine';
  allPill.onclick = () => { currentPage = null; setActivePill(allPill); loadChunks(); };
  pills.appendChild(allPill);
  pagesData.forEach(p => {
    const pill = document.createElement('button');
    pill.className = 'pill';
    pill.textContent = `p.${p}`;
    pill.dataset.page = p;
    pill.onclick = () => { currentPage = p; setActivePill(pill); loadChunks(); };
    pills.appendChild(pill);
  });
}

function setActivePill(el) {
  document.querySelectorAll('#page-pills .pill').forEach(p => p.classList.remove('active'));
  el.classList.add('active');
}

// ---------------------------------------------------------------------------
// Chunk loading
// ---------------------------------------------------------------------------
function debounceLoad() {
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(loadChunks, 300);
}

async function loadChunks() {
  if (!currentFileId) return;
  const list = document.getElementById('chunk-list');
  list.innerHTML = '<div class="loading"><div class="spinner"></div> Caricamento...</div>';
  const search = document.getElementById('text-filter').value.trim();
  let url = `/api/chunks?file_id=${encodeURIComponent(currentFileId)}`;
  if (currentPage != null) url += `&page=${currentPage}`;
  if (search) url += `&search=${encodeURIComponent(search)}`;

  try {
    const chunks = await fetch(url).then(r => r.json());
    document.getElementById('chunk-count').textContent = `${chunks.length} chunk`;
    if (!chunks.length) { list.innerHTML = '<div class="empty">Nessun chunk trovato</div>'; return; }
    list.innerHTML = '';
    chunks.forEach(c => list.appendChild(buildChunkCard(c)));
  } catch(e) {
    list.innerHTML = `<div class="empty">Errore: ${e.message}</div>`;
  }
}

function buildChunkCard(c) {
  const meta = c.cmetadata || {};
  const page = meta.page ?? '?';
  const source = meta.source_file || '';
  const imageIds = meta.image_ids || [];
  const images = meta.images || [];
  const text = c.document || '';
  const dims = c.embedding_dims;

  const card = document.createElement('div');
  card.className = 'chunk-card';

  const badges = [
    `<span class="badge badge-page">p.${page}</span>`,
    `<span class="badge badge-chars">${text.length} chars</span>`,
    dims ? `<span class="badge badge-chars">${dims}d</span>` : '',
    imageIds.length ? `<span class="badge badge-img">🖼 ${imageIds.length} img</span>` : '',
  ].join('');

  const imgSection = buildImageSection(imageIds, images);
  const uid = 'c_' + Math.random().toString(36).slice(2);

  card.innerHTML = `
    <div class="chunk-header">${badges}</div>
    ${source ? `<div class="chunk-file">${source}</div>` : ''}
    <div class="chunk-text" id="${uid}">${escHtml(text)}</div>
    ${text.length > 400 ? `<div class="expand-btn" onclick="toggleExpand('${uid}',this)">▼ Mostra tutto</div>` : ''}
    ${imgSection}
  `;
  return card;
}

function buildImageSection(imageIds, images) {
  if (!imageIds.length) return '';
  const imgs = imageIds.map(iid => {
    const rec = images.find(i => (i.image_id || i) === iid);
    const url = rec?.url || null;
    if (!url) return `<span style="font-size:0.7rem;color:#475569">${iid}</span>`;
    return `<img class="img-thumb" src="${url}" alt="${iid}" onclick="openLightbox('${url}')">`;
  }).join('');
  return `<div class="chunk-images">${imgs}</div>`;
}

function toggleExpand(id, btn) {
  const el = document.getElementById(id);
  el.classList.toggle('expanded');
  btn.textContent = el.classList.contains('expanded') ? '▲ Comprimi' : '▼ Mostra tutto';
}

// ---------------------------------------------------------------------------
// Semantic search
// ---------------------------------------------------------------------------
async function doSemanticSearch() {
  if (!currentFileId) { alert('Seleziona prima un file'); return; }
  const query = document.getElementById('sem-query').value.trim();
  if (!query) return;
  const k = parseInt(document.getElementById('sem-k').value);
  const results = document.getElementById('search-results');
  results.innerHTML = '<div class="loading"><div class="spinner"></div> Ricerca in corso...</div>';

  try {
    const resp = await fetch('/api/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, file_id: currentFileId, k }),
    });
    if (!resp.ok) throw new Error(await resp.text());
    const data = await resp.json();
    renderSearchResults(data, results);
  } catch(e) {
    results.innerHTML = `<div class="empty">Errore: ${e.message}</div>`;
  }
}

function renderSearchResults(data, container) {
  container.innerHTML = '';
  // data may be a list of groups or the full multimodal result
  const groups = data.context_groups || (Array.isArray(data) ? data : [data]);
  if (!groups.length) { container.innerHTML = '<div class="empty">Nessun risultato</div>'; return; }

  groups.forEach((g, gi) => {
    const div = document.createElement('div');
    div.className = 'search-group';
    const topScore = Math.max(0, ...((g.chunks || []).map(c => c.score || 0)));
    const pages = [...new Set((g.chunks || []).map(c => c.metadata?.page).filter(p => p != null))].join(', ');
    div.innerHTML = `<div class="search-group-header">Gruppo ${gi+1} · pagine: ${pages || '?'} · score: ${topScore.toFixed(4)}</div>`;

    (g.chunks || []).forEach(c => {
      const meta = c.metadata || {};
      const score = c.score || 0;
      const text = c.text || '';
      const imageIds = meta.image_ids || [];
      const images = meta.images || [];
      const sc = document.createElement('div');
      sc.className = 'search-chunk';
      const scoreClass = score > 0.5 ? 'badge-score' : score > 0 ? 'badge-chars' : 'badge-nf';
      const uid = 'sc_' + Math.random().toString(36).slice(2);
      sc.innerHTML = `
        <div class="chunk-header">
          <span class="badge badge-page">p.${meta.page ?? '?'}</span>
          <span class="badge ${scoreClass}">score: ${score.toFixed(4)}</span>
          <span class="badge badge-chars">${text.length} chars</span>
          ${imageIds.length ? `<span class="badge badge-img">🖼 ${imageIds.length}</span>` : ''}
        </div>
        <div class="chunk-text" id="${uid}">${escHtml(text)}</div>
        ${text.length > 400 ? `<div class="expand-btn" onclick="toggleExpand('${uid}',this)">▼ Mostra tutto</div>` : ''}
        ${buildImageSection(imageIds, images)}
      `;
      div.appendChild(sc);
    });
    container.appendChild(div);
  });
}

// ---------------------------------------------------------------------------
// Tab switching
// ---------------------------------------------------------------------------
function switchTab(tab) {
  document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.tab === tab));
  document.getElementById('tab-browse').style.display = tab === 'browse' ? 'flex' : 'none';
  document.getElementById('tab-search').style.display = tab === 'search' ? 'flex' : 'none';
}

// ---------------------------------------------------------------------------
// Lightbox
// ---------------------------------------------------------------------------
function openLightbox(url) {
  document.getElementById('lightbox-img').src = url;
  document.getElementById('lightbox').classList.add('open');
}
function closeLightbox() { document.getElementById('lightbox').classList.remove('open'); }
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeLightbox(); });

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function escHtml(s) {
  return (s || '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

init();
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(_HTML)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"pgvector Explorer -> http://localhost:{EXPLORER_PORT}")
    print(f"  Postgres: {PG_USER}@{PG_HOST}:{PG_PORT}/{PG_DB}")
    print(f"  rag_api:  {RAG_API_URL}")
    print(f"  JWT auth: {'yes (token generated)' if _RAG_TOKEN else 'NO - set JWT_SECRET env var'}")
    if not JWT_SECRET:
        print("  WARNING: JWT_SECRET not set - semantic search will return 401")
        print("  Set it: $env:JWT_SECRET='16f8c0ef4a5d391b26034086c628469d3f9f497f08163ab9b40137092f2909ef'; python utils/pgvector_explorer/app.py")
    uvicorn.run(app, host="0.0.0.0", port=EXPLORER_PORT, log_level="warning")
