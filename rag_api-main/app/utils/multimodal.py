# app/utils/multimodal.py
"""
Multimodal chunking and retrieval utilities for RAG with images.

Implements:
- Placeholder-aware text chunking that preserves <image_id>...</image_id> markers
- embedding_text enrichment (strips placeholders, adds image summaries)
- Context expansion (previous_chunk_id / next_chunk_id linking)
- Connected-node merging of overlapping result groups
"""

import json
import os
import re
import hashlib
import unicodedata
from difflib import SequenceMatcher
from typing import List, Dict, Optional, Any, Tuple

from langchain_core.documents import Document

from app.config import logger

# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------

_IMAGE_ID_RE = re.compile(r"<image_id>([^<]+)</image_id>")
_WORD_RE = re.compile(r"\w+", re.UNICODE)
_HEADING_RE = re.compile(r"^#{1,6}\s+(.+)", re.MULTILINE)
_DEFAULT_RERANK_MARGIN = 0.04
_INTENT_TERMS = {
    "how_to": {
        "come", "how", "procedura", "procedure", "usare", "use", "installare",
        "configurare", "attivare", "avviare", "accendere", "spegnere",
        "caricare", "pulire", "rimuovere",
    },
    "status": {
        "stato", "status", "controllare", "check", "verificare", "verifica",
        "livello", "batteria", "health", "monitorare",
    },
    "troubleshooting": {
        "errore", "error", "problema", "problem", "guasto", "warning",
        "avviso", "lampeggia", "lampeggiante", "reset", "risolvere",
        "fix", "issue",
    },
}
_STOPWORDS = {
    "a", "ad", "al", "alla", "allo", "all", "ai", "agli", "alle", "anche",
    "che", "chi", "ci", "coi", "col", "come", "con", "cosa", "da", "dal",
    "dalla", "de", "dei", "del", "dell", "della", "delle", "di", "do",
    "e", "ed", "for", "gli", "ha", "hai", "he", "how", "i", "il", "in",
    "is", "it", "la", "le", "lo", "mi", "nel", "nella", "of", "on", "o",
    "per", "piu", "puo", "quale", "quando", "se", "su", "sul", "sulla",
    "the", "to", "un", "una", "uno", "what", "when", "where", "why",
}


def extract_image_ids_from_text(text: str) -> List[str]:
    """Return all image_id values found in a markdown text block."""
    return _IMAGE_ID_RE.findall(text)


def strip_image_placeholders(text: str) -> str:
    """Remove <image_id>...</image_id> tags, leaving surrounding text intact."""
    return _IMAGE_ID_RE.sub("", text).strip()


def normalize_search_query(query: str) -> str:
    return _normalize_text(query)


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = normalized.lower().replace("’", "'").replace("`", "'")
    normalized = re.sub(r"[^\w\s']+", " ", normalized, flags=re.UNICODE)
    return re.sub(r"\s+", " ", normalized).strip()


def _tokens(text: str) -> List[str]:
    return [
        token
        for token in _WORD_RE.findall(_normalize_text(text))
        if len(token) > 2 and token not in _STOPWORDS
    ]


def _headings(text: str) -> List[str]:
    return [match.group(1).strip() for match in _HEADING_RE.finditer(text or "")]


def _ngrams(tokens: List[str], size: int) -> set:
    return {
        " ".join(tokens[idx : idx + size])
        for idx in range(0, max(len(tokens) - size + 1, 0))
    }


def _token_matches(query_terms: set, candidate_terms: set) -> int:
    if not query_terms or not candidate_terms:
        return 0

    matches = len(query_terms & candidate_terms)
    unmatched = query_terms - candidate_terms
    for query_term in unmatched:
        if len(query_term) < 5:
            continue
        for candidate in candidate_terms:
            if len(candidate) < 5:
                continue
            similarity = SequenceMatcher(None, query_term, candidate).ratio()
            if query_term[:4] == candidate[:4] and similarity >= 0.60:
                matches += 1
                break
            if similarity >= 0.84:
                matches += 1
                break
    return matches


def _flatten_hint_values(value: Any) -> List[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        result = []
        for item in value:
            result.extend(_flatten_hint_values(item))
        return result
    if isinstance(value, dict):
        result = []
        for key, item in value.items():
            result.append(str(key))
            result.extend(_flatten_hint_values(item))
        return result
    return []


def _hint_text(metadata: Dict[str, Any]) -> str:
    parts = []
    for key in ("domain_hints", "chunk_hints", "chunk_keywords", "chunk_headings"):
        parts.extend(_flatten_hint_values(metadata.get(key)))
    return " ".join(parts)


def detect_query_intent(query: str) -> Optional[str]:
    query_terms = set(_tokens(query))
    if not query_terms:
        return None

    scored = [
        (intent, len(query_terms & terms))
        for intent, terms in _INTENT_TERMS.items()
    ]
    intent, score = max(scored, key=lambda item: item[1])
    return intent if score > 0 else None


def _intent_hint_text(metadata: Dict[str, Any], intent: Optional[str]) -> str:
    if not intent:
        return ""
    domain_hints = metadata.get("domain_hints")
    if not isinstance(domain_hints, dict):
        return ""
    intents = domain_hints.get("intents")
    if not isinstance(intents, dict):
        return ""
    return " ".join(_flatten_hint_values(intents.get(intent)))


def rerank_results(
    query: str,
    results: List[Tuple[Document, float]],
) -> List[Tuple[Document, float]]:
    """
    Rerank pgvector distance results with generic document signals.

    pgvector returns distance, where lower is better. Vector distance remains the
    base signal; lexical overlap, heading overlap, image presence, and optional
    ingest-time domain hints only nudge otherwise close results.
    """
    normalized_query = normalize_search_query(query)
    query_tokens = _tokens(normalized_query)
    query_terms = set(query_tokens)
    if not query_terms or not results:
        return results

    query_intent = detect_query_intent(normalized_query)
    query_bigrams = _ngrams(query_tokens, 2)
    query_trigrams = _ngrams(query_tokens, 3)

    def fields_for(doc: Document) -> Dict[str, str]:
        metadata = doc.metadata or {}
        visible_text = strip_image_placeholders(metadata.get("text", ""))
        headings = " ".join(_headings(visible_text))
        hint_text = _hint_text(metadata)
        intent_hint_text = _intent_hint_text(metadata, query_intent)
        full_text = " ".join(
            str(part)
            for part in [
                metadata.get("section_title", ""),
                visible_text,
                doc.page_content or "",
                hint_text,
                intent_hint_text,
            ]
            if part
        )
        return {
            "full": full_text,
            "visible": visible_text,
            "headings": headings,
            "hints": hint_text,
            "intent_hints": intent_hint_text,
        }

    def adjusted_distance(doc: Document, distance: float) -> float:
        metadata = doc.metadata or {}
        fields = fields_for(doc)
        full_tokens = set(_tokens(fields["full"]))
        visible_tokens = set(_tokens(fields["visible"]))
        heading_tokens = set(_tokens(fields["headings"]))
        hint_tokens = set(_tokens(fields["hints"]))
        intent_hint_tokens = set(_tokens(fields["intent_hints"]))
        full_text = _normalize_text(fields["full"])
        visible_text = strip_image_placeholders(metadata.get("text", ""))

        lexical_hits = _token_matches(query_terms, full_tokens)
        visible_hits = _token_matches(query_terms, visible_tokens)
        heading_hits = _token_matches(query_terms, heading_tokens)
        hint_hits = _token_matches(query_terms, hint_tokens)
        intent_hint_hits = _token_matches(query_terms, intent_hint_tokens)
        full_token_list = _tokens(fields["full"])
        bigram_hits = len(query_bigrams & _ngrams(full_token_list, 2))
        trigram_hits = len(query_trigrams & _ngrams(full_token_list, 3))

        exact_phrase_bonus = 0.10 if normalized_query and normalized_query in full_text else 0.0
        has_images = bool(metadata.get("image_ids"))
        image_bonus = 0.025 if has_images else 0.0
        title_only = len(visible_text.strip()) < 80
        title_only_penalty = 0.0
        if title_only:
            title_only_penalty = 0.12 if has_images else 0.24
        no_visible_hit_penalty = 0.05 if visible_hits == 0 else 0.0

        lexical_bonus = min(lexical_hits * 0.018, 0.12)
        visible_bonus = min(visible_hits * 0.012, 0.08)
        heading_bonus = min(heading_hits * 0.045, 0.14)
        hint_bonus = min(hint_hits * 0.025, 0.10)
        intent_bonus = min(intent_hint_hits * 0.035, 0.10)
        phrase_bonus = min(bigram_hits * 0.035 + trigram_hits * 0.05, 0.12)

        return distance + (
            title_only_penalty
            + no_visible_hit_penalty
            - lexical_bonus
            - visible_bonus
            - heading_bonus
            - hint_bonus
            - intent_bonus
            - phrase_bonus
            - exact_phrase_bonus
            - image_bonus
        )

    reranked = []
    for doc, distance in results:
        doc.metadata["_rank_distance"] = adjusted_distance(doc, distance)
        reranked.append((doc, distance))

    return sorted(reranked, key=lambda item: item[0].metadata.get("_rank_distance", item[1]))


def _extract_chunk_hints(chunk_text: str, section_title: str = "") -> Dict[str, Any]:
    visible_text = strip_image_placeholders(chunk_text)
    headings = _headings(visible_text)
    weighted_text = " ".join([section_title, " ".join(headings), visible_text])
    terms = _tokens(weighted_text)
    frequencies: Dict[str, int] = {}
    for term in terms:
        frequencies[term] = frequencies.get(term, 0) + 1
    keywords = [
        term
        for term, _ in sorted(
            frequencies.items(),
            key=lambda item: (-item[1], -len(item[0]), item[0]),
        )[:12]
    ]
    return {
        "headings": headings[:6],
        "keywords": keywords,
    }


def apply_domain_hints(
    chunks: List[Dict[str, Any]],
    domain_hints: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Attach generic chunk hints plus optional OpenAI-created document hints."""
    safe_domain_hints = domain_hints or {}
    for chunk in chunks:
        chunk_hints = _extract_chunk_hints(
            chunk.get("text", ""),
            chunk.get("section_title", ""),
        )
        chunk["chunk_hints"] = chunk_hints
        chunk["chunk_keywords"] = chunk_hints.get("keywords", [])
        chunk["chunk_headings"] = chunk_hints.get("headings", [])
        chunk["domain_hints"] = safe_domain_hints
    return chunks


def filter_ranked_results(
    results: List[Tuple[Document, float]],
    min_results: int = 1,
    margin: Optional[float] = None,
) -> List[Tuple[Document, float]]:
    if not results:
        return []

    configured_margin = os.getenv("RAG_RERANK_MARGIN")
    if margin is None and configured_margin:
        try:
            margin = float(configured_margin)
        except ValueError:
            logger.warning("Invalid RAG_RERANK_MARGIN=%r; using default", configured_margin)

    active_margin = _DEFAULT_RERANK_MARGIN if margin is None else margin
    if active_margin <= 0:
        return results

    best_distance = results[0][0].metadata.get("_rank_distance", results[0][1])
    cutoff = best_distance + active_margin
    filtered = [
        item
        for item in results
        if item[0].metadata.get("_rank_distance", item[1]) <= cutoff
    ]
    if len(filtered) >= min_results:
        return filtered
    return results[:min_results]


def build_ingest_report(
    chunks: List[Dict[str, Any]],
    images_by_id: Dict[str, Any],
    domain_hints: Optional[Dict[str, Any]] = None,
    domain_hints_status: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    chunks_with_images = [chunk for chunk in chunks if chunk.get("image_ids")]
    short_chunks = [
        chunk
        for chunk in chunks
        if len(strip_image_placeholders(chunk.get("text", "")).strip()) < 80
    ]
    unresolved_image_ids = sorted({
        image_id
        for chunk in chunks
        for image_id in chunk.get("image_ids", [])
        if image_id not in images_by_id
    })
    chunks_with_hints = [
        chunk
        for chunk in chunks
        if chunk.get("chunk_keywords") or chunk.get("chunk_headings")
    ]

    return {
        "chunks": len(chunks),
        "images": len(images_by_id),
        "chunks_with_images": len(chunks_with_images),
        "chunks_without_images": len(chunks) - len(chunks_with_images),
        "short_chunks": len(short_chunks),
        "short_chunks_with_images": sum(1 for chunk in short_chunks if chunk.get("image_ids")),
        "chunks_with_hints": len(chunks_with_hints),
        "unresolved_image_ids": unresolved_image_ids[:20],
        "domain_hints_generated": bool(domain_hints),
        "domain_hints_status": domain_hints_status or {},
        "domain": (domain_hints or {}).get("domain", ""),
        "key_terms": (domain_hints or {}).get("key_terms", [])[:20],
    }

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

_DEFAULT_CHUNK_SIZE = 800  # characters (not tokens — fast, good enough for this use-case)
_DEFAULT_OVERLAP = 120


def _build_embedding_text(
    chunk_text: str,
    image_ids: List[str],
    images_by_id: Dict[str, Any],
    source_file: str,
    page: int,
) -> str:
    """
    Build the text that will be embedded (no placeholders, summaries included).
    """
    clean = strip_image_placeholders(chunk_text)
    parts = [clean]
    for iid in image_ids:
        img = images_by_id.get(iid)
        if not img:
            continue
        summary = img.get("image_summary") if isinstance(img, dict) else getattr(img, "image_summary", None)
        if summary:
            parts.append(f"[Image: {summary}]")
    parts.append(f"Source: {source_file}, page {page}.")
    return " ".join(p for p in parts if p).strip()


def chunk_page(
    page_markdown: str,
    page_num: int,
    file_id: str,
    source_file: str,
    images_by_id: Dict[str, Any],
    section_title: str = "",
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    overlap: int = _DEFAULT_OVERLAP,
) -> List[Dict[str, Any]]:
    """
    Split a single page's markdown into overlapping chunks, preserving image placeholders.

    Returns a list of raw chunk dicts (without previous/next links — those are added later).
    """
    # Simple paragraph/heading-aware split: try to break on double newlines first
    chunks: List[Dict[str, Any]] = []
    text = page_markdown

    # Try to infer section title from first heading in the page
    if not section_title:
        heading_match = re.search(r"^#{1,6}\s+(.+)", text, re.MULTILINE)
        if heading_match:
            section_title = heading_match.group(1).strip()

    # Split on paragraph boundaries
    paragraphs = re.split(r"\n{2,}", text)

    current: List[str] = []
    current_len = 0

    def flush():
        """Emit the current buffer as a chunk."""
        if not current:
            return
        chunk_text = "\n\n".join(current)
        image_ids = extract_image_ids_from_text(chunk_text)
        chunk_index = len(chunks) + 1
        chunk_id = f"{file_id}_p{page_num}_c{chunk_index:02d}"
        embedding_text = _build_embedding_text(
            chunk_text, image_ids, images_by_id, source_file, page_num
        )
        chunks.append(
            {
                "chunk_id": chunk_id,
                "file_id": file_id,
                "source_file": source_file,
                "page": page_num,
                "section_title": section_title,
                "text": chunk_text,
                "embedding_text": embedding_text,
                "image_ids": image_ids,
            }
        )

    def current_is_heading_only() -> bool:
        return bool(current) and all(
            re.match(r"^#{1,6}\s+", item.strip())
            for item in current
        )

    def current_is_image_only() -> bool:
        return bool(current) and all(
            _IMAGE_ID_RE.search(item.strip())
            and not strip_image_placeholders(item).strip()
            for item in current
        )

    def current_is_heading_image_only() -> bool:
        if not current:
            return False
        for item in current:
            stripped = item.strip()
            visible = strip_image_placeholders(stripped).strip()
            if _IMAGE_ID_RE.search(stripped) and not visible:
                continue
            if re.match(r"^#{1,6}\s+", visible):
                continue
            return False
        return True

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        is_img_para = bool(_IMAGE_ID_RE.search(para))
        is_heading_para = bool(re.match(r"^#{1,6}\s+", para))

        if is_heading_para and current and not current_is_image_only() and not current_is_heading_image_only():
            flush()
            current = []
            current_len = 0

        if is_img_para and current and not current_is_heading_only() and not current_is_image_only():
            # Always flush before an image placeholder paragraph so it lands in a
            # fresh chunk and is never duplicated via overlap.
            flush()
            current = []
            current_len = 0
        elif current_len + len(para) > chunk_size and current:
            flush()
            # Overlap: keep last paragraph(s) up to `overlap` chars.
            # Image-only paragraphs are excluded from overlap to prevent
            # the same <image_id> tag appearing in two consecutive chunks.
            overlap_buf: List[str] = []
            overlap_len = 0
            for p in reversed(current):
                if _IMAGE_ID_RE.search(p):
                    continue
                if overlap_len + len(p) <= overlap:
                    overlap_buf.insert(0, p)
                    overlap_len += len(p)
                else:
                    break
            current = overlap_buf
            current_len = overlap_len

        current.append(para)
        current_len += len(para)

    flush()

    # If no paragraphs produced chunks (e.g. single long line), force a single chunk
    if not chunks and text.strip():
        chunk_id = f"{file_id}_p{page_num}_c01"
        image_ids = extract_image_ids_from_text(text)
        embedding_text = _build_embedding_text(
            text, image_ids, images_by_id, source_file, page_num
        )
        chunks.append(
            {
                "chunk_id": chunk_id,
                "file_id": file_id,
                "source_file": source_file,
                "page": page_num,
                "section_title": section_title,
                "text": text,
                "embedding_text": embedding_text,
                "image_ids": image_ids,
            }
        )

    return chunks


def build_all_chunks(
    pages: List[Any],  # List[StructuredOCRPage]
    file_id: str,
    source_file: str,
    images_by_id: Dict[str, Any],
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    overlap: int = _DEFAULT_OVERLAP,
) -> List[Dict[str, Any]]:
    """
    Produce all multimodal chunks for a document, link previous/next, assign sequence_number.
    """
    all_chunks: List[Dict[str, Any]] = []

    for page in pages:
        page_num = page.page if hasattr(page, "page") else page.get("page", 0)
        markdown = page.markdown if hasattr(page, "markdown") else page.get("markdown", "")

        page_chunks = chunk_page(
            page_markdown=markdown,
            page_num=page_num,
            file_id=file_id,
            source_file=source_file,
            images_by_id=images_by_id,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        all_chunks.extend(page_chunks)

    # Link previous/next within the same document
    for i, chunk in enumerate(all_chunks):
        chunk["sequence_number"] = i + 1
        chunk["previous_chunk_id"] = all_chunks[i - 1]["chunk_id"] if i > 0 else None
        chunk["next_chunk_id"] = (
            all_chunks[i + 1]["chunk_id"] if i < len(all_chunks) - 1 else None
        )

    return apply_domain_hints(all_chunks)


def chunks_to_documents(
    chunks: List[Dict[str, Any]],
    user_id: str,
    images_by_id: Dict[str, Any],
    source_url_base: str = "",
) -> List[Document]:
    """
    Convert chunk dicts into LangChain Documents ready for vector store insertion.
    The `page_content` is the `embedding_text` (clean, no placeholders).
    All rich metadata is stored in `metadata`.
    """
    docs: List[Document] = []
    for chunk in chunks:
        page = chunk.get("page", 0)
        source_url = f"{source_url_base}#page={page}" if source_url_base else ""

        # Collect image records for chunks that reference images
        image_records = [
            _image_to_dict(images_by_id[iid])
            for iid in chunk.get("image_ids", [])
            if iid in images_by_id
        ]

        metadata = {
            "chunk_id": chunk["chunk_id"],
            "file_id": chunk["file_id"],
            "source_file": chunk["source_file"],
            "page": page,
            "section_title": chunk.get("section_title", ""),
            "source_url": source_url,
            "image_ids": chunk.get("image_ids", []),
            "images": image_records,
            "previous_chunk_id": chunk.get("previous_chunk_id"),
            "next_chunk_id": chunk.get("next_chunk_id"),
            "sequence_number": chunk.get("sequence_number", 0),
            "text": chunk["text"],  # original text with placeholders
            "ingest_run_id": chunk.get("ingest_run_id"),
            "chunk_hints": chunk.get("chunk_hints", {}),
            "chunk_keywords": chunk.get("chunk_keywords", []),
            "chunk_headings": chunk.get("chunk_headings", []),
            "domain_hints": chunk.get("domain_hints", {}),
            "user_id": user_id,
            "digest": hashlib.md5(chunk["embedding_text"].encode("utf-8", "ignore")).hexdigest(),
        }
        docs.append(Document(page_content=chunk["embedding_text"], metadata=metadata))

    return docs


def _image_to_dict(img: Any) -> Dict[str, Any]:
    """Normalise an image record (Pydantic model or dict) to a plain dict."""
    if hasattr(img, "model_dump"):
        return img.model_dump()
    if hasattr(img, "dict"):
        return img.dict()
    return dict(img) if isinstance(img, dict) else {}


# ---------------------------------------------------------------------------
# Context expansion and connected-node merging
# ---------------------------------------------------------------------------

def expand_results(
    results: List[Tuple[Document, float]],
    all_chunks_by_id: Dict[str, Document],
    depth_before: int = 1,
    depth_after: int = 1,
) -> List[List[Tuple[Document, Optional[float]]]]:
    """
    For each retrieved hit, fetch depth_before preceding chunks and depth_after following chunks.
    Returns a list of groups, each group being a list of (Document, score) pairs in doc order.
    """
    groups: List[List[Tuple[Document, Optional[float]]]] = []

    for doc, score in results:
        group_ids: List[str] = []

        # Walk backward
        prev_id = doc.metadata.get("previous_chunk_id")
        for _ in range(depth_before):
            if not prev_id or prev_id not in all_chunks_by_id:
                break
            group_ids.insert(0, prev_id)
            prev_id = all_chunks_by_id[prev_id].metadata.get("previous_chunk_id")

        current_id = doc.metadata.get("chunk_id")
        if current_id:
            group_ids.append(current_id)

        # Walk forward
        next_id = doc.metadata.get("next_chunk_id")
        for _ in range(depth_after):
            if not next_id or next_id not in all_chunks_by_id:
                break
            group_ids.append(next_id)
            next_id = all_chunks_by_id[next_id].metadata.get("next_chunk_id")

        # Build group with (doc, score) — expanded chunks get score 0 (not retrieved directly)
        group: List[Tuple[Document, Optional[float]]] = []
        for gid in group_ids:
            if gid == current_id:
                group.append((doc, score))
            elif gid in all_chunks_by_id:
                group.append((all_chunks_by_id[gid], None))
        groups.append(group)

    return groups


def _best_distance(group: List[Tuple[Document, Optional[float]]]) -> float:
    distances = [score for _, score in group if score is not None]
    return min(distances) if distances else float("inf")


def _best_rank_distance(group: List[Tuple[Document, Optional[float]]]) -> float:
    distances = [
        doc.metadata.get("_rank_distance", score)
        for doc, score in group
        if score is not None
    ]
    return min(distances) if distances else float("inf")


def _distance_to_relevance(distance: Optional[float]) -> float:
    if distance is None:
        return 0.0
    return max(0.0, 1.0 - distance)


def merge_groups(
    groups: List[List[Tuple[Document, Optional[float]]]]
) -> List[List[Tuple[Document, Optional[float]]]]:
    """
    Merge overlapping groups (groups sharing a chunk_id).
    Returns deduplicated, sorted groups.
    """
    if not groups:
        return []

    # Union-find on chunk_ids
    chunk_to_group: Dict[str, int] = {}
    merged: Dict[int, List[Tuple[Document, Optional[float]]]] = {}
    gid_counter = 0

    for group in groups:
        ids_in_group = [d.metadata.get("chunk_id") for d, _ in group if d.metadata.get("chunk_id")]
        # Find existing groups that overlap
        existing_gids = {chunk_to_group[cid] for cid in ids_in_group if cid in chunk_to_group}

        if not existing_gids:
            # New group
            gid = gid_counter
            gid_counter += 1
            merged[gid] = []
        else:
            # Merge all overlapping into the smallest gid
            gid = min(existing_gids)
            for other_gid in existing_gids - {gid}:
                merged[gid].extend(merged.pop(other_gid, []))
                for cid, og in list(chunk_to_group.items()):
                    if og == other_gid:
                        chunk_to_group[cid] = gid

        # Add new items
        existing_ids_in_gid = {d.metadata.get("chunk_id") for d, _ in merged[gid]}
        response_group = sorted(
            group,
            key=lambda item: (
                item[1] is None,
                item[0].metadata.get("sequence_number", 0),
            ),
        )

        for doc, score in response_group:
            cid = doc.metadata.get("chunk_id")
            if cid not in existing_ids_in_gid:
                merged[gid].append((doc, score))
                existing_ids_in_gid.add(cid)
            chunk_to_group[cid] = gid

    # Sort each group by sequence_number, drop empty groups
    result = []
    for group in merged.values():
        if not group:
            continue
        group.sort(key=lambda x: x[0].metadata.get("sequence_number", 0))
        result.append(group)

    result.sort(key=_best_rank_distance)
    return result


def groups_to_context_groups(
    merged_groups: List[List[Tuple[Document, Optional[float]]]],
) -> List[Dict[str, Any]]:
    """
    Convert merged chunk groups into the context_groups response format consumed by the gateway.
    """
    output = []
    for idx, group in enumerate(merged_groups):
        if not group:
            continue

        response_group = sorted(
            group,
            key=lambda item: (
                item[1] is None,
                item[0].metadata.get("_rank_distance", item[1] or float("inf")),
                item[0].metadata.get("sequence_number", 0),
            ),
        )

        first_doc = response_group[0][0]
        file_id = first_doc.metadata.get("file_id", "")
        source_file = first_doc.metadata.get("source_file", "")
        best_distance = _best_distance(group)
        best_distance_value = best_distance if best_distance != float("inf") else None
        best_rank_distance = _best_rank_distance(group)
        best_rank_distance_value = (
            best_rank_distance if best_rank_distance != float("inf") else None
        )
        best_score = _distance_to_relevance(best_distance_value)

        pages = sorted({d.metadata.get("page", 0) for d, _ in group})

        chunks_out = []
        images_seen: Dict[str, Any] = {}
        sources_seen: Dict[str, Dict[str, Any]] = {}

        for doc, score in response_group:
            m = doc.metadata
            chunk_images = m.get("images", [])
            for img in chunk_images:
                iid = img.get("image_id") if isinstance(img, dict) else getattr(img, "image_id", None)
                if iid and iid not in images_seen:
                    images_seen[iid] = img if isinstance(img, dict) else _image_to_dict(img)

            source_url = m.get("source_url", "")
            page = m.get("page", 0)
            src_key = f"{source_file}:{page}"
            if src_key not in sources_seen:
                sources_seen[src_key] = {
                    "source_file": source_file,
                    "page": page,
                    "source_url": source_url,
                }

            # Use original text (with placeholders) for the chunk payload
            chunk_text = m.get("text", doc.page_content)

            chunks_out.append(
                {
                    "chunk_id": m.get("chunk_id", ""),
                    "text": chunk_text,
                    "distance": score,
                    "score": _distance_to_relevance(score),
                    "metadata": {
                        "file_id": file_id,
                        "source_file": source_file,
                        "page": page,
                        "source_url": source_url,
                        "image_ids": m.get("image_ids", []),
                        "images": [
                            images_seen[iid]
                            for iid in m.get("image_ids", [])
                            if iid in images_seen
                        ],
                        "previous_chunk_id": m.get("previous_chunk_id"),
                        "next_chunk_id": m.get("next_chunk_id"),
                    },
                }
            )

        output.append(
            {
                "group_id": f"group_{idx + 1}",
                "file_id": file_id,
                "source_file": source_file,
                "pages": pages,
                "distance": best_distance_value,
                "rank_distance": best_rank_distance_value,
                "score": best_score,
                "chunks": chunks_out,
                "images": list(images_seen.values()),
                "sources": list(sources_seen.values()),
            }
        )

    return output
