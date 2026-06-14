"""
agents-gateway multimodal helpers.

Provides pure functions for parsing enriched tool messages, selecting
contextual images, and composing the final markdown response with sources.
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from logger import get_logger

log = get_logger("multimodal")

_IMAGE_ID_RE = re.compile(r"<image_id>([^<]+)</image_id>")
# Control characters that break JSON parsers (everything below 0x20 except \t \n \r)
_CTRL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


# ---------------------------------------------------------------------------
# Tool message parsing
# ---------------------------------------------------------------------------

def parse_tool_results(tool_content: str) -> Optional[Dict[str, Any]]:
    """
    Try to parse the tool message content as a multimodal_file_search_results payload.
    Returns the parsed dict on success, None on failure (caller should use legacy path).
    """
    stripped = (tool_content or "").strip()
    if not stripped.startswith("{"):
        log.info(
            "parse_tool_results.not_json",
            content_len=len(tool_content or ""),
            first_chars_repr=repr(stripped[:50]),
        )
        return None
    _lenient_decoder = json.JSONDecoder(strict=False)

    # Pre-clean: strip problematic control characters that survive strict=False
    cleaned = _CTRL_CHAR_RE.sub("", stripped)

    def _try_parse(s: str, lenient: bool = False) -> Optional[Dict[str, Any]]:
        data = (_lenient_decoder.decode(s) if lenient else json.loads(s))
        # Handle double-encoded JSON
        if isinstance(data, str):
            log.info("parse_tool_results.double_encoded", inner_start=repr(data[:30]))
            data = (_lenient_decoder.decode(data) if lenient else json.loads(data))
        return data if isinstance(data, dict) else None

    def _is_multimodal(data: Dict[str, Any]) -> bool:
        return data.get("type") == "multimodal_file_search_results" and "context_groups" in data

    # Fast path: strict JSON parse on cleaned string
    try:
        data = _try_parse(cleaned)
        if data is None:
            return None
        found_type = data.get("type")
        has_groups = "context_groups" in data
        log.info("parse_tool_results.parsed", found_type=found_type, has_context_groups=has_groups)
        if _is_multimodal(data):
            log.debug("parse_tool_results.multimodal_detected", groups=len(data["context_groups"]))
            return data
    except (json.JSONDecodeError, TypeError) as exc:
        log.info(
            "parse_tool_results.decode_failed_retrying",
            exc=str(exc),
            content_len=len(tool_content),
        )
        # Fallback: lenient parser on cleaned string, then on raw stripped string.
        for candidate in (cleaned, stripped):
            try:
                data = _try_parse(candidate, lenient=True)
                if data and _is_multimodal(data):
                    log.info(
                        "parse_tool_results.recovered_lenient",
                        groups=len(data["context_groups"]),
                    )
                    return data
            except (json.JSONDecodeError, TypeError):
                pass
        log.warning(
            "parse_tool_results.decode_failed",
            content_len=len(tool_content),
            first_chars_repr=repr(stripped[:80]),
        )
    return None


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def extract_image_placeholders(text: str) -> List[str]:
    """Return all image_id values found in placeholder tags within text."""
    return _IMAGE_ID_RE.findall(text)


def build_images_by_id(context_groups: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Build a flat {image_id → image_record} dict from all context_groups.
    Collects images from:
      1. group-level `images` list
      2. chunk-level `metadata.images` lists
    """
    images: Dict[str, Dict[str, Any]] = {}
    for group in context_groups:
        for img in group.get("images", []):
            iid = img.get("image_id")
            if iid and iid not in images:
                images[iid] = img
        for chunk in group.get("chunks", []):
            for img in chunk.get("metadata", {}).get("images", []):
                iid = img.get("image_id") if isinstance(img, dict) else None
                if iid and iid not in images:
                    images[iid] = img
    return images


def select_contextual_images(
    context_groups: List[Dict[str, Any]],
    images_by_id: Dict[str, Dict[str, Any]],
    max_images: int = 4,
) -> List[Dict[str, Any]]:
    """
    Return images from the single highest-scoring chunk that has images.

    We find the best chunk across all groups (score > 0, has image_ids, has valid URLs),
    then return up to max_images images from that chunk only.
    If no chunk with images is found, return an empty list.
    """
    best_chunk: Optional[Dict[str, Any]] = None
    best_priority: float = -1.0

    for group in context_groups:
        for chunk in group.get("chunks", []):
            chunk_score = chunk.get("score", 0.0)
            if chunk_score <= 0:
                continue
            chunk_image_ids = chunk.get("metadata", {}).get("image_ids", [])
            if not chunk_image_ids:
                continue
            # Verify at least one image_id resolves to a valid URL
            valid_ids = [iid for iid in chunk_image_ids if images_by_id.get(iid, {}).get("url")]
            if not valid_ids:
                continue
            chunk_text = chunk.get("text", "")
            text_per_image = len(chunk_text) / (len(valid_ids) + 1)
            richness_factor = 1.0 + text_per_image / 500.0
            priority = chunk_score * richness_factor
            if priority > best_priority:
                best_priority = priority
                best_chunk = chunk

    if best_chunk is None:
        return []

    image_ids = best_chunk.get("metadata", {}).get("image_ids", [])
    result: List[Dict[str, Any]] = []
    for iid in image_ids:
        img = images_by_id.get(iid)
        if img and img.get("url"):
            result.append(img)
        if len(result) >= max_images:
            break
    return result


def replace_valid_image_placeholders(
    answer: str,
    images_by_id: Dict[str, Dict[str, Any]],
) -> Tuple[str, List[str]]:
    """
    Replace <image_id>...</image_id> placeholders in answer with proper markdown image syntax.
    Returns (updated_answer, list_of_replaced_image_ids).

    Rules:
    - Only replace image_ids that exist in images_by_id AND have a non-empty URL.
    - Remove placeholders for image_ids with no valid record (strip tag).
    - Never invent new images.
    """
    replaced_ids: List[str] = []

    def _caption_for(img: Dict[str, Any], preceding_text: str = "") -> str:
        """Derive a human-readable caption for an image.

        Priority:
        1. image_summary / caption from Mistral OCR (if not just a filename)
        2. Last non-empty line of preceding_text (document context before the image)
        3. "Figura - pagina N" fallback
        """
        raw = (img.get("image_summary") or img.get("caption") or "").strip()
        # Reject filenames (e.g. "img-21.jpeg") as useless captions
        if raw and not re.search(r"\.\w{2,5}$", raw):
            return raw
        if preceding_text:
            # Take the last non-empty line before the placeholder
            for line in reversed(preceding_text.splitlines()):
                line = line.strip().lstrip("#* ")
                if len(line) > 5:
                    return line[:100]
        page = img.get("page")
        return f"Figura - pagina {page}" if page else "Figura"

    def _replace(match: re.Match) -> str:
        iid = match.group(1)
        img = images_by_id.get(iid)
        if not img or not img.get("url"):
            return ""  # Strip unknown/invalid placeholder
        if iid in replaced_ids:
            return ""
        replaced_ids.append(iid)
        preceding = answer[: match.start()]
        caption = _caption_for(img, preceding)
        return f"\n\n![{caption}]({img['url']})\n"

    updated = _IMAGE_ID_RE.sub(_replace, answer)
    return updated, replaced_ids


# ---------------------------------------------------------------------------
# Context and response building
# ---------------------------------------------------------------------------

def build_doc_kb_context(context_groups: List[Dict[str, Any]]) -> str:
    """
    Concatenate chunk texts in document order to form the context block
    sent to the doc_kb agent.  Placeholder tags are preserved so the agent
    can reference them in its answer.
    """
    parts: List[str] = []
    chunk_summary: List[Dict[str, Any]] = []
    for group in context_groups:
        source_file = group.get("source_file", "?")
        for chunk in group.get("chunks", []):
            text = chunk.get("text", "")
            score = chunk.get("score", 0.0)
            meta = chunk.get("metadata", {})
            page = meta.get("page")
            image_ids = meta.get("image_ids", [])
            if text.strip():
                parts.append(text.strip())
                chunk_summary.append({
                    "file": source_file,
                    "page": page,
                    "score": round(score, 4),
                    "chars": len(text),
                    "images": len(image_ids),
                    "text_preview": text.strip()[:120].replace("\n", " "),
                })
    log.info(
        "build_doc_kb_context.chunks",
        total_chunks=len(chunk_summary),
        total_chars=sum(c["chars"] for c in chunk_summary),
    )
    for i, c in enumerate(chunk_summary):
        log.info(
            "chunk",
            idx=i + 1,
            file=c["file"],
            page=c["page"],
            score=c["score"],
            chars=c["chars"],
            images=c["images"],
            preview=c["text_preview"],
        )
    return "\n\n---\n\n".join(parts)


def build_sources(
    context_groups: List[Dict[str, Any]],
    max_sources: int = 3,
) -> List[Dict[str, Any]]:
    """
    Collect source references from directly-retrieved chunks (score > 0),
    ranked by score descending so only the most relevant pages are shown.
    Expanded context chunks (score == 0) are excluded.
    Returns at most `max_sources` deduplicated {source_file, page, source_url} dicts.
    """
    # Collect all scored entries first so we can sort before dedup-truncating.
    candidates: List[tuple] = []  # (score, source_file, page, source_url)
    for group in context_groups:
        for chunk in group.get("chunks", []):
            score = chunk.get("score", 0)
            if score <= 0:
                continue
            m = chunk.get("metadata", {})
            source_file = m.get("source_file", group.get("source_file", ""))
            page = m.get("page")
            source_url = m.get("source_url", "")
            candidates.append((score, source_file, page, source_url))

    # Sort highest score first, then deduplicate by (file, page).
    candidates.sort(key=lambda x: x[0], reverse=True)
    log.info(
        "build_sources.candidates",
        total=len(candidates),
        top=[
            {"file": sf, "page": pg, "score": round(sc, 4)}
            for sc, sf, pg, _ in candidates[:6]
        ],
    )
    seen: set = set()
    sources: List[Dict[str, Any]] = []
    for score, source_file, page, source_url in candidates:
        key = f"{source_file}:{page}"
        if key in seen:
            continue
        seen.add(key)
        sources.append({"source_file": source_file, "page": page, "source_url": source_url})
        if len(sources) >= max_sources:
            break
    log.info("build_sources.selected", sources=[{"file": s["source_file"], "page": s["page"]} for s in sources])
    return sources


def build_markdown_response(
    answer: str,
    appended_images: List[Dict[str, Any]],
    replaced_image_ids: List[str],
    sources: List[Dict[str, Any]],
) -> str:
    """
    Compose the final markdown response:
    1. Cleaned answer text (placeholders already replaced or stripped)
    2. Up to max_images contextual images NOT already embedded inline
    3. Sources section
    """
    parts = [_dedupe_markdown_images(answer.strip())]

    for img in appended_images:
        iid = img.get("image_id", "")
        if iid in replaced_image_ids:
            continue  # Already inlined by placeholder replacement
        url = img.get("url", "")
        if not url:
            continue
        raw_caption = (img.get("image_summary") or img.get("caption") or "").strip()
        # Reject bare filenames as captions
        if raw_caption and re.search(r"\.\w{2,5}$", raw_caption):
            raw_caption = ""
        page = img.get("page")
        caption = raw_caption or (f"Figura - pagina {page}" if page else "Figura")
        parts.append(f"\n![{caption}]({url})")

    if sources:
        src_lines = ["", "## Sources"]
        for i, src in enumerate(sources, 1):
            sf = src.get("source_file", "")
            pg = src.get("page")
            url = src.get("source_url", "")
            label = f"{sf}" + (f", page {pg}" if pg else "")
            if url:
                src_lines.append(f"{i}. [{label}]({url})")
            else:
                src_lines.append(f"{i}. {label}")
        parts.append("\n".join(src_lines))

    return "\n\n".join(p for p in parts if p.strip())


def _dedupe_markdown_images(markdown: str) -> str:
    seen_urls: set = set()

    def _replace(match: re.Match) -> str:
        url = match.group(1)
        if url in seen_urls:
            return ""
        seen_urls.add(url)
        return match.group(0)

    return re.sub(r"!\[[^\]]*]\(([^)]+)\)", _replace, markdown)
