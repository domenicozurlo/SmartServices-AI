# app/utils/describer.py
"""
OpenAI vision-based image describer for multimodal document ingestion.

Adapted from azure-search-openai-demo MultimodalModelDescriber.
No Azure dependencies — uses openai.AsyncOpenAI with tenacity retry.

Usage:
    from app.utils.describer import describe_image
    summary = await describe_image(png_bytes)  # returns str | None
"""

import base64
import json
import os
import re
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI, BadRequestError, RateLimitError
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from app.config import logger

_client: Optional[AsyncOpenAI] = None
_last_domain_hints_status: Dict[str, Any] = {}


def get_domain_hints_status() -> Dict[str, Any]:
    return dict(_last_domain_hints_status)


def _set_domain_hints_status(**updates: Any) -> None:
    _last_domain_hints_status.clear()
    _last_domain_hints_status.update(updates)


def _get_client() -> Optional[AsyncOpenAI]:
    global _client
    if _client is not None:
        return _client
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("RAG_OPENAI_API_KEY")
    if not api_key:
        logger.warning("describer: OpenAI API key not set - description disabled")
        return None
    base_url = os.getenv("OPENAI_API_BASE") or os.getenv("RAG_OPENAI_BASEURL") or None
    _client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    return _client


async def _create_chat_completion(
    client: AsyncOpenAI,
    model: str,
    token_limit: int,
    messages: List[Dict[str, Any]],
) -> Any:
    try:
        return await client.chat.completions.create(
            model=model,
            max_completion_tokens=token_limit,
            messages=messages,
        )
    except BadRequestError as exc:
        if "max_completion_tokens" not in str(exc):
            raise
        return await client.chat.completions.create(
            model=model,
            max_tokens=token_limit,
            messages=messages,
        )


async def describe_image(image_bytes: bytes, model: Optional[str] = None) -> Optional[str]:
    """
    Generate a concise textual description of an image for vector indexing.

    Uses OpenAI vision API with exponential-backoff retry on rate limits.
    Returns None when the client is not configured or on unrecoverable errors.
    """
    client = _get_client()
    if client is None:
        return None

    vision_model = model or os.getenv("IMAGE_DESCRIPTION_MODEL", "gpt-5-mini")
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    image_datauri = f"data:image/png;base64,{image_b64}"

    try:
        async for attempt in AsyncRetrying(
            retry=retry_if_exception_type(RateLimitError),
            wait=wait_random_exponential(min=10, max=60),
            stop=stop_after_attempt(5),
            reraise=True,
        ):
            with attempt:
                response = await _create_chat_completion(
                    client,
                    vision_model,
                    300,
                    [
                        {
                            "role": "system",
                            "content": (
                                "You are a helpful assistant that describes images extracted from "
                                "organizational or technical documents. Be concise and precise. "
                                "If the image is a chart, diagram, or table, describe its key data. "
                                "Answer in the same language as any text visible in the image, "
                                "defaulting to Italian if no language is detectable."
                            ),
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": image_datauri}},
                                {
                                    "type": "text",
                                    "text": "Describe this image concisely for a document search index.",
                                },
                            ],
                        },
                    ],
                )
                return response.choices[0].message.content or None
    except Exception as exc:
        logger.warning("describer: failed to describe image: %s", exc)
        return None


async def describe_domain_hints(
    source_file: str,
    chunk_texts: List[str],
    model: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Generate optional document-level retrieval hints for ingest.

    The returned JSON is stored as metadata and used only as a generic reranking
    signal. The core retrieval logic must still work when this returns None.
    """
    if os.getenv("DOMAIN_HINTS_ENABLED", "true").lower() in ("false", "0", "no"):
        _set_domain_hints_status(enabled=False, generated=False, reason="disabled")
        return None

    hints_model = model or os.getenv("DOMAIN_HINTS_MODEL", "gpt-5-mini")
    has_api_key = bool(os.getenv("OPENAI_API_KEY") or os.getenv("RAG_OPENAI_API_KEY"))
    _set_domain_hints_status(
        enabled=True,
        generated=False,
        model=hints_model,
        has_api_key=has_api_key,
        base_url_configured=bool(os.getenv("OPENAI_API_BASE") or os.getenv("RAG_OPENAI_BASEURL")),
    )

    client = _get_client()
    if client is None:
        _set_domain_hints_status(
            enabled=True,
            generated=False,
            model=hints_model,
            has_api_key=False,
            reason="missing_api_key",
        )
        return None

    sample = "\n\n---\n\n".join(text[:1200] for text in chunk_texts[:18] if text.strip())
    if not sample:
        _set_domain_hints_status(
            enabled=True,
            generated=False,
            model=hints_model,
            has_api_key=has_api_key,
            reason="empty_sample",
        )
        return None

    try:
        async for attempt in AsyncRetrying(
            retry=retry_if_exception_type(RateLimitError),
            wait=wait_random_exponential(min=5, max=30),
            stop=stop_after_attempt(3),
            reraise=True,
        ):
            with attempt:
                response = await _create_chat_completion(
                    client,
                    hints_model,
                    700,
                    [
                        {
                            "role": "system",
                            "content": (
                                "You extract compact retrieval metadata from documents. "
                                "Return only valid JSON. Do not include explanations."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                "Analyze this document sample and return JSON with this schema:\n"
                                "{\n"
                                '  "language": "short language code",\n'
                                '  "domain": "short domain label",\n'
                                '  "key_terms": ["important exact terms"],\n'
                                '  "aliases": {"query term or typo": ["canonical or related terms"]},\n'
                                '  "intents": {"how_to": ["terms"], "status": ["terms"], "troubleshooting": ["terms"]}\n'
                                "}\n"
                                "Keep lists short. Include only terms present or strongly implied by the document. "
                                "Do not add brand-specific rules unless they appear in the document.\n\n"
                                f"Source file: {source_file}\n\nDocument sample:\n{sample}"
                            ),
                        },
                    ],
                )

        content = response.choices[0].message.content or ""
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not match:
            _set_domain_hints_status(
                enabled=True,
                generated=False,
                model=hints_model,
                has_api_key=has_api_key,
                reason="no_json_in_response",
            )
            return None
        data = json.loads(match.group(0))
        if not isinstance(data, dict):
            _set_domain_hints_status(
                enabled=True,
                generated=False,
                model=hints_model,
                has_api_key=has_api_key,
                reason="json_not_object",
            )
            return None
        hints = {
            "language": str(data.get("language", ""))[:16],
            "domain": str(data.get("domain", ""))[:80],
            "key_terms": _string_list(data.get("key_terms"), 40),
            "aliases": _string_map(data.get("aliases"), 40, 8),
            "intents": _string_map(data.get("intents"), 12, 20),
        }
        _set_domain_hints_status(
            enabled=True,
            generated=bool(hints["domain"] or hints["key_terms"] or hints["aliases"] or hints["intents"]),
            model=hints_model,
            has_api_key=has_api_key,
            reason="ok",
            domain=hints["domain"],
            key_terms=len(hints["key_terms"]),
        )
        return hints
    except Exception as exc:
        logger.warning("describer: failed to describe domain hints: %s", exc)
        _set_domain_hints_status(
            enabled=True,
            generated=False,
            model=hints_model,
            has_api_key=has_api_key,
            reason="exception",
            error=str(exc)[:300],
        )
        return None


def _string_list(value: Any, limit: int) -> List[str]:
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        if isinstance(item, str) and item.strip():
            result.append(item.strip()[:80])
        if len(result) >= limit:
            break
    return result


def _string_map(value: Any, key_limit: int, value_limit: int) -> Dict[str, List[str]]:
    if not isinstance(value, dict):
        return {}
    result: Dict[str, List[str]] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key.strip():
            continue
        values = _string_list(item, value_limit)
        if values:
            result[key.strip()[:80]] = values
        if len(result) >= key_limit:
            break
    return result
