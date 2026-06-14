"""
Document Knowledge Base Agent.

Synthesises answers from document context already injected into the conversation
by LibreChat's native file_search / rag_api pipeline.

This agent does NOT query rag_api directly — context enrichment and citation
tracking are handled upstream by LibreChat before the request reaches this gateway.

When the tool message carries multimodal_file_search_results (JSON), the handler
extracts structured context groups, selects relevant images, runs the agent over
the enriched context, then post-processes the answer to replace image placeholders
with real markdown image syntax and appends a sources section.
"""

import os
from typing import AsyncGenerator, List, Dict, Any, Optional

from agents import Agent, ModelSettings, Runner
from agents.stream_events import RawResponsesStreamEvent
from openai.types.shared.reasoning import Reasoning
from logger import get_logger
from handlers.multimodal import (
    parse_tool_results,
    build_images_by_id,
    select_contextual_images,
    replace_valid_image_placeholders,
    build_doc_kb_context,
    build_sources,
    build_markdown_response,
)

log = get_logger("doc_kb")


_INSTRUCTIONS = """
You are a Document Knowledge Base Agent.

LibreChat has already retrieved relevant document chunks from the knowledge base
and injected them into the conversation context before this request reached you.
Your job is to give a clear, complete answer based solely on that context.

Rules:
- Answer in the same language the user used.
- Be as detailed as the question requires: use bullet points or numbered steps for
  procedures, short prose for simple factual questions. Do not pad or repeat.
- For procedural answers, preserve the retrieved step labels/headings when they
  are present, especially numbered labels such as "1.", "2.", "3.". If multiple
  consecutive relevant steps are in context, include them in document order.
- If the relevant context lists multiple causes, options, warnings, or conditions
  joined by words like "oppure", "o", "or", or similar separators, include every
  listed item unless the user's question explicitly narrows the answer.
- Ground every claim in the provided context; do not invent information.
- If the context contains ANY information related to the question — even partial, even
  a single relevant sentence — answer with what is there. Never say "not found" when
  there is something relevant in the context, even if it is incomplete.
- Use the exact phrase "Le informazioni richieste non sono presenti nei documenti forniti."
  ONLY when the context contains absolutely nothing relevant to the question.
- Do NOT ask follow-up questions.
- Do NOT suggest uploading more documents or taking further actions.
- Do NOT offer alternatives or additional options.
- No markdown headings.

Images — CRITICAL:
- The context chunks contain <image_id>SOME_ID</image_id> placeholder tags embedded in the text.
- You MUST copy those placeholder tags VERBATIM into your answer at the point where the image is relevant.
- Do NOT omit or rewrite them. Do NOT use markdown image syntax. Copy the exact tag including the angle brackets.
- Example output: "Inserire l'Accessorio per la Pulizia IQOS e ruotare. <image_id>abc123_p7_imgimg-21.jpeg</image_id> Poi usare il Bastoncino. <image_id>abc123_p7_imgimg-22.jpeg</image_id>"
- If no image tag is present in the relevant passage, do not invent one.
"""

_doc_agent = Agent(
    name="Document KB Agent",
    model=os.getenv("DOC_KB_MODEL", "gpt-5-mini"),
    instructions=_INSTRUCTIONS,
    model_settings=ModelSettings(
        store=True,
        reasoning=Reasoning(effort="low", summary="detailed"),
    ),
)


async def run_doc_kb(
    conversation: list,
    rewritten_query: str,
    context_groups: Optional[List[Dict[str, Any]]] = None,
    images_by_id: Optional[Dict[str, Any]] = None,
    sources: Optional[List[Dict[str, Any]]] = None,
) -> AsyncGenerator[str, None]:
    """
    Run the doc_kb agent.

    When context_groups are provided (multimodal path), context is built from
    the structured groups and the answer is post-processed to replace image
    placeholders and append sources.

    When context_groups is None (legacy path), the caller has already injected
    the context into `conversation` as a user message.
    """
    log.info(
        "doc_kb.start",
        rewritten_query=rewritten_query,
        multimodal=context_groups is not None,
        context_groups=len(context_groups) if context_groups is not None else None,
        images_available=len(images_by_id) if images_by_id else 0,
        sources_available=len(sources) if sources else 0,
    )

    if context_groups is not None:
        # Multimodal path: build context from structured groups
        doc_context = build_doc_kb_context(context_groups)

        prompt_text = (
            f"[Retrieved document context]\n{doc_context}\n\n"
            f"Question: {rewritten_query}"
        )
        log.info(
            "doc_kb.prompt",
            prompt_chars=len(prompt_text),
            context_chars=len(doc_context),
            question=rewritten_query,
        )
        log.debug("doc_kb.prompt_full", prompt=prompt_text)

        input_messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": prompt_text,
                    }
                ],
            }
        ]

        # Collect the full answer + reasoning summary, then post-process (non-streaming internally)
        full_answer_chunks: List[str] = []
        reasoning_chunks: List[str] = []
        streamed = Runner.run_streamed(_doc_agent, input_messages)
        async for event in streamed.stream_events():
            if not isinstance(event, RawResponsesStreamEvent):
                continue
            raw = event.data
            event_type = getattr(raw, "type", None)
            if event_type == "response.reasoning_summary_text.delta":
                delta = getattr(raw, "delta", "")
                if delta:
                    reasoning_chunks.append(delta)
            elif event_type == "response.output_text.delta":
                delta = getattr(raw, "delta", "")
                if delta:
                    full_answer_chunks.append(delta)

        raw_answer = "".join(full_answer_chunks)
        reasoning_text = "".join(reasoning_chunks).strip()

        log.info(
            "doc_kb.agent_response",
            raw_answer_chars=len(raw_answer),
            reasoning_chars=len(reasoning_text),
            raw_answer_full=raw_answer,
        )
        log.info(
            "doc_kb.multimodal_post_process",
            images_by_id_count=len(images_by_id or {}),
            images_with_url=sum(1 for img in (images_by_id or {}).values() if img.get("url")),
            raw_answer_preview=raw_answer[:300],
            has_reasoning=bool(reasoning_text),
            reasoning_preview=reasoning_text[:300] if reasoning_text else None,
        )

        # When the agent signals that no relevant content was found, suppress
        # images and sources — skip expensive post-processing entirely.
        _NOT_FOUND_SENTINEL = "Le informazioni richieste non sono presenti nei documenti forniti"
        if _NOT_FOUND_SENTINEL in raw_answer:
            log.info("doc_kb.not_found", suppressing_images=True)
            yield _NOT_FOUND_SENTINEL
            return

        # Replace valid placeholders with markdown images; strip invalid ones
        answer_with_images, replaced_ids = replace_valid_image_placeholders(
            raw_answer, images_by_id or {}
        )

        # Select up to 2 contextual images not already inlined
        selected_images = select_contextual_images(
            context_groups, images_by_id or {}, max_images=2
        )

        final = build_markdown_response(
            answer=answer_with_images,
            appended_images=selected_images,
            replaced_image_ids=replaced_ids,
            sources=sources or [],
        )

        # Keep final document answers concise by default. Reasoning can still be
        # exposed for diagnostics when explicitly enabled in the gateway env.
        if reasoning_text and os.getenv("DOC_KB_INCLUDE_REASONING", "false").lower() in ("true", "1", "yes"):
            final = f":::thinking\n{reasoning_text}\n:::\n\n{final}"

        log.info(
            "doc_kb.multimodal_final",
            replaced_images=len(replaced_ids),
            appended_images=len(selected_images),
            sources=len(sources or []),
            final_len=len(final),
            has_img_markdown="![](" in final,
            final_preview=final[:400],
        )
        yield final
        return

    # Legacy path: conversation already contains injected context
    input_messages = [
        *conversation,
        {
            "role": "user",
            "content": [{"type": "input_text", "text": rewritten_query}],
        },
    ]
    streamed = Runner.run_streamed(_doc_agent, input_messages)
    async for event in streamed.stream_events():
        if not isinstance(event, RawResponsesStreamEvent):
            continue
        raw = event.data
        if getattr(raw, "type", None) == "response.output_text.delta":
            delta = getattr(raw, "delta", "")
            if delta:
                yield delta


