"""
Normalize OpenAI Chat Completions content blocks to OpenAI Responses API format,
which is what the Agents SDK (Runner) expects internally.

LibreChat sends (Chat Completions format):
  {"type": "text",      "text": "..."}
  {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
  {"type": "file",      "file": {"file_id": "...", "filename": "...", "file_data": "base64..."}}

Responses API format expected by the Agents SDK:
  {"type": "input_text",  "text": "..."}
  {"type": "input_image", "image_url": "..."}
  {"type": "input_file",  "file_id": "..."} | {"type": "input_file", "filename": "...", "file_data": "..."}
"""

_PASSTHROUGH_TYPES = {
    "input_text", "input_image", "input_file",
    "output_text", "refusal", "computer_screenshot", "summary_text",
}


def _normalize_block(block: dict) -> dict:
    t = block.get("type")

    if t in _PASSTHROUGH_TYPES:
        return block

    if t == "text":
        return {"type": "input_text", "text": block["text"]}

    if t == "image_url":
        return {"type": "input_image", "image_url": block["image_url"]["url"]}

    if t == "file":
        file_obj = block.get("file", {})
        if "file_id" in file_obj:
            return {"type": "input_file", "file_id": file_obj["file_id"]}
        result: dict = {"type": "input_file"}
        if "filename" in file_obj:
            result["filename"] = file_obj["filename"]
        if "file_data" in file_obj:
            result["file_data"] = file_obj["file_data"]
        return result

    # unknown block — pass through as-is and let the API surface the error
    return block


def normalize_content(content: str | list) -> str | list:
    if isinstance(content, str):
        return content
    return [_normalize_block(b) for b in content]


def normalize_messages(messages: list[dict]) -> list[dict]:
    result = []
    for m in messages:
        role = m.get("role")
        content = m.get("content")
        # Strip Chat Completions-only message types not supported by the Responses API:
        #   - role "tool"      → file_search / function results
        #   - role "assistant" with content=None → tool_call request messages
        if role == "tool":
            continue
        if role == "assistant" and content is None:
            continue
        result.append({"role": role, "content": normalize_content(content)})
    return result
