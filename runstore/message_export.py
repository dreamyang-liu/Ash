"""Export native histories for hint-free training, retaining tool causality."""

from copy import deepcopy
import json
import re

HINT_START = "<ash_training_hint>"
HINT_END = "</ash_training_hint>"
HINT_PATTERN = re.compile(re.escape(HINT_START) + r".*?" + re.escape(HINT_END), re.DOTALL)


def mark_hint(text):
    if HINT_START in text or HINT_END in text:
        raise ValueError("Branch hint contains reserved delimiters")
    return HINT_START + text + HINT_END


def clean_messages(messages):
    cleaned = []
    for original in messages:
        message = deepcopy(original)
        role = message.get("role")
        if role == "developer":
            message["role"] = role = "system"
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"Unsupported message role {role!r}")
        content = message.get("content", "")
        if not isinstance(content, str):
            raise ValueError("Training export requires text content")
        if role in {"system", "user"}:
            content = HINT_PATTERN.sub("", content)
            if HINT_START in content or HINT_END in content:
                raise ValueError("Incomplete hint boundary")
            if content != message.get("content", "") and not content.strip():
                continue
            message["content"] = content
        cleaned.append(message)
    pending, seen = set(), set()
    for message in cleaned:
        for call in message.get("tool_calls") or []:
            if call["id"] in seen:
                raise ValueError("Duplicate tool call")
            seen.add(call["id"])
            pending.add(call["id"])
        if message["role"] == "tool":
            if message.get("tool_call_id") not in pending:
                raise ValueError("Unmatched tool result")
            pending.remove(message["tool_call_id"])
    if pending:
        raise ValueError("Native history has incomplete tool calls")
    if not any(m["role"] == "assistant" for m in cleaned):
        raise ValueError("Native history has no assistant output")
    return cleaned


def text_content(content):
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise ValueError("Unsupported native content")
    texts = []
    for block in content:
        if not isinstance(block, dict) or block.get("type") not in {"text", "input_text", "output_text", "reasoning_text"}:
            raise ValueError("Unsupported native content block")
        texts.append(block["text"])
    return "".join(texts)


def codex_messages(entries):
    messages = []
    merge_assistant = False

    def append(message):
        nonlocal merge_assistant
        if message["role"] == "assistant" and merge_assistant:
            previous = messages[-1]
            previous["content"] = previous.get("content", "") + message.get("content", "")
            if message.get("reasoning_content"):
                previous["reasoning_content"] = previous.get("reasoning_content", "") + message["reasoning_content"]
            if message.get("tool_calls"):
                previous.setdefault("tool_calls", []).extend(message["tool_calls"])
        else:
            messages.append(message)
        merge_assistant = message["role"] == "assistant"

    for entry in entries:
        if (entry.get("type") == "compacted"
                or (entry.get("type") == "event_msg" and entry.get("payload", {}).get("type") == "thread_rolled_back")):
            raise ValueError("Compacted history is not a complete training trajectory")
        if entry.get("type") == "event_msg" and entry.get("payload", {}).get("type") == "token_count":
            merge_assistant = False
        if entry.get("type") != "response_item":
            continue
        item = entry["payload"]
        kind = item["type"]
        if kind == "message":
            content = text_content(item["content"])
            if item["role"] == "assistant" and item.get("channel") == "analysis":
                append({"role": "assistant", "content": "", "reasoning_content": content})
            else:
                append({"role": item["role"], "content": content})
        elif kind in {"function_call", "custom_tool_call"}:
            if kind == "custom_tool_call":
                raise ValueError("Custom tool grammar needs an explicit training template adapter")
            name = "__".join(filter(None, (item.get("namespace"), item["name"])))
            append({"role": "assistant", "content": "", "tool_calls": [{
                "id": item["call_id"], "type": "function",
                "function": {"name": name, "arguments": item["arguments"]},
            }]})
        elif kind in {"function_call_output", "custom_tool_call_output"}:
            append({"role": "tool", "tool_call_id": item["call_id"],
                             "content": text_content(item["output"])})
        elif kind == "reasoning":
            # Native encrypted reasoning is not recoverable text; do not invent it.
            content = item.get("content")
            if content:
                append({"role": "assistant", "content": "",
                                 "reasoning_content": text_content(content)})
        else:
            raise ValueError(f"Unsupported Codex response item {kind!r}")
    return messages


def claude_messages(entries):
    messages = []
    for entry in entries:
        if entry.get("subtype") == "compact_boundary":
            raise ValueError("Compacted history is not a complete training trajectory")
        if entry.get("type") not in {"user", "assistant"}:
            continue
        message = entry["message"]
        role = message["role"]
        content = message["content"]
        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue
        current = {"role": role, "content": ""}
        for block in content:
            kind = block["type"]
            if kind == "text":
                current["content"] += block["text"]
            elif kind == "thinking":
                current["reasoning_content"] = current.get("reasoning_content", "") + block["thinking"]
            elif kind == "tool_use":
                current.setdefault("tool_calls", []).append({
                    "id": block["id"], "type": "function",
                    "function": {"name": block["name"], "arguments": json.dumps(block["input"])},
                })
            elif kind == "tool_result":
                if current["content"]:
                    messages.append(current)
                    current = {"role": role, "content": ""}
                messages.append({"role": "tool", "tool_call_id": block["tool_use_id"],
                                 "content": text_content(block.get("content", ""))})
            else:
                raise ValueError(f"Unsupported Claude content {kind!r}")
        if current["content"] or current.get("tool_calls") or current.get("reasoning_content"):
            messages.append(current)
    return messages


def export_messages(directory, session_id, slot, events, *, allow_incomplete_tail=False):
    if not isinstance(session_id, str) or not session_id or slot not in {"codex", "claude-code", "mini-swe-agent"}:
        raise ValueError("Missing or unsupported native session identity")
    files = list((directory / "native-home").glob(f"**/*{session_id}.jsonl"))
    if len(files) != 1:
        raise ValueError("Expected one native history for the attempt")
    data = files[0].read_bytes()
    if not data.endswith(b"\n"):
        if not allow_incomplete_tail:
            raise ValueError("Native history has an incomplete final record")
        data = data[:data.rfind(b"\n") + 1]
    entries = [json.loads(line) for line in data.splitlines() if line.strip()]
    if slot == "mini-swe-agent":
        from harness.slots.mini_history import training_messages

        # mini writes its actual formatted observation before closing the turn.
        # Do not substitute the MCP's different presentation from tool.finished.
        return clean_messages(training_messages(entries))
    messages = (codex_messages if slot == "codex" else claude_messages)(entries)
    # Some SDK histories stop after the last tool call. Use only actual
    # observed results with the same call_id; never synthesize a tool response.
    returned = {m["tool_call_id"] for m in messages if m["role"] == "tool"}
    results = {e["call_id"]: e for e in events if e.get("type") == "tool.finished"}
    for call in messages[-1].get("tool_calls", []) if messages else []:
        if call["id"] not in returned and call["id"] in results:
            messages.append({"role": "tool", "tool_call_id": call["id"],
                             "content": text_content(results[call["id"]].get("output", ""))})
    return clean_messages(messages)


def export_tools(events):
    schemas = {}
    for event in events:
        if event.get("type") != "rollout.model_tools":
            continue
        tools = []
        for tool in event.get("tools", []):
            if tool.get("type") == "namespace":
                tools.extend({**nested, "name": f"{tool['name']}__{nested['name']}"}
                             for nested in tool.get("tools", []))
            else:
                tools.append(tool)
        for tool in tools:
            if event["shape"] == "messages":
                item = {"name": tool["name"], "description": tool.get("description", ""),
                        "parameters": tool.get("input_schema", {})}
            elif event["shape"] == "chat/completions" and tool.get("type") == "function":
                item = deepcopy(tool["function"])
            elif tool.get("type") == "function":
                item = {key: tool[key] for key in ("name", "description", "parameters", "strict") if key in tool}
            else:
                raise ValueError("Unsupported native tool schema for training")
            name = item["name"]
            if name in schemas and schemas[name] != item:
                raise ValueError("Tool schema changed during the trajectory")
            schemas[name] = item
    return [{"type": "function", "function": schema} for schema in schemas.values()]
