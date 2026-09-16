"""Bound cleaned training histories and grade the exact retained snapshot."""

from copy import deepcopy
from functools import lru_cache
import json
from pathlib import Path

from runstore.files import journal_events
from runstore.message_export import (
    claude_messages, clean_messages, codex_messages, export_messages, export_tools, text_content,
)
from runstore.native import index_native, read_prefix


SEQUENCE_STOP_REASON = "max_sequence_tokens"
FRAMING_RESERVE = 1024


@lru_cache(maxsize=4)
def _tokenizer(path):
    # Optional: ordinary v3 deployments do not need a local tokenizer.
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)


@lru_cache(maxsize=4)
def _training_format(path):
    """Deployment-side pairing with the learner's loss-mask serialization."""
    config = Path(path) / "ash_sequence_counter.json"
    if not config.exists():
        return "full"
    body = json.loads(config.read_text())
    if set(body) != {"training_format"} or body["training_format"] not in {"full", "qwen3"}:
        raise ValueError("Invalid ash_sequence_counter.json")
    return body["training_format"]


_MASK_PREFIX = {"role": "user", "content": "FOR CALCULATING LOSS MASK ONLY"}


@lru_cache(maxsize=4)
def _qwen3_framing(path):
    tokenizer = _tokenizer(path)
    prefix = tokenizer.apply_chat_template([_MASK_PREFIX], tokenize=True, return_dict=False)
    twice = tokenizer.apply_chat_template([_MASK_PREFIX, _MASK_PREFIX], tokenize=True, return_dict=False)
    body = twice[len(prefix):]
    if not body or twice[:len(prefix)] != prefix or prefix[-len(body):] != body:
        raise ValueError("Tokenizer does not preserve qwen3 mask framing")
    return prefix, body


@lru_cache(maxsize=4096)
def _qwen3_message_length(path, serialized_message, serialized_tools, first):
    # Cache message lengths: adjacent recovery points share almost all messages.
    tokenizer = _tokenizer(path)
    prefix, body = _qwen3_framing(path)
    message = json.loads(serialized_message)
    if first:
        tokens = tokenizer.apply_chat_template(
            [message, _MASK_PREFIX], tools=json.loads(serialized_tools),
            tokenize=True, return_dict=False,
        )
        if tokens[-len(body):] != body:
            raise ValueError("Tokenizer does not preserve trailing qwen3 mask prefix")
        return len(tokens) - len(body)
    tokens = tokenizer.apply_chat_template([_MASK_PREFIX, message], tokenize=True, return_dict=False)
    if tokens[:len(prefix)] != prefix:
        raise ValueError("Tokenizer does not preserve leading qwen3 mask prefix")
    return len(tokens) - len(prefix)


def token_count(messages, tools, tokenizer_path):
    normalized = deepcopy(messages)
    for message in normalized:
        for call in message.get("tool_calls") or []:
            arguments = call["function"].get("arguments", {})
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            if not isinstance(arguments, dict):
                raise ValueError("Tool arguments must be a JSON object for sequence counting")
            call["function"]["arguments"] = arguments
    # Durable result/ledger serialization sorts schema keys. HF templates may
    # preserve that order in tool JSON, changing BPE boundaries even though the
    # dictionaries are equal. Count the representation Miles will receive.
    durable_tools = json.loads(json.dumps(tools, sort_keys=True)) if tools else None
    if _training_format(tokenizer_path) == "qwen3":
        # Miles serializes each message independently to preserve assistant
        # reasoning and build masks. A full render merges adjacent tool results
        # and therefore undercounts histories containing parallel tool calls.
        serialized_tools = json.dumps(durable_tools, ensure_ascii=False)
        return sum(
            _qwen3_message_length(
                tokenizer_path, json.dumps(message, ensure_ascii=False),
                serialized_tools if index == 0 else "null", index == 0,
            )
            for index, message in enumerate(normalized)
        )
    tokens = _tokenizer(tokenizer_path).apply_chat_template(
        normalized, tools=durable_tools, tokenize=True, add_generation_prompt=False, return_dict=False,
    )
    return len(tokens)


def native_request_tokens(payload, shape, tokenizer_path):
    if shape == "messages":
        entries = [{"type": message["role"], "message": message} for message in payload["messages"]]
        messages = claude_messages(entries)
        if payload.get("system"):
            messages.insert(0, {"role": "system", "content": text_content(payload["system"])})
    elif shape == "responses":
        content = payload["input"]
        if isinstance(content, str):
            messages = [{"role": "user", "content": content}]
        else:
            entries = [
                {"type": "response_item", "payload": {"type": item.get("type", "message"), **item}}
                for item in content
            ]
            messages = codex_messages(entries)
        if payload.get("instructions"):
            messages.insert(0, {"role": "system", "content": payload["instructions"]})
    else:
        raise ValueError("Unsupported native API for sequence limit")
    tools = export_tools([{"type": "rollout.model_tools", "shape": shape, "tools": payload.get("tools", [])}])
    return token_count(messages, tools, tokenizer_path)


def remaining_output_budget(payload, shape, contract, journal):
    path = contract["sequence_tokenizer_path"]
    used = native_request_tokens(payload, shape, path)
    events = journal_events(journal.path)
    references = [e["native_session_id"] for e in events
                  if e.get("type") == "session.ref" and e.get("native_session_id")]
    if references:
        try:
            history = export_messages(
                journal.path.parent, references[-1], contract["native_slot"], events,
                allow_incomplete_tail=True,
            )
        except (ValueError, KeyError, TypeError, OSError):
            # The live writer may not yet have closed a response group. The
            # final export below always performs an exact check and safe cut.
            history = None
        if history:
            used = max(used, token_count(history, export_tools(events), path))
    remaining = contract["max_sequence_tokens"] - used - FRAMING_RESERVE
    journal.emit("rollout.sequence_budget", used_tokens=used,
                 max_sequence_tokens=contract["max_sequence_tokens"], remaining_output_tokens=max(0, remaining))
    return remaining


def _prefixes(directory, session_id, slot, events, recovery):
    transcripts = list((directory / "native-home").glob(f"**/*{session_id}.jsonl"))
    if len(transcripts) != 1:
        raise ValueError("Expected one owned transcript for sequence truncation")
    inherited_native = (recovery or {}).get("native")
    outputs = (inherited_native or {}).get("referenced_outputs", [])
    return index_native(
        directory / "trajectory.jsonl", transcripts[0], slot, session_id,
        tuple((item["path"], item["sha256"]) for item in outputs),
        events=events, inherited_native=inherited_native,
    )


def apply_sequence_limit(result, directory, slot, events, recovery, contract):
    limit = contract.get("max_sequence_tokens") if contract else None
    if limit is None:
        return result
    path = contract["sequence_tokenizer_path"]
    tools = result["training_tools"]
    original_count = token_count(result["training_messages"], tools, path)
    result["training_token_count"] = original_count
    result["max_sequence_tokens"] = limit
    points = _prefixes(directory, result["native_session_id"], slot, events, recovery)
    candidates, point_tokens = [], {}
    parser = claude_messages if slot == "claude-code" else codex_messages
    for point in points:
        entries = [json.loads(line) for line in read_prefix(point.native).splitlines()]
        messages = clean_messages(parser(entries))
        length = token_count(messages, tools, path)
        point_tokens[str(point.tool_depth)] = length
        if length <= limit:
            candidates.append((point, messages, length))
    result["training_point_tokens"] = point_tokens
    if original_count <= limit:
        return result
    if not candidates:
        raise ValueError("No complete native checkpoint prefix fits max_sequence_tokens")
    point, messages, length = max(candidates, key=lambda item: item[0].tool_depth)
    result.update(
        agent_status=result.get("agent_status", result.get("status")),
        status="truncated", stop_reason=SEQUENCE_STOP_REASON, failure_kind=None,
        raw_final_snapshot_id=result["final_snapshot_id"],
        final_snapshot_id=point.snapshot_id, training_messages=messages, training_token_count=length,
        sequence_truncation={
            "original_tokens": original_count, "retained_tokens": length,
            "tool_depth": point.tool_depth, "message_step": point.message_step,
            "snapshot_id": point.snapshot_id, "native_prefix_sha256": point.native["sha256"],
        },
    )
    return result
