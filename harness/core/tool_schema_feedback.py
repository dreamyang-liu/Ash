"""Tool-role feedback for an atomically rejected, unexecuted response."""
import json


def tool_schema_feedback(message: dict, errors: dict[str, str], tools: list[dict]) -> list[dict]:
    observations = []
    for call in message["tool_calls"]:
        identifier = call["id"]
        error = {
            "type": "tool_schema_error" if identifier in errors else "tool_batch_rejected",
            "message": errors.get(identifier, "Another call in this response failed schema validation."),
        }
        content = json.dumps({
            "error": error, "executed": False,
            "instruction": "No calls in this response were executed. Correct and resend the tool calls.",
            "available_tools": tools,
        }, ensure_ascii=False)
        observations.append({"role": "tool", "tool_call_id": identifier, "content": content})
    return observations
