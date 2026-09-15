"""Preserve Miles/SGLang controls through native API requests.

The message-rollout gateway speaks SGLang's native Responses/Messages
extensions. Unknown controls are rejected; values are never silently dropped.
"""

from copy import deepcopy
import math


def validate(value):
    allowed = {
        "temperature", "top_p", "top_k", "seed", "max_tokens",
        "max_new_tokens", "max_output_tokens", "stop",
    }
    if not isinstance(value, dict) or set(value) - allowed:
        raise ValueError("Unknown message sampling controls")
    for key in ("temperature", "top_p"):
        if key in value:
            number = value[key]
            if (type(number) not in (int, float) or not math.isfinite(number)
                    or number < 0 or (key == "top_p" and not 0 < number <= 1)):
                raise ValueError(f"Invalid {key}")
    if "top_k" in value and (type(value["top_k"]) is not int or value["top_k"] < -1):
        raise ValueError("top_k must be -1 or a nonnegative integer")
    if "seed" in value and type(value["seed"]) is not int:
        raise ValueError("seed must be an integer")
    limits = [value[k] for k in ("max_tokens", "max_new_tokens", "max_output_tokens") if k in value]
    if limits and any(type(n) is not int or n <= 0 or n != limits[0] for n in limits):
        raise ValueError("Output token limits must be positive and agree")
    stop = value.get("stop")
    if stop is not None and not isinstance(stop, str):
        if not isinstance(stop, list) or any(not isinstance(s, str) for s in stop):
            raise ValueError("stop must be text, a text list, or null")


def parameters(value, shape):
    validate(value)
    if shape not in {"responses", "messages"}:
        raise ValueError("Unsupported native API shape")
    result = deepcopy(value)
    limits = [result.pop(k) for k in ("max_tokens", "max_new_tokens", "max_output_tokens") if k in result]
    if limits:
        result["max_output_tokens" if shape == "responses" else "max_tokens"] = limits[0]
    if shape == "messages" and "stop" in result:
        stop = result.pop("stop")
        result["stop_sequences"] = [stop] if isinstance(stop, str) else stop
    return result
