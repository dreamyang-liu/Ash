"""Validate branch-specific guidance without changing inherited execution settings."""
from copy import deepcopy

from harness.core.assistant_turn import validate_assistant_turn
from harness.core.branch_guidance import BRANCH_GUIDANCE_MODES, resolve_guidance, validate_guidance
from harness.core.mini_tools import mini_tool_schema
from harness.slots.mini_history import load_prefix

GUIDANCE_KEYS = frozenset({"branch_guidance", "assistant_turn", "resume_without_hint"})
BRANCH_OVERRIDES = frozenset({"prompt", "model", "timeout_s", "budget_usd",
                              "branch_guidance", "assistant_turn"})


def recorded_guidance(extra):
    """Unmarked persisted jobs keep their historical user-hint interpretation."""
    if extra.get("branch_guidance") is not None:
        return extra["branch_guidance"]
    if extra.get("assistant_turn") is not None:
        return "assistant-turn"
    return "none" if extra.get("resume_without_hint") else "user-hint"


def validate_overrides(overrides):
    """Driver-facing contract validation; no snapshot reads or execution."""
    if not isinstance(overrides, dict) or set(overrides) - BRANCH_OVERRIDES:
        raise ValueError("Branches can change prompt, model, budgets and validated guidance only")
    mode = overrides.get("branch_guidance")
    if mode is not None and mode not in BRANCH_GUIDANCE_MODES:
        raise ValueError("Unknown branch_guidance")
    if "assistant_turn" in overrides:
        validate_assistant_turn(overrides["assistant_turn"])
        if mode not in (None, "assistant-turn") or "prompt" in overrides:
            raise ValueError("assistant_turn cannot be combined with user-hint or point-only guidance")
    elif mode == "assistant-turn":
        raise ValueError("assistant-turn requires assistant_turn")
    if mode == "none" and "prompt" in overrides:
        raise ValueError("Point-only guidance cannot add a prompt")


def without_guidance(spec):
    """Only these guidance keys may differ inside the inherited extra object."""
    spec = deepcopy(spec)
    extra = {k: v for k, v in spec.get("extra", {}).items() if k not in GUIDANCE_KEYS}
    if extra:
        spec["extra"] = extra
    else:
        spec.pop("extra", None)
    return spec


def validate_extra(spec, *, continuation):
    extra = spec.get("extra", {})
    slot = spec.get("slot", "claude-code")  # Historical raw RunStore requests retain their slot fallback.
    mode = extra.get("branch_guidance")
    turn = extra.get("assistant_turn")
    omit = extra.get("resume_without_hint", False)
    if type(omit) is not bool:
        raise ValueError("resume_without_hint must be a boolean")
    if mode is not None and mode not in BRANCH_GUIDANCE_MODES:
        raise ValueError("Unknown branch_guidance")
    if not continuation:
        if turn is not None or omit:
            raise ValueError("Assistant-turn and point-only guidance require a validated parent_point")
        if mode is not None and "slot" in spec:
            validate_guidance(mode, slot)
        return
    mode = recorded_guidance(extra)
    validate_guidance(mode, slot)
    if mode == "assistant-turn":
        if turn is None or omit:
            raise ValueError("assistant-turn branch requires assistant_turn and cannot omit it")
        validate_assistant_turn(turn)
    elif turn is not None:
        raise ValueError("assistant_turn requires assistant-turn guidance")
    if omit != (mode == "none"):
        raise ValueError("resume_without_hint must match point-only guidance")


def branch_spec(parent_spec, overrides, native_slot):
    validate_overrides(overrides)
    spec = deepcopy(parent_spec)
    if spec.get("slot", "claude-code") != native_slot:
        raise ValueError("Branch slot must match its native parent")
    inherited = spec.get("extra", {})
    mode = resolve_guidance(overrides.get("branch_guidance", inherited.get("branch_guidance")), native_slot)
    validate_guidance(mode, native_slot)
    extra = {k: v for k, v in inherited.items() if k not in GUIDANCE_KEYS}
    extra["branch_guidance"] = mode
    if mode in {"assistant-turn", "none"} and "prompt" in overrides:
        raise ValueError("Assistant-turn and point-only branches cannot add a user prompt")
    if mode == "assistant-turn":
        turn = overrides.get("assistant_turn")
        if turn is None:
            raise ValueError("Mini defaults to assistant-turn; supply assistant_turn or explicitly select user-hint/none")
        extra["assistant_turn"] = validate_assistant_turn(turn)
    elif "assistant_turn" in overrides:
        raise ValueError("assistant_turn requires assistant-turn guidance")
    if mode == "none":
        extra["resume_without_hint"] = True
    spec.update({k: v for k, v in overrides.items() if k not in GUIDANCE_KEYS})
    spec["extra"] = extra
    validate_extra(spec, continuation=True)
    return spec


def validate_at_point(spec, point):
    validate_extra(spec, continuation=True)
    extra = spec.get("extra", {})
    if recorded_guidance(extra) == "assistant-turn":
        entries = load_prefix(point["native"])
        history = [e["message"] for e in entries if e["type"] == "mini.message"]
        validate_assistant_turn(extra["assistant_turn"], history=history, tools=mini_tool_schema())


def execution_guidance(spec, recovery, job_id):
    """Resolve new branch delivery; do not replay a seed on an infrastructure retry."""
    extra = deepcopy(spec.get("extra", {}))
    slot = spec.get("slot", "claude-code")
    retry = bool(recovery.get("job_id")) and recovery["job_id"] == job_id
    # Old stored continuations have no guidance marker and retain their behavior.
    mode = extra.get("branch_guidance")
    if retry and slot == "mini-swe-agent":
        extra.pop("assistant_turn", None)
        extra.update(branch_guidance="none", resume_without_hint=True)
        mode = "none"
    elif mode is None:
        mode = recorded_guidance(extra)
    extra["branch_guidance"] = mode
    validate_at_point({**spec, "extra": extra}, recovery)
    origin = {"branch_guidance": mode, "hint_delivery": mode,
              "recovery_kind": "retry" if retry else "branch"}
    if mode == "assistant-turn":
        origin.update(assistant_turn_source="reviewer",
                      assistant_turn_call_ids=[c["id"] for c in extra["assistant_turn"]["tool_calls"]])
    return extra, mode, origin
