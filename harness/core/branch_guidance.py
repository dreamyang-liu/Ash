"""Shared guidance defaults and agent compatibility for evaluation and RL."""

BRANCH_GUIDANCE_MODES = ("user-hint", "assistant-turn", "none")


def resolve_guidance(mode: str | None, slot: str) -> str:
    if mode is not None:
        return mode
    return "assistant-turn" if slot == "mini-swe-agent" else "user-hint"


def validate_guidance(mode: str, slot: str, full_conversation: bool = False) -> None:
    if mode not in BRANCH_GUIDANCE_MODES:
        raise ValueError(f"unknown branch guidance mode: {mode!r}")
    if mode in {"assistant-turn", "none"} and (slot != "mini-swe-agent" or full_conversation):
        raise ValueError(f"{mode} requires mini-swe-agent and an exact conversation cut")
