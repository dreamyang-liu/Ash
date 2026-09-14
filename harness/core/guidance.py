"""Render a reviewer's unchanged direction without controller-only reports."""

from __future__ import annotations


def render_branch_note(hint: str, *, truncated: bool = True,
                       commit_required: bool = False) -> str:
    if not isinstance(hint, str) or not hint.strip():
        raise ValueError("branch hint must be a non-empty string")
    lines = []
    if not truncated:
        lines.extend([
            "The filesystem has been restored to an earlier checkpoint. Inspect the current",
            "files rather than assuming later edits in the conversation are still present.", "",
        ])
    lines.extend([
        hint, "",
        "Continue the task directly, without acknowledging or recapping this message.",
        "Keep reasoning and actions focused on the code and observable behavior.",
        "Check uncertain claims against the actual implementation; do not invent",
        "prior observations or treat a suggested cause as already established.",
    ])
    if commit_required:
        lines.append("Commit your changes when finished.")
    return "\n".join(lines)
