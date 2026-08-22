"""Which calls count as reading, editing, or running the tests.

Separate from the interceptor because the answers come from the tool contract, not from
any one rule: ``EDIT_COMMANDS`` and ``CONTENT_EDIT_COMMANDS`` live in
``agent/tools.py`` so that "this call is an edit" is stated once.
"""

from __future__ import annotations

from ...tools import CONTENT_EDIT_COMMANDS, EDIT_COMMANDS

__all__ = ["is_read", "is_edit", "is_content_edit", "is_test_run", "TEST_MARKERS"]

#: What a command has to mention to count as running the tests. Crude on purpose:
#: the streak counter only needs to know the agent has stopped editing and gone to
#: check, and a missed match costs one extra nudge rather than a wrong decision.
TEST_MARKERS = ("pytest", "test_", "assert")


def _command(tool_name: str, args: dict) -> str:
    """This call's text_editor command, or ``''``.

    A model can put anything in ``args`` — a list, a dict, a number. Membership
    tests against a frozenset raise ``TypeError`` on unhashable values, and in
    reject mode a fail-closed crash would turn malformed model output into a
    policy refusal. Normalize once, here, so junk simply does not match.
    """
    if tool_name != "text_editor":
        return ""
    command = args.get("command")
    return command if isinstance(command, str) else ""


def is_read(tool_name: str, args: dict) -> bool:
    return _command(tool_name, args) == "view"


def is_edit(tool_name: str, args: dict) -> bool:
    return _command(tool_name, args) in EDIT_COMMANDS


def is_content_edit(tool_name: str, args: dict) -> bool:
    """An edit to existing content — see ``tools.CONTENT_EDIT_COMMANDS``."""
    return _command(tool_name, args) in CONTENT_EDIT_COMMANDS


def is_test_run(tool_name: str, args: dict) -> bool:
    if tool_name != "shell":
        return False
    command = args.get("command", "")
    return any(marker in command for marker in TEST_MARKERS)
