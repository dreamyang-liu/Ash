"""Restore an exact mini history and AgentENV snapshot inside a Harbor trial."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from harness.core.assistant_turn import validate_assistant_turn
from harness.slots.mini_history import load_prefix
from runstore.mini_native import reference_at
from swebench.assistant_branch import actor_tools_at
from swebench.fork_eval import available_branch_points
from terminalbench.agentenv import AgentENVEnvironment
from terminalbench.agentenv_mini import AgentENVMini


def read_branch_context(path: str | Path) -> dict:
    context = json.loads(Path(path).read_text())
    if context.get("branch_guidance") != "assistant-turn":
        raise ValueError("TerminalBench mini branch requires assistant-turn guidance")
    if not re.fullmatch(r"[0-9a-f-]{36}", context.get("snapshot_id", "")):
        raise ValueError("A branch requires an explicit snapshot UUID")
    if context.get("checkpoint_mode") != "full":
        raise ValueError("TerminalBench branch requires a full snapshot")
    if "@sha256:" not in context.get("base_image", ""):
        raise ValueError("The original task image digest must be pinned")
    if not isinstance(context.get("image_config"), dict):
        raise ValueError("The original task image config must be recorded")
    if type(context.get("step")) is not int or context["step"] < 1:
        raise ValueError("A branch needs a positive parent tool step")
    if not isinstance(context.get("native_session_id"), str) or not context["native_session_id"]:
        raise ValueError("A branch needs the parent native session ID")
    if not isinstance(context.get("conversation_cut"), str) or not context["conversation_cut"]:
        raise ValueError("A branch needs the exact native cut")
    parent = Path(context.get("parent_journal", ""))
    if not parent.is_file() or hashlib.sha256(parent.read_bytes()).hexdigest() != context.get("parent_journal_sha256"):
        raise ValueError("Parent journal is missing or changed")
    return context


class SnapshotEnvironment(AgentENVEnvironment):
    """Boot Harbor's task environment from a selected full snapshot."""

    def __init__(self, *args, branch_context: str, **kwargs):
        self.branch_context_path = Path(branch_context)
        self.branch_context = read_branch_context(self.branch_context_path)
        super().__init__(*args, **kwargs)
        if self.checkpoint_mode != "full":
            raise ValueError("Branch environment must keep full checkpoint mode")

    def _prepare_image(self, force_build: bool):
        if force_build:
            raise ValueError("A branch must restore its snapshot, not rebuild the task image")
        return self.branch_context["snapshot_id"], self.branch_context["image_config"]

    async def _upload_environment_dir_after_start(self):
        # Restored task files are already in the snapshot. A fresh upload would
        # overwrite edits made before the selected checkpoint.
        return None

    async def start(self, force_build: bool = False) -> None:
        await super().start(force_build=force_build)
        receipt = {"snapshot_id": self.branch_context["snapshot_id"],
                   "sandbox_id": self.session.sandbox_id,
                   "base_image": self.branch_context["base_image"],
                   "branch_context": str(self.branch_context_path),
                   "checkpoint_mode": self.checkpoint_mode}
        (self.trial_paths.trial_dir / "restored-state.json").write_text(json.dumps(receipt, indent=2))


class BranchMini(AgentENVMini):
    """Continue a reviewer-authored assistant turn from the selected point."""

    def __init__(self, *args, branch_context: str, **kwargs):
        self.branch_context_path = Path(branch_context)
        self.branch_context = read_branch_context(self.branch_context_path)
        super().__init__(*args, **kwargs)

    @staticmethod
    def name() -> str:
        return "ash-agentenv-mini-branch"

    def _make_spec(self, prompt, workspace, journal, environment):
        context = self.branch_context
        if not isinstance(environment, SnapshotEnvironment) or environment.branch_context_path != self.branch_context_path:
            raise ValueError("Agent and environment must use the same branch context")
        parent = Path(context["parent_journal"])
        if hashlib.sha256(parent.read_bytes()).hexdigest() != context["parent_journal_sha256"]:
            raise ValueError("Parent journal changed after branch selection")
        point = available_branch_points(parent).get(context["step"])
        if (point is None or point.snapshot_id != context["snapshot_id"]
                or point.session_ckpt != context["native_session_id"]):
            raise ValueError("Branch snapshot/session does not match the parent journal")
        reference = reference_at(parent, context["step"], context["native_session_id"])
        if reference is None or reference.get("cut") != context["conversation_cut"]:
            raise ValueError("Branch native prefix does not match the selected cut")
        prefix = load_prefix(reference)
        turn = validate_assistant_turn(
            context.get("assistant_turn"),
            history=[row["message"] for row in prefix if row["type"] == "mini.message"],
            tools=actor_tools_at(parent, context["step"]),
        )
        spec = super()._make_spec("", workspace, journal, environment)
        spec.fork = True
        spec.extra.update(native_prefix=reference, assistant_turn=turn)
        spec.origin = {"parent_journal": str(parent),
                       "branch_step": context["step"],
                       "snapshot_id": context["snapshot_id"],
                       "conversation_cut": context["conversation_cut"],
                       "branch_guidance": "assistant-turn",
                       "assistant_turn_source": "reviewer",
                       "parent_journal_sha256": context["parent_journal_sha256"]}
        return spec
