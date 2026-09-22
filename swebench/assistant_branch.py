"""Reviewer context and response schemas for mini-native branch modes."""

from harness.core.journal import read_journal
from harness.core.mini_tools import validate_mini_tools
from harness.slots.mini_history import load_prefix, training_messages


BRANCH_GUIDANCE_MODES = ("user-hint", "assistant-turn", "none")


def resolve_guidance(mode: str | None, slot: str) -> str:
    """Default mini to assistant-turn; retain user hints for other agent slots."""
    if mode is not None:
        return mode
    return "assistant-turn" if slot == "mini-swe-agent" else "user-hint"


def validate_guidance(mode: str, slot: str, full_conversation: bool = False) -> None:
    if mode not in BRANCH_GUIDANCE_MODES:
        raise ValueError(f"unknown branch guidance mode: {mode!r}")
    if mode in {"assistant-turn", "none"} and (slot != "mini-swe-agent" or full_conversation):
        raise ValueError(f"{mode} requires mini-swe-agent and an exact conversation cut")


def require_mini_parent(journal) -> None:
    slots = {e["slot"] for e in read_journal(journal)
             if e.get("type") == "run.started" and e.get("slot")}
    if slots != {"mini-swe-agent"}:
        raise ValueError("This branch mode accepts only mini-swe-agent parent trajectories")


def selected_prefix(journal, checkpoint) -> list[dict]:
    from runstore.mini_native import reference_at

    require_mini_parent(journal)
    reference = reference_at(journal, checkpoint.step, checkpoint.session_ckpt)
    if reference is None:
        raise ValueError("No exact mini prefix at the selected checkpoint")
    return load_prefix(reference)


def actor_tools_at(journal, step: int) -> list[dict]:
    """Read the last actual tool declaration before the selected turn closed."""
    tools = None
    for event in read_journal(journal):
        if event.get("type") == "rollout.model_tools" and event.get("shape") == "chat/completions":
            tools = event.get("tools")
        if event.get("type") == "model.turn.completed" and event.get("step") == step:
            if tools is None:
                raise ValueError("No recorded actor tool schema at the selected checkpoint")
            return validate_mini_tools(tools)
    raise ValueError("No completed turn for the selected actor tool schema")


def reviewer_context(journal, checkpoints: dict) -> dict:
    """Supply each attempt once, with precise message offsets for eligible cuts."""
    if not checkpoints:
        return {"messages": [], "prefix_message_counts": {}}
    prefix = selected_prefix(journal, checkpoints[max(checkpoints)])
    closed = {e["turn_id"]: e["step"] for e in read_journal(journal)
              if e.get("type") == "model.turn.completed"}
    count, cuts = 0, {}
    for entry in prefix:
        if entry["type"] == "mini.message" and entry["message"].get("role") != "exit":
            count += 1
        elif entry["type"] == "mini.turn":
            step = closed.get(entry["turn_id"])
            if step in checkpoints:
                cuts[str(step)] = count
    if set(cuts) != {str(step) for step in checkpoints}:
        raise ValueError("Reviewer mini history does not cover every eligible checkpoint")
    workspace = next((e.get("workspace") for e in prefix if e["type"] == "mini.session"), None)
    return {"messages": training_messages(prefix), "prefix_message_counts": cuts,
            "tools": actor_tools_at(journal, max(checkpoints)), "workspace": workspace}


# Braces are escaped because the controller supplies variables with str.format.
ASSISTANT_REVIEW_PROMPT = """\
You are the REVIEWER for a failed coding task. Choose branches and author the
next assistant response for each branch. mini-swe-agent will execute its bash
tool calls immediately, record the REAL observations, then continue normally.
There is no user hint or opportunity for the actor to revise your first action.

## The problem
{problem}

## Every attempt so far
{reports}

## Branch count requirement
{count_rule}

## Selecting the retained context
- Each branch chooses its own base and branch_step from available_steps.
  Repeated positions are allowed; vary useful repair hypotheses, not wording.
- native_history.messages contains the attempt's native messages. For your
  chosen step, take ONLY the first native_history.prefix_message_counts[step]
  messages (the JSON map uses string keys). The restored disk is AFTER that
  step. Messages beyond that offset are discarded, NOT observations the
  continuing agent has already made.
- Analyses, grades and discarded suffixes are private diagnostic evidence.
  Use them to choose a promising action, not to invent prior observations.
  Prior assistant_turn_given records show directions already tried.

## The response you author
- native_history.tools contains the actor's recorded function definitions.
  Match the function names, required arguments and parameter types exactly.
  The mini executor accepts only command; never add timeout, working_dir, tail
  or other arguments, or use host/MCP tools absent from that list.
- native_history.workspace records the inherited working directory when
  available. To change directories, use cd inside the bash command.
- Write assistant_turn as a normal assistant message: role="assistant",
  ordinary content containing concise code-focused reasoning, and tool_calls.
  Reasoning belongs in content, NOT a provider-native reasoning/thinking field.
- Make it a plausible next response to the retained history. A later diagnosis
  can motivate a hypothesis to test; do not state it was already observed.
- Do not mention reviewers, guidance, other branches, scores, hidden test
  names/IDs, grader paths or outside feedback. Do not write an acknowledgment,
  a user instruction, a task restatement or invented tool observations.
- Never describe the continuation as a checkpoint, restore, replay, restart,
  injected turn or selected branch. Phrases such as "from this checkpoint",
  "the clean checkpoint" and "resuming this trajectory" expose controller-only
  knowledge. Speak only about the code and observations in the retained history:
  for example, "I've inspected the merge APIs; I'll check the working tree
  before implementing the merge logic." A task's own database checkpoint or
  application restore API may still be discussed as domain functionality.
- Claims that work is complete or tests have passed must be supported by
  observations BEFORE the selected cut. Otherwise describe a hypothesis and
  the check you will run, not a fact learned from discarded history or grading.
- Supply one or more bash function calls, in execution order. Every call needs
  a non-empty unique id not used anywhere in the retained history, type=function,
  and function.name=bash. function.arguments is a JSON STRING containing exactly
  one string field, command. Commands may inspect or modify the sandbox.
- bash is the ONLY allowed tool name. Calls named shell, python, apply_patch,
  text_editor, or any host/MCP tool are rejected before any branch executes;
  the validation error will be returned for you to correct the complete plan.
  Python, git and other available programs can be invoked INSIDE bash commands.
- A bash tool does not imply a shell executable named apply_patch exists.
  Do not invoke the host's apply_patch helper inside bash. Use ordinary shell
  redirection or Python file edits when those programs are supported by the
  retained context. Ensure a failed edit stops subsequent checks and commits,
  using explicit error checks, && chaining, or set -e as appropriate.
- Use only files, working-directory conventions and tools supported by the
  selected prefix. Preserve the original task's behavioral constraints.
- Do not include hint, reasoning_content, extra, tool results or additional
  fields in assistant_turn. The controller preserves your content and calls.
- Check that your reasoning still reads naturally immediately after the
  selected prefix, and that each claim about completed work is supported there.

Return exactly one fenced code block labelled branch-plan, containing only the
JSON plan. Do not emit another branch-plan block.

```branch-plan
{{"synthesis": "<why these directions>",
  "branches": [{{"name": "<slug>", "base": "<attempt name>",
    "branch_step": <int>, "why": "<why this checkpoint and action>",
    "assistant_turn": {{"role": "assistant",
      "content": "<reasoning naturally continuing the retained trajectory>",
      "tool_calls": [{{"id": "<unique call id>", "type": "function",
        "function": {{"name": "bash",
          "arguments": "<JSON string with the command field>"}}}}]}}}}]}}
```
"""


POINT_REVIEW_PROMPT = """\
You are the REVIEWER choosing restart points for a failed coding task.
Choose only each branch's base attempt and checkpoint. The actor resumes its
retained conversation and filesystem directly, without receiving any hint,
reviewer-authored assistant message, prescribed reasoning or tool call.

## The problem
{problem}

## Every attempt so far
{reports}

## Branch count requirement
{count_rule}

## Selection rules
- Choose each branch's base and branch_step from that base's available_steps.
  The snapshot is the environment AFTER that step.
- native_history.prefix_message_counts[step] is the exact number of messages
  retained from native_history.messages; map keys are strings. Later messages
  are discarded. Choose a useful place for the actor to continue independently.
- Preserve useful prior work and select a complete turn boundary from the
  supplied candidates. Do not invent steps or move a checkpoint.
- Positions may repeat; do not spread them artificially. Keep the existing
  branch-count requirement, including whether it is an upper bound or exact.
- synthesis and why are controller-only explanations of point selection.
  Nothing you write is sent to the actor. Do not produce a hint, assistant_turn,
  command, tool call or text to insert into the actor's conversation.

Return exactly one fenced code block labelled branch-plan. Every branch may
contain only name, base, branch_step and why. Do not add guidance fields.

```branch-plan
{{"synthesis": "<why these restart points>",
  "branches": [{{"name": "<slug>", "base": "<attempt name>",
    "branch_step": <int>, "why": "<why this checkpoint is useful>"}}]}}
```
"""
