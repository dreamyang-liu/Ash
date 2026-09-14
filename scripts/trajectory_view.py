#!/usr/bin/env python3
"""Render an Ash journal as a readable transcript (markdown).

    python3.11 scripts/trajectory_view.py runs/.../parent.jsonl                 # to stdout
    python3.11 scripts/trajectory_view.py runs/.../r1b2-xxx.jsonl --with-parent  # branch, preceded by
                                                                                 # its parent up to the fork
    python3.11 scripts/trajectory_view.py J.jsonl --max-output 4000 -o traj.md   # longer tool outputs

What you get, in order: the prompt the agent was given; then per step the
agent's thinking/message, the exact tool call (full arguments), the tool result
(head+tail, `--max-output` characters; `--full` for everything), and the
checkpoint (snapshot id) taken after it; finally the agent's closing message and
the run's usage/cost. A branch journal starts at its fork: with --with-parent
the parent's steps up to and including the fork step are printed first, then a
marker, then the branch -- which is exactly the conversation the branch model saw.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from swebench.branching import planned_branch


def load(path):
    with Path(path).open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def fork_metadata(path, events=None):
    """Prefer the executed origin; accept historical and per-branch plans."""
    path = Path(path)
    events = load(path) if events is None else events
    origin = next((e for e in events if e.get("type") == "fork.origin"), None)
    if origin is None:
        origin = next((e.get("origin") for e in events if e.get("type") == "run.started"), {}) or {}
    match = re.match(r"r(\d+)b\d+-", path.stem)
    planned = None
    if match:
        round_no = int(match.group(1))
        plan_path = path.parent / ("plan-round%d.json" % round_no)
        if plan_path.exists():
            plan = json.loads(plan_path.read_text())
            if not plan.get("validation_error"):
                planned = planned_branch(plan.get("review") or {}, path.stem, round_no)
    if not origin.get("parent_run_id") and not planned:
        return None
    info = dict(planned or {})
    info.update(base=origin.get("parent_run_id", info.get("base", "parent")),
                branch_step=origin.get("branch_step", info.get("branch_step")),
                hint=origin.get("actor_hint", info.get("hint")),
                why=origin.get("selection_reason", info.get("why", "")))
    for key in ("snapshot_id", "conversation_cut", "cut_note", "branch_policy", "branch_count_mode"):
        if key in origin:
            info[key] = origin[key]
    return info


def lineage_journals(path, seen=None):
    path = Path(path).resolve()
    seen = set() if seen is None else set(seen)
    if path in seen:
        raise ValueError("cyclic branch lineage: %s" % path)
    seen.add(path)
    info = fork_metadata(path)
    if not info:
        return [path]
    base = info["base"]
    if not isinstance(base, str) or Path(base).name != base:
        raise ValueError("invalid parent run id: %r" % base)
    return lineage_journals(path.parent / (base + ".jsonl"), seen) + [path]


def through_tool_result(events, step):
    count = 0
    target = None
    kept = []
    for event in events:
        kept.append(event)
        if event.get("type") == "tool.started":
            count += 1
            if count == step:
                target = event.get("call_id")
        elif event.get("type") == "tool.finished" and target and event.get("call_id") == target:
            return kept
    raise ValueError("no completed tool result at step %s" % step)


def render_with_ancestry(path, *, max_output, full):
    chain = lineage_journals(path)

    def visit(index, upto=None):
        current = chain[index]
        events = load(current)
        info = fork_metadata(current, events)
        lines, offset = [], 0
        if index:
            full_resume = info.get("cut_note") in {
                "explicit-full-conversation", "compacted-before-fork", "cut-refused-by-cli"}
            lines, offset = visit(index - 1, None if full_resume else info["branch_step"])
            lines += ["", "**Branch `%s` from `%s` at disk step %s**" %
                      (current.stem, info["base"], info["branch_step"]), ""]
            if full_resume:
                lines += ["_Full-conversation resume: history is not cut at the disk step; "
                          "this is a journal lineage view, not the native compacted prompt._", ""]
        if upto is not None:
            events = through_tool_result(events, int(upto))
        lines += render(events, max_output=max_output, full=full, step_offset=offset,
                        prompt_as="message" if index else "header")
        return lines, offset + sum(e.get("type") == "tool.started" for e in events)

    return visit(len(chain) - 1)[0]


def clip(text: str, limit: int, full: bool) -> str:
    text = text or ""
    if full or len(text) <= limit:
        return text
    head = limit * 2 // 3
    return text[:head] + "\n… [%d chars elided] …\n" % (len(text) - limit) + text[-(limit - head):]


def tool_output(e) -> str:
    out = e.get("output")
    if isinstance(out, str):
        # the runtime's JSON envelope (stdout/stderr/exit_code) reads better unpacked
        try:
            p = json.loads(out)
            if isinstance(p, dict) and "stdout" in p:
                parts = []
                if p.get("stdout"):
                    parts.append(p["stdout"])
                if p.get("stderr"):
                    parts.append("[stderr]\n" + p["stderr"])
                parts.append("[exit_code=%s]" % p.get("exit_code"))
                return "\n".join(parts)
        except (ValueError, TypeError):
            pass
        return out
    return json.dumps(out, ensure_ascii=False)


def render(events, *, max_output: int, full: bool, upto_step: int | None = None,
           title: str = "", step_offset: int = 0, prompt_as: str = "header") -> list:
    """``step_offset``: number steps from here (a branch continues its parent's
    count). ``prompt_as``: "header" = run metadata + prompt block; "message" =
    the prompt shown as a message arriving mid-conversation (a branch's
    system-reminder); "none" = omit."""
    lines = []
    step = step_offset
    started = {}
    if title:
        lines += ["# " + title, ""]
    for e in events:
        t = e.get("type")
        if t == "run.started":
            if prompt_as == "header":
                lines += ["## Run", "", "- slot: `%s` %s" % (e.get("slot"), e.get("slot_version", "")),
                          "- model: `%s`" % e.get("model"),
                          "- origin: `%s`" % json.dumps(e.get("origin")) if e.get("origin") else "",
                          "", "## Prompt given to the agent", "", "```", (e.get("task_prompt") or "").rstrip(), "```", ""]
            elif prompt_as == "message":
                lines += ["📩 **message received at this point:**", "", "```",
                          (e.get("task_prompt") or "").rstrip(), "```", ""]
        elif t == "agent.thinking":
            lines += ["> 🤔 **thinking:** " + (e.get("text") or "").strip().replace("\n", "\n> "), ""]
        elif t == "agent.message":
            lines += ["💬 " + (e.get("text") or "").strip(), ""]
        elif t == "tool.started":
            if upto_step is not None and step >= upto_step:
                # the checkpoint record for step N lands after step N+1 has
                # started; stop at the next call instead, so the marker follows
                # the last kept step and its snapshot
                snap = next((x.get("snapshot_id") for x in reversed(events[:events.index(e)])
                             if x.get("type") == "checkpoint.captured"), "?")
                lines += ["", "─" * 78, "**⋯ fork point: the branch below inherits the conversation up to here "
                          "(step %d), and starts from snapshot `%s` ⋯**" % (upto_step, snap), "─" * 78, ""]
                return lines
            step += 1
            started[e.get("call_id")] = step
            name = (e.get("name") or "").replace("mcp__ash__", "")
            lines += ["### step %d — `%s`" % (step, name), "", "```json",
                      json.dumps(e.get("args", {}), indent=2, ensure_ascii=False), "```", ""]
        elif t == "tool.finished":
            s = started.get(e.get("call_id"), step)
            status = e.get("status", "")
            lines += ["**result (step %d, %s):**" % (s, status), "", "```",
                      clip(tool_output(e), max_output, full).rstrip(), "```", ""]
        elif t == "checkpoint.captured":
            lines += ["_checkpoint after step %s → snapshot `%s` (%s)_" % (
                (e.get("step") or 0) + step_offset, e.get("snapshot_id"), e.get("reason")), ""]
        elif t == "run.result":
            lines += ["## Agent's closing message", "", (e.get("text") or "").rstrip(), ""]
        elif t == "run.finished":
            u = e.get("usage") or {}
            lines += ["## Run finished", "", "- status: `%s`%s" % (e.get("status"), " — %s" % e.get("error") if e.get("error") else ""),
                      "- steps (tool calls): %d" % (step - step_offset),
                      "- output tokens: %s · cached input: %s · cost: $%.2f" % (
                          u.get("output_tokens"), u.get("cached_input_tokens"), u.get("cost_usd") or 0), ""]
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("journal")
    ap.add_argument("--with-parent", action="store_true",
                    help="for a branch journal: print parent.jsonl (same dir) up to the fork step first")
    ap.add_argument("--max-output", type=int, default=1500, help="chars of each tool result to show")
    ap.add_argument("--full", action="store_true", help="never truncate tool results")
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args()

    journal = Path(args.journal)
    if args.with_parent:
        lines = render_with_ancestry(journal, max_output=args.max_output, full=args.full)
    else:
        lines = render(load(journal), max_output=args.max_output, full=args.full,
                       title="%s — %s" % (journal.stem, journal.parent.name))
    text = "\n".join(lines)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print("wrote %s (%d lines)" % (args.out, text.count("\n")))
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
