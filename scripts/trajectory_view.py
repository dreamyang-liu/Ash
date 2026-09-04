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
import sys
from pathlib import Path


def load(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


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
    events = load(journal)
    lines = []
    if args.with_parent:
        parent = journal.parent / "parent.jsonl"
        rs = next((e for e in events if e.get("type") == "run.started"), {})
        origin = rs.get("origin") or {}
        fork_step = origin.get("branch_step")
        if fork_step is None:
            plan_files = sorted(journal.parent.glob("plan-round*.json"))
            rnd = journal.name[1] if journal.name.startswith("r") else None
            for p in plan_files:
                if rnd and p.name.endswith("round%s.json" % rnd):
                    fork_step = (json.loads(p.read_text()).get("review") or {}).get("branch_step")
        if parent.exists() and fork_step:
            lines += render(load(parent), max_output=args.max_output, full=args.full,
                            upto_step=int(fork_step), title="PARENT — %s (up to fork step %s)" % (journal.parent.name, fork_step))
        else:
            lines += ["_(no parent.jsonl / fork step found next to %s)_" % journal.name, ""]
    lines += render(events, max_output=args.max_output, full=args.full,
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
