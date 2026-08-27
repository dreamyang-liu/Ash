"""``python -m harness`` — run a slot, inspect a journal, export ATIF.

    # smallest possible run (no sandbox wiring; agent works in --cwd)
    python -m harness run --slot opencode --cwd /tmp/work "list the files"

    # with the ash MCP execution plane as a stdio subprocess
    python -m harness run --slot claude-code --cwd /tmp/work \
        --mcp-stdio "--image python:3.11" "fix the failing test"

    # against a long-lived Execution Server
    python -m harness run --slot codex --mcp-url http://localhost:8400/mcp ...

    python -m harness show   runs/<id>.jsonl        # event summary
    python -m harness atif   runs/<id>.jsonl -o t.json
    python -m harness fork-plan runs/<id>.jsonl --step 12
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from collections import Counter
from pathlib import Path

from harness.atif import export_file
from harness.core.journal import JournalWriter, new_run_id, read_journal
from harness.core.slot import TaskSpec
from harness.execution.wiring import http_wiring, stdio_wiring
from harness.rollback import fork_plan
from harness.slots import available, load_slot


def _cmd_run(args: argparse.Namespace) -> int:
    slot_cls = load_slot(args.slot)
    slot = slot_cls()

    mcp = None
    if args.mcp_url:
        mcp = http_wiring(args.mcp_url, name=args.mcp_name, agent_id=args.agent_id)
    elif args.mcp_stdio is not None:
        mcp = stdio_wiring(name=args.mcp_name, args=shlex.split(args.mcp_stdio))

    run_id = args.run_id or new_run_id()
    journal_path = Path(args.journal or "runs/%s.jsonl" % run_id)

    extra = json.loads(args.extra) if args.extra else {}
    if args.slot == "claude-code":
        # Eval hygiene: ignore the developer's local CLAUDE.md / .claude config
        # unless explicitly asked for (equivalent of `claude --bare`).
        extra.setdefault("setting_sources", [])

    task = TaskSpec(
        prompt=args.prompt,
        cwd=args.cwd,
        model=args.model,
        timeout_s=args.timeout,
        extra=extra,
    )

    with JournalWriter(journal_path, run_id=run_id, agent_id=args.agent_id) as journal:
        result = slot.run(task, journal, mcp)

    print("run_id      %s" % run_id)
    print("journal     %s" % journal_path)
    print("status      %s" % result.status)
    if result.native_session_id:
        print("session     %s" % result.native_session_id)
    if result.usage:
        print("usage       %s" % json.dumps(result.usage))
    if result.error:
        print("error       %s" % result.error, file=sys.stderr)
    if result.final_text:
        print("---")
        print(result.final_text)
    return 0 if result.status == "completed" else 1


def _cmd_show(args: argparse.Namespace) -> int:
    records = read_journal(args.journal)
    counts = Counter(r.get("type") for r in records)
    print("events   %d" % len(records))
    for etype, count in counts.most_common():
        print("  %-24s %d" % (etype, count))
    raw = [t for t in counts if str(t).startswith("raw.")]
    if raw:
        print("\nunmapped native payloads present: %s" % ", ".join(sorted(raw)))
        print("(normalizer gap -- see contracts/ before trusting derived metrics)")
    return 0


def _cmd_atif(args: argparse.Namespace) -> int:
    document = export_file(args.journal, copied_through_seq=args.copied_through_seq)
    payload = json.dumps(document, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
        print("wrote %s (%d steps)" % (args.output, len(document["steps"])))
    else:
        print(payload)
    return 0


def _cmd_fork_plan(args: argparse.Namespace) -> int:
    plan = fork_plan(args.journal, args.step)
    print(json.dumps(plan, indent=2))
    if not plan["complete"]:
        print(
            "warning: incomplete pair (missing %s half) -- branch will diverge"
            % ("session" if plan["snapshot_id"] else "environment"),
            file=sys.stderr,
        )
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="harness", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run a task through a slot")
    run.add_argument("prompt")
    run.add_argument("--slot", required=True, choices=available())
    run.add_argument("--cwd", default=".")
    run.add_argument("--model")
    run.add_argument("--timeout", type=float, default=3600.0)
    run.add_argument("--journal")
    run.add_argument("--run-id")
    run.add_argument("--agent-id", default="agent")
    run.add_argument("--mcp-name", default="ash")
    run.add_argument("--mcp-stdio", help="args for `python -m swebench.mcp_server`")
    run.add_argument("--mcp-url", help="remote MCP endpoint")
    run.add_argument("--extra", help="JSON dict of slot-specific options")
    run.set_defaults(func=_cmd_run)

    show = sub.add_parser("show", help="summarize a journal")
    show.add_argument("journal")
    show.set_defaults(func=_cmd_show)

    atif = sub.add_parser("atif", help="export a journal as ATIF v1.8")
    atif.add_argument("journal")
    atif.add_argument("-o", "--output")
    atif.add_argument("--copied-through-seq", type=int, default=0)
    atif.set_defaults(func=_cmd_atif)

    fork = sub.add_parser("fork-plan", help="resolve the rollback pair at a step")
    fork.add_argument("journal")
    fork.add_argument("--step", type=int, required=True)
    fork.set_defaults(func=_cmd_fork_plan)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
