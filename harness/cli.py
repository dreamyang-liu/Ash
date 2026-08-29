"""``python -m harness`` — run a slot, inspect a journal, export ATIF.

    # smallest possible run (no sandbox wiring; agent works in --cwd)
    python -m harness run --slot opencode --cwd /tmp/work "list the files"

    # with the ash MCP execution plane as a stdio subprocess
    python -m harness run --slot claude-code --cwd /tmp/work \
        --mcp-stdio "--image python:3.11" "fix the failing test"

    # against a long-lived Execution Server
    python -m harness run --slot codex --mcp-url http://localhost:8400/mcp ...

    # through an inference gateway: model swap + wire tap + enforced budget
    python -m harness run --slot claude-code --gateway --routes routes.json \
        --budget-usd 2.50 --model ash-rl-ckpt-42 "..."

    python -m harness show   runs/<id>.jsonl        # event summary
    python -m harness atif   runs/<id>.jsonl -o t.json
    python -m harness fork-plan runs/<id>.jsonl --step 12

    # stand a gateway up on its own (many runs, one gateway)
    python -m harness gateway --routes routes.json --port 8787

    # a batch: bounded concurrency, per-task isolation, resumable by skipping
    python -m harness batch tasks.jsonl --slot codex --workers 8 --out runs/b1

    # reclaim sandboxes/snapshots left behind by killed runs
    python -m harness reap --dry-run

    # extract a run's answer from its snapshots, after the fact
    python -m harness extract runs/<id>.jsonl --extractor swebench:patch
    python -m harness extract runs/<id>.jsonl --every-step -o answers.json
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from collections import Counter
from pathlib import Path

from harness.atif import export_file
from harness.orchestrator import (AgentEnvClient, BatchRunner, Orchestrator,
                                  Reaper, ResourceLedger, RunSpec, load_tasks,
                                  parse_duration)
from harness.core.journal import JournalWriter, new_run_id, read_journal
from harness.core.slot import TaskSpec
from harness.execution.provision import provision_http
from harness.extract import patch_extractor, run_extract
from harness.execution.wiring import http_wiring, stdio_wiring
from harness.gateway import GatewayServer, RoutingTable
from harness.rollback import fork_plan
from harness.slots import available, load_slot


def _backend_section(name: "str | None", runtime_bin: "str | None") -> dict:
    """A backend config from CLI flags. Empty means Docker, the pool's default.

    ``server_url`` and ``api_key`` are left out on purpose: ``build_pool`` reads
    AENV_SERVER_URL / AENV_API_KEY itself, so a key never has to be typed on a
    command line where it would land in shell history.

    microvm gets ``from_image``, because a name given on a command line is an OCI
    image reference and only the cold-start path accepts one. With
    ``--runtime-bin`` a per-image template is built instead, which is what a plain
    image needs: the backend runs no startup command for one, so a cold-started
    image comes up without the runtime and answers every tool call with a 502.
    """
    if not name:
        return {}
    config: dict = {"backend": name}
    if name == "microvm":
        section: dict = {"from_image": True}
        if runtime_bin:
            section["runtime_bin"] = runtime_bin
        config["microvm"] = section
    return config


def _cmd_run(args: argparse.Namespace) -> int:
    """Translate flags into a RunSpec and let the orchestrator drive.

    The sequence (provision, gateway, checkpoints, teardown) used to be inline
    here, which meant nothing else could reuse it. This function is now argument
    parsing and printing.
    """
    spec = RunSpec(
        prompt=args.prompt,
        slot=args.slot,
        cwd=args.cwd,
        model=args.model,
        timeout_s=args.timeout,
        agent_id=args.agent_id,
        run_id=args.run_id,
        journal_path=args.journal,
        mcp_url=args.mcp_url,
        sandbox_image=args.sandbox_image,
        sandbox_id=args.sandbox_id,
        keep_sandbox=args.keep_sandbox,
        transport=args.transport,
        tools=args.tools,
        backend=_backend_section(args.backend, args.runtime_bin),
        runtime_bin=args.runtime_bin,
        snapshot_every_step=args.snapshot_every_step,
        mcp_stdio_args=shlex.split(args.mcp_stdio) if args.mcp_stdio is not None else None,
        use_gateway=args.gateway,
        routes_file=args.routes,
        gateway_port=args.gateway_port,
        budget_usd=args.budget_usd,
        resume_session_id=args.resume_session,
        fork=args.fork,
        extra=json.loads(args.extra) if args.extra else {},
    )

    def report(kind: str, payload: dict) -> None:
        if kind == "sandbox":
            print("sandbox     %s (bound)" % payload["sandbox_id"])
        elif kind == "mcp":
            print("mcp         %s (%s)" % (payload["url"], payload["transport"]))
        elif kind == "gateway":
            print("gateway     %s (budget %s)"
                  % (payload["url"], payload.get("budget_usd") or "none"))
        elif kind == "checkpoint.unavailable":
            # Loud on purpose. A run that reports success while having recorded no
            # snapshots cannot be branched later, and the person who finds that out
            # is doing so from the outside, hours afterwards.
            print("checkpoints NONE -- %d opportunities skipped: %s"
                  % (payload["skipped"], payload["reason"]))

    # Any sandbox this run acquires, however it acquires it: `harness reap` reads
    # the ledger to reclaim what a killed process could not release. This used to
    # be created only for a sandbox on a remote server, so an orchestrator-owned
    # one -- the common case now -- was invisible to reap.
    ledger = ResourceLedger() if args.sandbox_image else None
    outcome = Orchestrator(ledger=ledger, on_event=report).run(spec)

    print("run_id      %s" % outcome.run_id)
    print("journal     %s" % outcome.journal_path)
    print("status      %s" % outcome.status)
    if outcome.native_session_id:
        print("session     %s" % outcome.native_session_id)
    if outcome.usage:
        print("usage       %s" % json.dumps(outcome.usage))
    if outcome.checkpoints:
        print("checkpoints %d" % outcome.checkpoints)
    if outcome.error:
        print("error       %s" % outcome.error, file=sys.stderr)
    if outcome.final_text:
        print("---")
        print(outcome.final_text)
    return 0 if outcome.ok else 1


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


def _cmd_extract(args: argparse.Namespace) -> int:
    from harness.execution.backends import build_pool

    if args.extractor != "swebench:patch":
        print("unknown extractor %r (known: swebench:patch)" % args.extractor,
              file=sys.stderr)
        return 2
    extractor = patch_extractor()

    runtime_bin = Path(args.runtime_bin).expanduser()
    pool = build_pool(
        {"backend": args.backend},
        runtime_bin=str(runtime_bin.resolve()) if runtime_bin.exists() else None,
    )
    if not getattr(pool, "supports_snapshot", lambda: False)():
        print(
            "backend %r cannot restore snapshots, so there is nothing to extract "
            "from after the fact; extract from the live sandbox before teardown "
            "instead (see harness/execution/server.py)." % args.backend,
            file=sys.stderr,
        )
        return 2

    results = run_extract(
        pool, args.journal, extractor,
        step=args.step, every_step=args.every_step, with_pristine=args.pristine,
    )
    if not results:
        print("no checkpoints in %s -- was the run snapshotted?" % args.journal,
              file=sys.stderr)
        return 1

    payload = [
        {"step": r.step, "snapshot_id": r.snapshot_id,
         "ok": r.ok, "error": r.error, "answer": r.answer}
        for r in results
    ]
    if args.output:
        Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print("wrote %s (%d step(s))" % (args.output, len(payload)))
    for r in results:
        size = len(r.answer or "") if isinstance(r.answer, str) else "-"
        print("step %-4s %-28s %s %s" % (
            r.step, r.snapshot_id, "ok" if r.ok else "FAILED", r.error or "%s chars" % size))
    return 0 if all(r.ok for r in results) else 1


def _cmd_gateway(args: argparse.Namespace) -> int:
    table = RoutingTable.from_file(args.routes) if args.routes else RoutingTable()
    journal = (
        JournalWriter(args.journal, run_id="gateway", agent_id="gateway")
        if args.journal
        else None
    )
    server = GatewayServer(
        table,
        journal=journal,
        host=args.host,
        port=args.port,
        require_token=not args.open,
    )
    for agent_id in args.mint or []:
        token = table.mint(agent_id, budget_usd=args.budget_usd)
        print("token %-16s %s" % (agent_id, token.token))
    print("gateway  %s" % server.base_url)
    print("routes   %s" % (", ".join(table.models()) or "(default only)"))
    if args.open:
        print("WARNING  --open: no token required; do not expose this port")
    print("env      ANTHROPIC_BASE_URL=%s" % server.base_url)
    server.start()
    try:
        import time

        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        server.stop()
        if journal is not None:
            journal.close()
    return 0


def _cmd_batch(args: argparse.Namespace) -> int:
    tasks = load_tasks(args.tasks)
    mcp = None
    if args.mcp_url:
        mcp = http_wiring(args.mcp_url)
    elif args.mcp_stdio is not None:
        mcp = stdio_wiring(args=shlex.split(args.mcp_stdio))

    done = {"n": 0}
    total = len(tasks)

    def report(task_id, outcome):
        done["n"] += 1
        mark = "skip" if outcome.skipped else outcome.status
        print("[%d/%d] %-28s %-10s %5.0fs%s" % (
            done["n"], total, task_id[:28], mark, outcome.seconds,
            "" if outcome.attempts <= 1 else "  (attempt %d)" % outcome.attempts,
        ))

    runner = BatchRunner(
        args.slot,
        args.out,
        workers=args.workers,
        max_attempts=args.max_attempts,
        timeout_s=args.timeout,
        mcp_factory=(lambda _task: mcp) if mcp else None,
        on_update=report,
        resume=not args.no_resume,
    )
    print("batch    %d tasks, %d workers, slot=%s" % (total, runner.workers, args.slot))
    try:
        runner.run(tasks)
    except KeyboardInterrupt:
        runner.stop()
        print("\ninterrupted; in-flight tasks finishing", file=sys.stderr)

    counts = runner.counts()
    print("\ncounts   %s" % json.dumps(counts))
    print("summary  %s" % (Path(args.out) / "summary.json"))
    print("cleanup  python -m harness reap --ledger %s"
          % (Path(args.out) / "resources.jsonl"))
    return 0 if counts.get("completed", 0) == total else 1


def _cmd_reap(args: argparse.Namespace) -> int:
    ledger = ResourceLedger(args.ledger) if args.ledger else ResourceLedger()
    if args.compact:
        dropped = ledger.compact()
        print("compacted ledger, dropped %d entries" % dropped)

    older_than = parse_duration(args.older_than) if args.older_than else None
    if args.include_unknown and older_than is None:
        print("refusing --include-unknown without --older-than", file=sys.stderr)
        return 2

    reaper = Reaper(AgentEnvClient(), ledger)
    plan = reaper.plan(include_unknown=args.include_unknown, older_than=older_than)

    if plan.unsupported:
        print("%d snapshot(s) are reclaimable but this backend has no snapshot"
              % len(plan.unsupported))
        print("delete API (DELETE /snapshots/{id} -> 405). They must be freed")
        print("server-side; a reaper cannot do it from outside. Ids:")
        for resource_id in plan.unsupported[:10]:
            print("  %s" % resource_id)
        if len(plan.unsupported) > 10:
            print("  ... and %d more" % (len(plan.unsupported) - 10))
        print()

    if not plan.total():
        print("nothing to reap (%d kept)" % len(plan.kept))
        return 0
    for kind, items in (("sandbox", plan.sandboxes), ("snapshot", plan.snapshots)):
        for resource_id in items:
            print("  %-9s %s  %s" % (kind, resource_id, plan.reasons.get(resource_id, "")))
    if args.dry_run:
        print("\n%d resource(s) would be freed (dry run)" % plan.total())
        return 0

    done = reaper.apply(plan)
    print("\nfreed %d sandbox(es), %d snapshot(s)"
          % (len(done["sandboxes"]), len(done["snapshots"])))
    if done["failed"]:
        print("failed: %s" % ", ".join(done["failed"]), file=sys.stderr)
        return 1
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
    run.add_argument("--mcp-stdio", help="args for `python -m harness.execution.server`")
    run.add_argument("--mcp-url", help="remote MCP endpoint")
    run.add_argument(
        "--sandbox-image",
        help="create a sandbox from this image and bind the slot to it (the agent "
             "then sees no sandbox_id argument). Without --mcp-url this process "
             "owns the sandbox and runs the server itself, which is what makes "
             "checkpoints available; with it, the sandbox is created on that server",
    )
    run.add_argument(
        "--transport", default="stdio", choices=("stdio", "http"),
        help="how the agent reaches a sandbox this process owns: a server "
             "subprocess (stdio) or one in-process on an ephemeral port (http). "
             "Ignored with --mcp-url",
    )
    run.add_argument(
        "--tools", metavar="PANEL",
        help="tool panel the agent is offered: a shipped name (default, full, "
             "bash_only, no_web) or a path to a manifest",
    )
    run.add_argument(
        "--backend", choices=sorted(("docker", "microvm", "k8s")),
        help="where a sandbox this process creates comes from (default: docker). "
             "microvm reads AENV_SERVER_URL / AENV_API_KEY and is the one that "
             "can snapshot",
    )
    run.add_argument(
        "--runtime-bin", default=None,
        help="ash-runtime binary to provision into a bare image (microvm)",
    )
    run.add_argument(
        "--snapshot-every-step", action="store_true",
        help="checkpoint at every quiesce point, not only after a mutation",
    )
    run.add_argument(
        "--sandbox-id",
        help="with --mcp-url: bind to an existing sandbox instead of creating one",
    )
    run.add_argument(
        "--keep-sandbox", action="store_true",
        help="do not destroy a provisioned sandbox on exit (grading, inspection)",
    )
    run.add_argument("--extra", help="JSON dict of slot-specific options")
    run.add_argument("--gateway", action="store_true", help="route LLM traffic through a gateway")
    run.add_argument("--routes", help="gateway routing table JSON (implies --gateway)")
    run.add_argument("--gateway-port", type=int, default=0, help="0 = ephemeral")
    run.add_argument(
        "--budget-usd", type=float, help="hard ceiling enforced by the gateway (implies it)"
    )
    run.add_argument("--resume-session", help="native session id to continue")
    run.add_argument("--fork", action="store_true", help="branch the resumed session")
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

    ext = sub.add_parser(
        "extract",
        help="extract a run's answer from its snapshots (re-runnable, any step)",
    )
    ext.add_argument("journal")
    ext.add_argument(
        "--extractor", default="swebench:patch",
        help="what an answer is. 'swebench:patch' = this repo's git diff.",
    )
    ext.add_argument("--step", type=int, help="one step (default: the last)")
    ext.add_argument("--every-step", action="store_true",
                     help="walk every distinct snapshot -- a per-step curve")
    ext.add_argument("--pristine", action="store_true",
                     help="also restore step 0, so an extractor can diff against "
                          "the untouched environment instead of guessing a baseline")
    ext.add_argument("--backend", default="microvm", choices=("microvm", "docker", "k8s"))
    ext.add_argument("--runtime-bin", default="runtime/ash-runtime")
    ext.add_argument("-o", "--output", help="write results as JSON")
    ext.set_defaults(func=_cmd_extract)

    gw = sub.add_parser("gateway", help="run an inference gateway in the foreground")
    gw.add_argument("--routes", help="routing table JSON")
    gw.add_argument("--host", default="127.0.0.1")
    gw.add_argument("--port", type=int, default=8787)
    gw.add_argument("--journal", help="tap wire-level events into this journal")
    gw.add_argument(
        "--mint", action="append", metavar="AGENT_ID",
        help="mint a slot token and print it (repeatable)",
    )
    gw.add_argument("--budget-usd", type=float)
    gw.add_argument(
        "--open", action="store_true",
        help="accept unauthenticated requests (local debugging only)",
    )
    gw.set_defaults(func=_cmd_gateway)

    batch = sub.add_parser("batch", help="run a task file with bounded concurrency")
    batch.add_argument("tasks", help="JSONL: one {id, prompt, cwd, ...} per line")
    batch.add_argument("--slot", required=True, choices=available())
    batch.add_argument("--out", default="runs/batch")
    batch.add_argument("--workers", type=int, default=4)
    batch.add_argument("--max-attempts", type=int, default=2)
    batch.add_argument("--timeout", type=float, default=1800.0)
    batch.add_argument("--mcp-stdio", help="args for `python -m harness.execution.server`")
    batch.add_argument("--mcp-url")
    batch.add_argument(
        "--no-resume", action="store_true",
        help="re-run tasks that already have a terminal status",
    )
    batch.set_defaults(func=_cmd_batch)

    reap = sub.add_parser("reap", help="reclaim resources left by dead runs")
    reap.add_argument("--ledger", default=None, help="path to resources.jsonl")
    reap.add_argument("--dry-run", action="store_true")
    reap.add_argument(
        "--include-unknown", action="store_true",
        help="also consider backend resources the ledger never saw (needs care)",
    )
    reap.add_argument("--older-than", help="with --include-unknown, e.g. 24h")
    reap.add_argument("--compact", action="store_true", help="prune released entries")
    reap.set_defaults(func=_cmd_reap)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
