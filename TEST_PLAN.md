# Ash functional test plan — identity, events, tools, provisioning

Scope: everything changed in `0626674`, `b980dc9`, `0d93339` — agent identity
binding, the event log's opt-in delivery, output bounds, manifest-defined custom
tools, and pool provisioning.

Method: **execute, do not read**. Every finding must come with the command that
produced it and the actual output. Three of this session's bugs (a boolean
compiling to `--fix True`, `Sandbox.connect(url, tools=...)` raising TypeError,
debian-slim having no CA roots) were invisible to code review and only appeared
when something ran. A test that cannot fail proves nothing: where a suite
already passes, verify it *bites* by breaking the implementation and watching it
go red.

## Shared setup

- Runtime binary: `/tmp/ash-runtime-test` (already built from `0d93339`)
- Repo: `/opt/dlami/nvme/projects/Ash`, SDK import via `PYTHONPATH=sdk`
- Each agent owns a port range and container name prefix, so parallel runs
  cannot collide:

| Agent | Area | Ports | Container prefix |
|---|---|---|---|
| A | Identity & isolation | 3410–3419 | `ash_t_a_` |
| B | Events & async | 3420–3429 | `ash_t_b_` |
| C | Tools & bounds | 3430–3439 | `ash_t_c_` |
| D | Provisioning & docs | 3440–3449 | `ash_t_d_` |

Rules for every agent:

- Clean up your containers (`docker rm -f`) on the way out, including on failure.
- Do not modify tracked files. Scratch work goes in `/tmp`.
- Report per case: PASS / FAIL / BLOCKED, the command, and the real output.
- A test that fails because the *test* was wrong is not a product bug — say so
  explicitly and correct it. (This session had two such: `mode="background"`
  instead of `background=True`, and waiting on a broadcast kind while claiming
  to test targeted delivery.)

## Agent A — identity and isolation

Fixed this session: identity used to be readable from tool arguments, so a
caller could name another agent and read its events.

1. **Transport wins over arguments.** Call any tool with `agent_id` in the
   arguments and a different identity on the connection. The transport's must
   win.
2. **Anonymous cannot be upgraded.** With no identity on the connection, put
   `agent_id: "victim"` in the arguments and try to read events targeted at
   `victim` (produce one with a backgrounded `shell`, `background=True`, whose
   `process_exited` is targeted). Expect zero events read, and the rightful
   owner still receives it.
3. **Schema silence.** `agent_id` must not appear in any tool's schema in any
   of the three formats (`openai` / `anthropic` / `raw`) — a property is an
   invitation to the model.
4. **Independent cursors.** Two named subscribers must *each* see a third
   party's call, not one stealing it from the other. Then repeat with both
   anonymous and show the difference (this is the failure mode that motivated
   the work).
5. **Same name shares a cursor.** Two connections with the *same* id split
   events. Confirm and quantify: this is a known sharp edge with no validation,
   and I want its exact behaviour on record.
6. **Own actions are not echoed.** A caller subscribed to `tool:shell` must not
   receive its own calls back in a later response's piggyback.
7. **Cross-transport.** Repeat case 1 over MCP (`Sandbox.mcp`) and CLI
   (`Sandbox.local`), not just HTTP. All three share one dispatch path; prove it.
8. **Harness separation.** In `swebench`, `AshSession.execute` is the harness's
   channel and `executor_for(id)` an agent's. Verify bookkeeping (`get_patch`)
   is attributed to `harness`, and that `swebench/tests/test_harness_trace_identity.py`
   bites: revert `executor_for("manager")` to `session.execute` and confirm red.

## Agent B — events and async

The delivery model is opt-in subscription plus per-event TTL, with long-polling
instead of push.

1. **Nothing without a subscription.** An identity that subscribed to nothing
   receives nothing on ordinary tool responses.
2. **Subscribe then receive.** After `action=subscribe`, matching events arrive
   with later responses. After `unsubscribe`, they stop.
3. **`timeout: 0` polls.** Must return immediately (this regressed once: an
   explicit 0 was treated as unset and blocked 30s). Verify it still *returns
   queued events*, and that a real wait still blocks until the event.
4. **Targeted waiting.** `sources=[pid]` waits for one specific background
   process, not any exit. Start three, wait for the middle one.
5. **Wakeup latency.** A wait must return when the event happens, not when the
   timeout expires. Measure.
6. **Delivered once.** A second wait must not repeat what the first returned.
7. **Loss is reported.** Overflow the log (`ASH_EVENT_QUEUE_MAX_EVENTS`) and
   confirm `missed` is non-zero rather than events vanishing silently. Same for
   TTL expiry (`ASH_EVENT_TTL_SECONDS`).
8. **Concurrency.** ~20 concurrent waiters and pushers; no hang, no panic, no
   event delivered twice to one identity. Also run `go test -race ./events`.
9. **Piggyback does not steal.** An event a waiter is blocked on must not be
   consumed by an unrelated tool call's piggyback drain.

## Agent C — tools and output bounds

Eight builtin tools; bounds exist to stop a runaway command OOM-ing the sandbox
(which kills the rollout), not as a token budget.

1. **All eight present and callable.** `shell`, `text_editor`, `grep_files`,
   `process`, `web_search`, `web_fetch`, `artifact`, `wait_for_events`.
   Exercise each with a real call; note which need network.
2. **`shell` stdin and env.** Structured, no shell quoting. Include a value
   containing spaces and one containing `$(...)` — it must not be evaluated.
3. **Bounds and truncation.** `max_output_bytes` with `truncate_mode` `H1T1`,
   `H2T3`, `T1`. Confirm the byte budget is respected, the split matches the
   weights, and truncation is *announced*. Check the 1 KiB per-stream floor
   (`minMaxOutputBytes`) is real: ask for 200 and see what you get.
4. **Bounds on other producers.** `text_editor` view, `grep_files`, `web_fetch`
   — a byte-producing tool that ignores the bound is a hole.
5. **Custom tool argv.** Booleans: `{flag: "--fix"}` → `--fix` when true,
   nothing when false; `style: value` → `--flag true` (lowercase); unknown
   `style` rejected at parse time. Positionals ordered. Defaults applied.
6. **Injection is inert.** `"; rm -rf / #"` and `$(whoami)` as argument values
   must reach the binary as single argv slots, unevaluated. Prove with a binary
   that echoes its argv (`/bin/echo`, or `printf '%s\n'`).
7. **Artifact provisioning.** `artifact` with a correct sha256 installs; with a
   *wrong* sha256 it must refuse; with none it downloads and trusts. Verify the
   cache means the second call does not re-download.
8. **Custom tool end to end.** Register a manifest, `prepare_tools()`, call it,
   confirm one artifact step then shell. Then delete the cached binary inside
   the sandbox and confirm the stale-path retry re-resolves rather than
   reporting "not found".
9. **Process control.** `process` list/kill against a real background process.

## Agent D — provisioning and documentation

1. **README executes.** Every runnable snippet in `sdk/README.md`, verbatim
   where possible. This is the check that found three bugs; treat a snippet
   that cannot run as a defect.
2. **The example runs.** `sdk/examples/multi_agent_shared_sandbox.py` against a
   fresh runtime. Confirm the runner attributes each write to the right agent.
3. **Pool identity.** `DockerPool.spawn(agent_id=...)` — the handle carries it
   and calls are attributed. Then two sandboxes from one pool with the *same*
   agent_id: confirm they do not interfere (separate event logs), which is what
   makes reuse across sandboxes safe.
4. **Pool lifecycle.** `spawn` / `list` / `destroy` / `destroy_all` / context
   manager. No container left behind — check `docker ps -a` before and after.
   Include the failure path: destroy an already-destroyed sandbox.
5. **Capabilities are honest.** `DockerPool.supports_pause()/supports_fork()`
   return False and the operations raise `NotImplementedError` naming the
   check. `MicroVMPool` declares True (its server is unavailable here — assert
   the declaration and the client's request shape, not the hypervisor).
6. **Image portability.** Spawn on `debian:bookworm-slim` (no CA certificates)
   and confirm `web_fetch` over HTTPS works — this is the embedded-CA-roots
   regression, invisible except in a bare image. Also try `alpine` (musl) and
   `python:3.12-slim`.
7. **Cross-backend parity.** The same script over HTTP, MCP and CLI backends
   gives the same results.
8. **Fresh-install honesty.** `pip install ./sdk` into a clean venv, then run a
   script that imports only `ash_sandbox` from a directory outside the repo.
   The package claims to be independently distributable; verify that claim.

## Reporting

Each agent returns a table of case → verdict → evidence, then:

- Bugs found, each with a minimal reproduction.
- Anything **BLOCKED** and why (a blocked case is not a passing case).
- Any case where the test itself was wrong, and the correction.
