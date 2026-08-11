"""Multiple agents operating on ONE shared sandbox.

The sandbox is stateless routing: every client that targets the same address
(HTTP: same URL / Gateway: same sandbox_id) lands in the *same* container and
shares its filesystem + process space. So "N agents on one sandbox" just means
N Sandbox clients pointing at the same target.

Two things this demonstrates beyond sharing a filesystem:

1. Each client is *named*. The runtime keeps a per-identity cursor over its
   event log, so two anonymous clients would share one cursor and silently
   split events between them -- each seeing roughly half. The identity is
   supplied by the client, never by the model: an agent that could name itself
   could read events addressed to another one.

2. The runner coordinates through the *sandbox*, not through this process. It
   subscribes to the writes it is waiting for and blocks in wait_for_events.
   Using an asyncio.Event here instead would only work because all three
   happen to run in one Python process -- which is exactly what stops being
   true as soon as an agent is an LLM in a loop somewhere else.

Run:
    # 1. start ONE runtime (or: docker run ... ash-runtime --port 3000)
    ./runtime/ash-runtime --port 3000

    # 2. in another shell
    python sdk/examples/multi_agent_shared_sandbox.py
"""

from __future__ import annotations

import asyncio

from ash_sandbox import Sandbox

SANDBOX_URL = "http://localhost:3000"
WORKDIR = "/tmp/shared_demo"

IMPL = f"{WORKDIR}/calc.py"
TESTS = f"{WORKDIR}/test_calc.py"


async def coder() -> None:
    """Agent 1: create the implementation. Its own identity, shared sandbox."""
    async with Sandbox.connect(SANDBOX_URL, agent_id="coder") as sb:
        await sb.call("shell", command=f"mkdir -p {WORKDIR}")
        await sb.call(
            "text_editor",
            command="write",
            path=IMPL,
            file_text="def add(a, b):\n    return a + b\n",
        )
        print("[coder]  wrote calc.py")


async def tester() -> None:
    """Agent 2: write tests in parallel — same filesystem as the coder."""
    async with Sandbox.connect(SANDBOX_URL, agent_id="tester") as sb:
        await sb.call("shell", command=f"mkdir -p {WORKDIR}")
        await sb.call(
            "text_editor",
            command="write",
            path=TESTS,
            file_text=(
                "from calc import add\n\n"
                "def test_add():\n"
                "    assert add(2, 3) == 5\n"
            ),
        )
        print("[tester] wrote test_calc.py")


async def runner(subscribed: asyncio.Event) -> None:
    """Agent 3: wait for both writes to land, then run the shared tests.

    It waits on facts from the sandbox rather than on the other coroutines, so
    the same code works when the writers are separate processes or models.
    """
    async with Sandbox.connect(SANDBOX_URL, agent_id="runner") as sb:
        # Delivery is opt-in: without a subscription the runtime keeps nothing
        # for this identity, so subscribe *before* the writers start.
        await sb.call(
            "wait_for_events",
            action="subscribe",
            kinds=["tool:text_editor"],
        )
        subscribed.set()

        seen: set[str] = set()
        while not {IMPL, TESTS} <= seen:
            batch = await sb.wait_for_events(kinds=["tool:text_editor"], timeout=30)
            if batch.timed_out and not batch:
                print("[runner] gave up waiting")
                return
            if batch.missed:
                # Loss is reported rather than hidden: the events expired or
                # were trimmed before this consumer read them.
                print(f"[runner] missed {batch.missed} event(s)")
            for event in batch:
                print(f"[runner] saw {event.origin} touch {event.source}")
                seen.add(event.source)

        result = await sb.call("shell", command=f"cd {WORKDIR} && python -m pytest -q")
        print("[runner] pytest output:\n" + result.output)


async def main() -> None:
    # The only local synchronisation is "the observer is listening"; everything
    # the agents learn about each other comes from the sandbox.
    subscribed = asyncio.Event()

    async def write_after_subscribe(agent):
        await subscribed.wait()
        await agent()

    await asyncio.gather(
        runner(subscribed),
        write_after_subscribe(coder),
        write_after_subscribe(tester),
    )


if __name__ == "__main__":
    asyncio.run(main())
