"""Multiple agents operating on ONE shared sandbox.

The sandbox is stateless routing: every client that targets the same address
(HTTP: same URL / Gateway: same sandbox_id) lands in the *same* container and
shares its filesystem + process space. So "N agents on one sandbox" just means
N Sandbox clients pointing at the same target.

This demo runs 3 cooperating "agents" concurrently against one runtime:
  - coder     : writes the implementation file
  - tester    : writes a pytest file
  - runner    : waits for both, then runs the tests

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


async def coder(ready: asyncio.Event) -> None:
    """Agent 1: create the implementation. Its own client, shared sandbox."""
    async with Sandbox.connect(SANDBOX_URL) as sb:
        await sb.call("shell", command=f"mkdir -p {WORKDIR}")
        await sb.call(
            "text_editor",
            command="write",
            path=f"{WORKDIR}/calc.py",
            file_text="def add(a, b):\n    return a + b\n",
        )
        print("[coder]  wrote calc.py")
        ready.set()


async def tester(ready: asyncio.Event) -> None:
    """Agent 2: write tests in parallel — same filesystem as the coder."""
    async with Sandbox.connect(SANDBOX_URL) as sb:
        await sb.call("shell", command=f"mkdir -p {WORKDIR}")
        await sb.call(
            "text_editor",
            command="write",
            path=f"{WORKDIR}/test_calc.py",
            file_text=(
                "from calc import add\n\n"
                "def test_add():\n"
                "    assert add(2, 3) == 5\n"
            ),
        )
        print("[tester] wrote test_calc.py")
        ready.set()


async def runner(coder_done: asyncio.Event, tester_done: asyncio.Event) -> None:
    """Agent 3: barrier on the other two, then run the shared tests."""
    await asyncio.gather(coder_done.wait(), tester_done.wait())
    async with Sandbox.connect(SANDBOX_URL) as sb:
        r = await sb.call("shell", command=f"cd {WORKDIR} && python -m pytest -q")
        print("[runner] pytest output:\n" + r.output)


async def main() -> None:
    coder_done = asyncio.Event()
    tester_done = asyncio.Event()
    await asyncio.gather(
        coder(coder_done),
        tester(tester_done),
        runner(coder_done, tester_done),
    )


if __name__ == "__main__":
    asyncio.run(main())
