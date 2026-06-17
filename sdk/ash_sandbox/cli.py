"""
Ash Sandbox CLI.

Usage:
    ash-sandbox spawn [--image IMAGE] [--entrypoint CMD] [--port PORT]
    ash-sandbox call <tool> [args as JSON]
    ash-sandbox list
    ash-sandbox destroy [--all]
    ash-sandbox shell <command>
"""

import argparse
import asyncio
import json
import os
import sys

from . import DockerPool, Sandbox, HTTPBackend

STATE_FILE = os.path.expanduser("~/.ash/sandboxes.json")


def _load_state() -> dict:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {}


def _save_state(state: dict):
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def _get_url(sandbox_id: str | None = None) -> str:
    state = _load_state()
    sandboxes = state.get("sandboxes", {})

    if sandbox_id:
        matches = [k for k in sandboxes if k.startswith(sandbox_id)]
        if len(matches) == 1:
            return sandboxes[matches[0]]["url"]
        elif len(matches) > 1:
            print(f"Ambiguous ID '{sandbox_id}', matches: {matches}", file=sys.stderr)
            sys.exit(1)
        else:
            print(f"Sandbox '{sandbox_id}' not found", file=sys.stderr)
            sys.exit(1)

    # No ID specified
    if len(sandboxes) > 1:
        print("Multiple sandboxes running. Specify one with -s <id>:", file=sys.stderr)
        for sid, info in sandboxes.items():
            print(f"  {sid}  {info['image']:20s}  {info['url']}", file=sys.stderr)
        sys.exit(1)

    if len(sandboxes) == 1:
        return next(iter(sandboxes.values()))["url"]

    return os.getenv("ASH_RUNTIME_URL", "http://localhost:3000")


async def cmd_spawn(args):
    runtime_bin = os.getenv("ASH_RUNTIME_BIN") or _which("ash-runtime")
    pool = DockerPool(runtime_bin=runtime_bin, port=args.port)

    sb = await pool.spawn(image=args.image, entrypoint=args.entrypoint)
    url = sb.backend.url
    cid = sb._container_id

    state = _load_state()
    state.setdefault("sandboxes", {})[cid[:12]] = {
        "url": url,
        "container_id": cid,
        "image": args.image,
    }
    state["active"] = cid[:12]
    _save_state(state)

    print(f"Sandbox started: {cid[:12]}")
    print(f"  Image: {args.image}")
    print(f"  URL:   {url}")
    if args.entrypoint:
        print(f"  Setup: {args.entrypoint}")
    print(f"\nUse: ash-sandbox shell 'echo hello'")


async def cmd_call(args):
    url = _get_url(args.sandbox)
    sb = Sandbox(backend=HTTPBackend(url))

    try:
        tool_args = json.loads(args.args) if args.args else {}
    except json.JSONDecodeError:
        tool_args = {"command": args.args} if args.tool == "shell" else {}

    result = await sb.call(args.tool, **tool_args)

    if result.is_error:
        print(f"[ERROR] {result.output}", file=sys.stderr)
        sys.exit(1)
    print(result.output, end="" if result.output.endswith("\n") else "\n")

    if result.notifications:
        for n in result.notifications:
            print(f"  [{n['kind']}] {json.dumps(n['data'])}", file=sys.stderr)


async def cmd_shell(args):
    url = _get_url(args.sandbox)
    sb = Sandbox(backend=HTTPBackend(url))
    command = " ".join(args.command)
    result = await sb.call("shell", command=command)

    if result.is_error:
        print(result.output, end="" if result.output.endswith("\n") else "\n", file=sys.stderr)
        sys.exit(1)
    print(result.output, end="" if result.output.endswith("\n") else "\n")


async def cmd_list(args):
    state = _load_state()
    sandboxes = state.get("sandboxes", {})
    if not sandboxes:
        print("No sandboxes. Run: ash-sandbox spawn")
        return

    active = state.get("active")
    for sid, info in sandboxes.items():
        marker = "*" if sid == active else " "
        print(f"  {marker} {sid}  {info['image']:25s}  {info['url']}")


async def cmd_destroy(args):
    state = _load_state()
    sandboxes = state.get("sandboxes", {})

    if args.all:
        for sid, info in sandboxes.items():
            cid = info["container_id"]
            proc = await asyncio.create_subprocess_exec(
                "docker", "rm", "-f", cid,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()
            print(f"  destroyed {sid}")
        state["sandboxes"] = {}
        state.pop("active", None)
    elif args.sandbox_id:
        # Match by prefix
        matches = [k for k in sandboxes if k.startswith(args.sandbox_id)]
        if not matches:
            print(f"Sandbox '{args.sandbox_id}' not found")
            return
        for sid in matches:
            cid = sandboxes[sid]["container_id"]
            proc = await asyncio.create_subprocess_exec(
                "docker", "rm", "-f", cid,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()
            del sandboxes[sid]
            if state.get("active") == sid:
                state.pop("active", None)
            print(f"  destroyed {sid}")
        if sandboxes and not state.get("active"):
            state["active"] = next(iter(sandboxes))
    else:
        active = state.get("active")
        if not active or active not in sandboxes:
            print("No active sandbox. Specify ID or use --all")
            return
        cid = sandboxes[active]["container_id"]
        proc = await asyncio.create_subprocess_exec(
            "docker", "rm", "-f", cid,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.wait()
        del sandboxes[active]
        state.pop("active", None)
        if sandboxes:
            state["active"] = next(iter(sandboxes))
        print(f"  destroyed {active}")

    _save_state(state)


def _which(name: str) -> str | None:
    import shutil
    return shutil.which(name)


def main():
    parser = argparse.ArgumentParser(prog="ash-sandbox", description="Ash Sandbox CLI")
    sub = parser.add_subparsers(dest="subcmd")

    # spawn
    sp = sub.add_parser("spawn", help="Spawn a new sandbox container")
    sp.add_argument("--image", "-i", default="ubuntu:24.04", help="Container image")
    sp.add_argument("--entrypoint", "-e", default=None, help="Setup command before runtime starts")
    sp.add_argument("--port", "-p", type=int, default=3000, help="Runtime port")

    # call
    cp = sub.add_parser("call", help="Call a tool")
    cp.add_argument("-s", "--sandbox", default=None, help="Sandbox ID (prefix match)")
    cp.add_argument("tool", help="Tool name")
    cp.add_argument("args", nargs="?", default=None, help="Tool arguments as JSON")

    # shell (shortcut)
    shp = sub.add_parser("shell", help="Run a shell command in the sandbox")
    shp.add_argument("-s", "--sandbox", default=None, help="Sandbox ID (prefix match)")
    shp.add_argument("command", nargs="+", help="Command to run")

    # list
    sub.add_parser("list", aliases=["ls"], help="List running sandboxes")

    # destroy
    dp = sub.add_parser("destroy", aliases=["rm"], help="Destroy sandbox(es)")
    dp.add_argument("sandbox_id", nargs="?", default=None, help="Sandbox ID to destroy (prefix match)")
    dp.add_argument("--all", "-a", action="store_true", help="Destroy all sandboxes")

    args = parser.parse_args()
    if not args.subcmd:
        parser.print_help()
        sys.exit(1)

    cmd_map = {
        "spawn": cmd_spawn,
        "call": cmd_call,
        "shell": cmd_shell,
        "list": cmd_list,
        "ls": cmd_list,
        "destroy": cmd_destroy,
        "rm": cmd_destroy,
    }

    asyncio.run(cmd_map[args.subcmd](args))


if __name__ == "__main__":
    main()
