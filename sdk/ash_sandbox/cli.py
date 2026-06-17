"""
Ash Sandbox CLI.

Usage:
    ash-sandbox config --mode local --url http://localhost:3000
    ash-sandbox config --mode docker
    ash-sandbox config --mode k8s --gateway URL --control-plane URL
    ash-sandbox info
    ash-sandbox spawn [--image IMAGE] [--entrypoint CMD]
    ash-sandbox shell [-s ID] <command>
    ash-sandbox call [-s ID] <tool> [args JSON]
    ash-sandbox list
    ash-sandbox destroy [ID | --all]
"""

import argparse
import asyncio
import json
import os
import sys

from . import DockerPool, SandboxPool, Sandbox, HTTPBackend, GatewayBackend

CONFIG_FILE = os.path.expanduser("~/.ash/config.json")
STATE_FILE = os.path.expanduser("~/.ash/sandboxes.json")


# ==================== State ====================

def _load_config() -> dict:
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE) as f:
            return json.load(f)
    return {"mode": "local", "url": "http://localhost:3000"}


def _save_config(config: dict):
    os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)


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
    config = _load_config()
    state = _load_state()
    sandboxes = state.get("sandboxes", {})

    # Local mode: always use config URL
    if config["mode"] == "local":
        return config.get("url", "http://localhost:3000")

    # Docker/K8s mode: must select a sandbox
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

    if len(sandboxes) > 1:
        print("Multiple sandboxes running. Specify with -s <id>:", file=sys.stderr)
        for sid, info in sandboxes.items():
            print(f"  {sid}  {info.get('image',''):20s}  {info['url']}", file=sys.stderr)
        sys.exit(1)

    if len(sandboxes) == 1:
        return next(iter(sandboxes.values()))["url"]

    print("No sandbox running. Run: ash-sandbox spawn", file=sys.stderr)
    sys.exit(1)


# ==================== Commands ====================

async def cmd_config(args):
    config = _load_config()

    if args.mode:
        config["mode"] = args.mode
    if args.url:
        config["url"] = args.url
    if args.gateway:
        config["gateway_url"] = args.gateway
    if args.control_plane:
        config["control_plane_url"] = args.control_plane
    if args.runtime_bin:
        config["runtime_bin"] = args.runtime_bin

    _save_config(config)
    print(f"Config saved: {CONFIG_FILE}")
    _print_config(config)


async def cmd_info(args):
    config = _load_config()
    state = _load_state()
    sandboxes = state.get("sandboxes", {})

    _print_config(config)
    print()

    if sandboxes:
        print(f"Sandboxes: {len(sandboxes)} running")
        for sid, info in sandboxes.items():
            print(f"  {sid}  {info.get('image',''):20s}  {info['url']}")
    else:
        print("Sandboxes: none")


def _print_config(config: dict):
    mode = config.get("mode", "local")
    print(f"  Mode:    {mode}")
    if mode == "local":
        print(f"  URL:     {config.get('url', 'http://localhost:3000')}")
    elif mode == "docker":
        print(f"  Runtime: {config.get('runtime_bin', '(auto-detect)')}")
    elif mode == "k8s":
        print(f"  Gateway: {config.get('gateway_url', '(not set)')}")
        print(f"  Control: {config.get('control_plane_url', '(not set)')}")


async def cmd_spawn(args):
    config = _load_config()
    mode = config.get("mode", "local")

    if mode == "local":
        print("Mode is 'local' — no container to spawn. Use: ash-sandbox config --mode docker")
        return

    image = args.image
    entrypoint = args.entrypoint

    if mode == "docker":
        runtime_bin = config.get("runtime_bin") or _which("ash-runtime")
        pool = DockerPool(runtime_bin=runtime_bin, port=args.port)
        sb = await pool.spawn(image=image, entrypoint=entrypoint)
        url = sb.backend.url
        cid = sb._container_id

    elif mode == "k8s":
        gw = config.get("gateway_url")
        cp = config.get("control_plane_url")
        if not gw or not cp:
            print("K8s mode requires --gateway and --control-plane in config", file=sys.stderr)
            sys.exit(1)
        pool = SandboxPool(control_plane_url=cp, gateway_url=gw)
        sb = await pool.spawn(image=image, entrypoint=entrypoint)
        url = gw
        cid = sb._container_id
    else:
        print(f"Unknown mode: {mode}", file=sys.stderr)
        sys.exit(1)

    state = _load_state()
    state.setdefault("sandboxes", {})[cid[:12]] = {
        "url": url,
        "container_id": cid,
        "image": image,
        "mode": mode,
    }
    _save_state(state)

    print(f"Sandbox started: {cid[:12]}")
    print(f"  Image: {image}")
    print(f"  URL:   {url}")
    if entrypoint:
        print(f"  Setup: {entrypoint}")


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
    config = _load_config()
    state = _load_state()
    sandboxes = state.get("sandboxes", {})

    if not sandboxes:
        if config["mode"] == "local":
            print(f"Mode: local → {config.get('url', 'http://localhost:3000')}")
        else:
            print("No sandboxes running. Run: ash-sandbox spawn")
        return

    for sid, info in sandboxes.items():
        print(f"  {sid}  {info.get('image',''):20s}  {info['url']}")


async def cmd_destroy(args):
    state = _load_state()
    sandboxes = state.get("sandboxes", {})

    if not sandboxes:
        print("No sandboxes to destroy")
        return

    if args.all:
        for sid, info in list(sandboxes.items()):
            await _kill_container(info["container_id"])
            print(f"  destroyed {sid}")
        state["sandboxes"] = {}
    elif args.sandbox_id:
        matches = [k for k in sandboxes if k.startswith(args.sandbox_id)]
        if not matches:
            print(f"Sandbox '{args.sandbox_id}' not found")
            return
        for sid in matches:
            await _kill_container(sandboxes[sid]["container_id"])
            del sandboxes[sid]
            print(f"  destroyed {sid}")
    else:
        if len(sandboxes) > 1:
            print("Multiple sandboxes. Specify ID or use --all:", file=sys.stderr)
            for sid, info in sandboxes.items():
                print(f"  {sid}  {info.get('image','')}", file=sys.stderr)
            sys.exit(1)
        sid = next(iter(sandboxes))
        await _kill_container(sandboxes[sid]["container_id"])
        del sandboxes[sid]
        print(f"  destroyed {sid}")

    _save_state(state)


async def _kill_container(cid: str):
    proc = await asyncio.create_subprocess_exec(
        "docker", "rm", "-f", cid,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    await proc.wait()


def _which(name: str) -> str | None:
    import shutil
    return shutil.which(name)


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(prog="ash-sandbox", description="Ash Sandbox CLI")
    sub = parser.add_subparsers(dest="subcmd")

    # config
    cfp = sub.add_parser("config", help="Configure mode and connection settings")
    cfp.add_argument("--mode", "-m", choices=["local", "docker", "k8s"], help="Execution mode")
    cfp.add_argument("--url", help="Runtime URL (local mode)")
    cfp.add_argument("--gateway", help="Gateway URL (k8s mode)")
    cfp.add_argument("--control-plane", help="Control plane URL (k8s mode)")
    cfp.add_argument("--runtime-bin", help="Path to ash-runtime binary (docker mode)")

    # info
    sub.add_parser("info", help="Show current config and running sandboxes")

    # spawn
    sp = sub.add_parser("spawn", help="Spawn a new sandbox")
    sp.add_argument("--image", "-i", default="ubuntu:24.04", help="Container image")
    sp.add_argument("--entrypoint", "-e", default=None, help="Setup command before runtime starts")
    sp.add_argument("--port", "-p", type=int, default=3000, help="Runtime port")

    # call
    cp = sub.add_parser("call", help="Call a tool")
    cp.add_argument("-s", "--sandbox", default=None, help="Sandbox ID (prefix match)")
    cp.add_argument("tool", help="Tool name")
    cp.add_argument("args", nargs="?", default=None, help="Tool arguments as JSON")

    # shell
    shp = sub.add_parser("shell", help="Run a shell command")
    shp.add_argument("-s", "--sandbox", default=None, help="Sandbox ID (prefix match)")
    shp.add_argument("command", nargs="+", help="Command to run")

    # list
    sub.add_parser("list", aliases=["ls"], help="List running sandboxes")

    # destroy
    dp = sub.add_parser("destroy", aliases=["rm"], help="Destroy sandbox(es)")
    dp.add_argument("sandbox_id", nargs="?", default=None, help="Sandbox ID (prefix match)")
    dp.add_argument("--all", "-a", action="store_true", help="Destroy all")

    args = parser.parse_args()
    if not args.subcmd:
        parser.print_help()
        sys.exit(1)

    cmd_map = {
        "config": cmd_config,
        "info": cmd_info,
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
