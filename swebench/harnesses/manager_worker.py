"""Manager–worker harness (topology A: shared sandbox + file partitioning).

Flow
----
1. **Manager** (read-oriented) explores the repo in ONE sandbox and decomposes the
   task into subtasks, each *owning a disjoint set of files*. It writes the plan as
   JSON to ``/tmp/ash_plan.json``.
2. The working tree is reset to the base commit (discarding any stray manager edits).
3. **Workers** run in parallel — one per subtask — each editing only its owned files.
   Because the runtime's ``executor`` is synchronous and drives a single event loop,
   we cannot ``asyncio.gather`` several agents on one ``AshSession``. Instead each
   worker gets its *own* HTTP client to the *same* container (``Sandbox.connect(url)``)
   and runs in its *own thread* (own event loop). They share the container filesystem
   over HTTP, so disjoint-file edits never conflict.
4. The combined patch is ``session.get_patch()`` — one tree, one diff, for free.

This showcases Ash's core strength (isolated/parallel execution over one MCP-native
runtime). It is a capability demo more than a SWE-bench score maximizer: typical
Verified instances are small localized fixes where decomposition rarely helps.
"""

import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ash_sandbox import Sandbox

from .base import BaseHarness
from ..dataset import resolve_image, image_registry_for_subset
from ..sandbox import AshSession
from ..models import AgentConfig, ToolResult
from ..agent import AshAgent, TOOLS_SCHEMA
from ..agent.waggle import CoordinatedExecutor, WorkspaceCoordinator
from .. import style as S


MANAGER_SYSTEM = (
    "You are the MANAGER of a small engineering team solving a software issue.\n"
    "Your job is to EXPLORE the repository (read-only) and DECOMPOSE the work into "
    "independent subtasks that can be executed in parallel.\n\n"
    "Rules:\n"
    "- Investigate using read_file / grep_files. Do NOT fix the bug yourself and do "
    "NOT edit source files.\n"
    "- Partition the work so that each subtask OWNS a DISJOINT set of files "
    "(no two subtasks touch the same file). Overlapping files break parallel execution.\n"
    "- Prefer 2-4 subtasks. If the fix is inherently single-file, return ONE subtask.\n"
    "- When done, write the plan as RAW JSON (no markdown fences) to /tmp/ash_plan.json "
    "using text_editor write, then stop.\n\n"
    "Plan JSON schema:\n"
    '{"subtasks": [{"id": "t1", "description": "what to change and why", '
    '"files": ["/testbed/path/a.py"], "acceptance": "how to know it is done"}]}\n'
)

WORKER_SYSTEM = (
    "You are a WORKER on an engineering team. You have been assigned ONE subtask and a "
    "set of files you OWN. Implement the change.\n\n"
    "Rules:\n"
    "- Edit ONLY the files you own (listed below). Do NOT touch any other file — a "
    "teammate is editing those concurrently.\n"
    "- Do NOT modify test files. Run relevant tests to check your own change.\n"
    "- Keep the patch minimal: only the source changes needed. Clean up temp files.\n"
    "- When your subtask is complete, stop.\n"
)

WAGGLE_MANAGER_NOTE = (
    "\nA coordination layer arbitrates concurrent edits, so file overlap between "
    "subtasks is PERMITTED (a stale write is rejected with a diff and can be "
    "retried). Still prefer naturally disjoint ownership — less contention "
    "means less rework.\n"
)

WAGGLE_WORKER_NOTE = (
    "\nA coordination layer arbitrates concurrent edits. If a write is rejected "
    "because a teammate changed the file, re-read it and re-apply YOUR change on "
    "top of the latest version. Always read with read_file and edit with "
    "text_editor (do not edit files via shell).\n"
)

_PLAN_PATH = "/tmp/ash_plan.json"


class _WorkerExecutor:
    """Thread-local executor: its own event loop + HTTP client to the SAME container.

    Constructed INSIDE the worker thread so the loop and httpx client belong to that
    thread. Multiple instances share one container via HTTP (disjoint-file safe).
    """

    def __init__(self, url: str):
        import asyncio
        self._loop = asyncio.new_event_loop()
        self._sb = Sandbox.connect(url)

    def __call__(self, tool_name: str, args: dict) -> ToolResult:
        try:
            r = self._loop.run_until_complete(self._sb.call(tool_name, **args))
        except Exception as e:  # noqa: BLE001 - surface as tool error to the agent
            return ToolResult(success=False, output="", error=str(e))
        if r.is_error:
            return ToolResult(success=False, output=r.output, error=r.output)
        return ToolResult(success=True, output=r.output)

    def close(self):
        try:
            self._loop.run_until_complete(self._sb.backend.close())
        except Exception:  # noqa: BLE001
            pass
        finally:
            self._loop.close()


def _extract_json(text: str) -> dict | None:
    """Tolerantly pull the first {...} JSON object out of ``text``."""
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _normalize_subtasks(plan: dict) -> list[dict]:
    """Validate + enforce disjoint file ownership (first claimant wins)."""
    subtasks = plan.get("subtasks") if isinstance(plan, dict) else None
    if not isinstance(subtasks, list) or not subtasks:
        return []
    claimed: set[str] = set()
    normalized: list[dict] = []
    for i, st in enumerate(subtasks):
        if not isinstance(st, dict):
            continue
        files = [f for f in (st.get("files") or []) if isinstance(f, str)]
        owned = [f for f in files if f not in claimed]  # drop overlaps
        claimed.update(owned)
        normalized.append({
            "id": str(st.get("id") or f"t{i + 1}"),
            "description": str(st.get("description") or "").strip(),
            "files": owned,
            "acceptance": str(st.get("acceptance") or "").strip(),
        })
    return [st for st in normalized if st["description"]]


class ManagerWorkerHarness(BaseHarness):
    """Explore → decompose → parallel workers on one shared sandbox (topology A)."""

    def _agent_config(self, c: dict, system: str, step_limit: int) -> AgentConfig:
        return AgentConfig(
            model=c.get("model", "anthropic/claude-sonnet-4-5-20250929"),
            api_base=c.get("api_base"),
            api_key=c.get("api_key"),
            max_tokens=c.get("max_tokens", 16384),
            step_limit=step_limit,
            cost_limit=c.get("cost_limit", 5.0),
            temperature=c.get("temperature"),
            reasoning_effort=c.get("reasoning_effort"),
            prompt_cache=c.get("prompt_cache", True),
            system_template=system,
            instance_template="{{task}}",  # pass the fully-built briefing through
        )

    def run_instance(self, instance: dict, output_dir: Path) -> dict:
        c = self.config
        iid = instance["instance_id"]
        subset = c.get("subset", "verified")
        registry = image_registry_for_subset(subset)
        image = resolve_image(instance, template=c.get("image_template", ""), registry=registry)
        problem = instance.get("problem_statement", "")

        n_workers = int(c.get("n_workers", 3))
        mgr_steps = int(c.get("manager_step_limit", 60))
        wrk_steps = int(c.get("worker_step_limit", 120))
        traj_dir = output_dir / "trajectories"
        waggle_state = WorkspaceCoordinator(ttl=float(c.get("waggle_ttl", 120.0))) if c.get("waggle") else None

        session = AshSession(runtime_bin=c.get("runtime_bin"), quiet=True)
        try:
            if not session.create(image):
                return self._fail(iid, c, "session_failed")
            url = session._sandbox.backend.url  # shared container URL for workers

            # --- 1. Manager: explore + decompose --------------------------------
            subtasks = self._run_manager(session, c, problem, n_workers, mgr_steps, iid, traj_dir)
            if not subtasks:
                # Degrade to a single unrestricted worker (== plain agent).
                subtasks = [{"id": "solo", "description": problem, "files": [], "acceptance": ""}]

            # --- 2. Reset tree to base (discard any stray manager edits) ---------
            base = session._base_commit or "HEAD"
            session.execute("shell", {"command": f"git reset --hard {base} && git clean -fd",
                                      "working_dir": "/testbed"})

            # --- 3. Workers execute in parallel on the shared container ----------
            worker_system = WORKER_SYSTEM + (WAGGLE_WORKER_NOTE if waggle_state else "")
            wrk_cfg = self._agent_config(c, worker_system, wrk_steps)
            results = self._run_workers(url, wrk_cfg, problem, subtasks, iid, traj_dir,
                                        waggle_state)

            # --- 4. Combined patch = one tree, one diff -------------------------
            patch = session.get_patch()
            if waggle_state:
                audit_path = traj_dir / f"{iid}-waggle.json"
                audit_path.parent.mkdir(parents=True, exist_ok=True)
                audit_path.write_text(json.dumps(waggle_state.dump(), indent=2))
            status = "completed" if patch.strip() else "no_patch"
            self._log(iid, subtasks, results, patch, status)

            return {
                "instance_id": iid,
                "model_patch": patch,
                "model_name_or_path": wrk_cfg.model,
                "exit_status": status,
            }
        except Exception as e:  # noqa: BLE001
            return self._fail(iid, c, f"error: {e}")
        finally:
            session.destroy()

    def _run_manager(self, session, c, problem, n_workers, steps, iid, traj_dir) -> list[dict]:
        system = MANAGER_SYSTEM + (WAGGLE_MANAGER_NOTE if c.get("waggle") else "")
        cfg = self._agent_config(c, system, steps)
        agent = AshAgent(cfg, executor=session.execute)
        agent.stream = False
        agent.set_tools_schema(TOOLS_SCHEMA)
        briefing = (
            f"Issue to solve:\n{problem}\n\n"
            f"Explore the repository and decompose the fix into at most {n_workers} "
            f"parallel subtasks with disjoint file ownership. Write the plan to "
            f"{_PLAN_PATH} as raw JSON, then stop."
        )
        agent.run(briefing, instance_id=f"{iid}-manager")
        agent.trajectory.info = {"exit_status": "manager", "model": cfg.model}
        agent.trajectory.cost = agent.cost
        agent.trajectory.save(traj_dir / f"{iid}-manager.json")

        raw = session.execute("shell", {"command": f"cat {_PLAN_PATH} 2>/dev/null"})
        plan = _extract_json(raw.output if raw.success else "")
        return _normalize_subtasks(plan) if plan else []

    def _run_workers(self, url, cfg, problem, subtasks, iid, traj_dir,
                     waggle_state=None) -> list[tuple]:
        def _run_one(st: dict) -> tuple:
            ex = _WorkerExecutor(url)  # created in this thread
            if waggle_state:
                ex = CoordinatedExecutor(ex, waggle_state, agent_id=st["id"])
            try:
                agent = AshAgent(cfg, executor=ex)
                agent.stream = False
                agent.set_tools_schema(TOOLS_SCHEMA)
                owned = "\n".join(f"  - {f}" for f in st["files"]) or "  (any file needed for your subtask)"
                briefing = (
                    f"Overall issue:\n{problem}\n\n"
                    f"YOUR SUBTASK ({st['id']}): {st['description']}\n\n"
                    f"Files you OWN (edit only these):\n{owned}\n\n"
                    f"Acceptance: {st['acceptance'] or 'the subtask is correctly implemented'}\n\n"
                    f"Implement your subtask now. Do not touch files owned by teammates."
                )
                status = agent.run(briefing, instance_id=f"{iid}-{st['id']}")
                agent.trajectory.info = {"exit_status": status, "model": cfg.model}
                agent.trajectory.cost = agent.cost
                agent.trajectory.save(traj_dir / f"{iid}-{st['id']}.json")
                return (st["id"], status, round(agent.cost.total_cost, 4))
            except Exception as e:  # noqa: BLE001
                return (st["id"], f"error: {e}", 0.0)
            finally:
                ex.close()

        with ThreadPoolExecutor(max_workers=max(1, len(subtasks))) as pool:
            return list(pool.map(_run_one, subtasks))

    def _log(self, iid, subtasks, results, patch, status):
        print(S.header(iid))
        print(S.kv("subtasks", S.dim(f"{len(subtasks)} → " + ", ".join(st["id"] for st in subtasks))))
        for sid, st, cost in results:
            print(S.kv(f"  {sid:<6}", f"{st}  ${cost}"))
        print(S.kv("patch   ", S.patch_info(patch)))
        print(S.kv("exit    ", S.green(status) if status == "completed" else S.yellow(status)))

    def _fail(self, iid: str, c: dict, status: str) -> dict:
        return {
            "instance_id": iid,
            "model_patch": "",
            "model_name_or_path": c.get("model", "unknown"),
            "exit_status": status,
        }
