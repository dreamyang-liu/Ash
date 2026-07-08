"""Best-of-N harness (topology B: fully isolated parallel candidates).

Flow
----
1. N candidate rollouts run in PARALLEL, each inside its OWN sandbox — own
   container, own ``AshSession`` (spawned and destroyed inside the candidate's
   thread), own agent, own budget. Full isolation is the point: candidates
   never share state, so there is nothing to coordinate (no Waggle, no shared
   files — per docs/ARCHITECTURE.md, a harness is topology only).
2. Diversity comes from temperature laddering: candidate *i* samples at
   ``temperature + i * temperature_jitter``.
3. Each candidate yields a patch; ONE is selected as the prediction:

   - ``selection: tests`` (default) — before a candidate's sandbox is
     destroyed, the instance's FAIL_TO_PASS tests are run inside it; score =
     number passing (ties: fewer changed files, then shorter patch).
     **RESEARCH MODE — not leaderboard-fair.** FAIL_TO_PASS is part of the
     benchmark's *evaluation*; selecting with it is test-aware. It measures
     the topology's upper bound (an oracle-verifier best-of-n), nothing more.
   - ``selection: heuristic`` — no test execution: prefer non-empty patches,
     then the changed-file set most candidates agree on, then the shortest
     patch. Leaderboard-fair.
   - TODO(selection=judge): LLM-judge ranking of candidate patches
     (leaderboard-fair, costs extra model calls) — deliberately unimplemented.

Selection logic is pure functions over ``Candidate`` values so it is testable
without sandboxes or model calls (see swebench/tests/test_best_of_n.py).
"""

import json
import re
import shlex
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

from .base import BaseHarness
from ..dataset import resolve_image, format_task_prompt, image_registry_for_subset
from ..sandbox import AshSession
from ..models import AgentConfig
from ..agent import AshAgent, TOOLS_SCHEMA, BASH_ONLY_SCHEMA
from .. import style as S

# FAIL_TO_PASS lists are usually 1-10 entries; cap sandbox test runs per candidate.
MAX_SCORED_TESTS = 20
# Anthropic/Bedrock reject temperature > 1.0; clamp the jitter ladder there.
MAX_TEMPERATURE = 1.0

_DIFF_HEADER = re.compile(r"^diff --git a/(\S+) b/(\S+)", re.MULTILINE)
_DJANGO_TEST_ID = re.compile(r"^(\S+)\s+\((\S+)\)$")  # "test_name (module.Class)"


@dataclass(frozen=True)
class Candidate:
    """One rollout's outcome — the value the pure selection functions rank."""
    index: int
    patch: str
    exit_status: str
    cost: float = 0.0
    tests_passed: Optional[int] = None
    tests_total: Optional[int] = None


# --------------------------------------------------------------------------- #
#  Pure helpers (unit-tested without sandboxes)
# --------------------------------------------------------------------------- #

def parse_test_list(raw: object) -> list[str]:
    """Parse FAIL_TO_PASS / PASS_TO_PASS — a JSON list *string* in SWE-bench."""
    if isinstance(raw, list):
        return [str(t) for t in raw]
    if not isinstance(raw, str) or not raw.strip():
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []
    return [str(t) for t in parsed] if isinstance(parsed, list) else []


def changed_files(patch: str) -> frozenset[str]:
    """File paths touched by a unified diff (b/ side of ``diff --git`` headers)."""
    return frozenset(m.group(2) for m in _DIFF_HEADER.finditer(patch))


def candidate_temperature(base: Optional[float], jitter: Optional[float],
                          index: int) -> Optional[float]:
    """Candidate *i* samples at ``base + i * jitter``, clamped to MAX_TEMPERATURE.

    Without jitter the configured value passes through untouched (None = model
    default). With jitter but no base, the ladder starts at 0.0 (one greedy
    candidate, the rest increasingly diverse).
    """
    if not jitter:
        return base
    return round(min(MAX_TEMPERATURE, (base or 0.0) + index * jitter), 4)


def build_test_command(repo: str, test_id: str) -> str:
    """Shell command whose exit code says whether one FAIL_TO_PASS test passes.

    django ids look like ``test_name (module.Class)`` and need django's own
    runner; the other SWE-bench Verified repos are pytest-collectable. A test
    id the runner cannot collect simply fails, degrading that candidate to the
    tiebreaks — this is a selection signal, never the evaluation itself.
    """
    if repo == "django/django":
        m = _DJANGO_TEST_ID.match(test_id.strip())
        spec = f"{m.group(2)}.{m.group(1)}" if m else test_id
        return ("./tests/runtests.py --verbosity 0 --settings=test_sqlite "
                f"--parallel 1 {shlex.quote(spec)}")
    return f"python -m pytest -x -q {shlex.quote(test_id)}"


def select_by_tests(candidates: Sequence[Candidate]) -> Optional[int]:
    """RESEARCH MODE (test-aware; see module docstring): winner = most
    FAIL_TO_PASS tests passing; ties -> fewer changed files, then shorter patch.

    Returns the winning ``Candidate.index``, or None if every patch is empty.
    """
    eligible = [c for c in candidates if c.patch.strip()]
    if not eligible:
        return None

    def rank(c: Candidate) -> tuple:
        return (-(c.tests_passed or 0), len(changed_files(c.patch)),
                len(c.patch), c.index)

    return min(eligible, key=rank).index


def select_by_heuristic(candidates: Sequence[Candidate]) -> Optional[int]:
    """Leaderboard-fair, no test execution: prefer non-empty patches, then the
    changed-file set most candidates agree on, then the shortest patch.

    Returns the winning ``Candidate.index``, or None if every patch is empty.
    """
    eligible = [c for c in candidates if c.patch.strip()]
    if not eligible:
        return None
    votes = Counter(changed_files(c.patch) for c in eligible)

    def rank(c: Candidate) -> tuple:
        return (-votes[changed_files(c.patch)], len(c.patch), c.index)

    return min(eligible, key=rank).index


# TODO(selection=judge): add an LLM-judge selector (rank candidate patches with
# a model call). Deliberately not implemented — extra spend, needs its own eval.
SELECTORS: dict[str, Callable[[Sequence[Candidate]], Optional[int]]] = {
    "tests": select_by_tests,
    "heuristic": select_by_heuristic,
}


def selection_report(instance_id: str, mode: str, candidates: Sequence[Candidate],
                     winner: Optional[int]) -> dict:
    """JSON-friendly per-candidate scores + winner (saved as <iid>-selection.json)."""
    return {
        "instance_id": instance_id,
        "selection": mode,
        "winner": winner,
        "candidates": [
            {
                "candidate": c.index,
                "exit_status": c.exit_status,
                "patch_bytes": len(c.patch),
                "changed_files": sorted(changed_files(c.patch)),
                "tests_passed": c.tests_passed,
                "tests_total": c.tests_total,
                "cost": round(c.cost, 4),
            }
            for c in candidates
        ],
    }


def overall_status(candidates: Sequence[Candidate], winner: Optional[int]) -> str:
    """completed | no_patch | the first failure mode when every rollout failed."""
    if winner is not None:
        return "completed"
    failed = [c for c in candidates
              if c.exit_status.startswith(("error", "session_failed"))]
    if candidates and len(failed) == len(candidates):
        return failed[0].exit_status
    return "no_patch"


# --------------------------------------------------------------------------- #
#  Harness
# --------------------------------------------------------------------------- #

class BestOfNHarness(BaseHarness):
    """N isolated candidates in parallel; select one patch (topology B)."""

    def __init__(self, config: dict):
        super().__init__(config)
        mode = str(config.get("selection", "tests"))
        if mode not in SELECTORS:
            available = ", ".join(SELECTORS)
            raise ValueError(f"best-of-n: unknown selection '{mode}' (available: {available})")
        self._selection = mode

    def _agent_config(self, c: dict, temperature: Optional[float]) -> AgentConfig:
        return AgentConfig(
            model=c.get("model", "anthropic/claude-sonnet-4-5-20250929"),
            api_base=c.get("api_base"),
            api_key=c.get("api_key"),
            max_tokens=int(c.get("max_tokens", 16384)),
            step_limit=int(c.get("step_limit", 250)),
            cost_limit=float(c.get("cost_limit", 3.0)),  # per-candidate budget
            temperature=temperature,
            reasoning_effort=c.get("reasoning_effort"),
            prompt_cache=c.get("prompt_cache", True),
            tools=c.get("tools", "default"),
            system_template=c.get("system_template"),
            instance_template=c.get("instance_template"),
        )

    def run_instance(self, instance: dict, output_dir: Path) -> dict:
        c = self.config
        iid = instance["instance_id"]
        registry = image_registry_for_subset(c.get("subset", "verified"))
        image = resolve_image(instance, template=c.get("image_template", ""),
                              registry=registry)
        n = max(1, int(c.get("n_candidates", 3)))
        traj_dir = output_dir / "trajectories"

        try:
            with ThreadPoolExecutor(max_workers=n) as pool:
                candidates = list(pool.map(
                    lambda i: self._run_candidate(i, instance, image, traj_dir),
                    range(n),
                ))

            winner = SELECTORS[self._selection](candidates)
            report = selection_report(iid, self._selection, candidates, winner)
            report_path = traj_dir / f"{iid}-selection.json"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(report, indent=2))

            status = overall_status(candidates, winner)
            patch = "" if winner is None else \
                next(x.patch for x in candidates if x.index == winner)
            self._log(iid, candidates, winner, patch, status)
            return {
                "instance_id": iid,
                "model_patch": patch,
                "model_name_or_path": c.get("model", "unknown"),
                "exit_status": status,
            }
        except Exception as e:  # noqa: BLE001
            return self._fail(iid, f"error: {e}")

    def _run_candidate(self, index: int, instance: dict, image: str,
                       traj_dir: Path) -> Candidate:
        """One fully-isolated rollout: own sandbox + agent, spawned and destroyed
        inside this thread (AshSession builds its event loop on first use here)."""
        c = self.config
        iid = instance["instance_id"]
        temp = candidate_temperature(c.get("temperature"),
                                     c.get("temperature_jitter"), index)
        session = AshSession(runtime_bin=c.get("runtime_bin"), quiet=True)
        try:
            if not session.create(image):
                return Candidate(index=index, patch="", exit_status="session_failed")
            cfg = self._agent_config(c, temp)
            agent = AshAgent(cfg, executor=session.execute)
            agent.stream = False
            agent.set_tools_schema(BASH_ONLY_SCHEMA if cfg.tools == "bash_only"
                                   else TOOLS_SCHEMA)
            task = (instance.get("problem_statement", "") if cfg.instance_template
                    else format_task_prompt(instance))
            exit_status = agent.run(task, instance_id=f"{iid}-c{index}")
            patch = session.get_patch()

            passed = total = None
            if self._selection == "tests" and patch.strip():
                passed, total = self._score_in_sandbox(session, instance)

            agent.trajectory.info = {"exit_status": exit_status,
                                     "submission": patch, "model": cfg.model}
            agent.trajectory.cost = agent.cost
            agent.trajectory.save(traj_dir / f"{iid}-c{index}.json")
            return Candidate(index=index, patch=patch, exit_status=exit_status,
                             cost=round(agent.cost.total_cost, 4),
                             tests_passed=passed, tests_total=total)
        except Exception as e:  # noqa: BLE001
            return Candidate(index=index, patch="", exit_status=f"error: {e}")
        finally:
            session.destroy()

    def _score_in_sandbox(self, session: AshSession,
                          instance: dict) -> tuple[Optional[int], Optional[int]]:
        """RESEARCH MODE: run the instance's FAIL_TO_PASS tests in the candidate's
        own sandbox before it is destroyed (test-aware — see module docstring)."""
        tests = parse_test_list(instance.get("FAIL_TO_PASS"))[:MAX_SCORED_TESTS]
        if not tests:
            return None, None
        repo = instance.get("repo", "")
        passed = 0
        for test_id in tests:
            r = session.execute("shell",
                                {"command": build_test_command(repo, test_id),
                                 "working_dir": "/testbed"})
            passed += 1 if r.success else 0
        return passed, len(tests)

    def _log(self, iid: str, candidates: Sequence[Candidate],
             winner: Optional[int], patch: str, status: str) -> None:
        print(S.header(iid))
        print(S.kv("select  ", S.dim(f"{self._selection} over {len(candidates)} candidates")))
        for c in candidates:
            score = (f"{c.tests_passed}/{c.tests_total} f2p  "
                     if c.tests_total is not None else "")
            mark = "  <- winner" if c.index == winner else ""
            print(S.kv(f"  c{c.index:<5}", f"{c.exit_status}  {score}${c.cost}{mark}"))
        print(S.kv("patch   ", S.patch_info(patch)))
        print(S.kv("exit    ", S.green(status) if status == "completed" else S.yellow(status)))

    def _fail(self, iid: str, status: str) -> dict:
        return {
            "instance_id": iid,
            "model_patch": "",
            "model_name_or_path": self.config.get("model", "unknown"),
            "exit_status": status,
        }
