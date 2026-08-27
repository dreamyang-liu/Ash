"""Run lifecycle: start one, run many, reclaim what dies.

    run.py        one run start to finish -- provision, gateway, checkpoints,
                  drive the slot, tear down whichever way it ends
    batch.py      many runs: concurrency cap, retries, per-task isolation,
                  resume by skipping what finished
    resources.py  what a run holds, written before it is used
    reap.py       reclaim sandboxes and snapshots whose run is gone

This is the *run* granularity, and the boundary is worth stating because two
things that look adjacent are deliberately not here:

- ``harness/execution/backends.py`` -- which pool sandboxes come from. The
  orchestrator is one of its callers, not its owner (``harness extract`` is
  another).
- ``harness/execution/interceptors/mutation.py`` -- an interceptor, so it runs
  per *tool call*. Putting per-call machinery in a per-run component would be a
  layer mistake. What lives here is the decision to snapshot; what decides whether
  a call could have changed anything is on the tool path.

There is no registry of live agents. Nothing yet asks "what else is running", and
a state machine with one state is worse than none -- when subagent spawning
arrives, this is the package it goes in, and `run.py`'s sequence is already what a
child needs.
"""

from harness.orchestrator.batch import BatchRunner, TaskOutcome, is_retryable, load_tasks
from harness.orchestrator.reap import AgentEnvClient, Reaper, parse_duration
from harness.orchestrator.resources import Resource, ResourceLedger
from harness.orchestrator.run import Orchestrator, RunOutcome, RunSpec

__all__ = [
    "Orchestrator", "RunSpec", "RunOutcome",
    "BatchRunner", "TaskOutcome", "load_tasks", "is_retryable",
    "ResourceLedger", "Resource",
    "Reaper", "AgentEnvClient", "parse_duration",
]
