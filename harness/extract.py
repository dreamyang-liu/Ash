"""Extract a run's answer from a snapshot, after the fact.

With a snapshot per step, "what did this run produce" does not have to be decided
while the agent runs. The harness owns the sandbox lifecycle, so it can restore
any step and ask -- which is strictly better than extracting in-band:

- **Re-runnable.** A fix to what counts as part of an answer applies to runs that
  already finished, instead of invalidating them.
- **No baseline probe.** Step 0 is a snapshot of the pristine environment, so the
  untracked files an image shipped can be read from it directly. Extracting
  in-band had to guess a baseline before the agent started because it had nothing
  to compare against.
- **Any step, not just the end.** "What would this run have submitted at step N?"
  is a question only post-hoc extraction can answer, and it is what per-step
  scoring and reward shaping need.
- **A killed run is still recoverable**, because its snapshots outlive it.

The cost is a sandbox per extraction: AgentENV has no API for reading a snapshot's
files without booting one, so this restores (disk-only snapshots cold-boot, which
needs the template's startup command -- see AGENTS.md). Extract once at the end
for grading; walk every step only when you actually want a per-step curve.

Backends that cannot snapshot (Docker) have nothing to restore. There the owner
extracts from the live sandbox before teardown -- same extractor function, called
at a different moment.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from harness.rollback import load_checkpoints

#: ``async (sandbox, context) -> answer``. The context carries whatever the
#: extractor needs that is not in the sandbox (a base commit, a pristine sandbox
#: to compare against).
Extractor = Callable[[Any, "ExtractContext"], Any]


@dataclass
class ExtractContext:
    step: int
    snapshot_id: str
    run_id: Optional[str] = None
    session_ckpt: Optional[str] = None
    #: A sandbox restored from step 0, when the caller asked for one. Present so
    #: an extractor can diff against the pristine environment instead of
    #: reconstructing a baseline.
    pristine: Any = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Extraction:
    step: int
    snapshot_id: str
    answer: Any = None
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None


class SnapshotExtractor:
    """Restores snapshots and runs an extractor against them.

    ``pool`` only needs ``spawn(image=...)`` and ``destroy(sandbox)`` -- a
    ``MicroVMPool``, or a small double in tests. Each extraction gets its own
    sandbox and destroys it, so a batch cannot leak one per step; the caller's own
    run is never touched.
    """

    def __init__(self, pool: Any, *, with_pristine: bool = False) -> None:
        self.pool = pool
        self.with_pristine = with_pristine

    async def extract_at(
        self,
        snapshot_id: str,
        extractor: Extractor,
        *,
        step: int = -1,
        pristine_snapshot: Optional[str] = None,
        run_id: Optional[str] = None,
        extra: Optional[dict] = None,
    ) -> Extraction:
        sandbox = None
        pristine = None
        try:
            sandbox = await self.pool.spawn(image=snapshot_id)
            if pristine_snapshot and self.with_pristine:
                pristine = await self.pool.spawn(image=pristine_snapshot)
            context = ExtractContext(
                step=step, snapshot_id=snapshot_id, run_id=run_id,
                pristine=pristine, extra=dict(extra or {}),
            )
            answer = await extractor(sandbox, context)
            return Extraction(step=step, snapshot_id=snapshot_id, answer=answer)
        except Exception as exc:  # noqa: BLE001 - one bad step must not stop a sweep
            return Extraction(step=step, snapshot_id=snapshot_id,
                              error="%s: %s" % (type(exc).__name__, exc))
        finally:
            for handle in (pristine, sandbox):
                if handle is not None:
                    try:
                        await self.pool.destroy(handle)
                    except Exception:  # noqa: BLE001 - `harness reap` is the backstop
                        pass

    async def extract_journal(
        self,
        journal: Union[str, Path],
        extractor: Extractor,
        *,
        step: Optional[int] = None,
        every_step: bool = False,
        run_id: Optional[str] = None,
    ) -> List[Extraction]:
        """Extract from a journal's checkpoints.

        Default: the last step only, which is what grading wants. ``step`` picks
        one; ``every_step`` walks them all (one sandbox per *distinct* snapshot --
        clean steps reuse the previous capture, so repeats are skipped rather than
        restored again).
        """
        checkpoints = [c for c in load_checkpoints(journal) if c.snapshot_id]
        if not checkpoints:
            return []

        pristine_snapshot = None
        first = min(checkpoints, key=lambda c: c.step)
        if self.with_pristine and first.step == 0:
            pristine_snapshot = first.snapshot_id

        if step is not None:
            wanted = [c for c in checkpoints if c.step == step]
            if not wanted:
                raise KeyError("no checkpoint at step %s" % step)
        elif every_step:
            wanted = []
            seen: set = set()
            for checkpoint in sorted(checkpoints, key=lambda c: c.step):
                if checkpoint.snapshot_id in seen:
                    continue           # a clean step: same environment
                seen.add(checkpoint.snapshot_id)
                wanted.append(checkpoint)
        else:
            wanted = [max(checkpoints, key=lambda c: c.step)]

        results = []
        for checkpoint in wanted:
            results.append(await self.extract_at(
                checkpoint.snapshot_id, extractor,
                step=checkpoint.step,
                pristine_snapshot=pristine_snapshot,
                run_id=run_id,
                extra={"session_ckpt": checkpoint.session_ckpt},
            ))
        return results


# --- extractors ------------------------------------------------------------
def patch_extractor() -> Extractor:
    """SWE-bench's answer: this repository's diff.

    Imported lazily so ``harness`` keeps no import edge into a benchmark; the
    caller names the extractor, the harness only runs it.
    """

    async def extract(sandbox, context: ExtractContext):
        from swebench.patch import extract_patch_async, untracked_baseline

        baseline = None
        if context.pristine is not None:
            # The real pristine environment, rather than a baseline guessed
            # before the agent started.
            baseline = await untracked_baseline(context.pristine)
        base_commit = context.extra.get("base_commit") or await _base_commit(sandbox)
        return await extract_patch_async(sandbox, base_commit,
                                         baseline_untracked=baseline)

    return extract


async def _base_commit(sandbox) -> str:
    result = await sandbox.call("shell", command="git -C /testbed rev-parse HEAD")
    return "" if result.is_error else (result.output or "").strip()


def run_extract(
    pool: Any,
    journal: Union[str, Path],
    extractor: Extractor,
    **kwargs,
) -> List[Extraction]:
    """Synchronous wrapper, for CLI use."""
    extractor_obj = SnapshotExtractor(pool, with_pristine=kwargs.pop("with_pristine", False))
    return asyncio.run(extractor_obj.extract_journal(journal, extractor, **kwargs))
