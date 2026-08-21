"""Generic tool-call interceptor pipeline — the L2 proxy's core abstraction.

Governance of the tool-call path (coordination, guardrails, ACLs, audit,
budgets) is expressed as an ordered chain of ``ToolInterceptor``s wrapped
around one inner executor (docs/ARCHITECTURE.md, ADR-2)::

    executor: (tool_name: str, args: dict) -> ToolResult

Onion semantics
---------------
``before`` hooks run in list order on the way in; ``after`` hooks run in
REVERSE order on the way out::

    A.before -> B.before -> inner -> B.after -> A.after

- An interceptor is *entered* once its ``before`` returns ``Continue`` or
  ``Rewrite`` (it let the call pass). Short-circuits still unwind the onion:
  when a ``before`` returns ``Reject`` or ``ShortCircuit``, the ``after``
  hooks of interceptors already entered still run on the terminal result — an
  audit interceptor placed first therefore sees rejected calls. The
  terminating interceptor's own ``after`` does not run (its ``before``
  produced the result), and deeper interceptors are never reached.
- ``Rewrite(new_args)`` replaces the args seen by deeper interceptors and by
  the inner executor. Contexts are immutable: each rewrite derives a new
  ``CallContext``; every ``after`` receives the context its own ``before``
  saw. ``metadata`` is one shared dict for the whole call by design.
- Fail-safety is per interceptor. ``fail_mode="closed"`` turns an interceptor
  crash into a rejection (safety interceptors); ``fail_mode="open"`` logs to
  stderr and passes the call through unchanged (observability interceptors).
  A failing ``after`` never aborts the unwind of outer interceptors.
- An inner-executor exception becomes a failed ``ToolResult`` and still
  unwinds the entered ``after`` hooks.
- The framework is stateless: the pipeline holds only its interceptor list;
  per-call state lives in locals, durable state (locks, versions) inside
  interceptors — the pipeline itself is never a concurrency bottleneck.
"""

from __future__ import annotations

import importlib.util
import logging
from dataclasses import dataclass, field, replace
from typing import Callable, Literal, Union

from ..models import ToolResult

logger = logging.getLogger("ash.pipeline")  # unconfigured -> WARNING+ to stderr

Executor = Callable[[str, dict], ToolResult]

#: Reserved ``CallContext.metadata`` keys — the conventions hosts and
#: interceptors use to talk to each other about one call.
#:
#: ``EXECUTOR``   the raw sandbox executor, for interceptors needing probe
#:               traffic or arbitrated writes that must not re-enter the chain.
#: ``RAW_OUTPUT`` / ``RAW_ERROR``
#:               what the runtime actually returned, recorded before any
#:               interceptor rewrote it. Interception is for what the *model*
#:               sees; a trace must still be able to record ground truth.
EXECUTOR = "executor"
RAW_OUTPUT = "raw_output"
RAW_ERROR = "raw_error"


# --------------------------------------------------------------------------- #
#  Call context and verdicts
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class CallContext:
    """One tool call travelling through the pipeline.

    ``metadata`` is a scratch space shared by every hook of one call; the
    proxy uses it to hand interceptors the raw sandbox executor (key:
    ``"executor"``) for probe traffic that must not re-enter the pipeline.
    """
    agent_id: str
    sandbox_id: str
    tool_name: str
    args: dict
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Continue:
    """Let the call proceed unchanged."""


@dataclass(frozen=True)
class Reject:
    """Refuse the call; ``message`` becomes the failed result's output."""
    message: str


@dataclass(frozen=True)
class Rewrite:
    """Let the call proceed with ``new_args`` instead of ``ctx.args``."""
    new_args: dict


@dataclass(frozen=True)
class ShortCircuit:
    """Answer the call with ``result`` without reaching the inner executor."""
    result: ToolResult


Verdict = Union[Continue, Reject, Rewrite, ShortCircuit]
_VERDICT_TYPES = (Continue, Reject, Rewrite, ShortCircuit)


# --------------------------------------------------------------------------- #
#  Interceptor base class
# --------------------------------------------------------------------------- #

class ToolInterceptor:
    """One seat on the tool-call path. Subclass and override the hooks."""

    tools: "set[str] | Literal['*']" = "*"  # which tools this interceptor sees
    fail_mode: Literal["open", "closed"] = "open"

    @property
    def name(self) -> str:
        return type(self).__name__

    def applies_to(self, tool_name: str) -> bool:
        return self.tools == "*" or tool_name in self.tools

    def before(self, ctx: CallContext) -> Verdict:
        """Runs before the call reaches the runtime. Returns a ``Verdict``."""
        return Continue()

    def after(self, ctx: CallContext, result: ToolResult) -> ToolResult:
        """Runs on the way out (reverse order). Returns the (possibly new) result."""
        return result


# --------------------------------------------------------------------------- #
#  Pipeline
# --------------------------------------------------------------------------- #

class ToolPipeline:
    """Ordered interceptor chain around one executor. Order is semantics."""

    def __init__(self, interceptors: "list[ToolInterceptor] | None" = None) -> None:
        self._interceptors: tuple[ToolInterceptor, ...] = tuple(interceptors or ())

    @property
    def interceptors(self) -> tuple[ToolInterceptor, ...]:
        return self._interceptors

    def execute(self, ctx: CallContext, inner: Executor) -> ToolResult:
        """Run one call through the onion: befores in order, inner, afters in
        reverse. Never raises — every failure surfaces as a ``ToolResult``."""
        entered: list[tuple[ToolInterceptor, CallContext]] = []
        current = ctx
        result: "ToolResult | None" = None

        for interceptor in self._interceptors:
            if not interceptor.applies_to(current.tool_name):
                continue
            verdict = self._safe_before(interceptor, current)
            if isinstance(verdict, Reject):
                result = _rejection(interceptor, verdict.message)
                break
            if isinstance(verdict, ShortCircuit):
                result = verdict.result
                break
            entered.append((interceptor, current))
            if isinstance(verdict, Rewrite):
                current = replace(current, args=dict(verdict.new_args))

        if result is None:
            result = self._run_inner(current, inner)

        for interceptor, seen_ctx in reversed(entered):
            result = self._safe_after(interceptor, seen_ctx, result)
        return result

    # -- guarded hook invocation --------------------------------------------- #

    @staticmethod
    def _safe_before(interceptor: ToolInterceptor, ctx: CallContext) -> Verdict:
        """Run ``before``; a crash rejects (fail-closed) or logs and lets the
        call continue (fail-open)."""
        try:
            verdict = interceptor.before(ctx)
            if isinstance(verdict, _VERDICT_TYPES):
                return verdict
            raise TypeError(f"before() returned {verdict!r}, expected a Verdict")
        except Exception as exc:
            if interceptor.fail_mode == "closed":
                return Reject(f"interceptor {interceptor.name} failed closed: {exc}")
            logger.warning("interceptor %s before() failed open (%s); continuing",
                           interceptor.name, exc, exc_info=True)
            return Continue()

    @staticmethod
    def _safe_after(interceptor: ToolInterceptor, ctx: CallContext,
                    result: ToolResult) -> ToolResult:
        """Run ``after``; failures never abort the unwind of outer interceptors."""
        try:
            processed = interceptor.after(ctx, result)
            if isinstance(processed, ToolResult):
                return processed
            raise TypeError(f"after() returned {processed!r}, expected a ToolResult")
        except Exception as exc:
            if interceptor.fail_mode == "closed":
                return _rejection(
                    interceptor, f"interceptor {interceptor.name} failed closed: {exc}")
            logger.warning("interceptor %s after() failed open (%s); result passed through",
                           interceptor.name, exc, exc_info=True)
            return result

    @staticmethod
    def _run_inner(ctx: CallContext, inner: Executor) -> ToolResult:
        """Execute the tool; an exception becomes a failed result so the
        entered ``after`` hooks (audit) still see the call."""
        try:
            return inner(ctx.tool_name, ctx.args)
        except Exception as exc:
            logger.warning("inner executor failed for %s: %s",
                           ctx.tool_name, exc, exc_info=True)
            return ToolResult(success=False, output="", error=str(exc))


def _rejection(interceptor: ToolInterceptor, message: str) -> ToolResult:
    return ToolResult(success=False, output=message,
                      error=f"rejected by {interceptor.name}")


# --------------------------------------------------------------------------- #
#  Mounting (fold a pipeline into an executor)
# --------------------------------------------------------------------------- #

def piped_executor(pipeline: ToolPipeline, inner: Executor, agent_id: str,
                   sandbox_id: "str | Callable[[], str]" = "default") -> Executor:
    """Fold ``pipeline`` around ``inner`` into a plain executor.

    This is the harness-side mount: the same onion the MCP proxy runs
    (``mcp_server._exec_via_pipeline``) exposed through the one seam every
    agent already consumes -- ``(tool_name, args) -> ToolResult`` -- so an
    ``AshAgent`` (or anything else holding an executor) gets L2 governance
    without knowing the pipeline exists.

    - Identity is bound at mount time, like ``AshSession.executor_for``:
      every call through the returned executor is attributed to ``agent_id``,
      so mount one executor per agent.
    - ``sandbox_id`` may be a zero-arg callable, resolved per call, for
      sessions whose sandbox appears (or is recreated) after the executor is
      handed out -- a stale id would silently key coordination state to the
      wrong workspace.
    - ``ctx.metadata["executor"]`` carries ``inner`` so interceptors needing
      probe traffic or arbitrated writes reach the sandbox without
      re-entering the pipeline.
    - Agents whose calls must be arbitrated together must share ONE pipeline
      instance (coordination state lives inside its interceptors). Blocking
      interceptors (one waiting on a lock, say) block the calling thread, so
      give each agent its own thread -- the manager-worker layout.
    - The returned executor carries ``ash_pipeline``, so a host that would
      otherwise mount a chain of its own can see one is already in force and
      not stack a second (``AshAgent`` checks this). Two seats enforcing one
      rule state it to the model twice.
    """
    def run(tool_name: str, args: dict) -> ToolResult:
        target = sandbox_id() if callable(sandbox_id) else sandbox_id
        ctx = CallContext(agent_id=agent_id, sandbox_id=target,
                          tool_name=tool_name, args=dict(args),
                          metadata={EXECUTOR: inner})
        return pipeline.execute(ctx, inner)
    run.ash_pipeline = pipeline          # type: ignore[attr-defined]
    return run


def mounted_pipeline(executor: Executor) -> "ToolPipeline | None":
    """The chain already folded into ``executor`` by :func:`piped_executor`."""
    return getattr(executor, "ash_pipeline", None)


# --------------------------------------------------------------------------- #
#  Plugin loading (assembly is plain Python: a list is the configuration)
# --------------------------------------------------------------------------- #

def load_pipeline(path: str) -> ToolPipeline:
    """Load ``PIPELINE: list[ToolInterceptor]`` from a plugins file.

    The module is executed from ``path`` (ADR-3: policy and assembly are
    Python code, not a DSL). Raises ``ValueError`` if the file cannot be
    imported or does not define a module-level ``PIPELINE`` list.
    """
    spec = importlib.util.spec_from_file_location("ash_mcp_plugins", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import plugins module from {path!r}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    interceptors = getattr(module, "PIPELINE", None)
    if not isinstance(interceptors, (list, tuple)):
        raise ValueError(f"{path!r} must define a module-level PIPELINE list")
    return ToolPipeline(list(interceptors))
