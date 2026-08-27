"""Unified journal event model (v2).

Extends the v1 tool-trace schema (swebench/agent/trace.py, docs/TRACE_SCHEMA.md)
with agent/turn/session/checkpoint event types so a single append-only stream
captures the full trajectory regardless of which slot produced it.

Envelope (added by JournalWriter, see harness/core/journal.py)::

    {"v": 2, "type": "<event type>", "ts": "...Z", "seq": N,
     "run_id": ..., "agent_id": ..., "sandbox_id": ..., **payload}

Conventions:
- Event types are dotted, lowercase: ``<domain>.<what>``.
- Payload keys are snake_case, JSON-safe.
- Slot-native payloads that have no mapping yet are preserved verbatim under
  ``raw.<slot>`` events -- never silently dropped (normalize/* contract).
- Framework-private additions go under ``extensions.*`` (v1 convention kept).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

JOURNAL_SCHEMA_VERSION = 2

# --- run lifecycle ---------------------------------------------------------
RUN_STARTED = "run.started"          # {slot, slot_version, model, task_prompt, cwd, config}
RUN_FINISHED = "run.finished"        # {status, exit_code, usage, error}
RUN_RESULT = "run.result"            # {text, native}  final answer as reported by the agent

# --- agent output ----------------------------------------------------------
AGENT_MESSAGE = "agent.message"      # {text, partial?, part_id?}
AGENT_THINKING = "agent.thinking"    # {text, part_id?}
AGENT_ERROR = "agent.error"          # {message, native?}

# --- tools (aligned with v1 tool.started/tool.finished) --------------------
TOOL_STARTED = "tool.started"        # {call_id, name, args, parent_call_id?}
TOOL_FINISHED = "tool.finished"      # {call_id, name?, status: ok|error, output, exit_code?, duration_ms?}

# --- turn / usage ----------------------------------------------------------
# Turn boundaries are the quiesce points for snapshotting (harness/rollback.py):
# between turn.completed and the next turn.started there is no in-flight call.
TURN_STARTED = "turn.started"        # {turn?}
TURN_COMPLETED = "turn.completed"    # {turn?, usage: Usage.as_dict()}

# --- session & rollback ----------------------------------------------------
SESSION_REF = "session.ref"          # {native_session_id, transcript_path?, capabilities?}
CHECKPOINT_CAPTURED = "checkpoint.captured"  # {step, snapshot_id, reason, disk_only, session_ckpt?}

# --- fallbacks --------------------------------------------------------------
SLOT_LOG = "slot.log"                # {stream: stdout|stderr, text}  non-JSON slot output


def raw_event(slot_name: str) -> str:
    """Event type for unmapped native payloads of a given slot."""
    return "raw.%s" % slot_name


@dataclass
class Usage:
    """Normalized token/cost accounting.

    Dimensions are kept separate on purpose (providers disagree on the
    breakdown; collapsing to a single total loses cache/reasoning economics).
    ``cached_input_tokens`` counts tokens read from cache (a subset of input,
    ATIF ``cached_tokens`` semantics). ``cache_creation_tokens`` is the
    Anthropic cache-write count; 0 elsewhere.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    cache_creation_tokens: int = 0
    reasoning_output_tokens: int = 0
    cost_usd: float = 0.0

    def add(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cached_input_tokens += other.cached_input_tokens
        self.cache_creation_tokens += other.cache_creation_tokens
        self.reasoning_output_tokens += other.reasoning_output_tokens
        self.cost_usd += other.cost_usd

    def add_dict(self, payload: dict) -> None:
        self.add(Usage(**{k: payload.get(k) or 0 for k in _USAGE_FIELDS}))

    def as_dict(self) -> dict:
        return asdict(self)


_USAGE_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cached_input_tokens",
    "cache_creation_tokens",
    "reasoning_output_tokens",
    "cost_usd",
)
