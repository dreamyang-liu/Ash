from __future__ import annotations

import hashlib
import json
import re
import shlex
from pathlib import PurePosixPath
from dataclasses import dataclass
from typing import Any, Iterable, Iterator

from .checkpoints import canonical_prefix, trajectory_prefix_chain_hash


SAFE_READ = "safe_read"
MUTATION = "mutation"
BARRIER = "barrier"
DEFAULT_WORKSPACE_ROOTS = ("/testbed", "/app")
_SAFE_SHELL_PROGRAMS = {
    "pwd", "ls", "cat", "head", "tail", "wc", "grep", "rg", "stat", "file",
    "sort", "uniq", "tr", "echo", "sed", "find", "which",
}


@dataclass(frozen=True)
class RelaxedProjection:
    """Conservative environment-state projection of a trajectory prefix.

    ``state_events`` contains only events that may change environment state or
    whose side effects are unknown. ``safe_reads`` are omitted from the state
    key but remain available for read-result caching/replay. This projection says
    nothing about model-history/KV equivalence.
    """

    state_events: tuple[dict[str, Any], ...]
    safe_reads: tuple[dict[str, Any], ...]
    tool_steps: int

    @property
    def ignored_read_steps(self) -> int:
        return len(self.safe_reads)

    @property
    def state_steps(self) -> int:
        return len(self.state_events)

    @property
    def state_hash(self) -> str:
        return trajectory_prefix_chain_hash(list(self.state_events))

    @property
    def model_prefix_reusable(self) -> bool:
        return False

    @property
    def kv_reuse(self) -> bool:
        return False


def _path_token_within_workspace(token: str, roots: tuple[str, ...]) -> bool:
    if not token or token.startswith("-"):
        return True
    candidates = [token]
    if "=" in token:
        candidates.append(token.split("=", 1)[1])
    for candidate in candidates:
        if not candidate.startswith("/"):
            parts = PurePosixPath(candidate).parts
            if ".." in parts:
                return False
            continue
        if not any(candidate == root or candidate.startswith(root.rstrip("/") + "/") for root in roots):
            return False
    return True


def _strip_safe_devnull_redirections(command: str) -> str | None:
    """Remove only stderr/stdout redirections proven to target ``/dev/null``.

    Any other redirection remains a hard barrier. The replacement happens before
    shell tokenization so common forms such as ``2>/dev/null`` are accepted while
    ``> /tmp/out`` and input redirection remain rejected.
    """
    normalized = re.sub(
        r"(?<!\S)(?:[012])?>\s*/dev/null(?=\s|[;&|]|$)",
        " ",
        command,
    )
    if any(ch in normalized for ch in (">", "<")):
        return None
    return normalized


def _safe_shell_segment(
    segment: list[str],
    *,
    roots: tuple[str, ...],
    current_root: str | None,
) -> bool:
    if not segment:
        return False
    program = PurePosixPath(segment[0]).name
    args = segment[1:]
    if program not in _SAFE_SHELL_PROGRAMS:
        return False

    # Explicitly reject options/actions that can mutate or execute arbitrary code.
    if program == "tail" and any(arg == "-f" or arg.startswith("--follow") for arg in args):
        return False
    if program == "rg" and any(arg == "--pre" or arg.startswith("--pre=") for arg in args):
        return False
    if program == "sed":
        if any(arg == "-i" or arg.startswith("-i") or arg.startswith("--in-place") for arg in args):
            return False
        # Restrict sed to simple print-only range scripts used for source inspection.
        non_options = [arg for arg in args if not arg.startswith("-")]
        if not non_options or not re.fullmatch(r"[0-9$]+(?:,[0-9$]+)?p", non_options[0]):
            return False
    if program == "find":
        forbidden = {"-delete", "-exec", "-execdir", "-ok", "-okdir", "-fls", "-fprint", "-fprintf"}
        if any(arg in forbidden for arg in args):
            return False
    if program == "echo" and any(arg in {"-e", "-E"} for arg in args):
        # Keep echo as a literal stream producer; escape interpretation is not
        # needed for repository inspection and complicates proof semantics.
        return False

    # Absolute operands must stay inside explicitly trusted read roots. Relative
    # tokens are safe only if they cannot traverse upward; they may be grep
    # patterns, sed scripts, stream-filter arguments, or paths under the current
    # workspace cwd.
    for arg in args:
        if arg.startswith("-"):
            continue
        if arg.startswith("/") and not _path_token_within_workspace(arg, roots):
            return False
        if not arg.startswith("/") and ".." in PurePosixPath(arg).parts:
            return False

    # File-oriented commands with only relative operands require a known trusted
    # cwd. Stream filters (sort/uniq/tr/echo) do not.
    file_oriented = {"ls", "cat", "head", "tail", "wc", "grep", "rg", "stat", "file", "sed", "find"}
    has_absolute = any(arg.startswith("/") for arg in args)
    non_option_args = [arg for arg in args if not arg.startswith("-")]
    if program in file_oriented and not has_absolute and current_root is None:
        # A few readers can consume stdin without naming any file. grep/rg use one
        # non-option token as the pattern; sed uses one print-only script. Other
        # file-oriented readers remain scoped to an explicit trusted cwd/path.
        stdin_only = (
            (program in {"cat", "head", "tail", "wc"} and not non_option_args)
            or (program in {"grep", "rg"} and len(non_option_args) <= 1)
            or (program == "sed" and len(non_option_args) == 1)
        )
        if not stdin_only:
            return False
    return True


def is_proven_workspace_read_shell(
    command: str,
    *,
    workspace_roots: tuple[str, ...] = DEFAULT_WORKSPACE_ROOTS,
    working_dir: str | None = None,
) -> bool:
    """Recognize a conservative read-only shell grammar.

    The grammar permits ``;``, ``&&`` and ``|`` composition only when *every*
    component command is independently proven read-only. Arbitrary scripts,
    backgrounding, ``||``, command substitution, variables, write redirection,
    mutating ``find`` actions, in-place ``sed``, and unknown programs remain hard
    barriers. This expands practical coverage without treating generic shell as
    safe.
    """
    command = str(command or "").strip()
    roots = tuple(str(root).rstrip("/") or "/" for root in workspace_roots)
    if not command or not roots:
        return False
    if "\n" in command or "`" in command or "$(" in command or "$" in command:
        return False
    command = _strip_safe_devnull_redirections(command)
    if command is None:
        return False
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|")
        lexer.whitespace_split = True
        lexer.commenters = ""
        tokens = list(lexer)
    except ValueError:
        return False
    if not tokens:
        return False

    segments: list[list[str]] = [[]]
    separators: list[str] = []
    for token in tokens:
        if token in {";", "&&", "|"}:
            if not segments[-1]:
                return False
            separators.append(token)
            segments.append([])
            continue
        if token in {"&", "||"}:
            return False
        segments[-1].append(token)
    if not segments[-1]:
        return False

    current_root: str | None = None
    if working_dir:
        normalized_working_dir = str(working_dir).rstrip("/") or "/"
        current_root = next(
            (
                root for root in roots
                if normalized_working_dir == root
                or normalized_working_dir.startswith(root.rstrip("/") + "/")
            ),
            None,
        )
        if current_root is None:
            return False
    for index, segment in enumerate(segments):
        program = PurePosixPath(segment[0]).name if segment else ""
        if program == "cd":
            # cd changes only shell cwd and is safe here, but never inside a pipe.
            if len(segment) != 2 or (index > 0 and separators[index - 1] == "|"):
                return False
            target = segment[1]
            if not _path_token_within_workspace(target, roots):
                return False
            if target.startswith("/"):
                current_root = next(
                    (root for root in roots if target == root or target.startswith(root.rstrip("/") + "/")),
                    None,
                )
            else:
                current_root = roots[0]
            if current_root is None:
                return False
            continue
        if not _safe_shell_segment(segment, roots=roots, current_root=current_root):
            return False
    return True


def classify_tool_effect(
    tool_name: str,
    tool_args: dict[str, Any] | None,
    *,
    allow_safe_shell: bool = False,
    workspace_roots: tuple[str, ...] = DEFAULT_WORKSPACE_ROOTS,
) -> str:
    """Return a deliberately narrow state-effect class for relaxed matching."""

    name = str(tool_name or "").strip()
    args = tool_args or {}
    if name == "grep_files":
        return SAFE_READ
    if name == "text_editor":
        command = str(args.get("command") or "").strip().lower()
        if command in {"view", "read"}:
            return SAFE_READ
        if command in {"write", "create", "replace", "str_replace", "insert", "delete", "patch"}:
            return MUTATION
        return BARRIER
    if name == "shell" and allow_safe_shell:
        if is_proven_workspace_read_shell(
            str(args.get("command") or ""),
            workspace_roots=workspace_roots,
            working_dir=(str(args.get("working_dir")) if args.get("working_dir") else None),
        ):
            return SAFE_READ
    return BARRIER


def _content_digest(content: Any) -> str:
    if isinstance(content, str):
        payload = content
    else:
        payload = canonical_prefix(content)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _tool_call_payload(call: Any) -> tuple[str, str, dict[str, Any]]:
    """Return ``(call_id, tool_name, arguments)`` from an OpenAI-style tool call.

    LiteLLM may hand the conversation either plain dictionaries or pydantic-like
    objects. Reusing ``canonical_prefix`` gives us one deterministic conversion
    path for both without importing provider-specific message classes.
    """
    try:
        value = json.loads(canonical_prefix(call))
    except Exception:
        return "", "__unknown_tool__", {}
    if not isinstance(value, dict):
        return "", "__unknown_tool__", {}
    call_id = str(value.get("id") or "")
    function = value.get("function") or {}
    if not isinstance(function, dict):
        function = {}
    name = str(function.get("name") or value.get("name") or "__unknown_tool__")
    raw_args = function.get("arguments", value.get("arguments", {}))
    if isinstance(raw_args, dict):
        args = raw_args
    elif isinstance(raw_args, str):
        try:
            parsed = json.loads(raw_args)
            args = parsed if isinstance(parsed, dict) else {"__raw_arguments__": raw_args}
        except json.JSONDecodeError:
            args = {"__raw_arguments__": raw_args}
    else:
        args = {"__raw_arguments__": canonical_prefix(raw_args)}
    return call_id, name, args


def normalize_tool_result_messages(messages: Iterable[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    """Yield canonical tool-result records from either saved or model-facing history.

    Saved trajectories already contain ``role=tool_result`` with tool metadata.
    Exact checkpoints instead store OpenAI model-facing messages, where tool name
    and arguments live on the preceding assistant ``tool_calls`` entry and the
    observation is a later ``role=tool`` message. Relaxed matching must understand
    both forms or it would silently ignore real environment actions.
    """
    pending: dict[str, tuple[str, dict[str, Any]]] = {}
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "")
        if role == "assistant":
            for call in message.get("tool_calls") or []:
                call_id, name, args = _tool_call_payload(call)
                if call_id:
                    pending[call_id] = (name, args)
            continue
        if role == "tool_result":
            yield message
            continue
        if role != "tool":
            continue
        call_id = str(message.get("tool_call_id") or "")
        tool_name, tool_args = pending.pop(call_id, ("__unknown_tool__", {}))
        content = message.get("content", "")
        success = not (isinstance(content, str) and content.lstrip().startswith("Error:"))
        yield {
            "role": "tool_result",
            "tool_name": tool_name,
            "tool_args": tool_args,
            "content": content,
            "success": success,
        }


def canonical_tool_event(
    message: dict[str, Any],
    *,
    allow_safe_shell: bool = False,
    workspace_roots: tuple[str, ...] = DEFAULT_WORKSPACE_ROOTS,
) -> dict[str, Any]:
    """Canonical environment event from one serialized tool-result message."""

    if message.get("role") != "tool_result":
        raise ValueError("message is not a tool_result")
    tool_name = str(message.get("tool_name") or "")
    tool_args = message.get("tool_args") or {}
    if not isinstance(tool_args, dict):
        raise TypeError("tool_args must be a mapping")
    effect = classify_tool_effect(
        tool_name, tool_args, allow_safe_shell=allow_safe_shell, workspace_roots=workspace_roots
    )
    event: dict[str, Any] = {
        "tool_name": tool_name,
        "tool_args": tool_args,
        "effect": effect,
        "success": bool(message.get("success", True)),
    }
    # For state-changing/unknown events, include the observed outcome. This keeps
    # failed edits, nondeterministic shell output, and other divergent outcomes
    # from collapsing to the same relaxed state key.
    if effect != SAFE_READ:
        event["result_digest"] = _content_digest(message.get("content", ""))
    return json.loads(canonical_prefix(event))


def project_environment_prefix(
    messages: Iterable[dict[str, Any]],
    *,
    allow_safe_shell: bool = False,
    workspace_roots: tuple[str, ...] = DEFAULT_WORKSPACE_ROOTS,
) -> RelaxedProjection:
    state_events: list[dict[str, Any]] = []
    safe_reads: list[dict[str, Any]] = []
    tool_steps = 0
    for message in normalize_tool_result_messages(messages):
        tool_steps += 1
        event = canonical_tool_event(
            message, allow_safe_shell=allow_safe_shell, workspace_roots=workspace_roots
        )
        if event["effect"] == SAFE_READ:
            safe_reads.append(event)
        else:
            state_events.append(event)
    return RelaxedProjection(
        state_events=tuple(state_events),
        safe_reads=tuple(safe_reads),
        tool_steps=tool_steps,
    )


def read_result_cache_key(
    state_hash: str,
    tool_name: str,
    tool_args: dict[str, Any] | None,
    *,
    allow_safe_shell: bool = False,
    workspace_roots: tuple[str, ...] = DEFAULT_WORKSPACE_ROOTS,
) -> str:
    """Content-address one read result under the exact environment state."""

    if len(state_hash) != 64:
        raise ValueError("state_hash must be a SHA-256 hex digest")
    try:
        bytes.fromhex(state_hash)
    except ValueError as exc:
        raise ValueError("state_hash must be hexadecimal") from exc
    effect = classify_tool_effect(
        tool_name, tool_args, allow_safe_shell=allow_safe_shell, workspace_roots=workspace_roots
    )
    if effect != SAFE_READ:
        raise ValueError("read-result cache accepts only proven safe reads")
    payload = canonical_prefix({
        "state_hash": state_hash,
        "tool_name": tool_name,
        "tool_args": tool_args or {},
    })
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def external_barrier_hash(
    messages: Iterable[dict[str, Any]],
    *,
    allow_safe_shell: bool = False,
    workspace_roots: tuple[str, ...] = DEFAULT_WORKSPACE_ROOTS,
) -> str:
    """Hash only events whose side effects are outside trusted file mutations."""
    barriers: list[dict[str, Any]] = []
    for message in normalize_tool_result_messages(messages):
        event = canonical_tool_event(
            message, allow_safe_shell=allow_safe_shell, workspace_roots=workspace_roots
        )
        if event["effect"] == BARRIER:
            barriers.append(event)
    return trajectory_prefix_chain_hash(barriers)


def workspace_convergence_key(
    *,
    env_fingerprint: str,
    workspace_digest: str,
    messages: Iterable[dict[str, Any]],
    allow_safe_shell: bool = False,
    workspace_roots: tuple[str, ...] = DEFAULT_WORKSPACE_ROOTS,
) -> str:
    """Exact convergence key for relaxed file-mutation matching.

    Different structured file-edit histories may converge only when the final
    Git workspace digest is identical and every untrusted/external side-effect
    barrier has the same canonical history/outcome. This key is environment-only;
    it never implies model-history or KV-cache equivalence.
    """
    if not env_fingerprint:
        raise ValueError("env_fingerprint must be non-empty")
    if len(workspace_digest) != 64:
        raise ValueError("workspace_digest must be a SHA-256 hex digest")
    try:
        bytes.fromhex(workspace_digest)
    except ValueError as exc:
        raise ValueError("workspace_digest must be hexadecimal") from exc
    payload = canonical_prefix({
        "env_fingerprint": env_fingerprint,
        "workspace_digest": workspace_digest,
        "external_barrier_hash": external_barrier_hash(
            messages, allow_safe_shell=allow_safe_shell, workspace_roots=workspace_roots
        ),
        "model_prefix_reusable": False,
        "kv_reuse": False,
    })
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
