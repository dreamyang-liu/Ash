# Custom Tools: Registry Ownership Moves to the Harness

**Date:** 2026-08-12
**Status:** Approved
**Follow-up to:** PR #19 (custom tools, agent identity, event delivery)

## Problem

`swebench/agent/custom_tools.py` is a compatibility shim left over from
moving the tool data layer into the SDK (`ash_sandbox.toolset`). It holds a
module-level `DEFAULT_REGISTRY = ToolRegistry()` singleton plus module-level
functions (`load_custom_tools`, `custom_agent_schemas`, `plan_custom_tool`,
`CUSTOM_TOOL_SPECS`) that all implicitly mutate or read that singleton.

Consequences:

- **Invisible data flow.** `runner.py` calls `load_custom_tools(...)` purely
  for its side effect; the agent dispatch code hundreds of lines away reaches
  the same singleton via import. Loading and use are connected only by hidden
  global state.
- **Process-wide pollution.** All agents in one process share one registry.
  Per-task or per-agent tool panels are impossible. It only works today
  because just one code path (runner.py, single-agent) loads custom tools.
- **Direction mismatch.** Prior decision: the SDK owns the mechanism
  (`ToolRegistry`, manifest parsing, schema generation, execution planning);
  the harness owns the policy (which manifests, default directory, per-run
  loading). The singleton keeps policy stuck in a pseudo-SDK shim.

Current global-state touch points (exhaustive, verified by grep):

1. `swebench/runner.py:268-270` — `load_custom_tools(config.custom_tools_dir)`
   then `custom_agent_schemas()` appended to `TOOLS_SCHEMA`.
2. `swebench/agent/__init__.py:86-88` — `plan_custom_tool(name, args)` during
   tool dispatch.
3. `swebench/agent/tools.py:51-53` — `is_custom_tool(name)` reads
   `CUSTOM_TOOL_SPECS`.
4. Tests: `swebench/tests/test_custom_tools.py`,
   `swebench/tests/test_custom_tool_dispatch.py`.

Of the four harnesses, only the runner.py path (litellm single-agent) loads
custom tools. `best_of_n`, `manager_worker`, and `claude_code` use the fixed
`TOOLS_SCHEMA` only.

## Design

Replace the hidden singleton with an explicitly passed `ToolRegistry`
instance. Delete the shim.

### 1. `AshAgent` accepts a registry

`AshAgent.__init__` gains `registry: ToolRegistry | None = None`; `None`
means an empty `ToolRegistry()`. Dispatch changes:

- `is_custom_tool(name)` becomes an instance check:
  `name in self.registry.custom_specs`.
- `plan_custom_tool(name, args)` becomes
  `self.registry.plan_custom_tool(name, args)`.

Harnesses that never use custom tools pass nothing and behave identically
(empty registry → no name ever matches).

### 2. Harness-level loader helper

New function in `swebench/agent/tools.py`:

```python
def build_tool_registry(custom_tools_dir: str | Path | None = None) -> ToolRegistry
```

Owns the default-directory policy previously in the shim:

- explicit directory: must exist, else `ManifestError` (a typo in a
  user-passed path is an error)
- `None`: load `<repo>/configs/custom_tools` if present, silently skip if
  absent (opt-in feature)

Returns a fresh `ToolRegistry` per call — no shared state.

### 3. Runner wiring

`runner.py` (default tools mode):

```python
registry = build_tool_registry(getattr(config, "custom_tools_dir", None))
agent = AshAgent(config, ..., registry=registry)
agent.set_tools_schema(TOOLS_SCHEMA + registry.custom_agent_schemas())
```

### 4. Delete the shim

`swebench/agent/custom_tools.py` is removed. All former importers move to
`ash_sandbox.toolset` (types) or `swebench.agent.tools.build_tool_registry`
(loading policy). This repo has no external consumers of the shim.

### 5. Out of scope

- SDK (`sdk/ash_sandbox/`) is untouched — its boundary is already correct.
- Manifest schema and `configs/custom_tools/README.md` semantics unchanged
  (README's pointer to the loader module is updated to the new location).
- The other three harnesses gain no custom-tools support; they just keep
  working via the empty-registry default.

## Error Handling

- Missing explicit `--custom-tools-dir`: `ManifestError` at run start
  (unchanged semantics, now raised from `build_tool_registry`).
- Malformed manifest: `ManifestError` from SDK parsing, propagates at load
  time — before any LLM call.
- Unknown/invalid custom tool args at dispatch: unchanged — registry raises,
  agent converts to a routing-error `ToolResult`.

## Testing

- Rewrite `test_custom_tools.py` / `test_custom_tool_dispatch.py` against
  instance-based APIs: `build_tool_registry` (default-dir semantics: explicit
  missing dir errors, absent default dir yields empty registry) and
  `AshAgent` with an injected registry (custom tool dispatch, empty-registry
  fallthrough).
- No global-state cleanup needed in tests anymore — each test builds its own
  registry.
- Existing SDK tests (`sdk/tests/test_toolset.py`) already cover manifest
  parsing and planning; not duplicated.

## Migration Notes

Single PR, no compatibility window: move consumers, delete shim, rewrite
tests. Grep for `custom_tools` must show no remaining imports of the deleted
module.
