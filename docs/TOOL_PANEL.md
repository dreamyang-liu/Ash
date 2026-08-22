# The tool panel is compiled, not written

The set of tools a model sees is a build product: the runtime declares what exists,
manifests declare what to offer and how, and a compiler produces the panel. Nothing
about the panel is hand-maintained, because a hand-maintained copy drifts and this
one had.

## Why

Two places described tools and one of them was written by hand:

| | source | answers |
|---|---|---|
| runtime (Go) | `runtime/tools/*.go` `Schema()` | authoritative — it executes the calls |
| agent panel | `agent/tools.py` `TOOLS_SCHEMA` | **hand-written Python literal** |

`Sandbox.tool_schemas()` is not a third copy and is not replaced by this. It answers
"what can this sandbox do" -- every runtime tool plus the registry's custom ones,
with no policy applied, `artifact` included. The compiled panel answers "what should
this model be offered", which is a narrower thing on purpose. Both exist and neither
is a stale copy of the other.

Measured drift on 7 tools, 4 of them wrong:

```
shell            ✓
text_editor      ✗  panel missing: max_output_bytes, truncate_mode
grep_files       ✗  panel missing: max_output_bytes, truncate_mode
process          ✓
web_fetch        ✗  panel has max_length, which the runtime does not accept
web_search       ✓
wait_for_events  ✗  panel missing: include_own
```

Both directions are defects, and they fail differently.

**A parameter the panel omits is a capability the model cannot reach.** Some of
these omissions are deliberate — `truncate_mode` and `max_output_bytes` let a model
raise its own output budget, which would go around `TruncateInterceptor` — but
nothing recorded that, so a reader cannot tell a decision from an oversight. Under a
compiler an omission has to be declared, which turns it into a decision on the
record.

**A parameter the panel invents is worse, because it fails silently.** `web_fetch`
offers `max_length`, a name the runtime dropped in favour of `max_output_bytes`.
Verified against a live runtime: asking for `max_length: 50` returns 168 characters,
`isError: false`. The model asked for a bound, believes it got one, and did not. No
test caught this because the contract test compared `TOOLS_SCHEMA` against
`EXPECTED_TOOL_NAMES` — its own second copy of the same list.

## The model

```
        runtime Schema()                    ← the base draft: what exists
                │
                ▼
        manifests (data)                    ← what to offer, and how
          ├── agent tool:  maps onto a runtime tool
          │     rename, argument mapping, description, parameter subset
          └── custom tool: an external binary
                argv slots (positional / flag / switch), no shell templating
                │
                ▼
        compile  ───►  the panel the model sees
```

An **agent tool** is a view of a runtime tool. It may rename it, hide parameters,
rename or remap arguments, and give a description written for the task rather than
for the tool. `bash_only` mode is one such view: a single tool with only `command`.

A **custom tool** is not a view of anything — it is an external binary, expanded by
the SDK into `artifact` (fetch + verify) then `shell`. Its parameters compile into
discrete argv slots and are never interpreted by a shell.

## Rules the compiler enforces

1. **Every agent tool names a runtime tool that exists.** Checked against the
   runtime's own declaration, so a typo or a dropped tool fails at compile time
   rather than mid-run. This replaces the routing table: no identity entries, no
   empty indirection layer — a mapping exists only where something is actually
   mapped.

2. **Every exposed parameter exists on the target runtime tool.** This is the
   `max_length` class of defect, and it is the reason to compile at all.

3. **Hiding a parameter is allowed and must be stated.** A panel narrower than the
   runtime is a decision (`truncate_mode` belongs to the interceptor, not the
   model); an accidental omission is a bug. Declaring it separates them.

4. **A custom tool may not collide with an agent tool name.** Already enforced by
   `ToolRegistry.register`; carried over.

## What this replaces

- `TOOLS_SCHEMA` / `BASH_ONLY_SCHEMA` in `agent/tools.py` — hand-written literals
- `BUILTIN_ROUTES` in `sdk/ash_sandbox/toolset.py` — seven identity entries and one
  real mapping (`bash` → `shell`), which existed so an interceptor keyed on `shell`
  would not go blind in `bash_only` mode. With `bash_only` expressed as an agent
  tool over `shell`, the mapping is data and the table is not needed.

  What the table was really doing was rejecting names the runtime does not serve, and
  the runtime already knows which those are: `Sandbox.execute_tool_call` now checks
  against its declaration. That also fixed a case the table got wrong — `artifact`
  was missing from it, so calling the runtime's own `artifact` tool through dispatch
  was rejected as unknown. `ToolRegistry(aliases=…)` survives for genuine renames and
  is empty by default; an empty declaration is treated as unverifiable rather than as
  a runtime with no tools, since refusing every call would be worse than the table.

Compile-time validation against the runtime is the whole point, so it needs a
runtime to validate against — either a running one or its declared schema. Where
neither is available the panel cannot be built, and that has to be a loud failure:
a governance or capability surface that silently falls back to a stale copy is how
this drifted in the first place.
