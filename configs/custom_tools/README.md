<!-- Documents the default manifest directory consumed by
swebench/agent/custom_tools.py load_custom_tools() (DEFAULT_MANIFEST_DIR), reached via
agent/tools.py build_panel(). User instruction: manifest 用户提供，目录作为参数传进去 +
有一个默认位置. -->

# Custom Tool Manifests

Two places can declare one, and they load together:

- **In the panel manifest**, under `custom_tools:` — one file then describes the whole
  surface a model is offered, views and binaries alike. See docs/TOOL_PANEL.md.
- **Here**, one tool per file — this directory is the default location, or pass
  `--custom-tools-dir <dir>`. Use it for a set shared across panels.

A name declared in both is an error rather than a silent overwrite: a registry keys by
name, so whichever loaded second would have won quietly.

Either way the shape is the same — one agent-facing tool backed by a static binary:

```yaml
name: code_complexity                 # tool name the agent sees
description: Analyze cyclomatic complexity of a source file
binary:
  # Source 1: remote static binary (CGO_ENABLED=0 / musl), downloaded once
  url: https://example.com/analyzer-linux-amd64
  sha256: "0c6207dc05ef13183cbb422a32682328deb95472d1b56c6275529c4fd6cc8d83"
                                      # optional but recommended — when set, content
                                      # is verified before execution; when omitted,
                                      # the download is trusted as-is (cached per URL)
  # Source 2 (alternative): binary already baked into the sandbox image
  # path: /opt/tools/analyzer          # absolute path; mutually exclusive with url
parameters:
  file:      {type: string, required: true, map: {positional: 0}}
  threshold: {type: integer, default: 10, map: {flag: "--threshold"}}
  verbose:   {type: boolean, map: {flag: "--verbose"}}
timeout: 30                           # seconds, max 600
```

A boolean behind a flag is a switch: `verbose: true` emits `--verbose` and
`false` emits nothing. Add `style: value` for the rare tool that wants
`--flag true` instead. Anything else in `style` is rejected when the manifest
is parsed, rather than becoming a stray argument at run time.

How it executes: url-sourced binaries are downloaded once (cached by
sha256, hash-verified before install); path-sourced binaries run directly
from the image. Either way, arguments are compiled into discrete argv
slots — agent input is never interpreted by the shell.

Manifests here are loaded on every run; the directory being absent or empty is fine.

They land in the registry of the `AshSession` doing the run, not in a process-wide one,
so two configurations in one process do not see each other's tools. The agent loop asks
its panel what is custom and the session executor dispatches it, and both read that one
registry -- otherwise the loop could wave a tool through that the executor cannot expand.
