<!-- Documents the default manifest directory consumed by
swebench/agent/custom_tools.py load_custom_tools() (DEFAULT_MANIFEST_DIR).
User instruction: manifest 用户提供，目录作为参数传进去 + 有一个默认位置. -->

# Custom Tool Manifests

Drop `*.yaml` / `*.json` manifests here (the default location) or pass
`--custom-tools-dir <dir>` to load from elsewhere. Each manifest defines
one agent-facing tool backed by a static binary:

```yaml
name: code_complexity                 # tool name the agent sees
description: Analyze cyclomatic complexity of a source file
binary:
  url: https://example.com/analyzer-linux-amd64   # static binary (CGO_ENABLED=0 / musl)
  sha256: "ab34..."                   # REQUIRED — content is verified before execution
parameters:
  file:      {type: string, required: true, map: {positional: 0}}
  threshold: {type: integer, default: 10, map: {flag: "--threshold"}}
  verbose:   {type: boolean, map: {flag: "--verbose", style: switch}}
timeout: 30                           # seconds, max 600
```

How it executes: the runtime downloads the binary once (cached by sha256,
hash-verified before install), then runs it with arguments compiled into
discrete argv slots — agent input is never interpreted by the shell.

Manifests in this directory are loaded automatically when the run uses the
default tools mode; the directory being absent or empty is fine.
