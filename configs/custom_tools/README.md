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
  # Source 1: remote static binary (CGO_ENABLED=0 / musl), downloaded once
  url: https://example.com/analyzer-linux-amd64
  sha256: "ab34..."                   # optional but recommended — when set, content
                                      # is verified before execution; when omitted,
                                      # the download is trusted as-is (cached per URL)
  # Source 2 (alternative): binary already baked into the sandbox image
  # path: /opt/tools/analyzer          # absolute path; mutually exclusive with url
parameters:
  file:      {type: string, required: true, map: {positional: 0}}
  threshold: {type: integer, default: 10, map: {flag: "--threshold"}}
  verbose:   {type: boolean, map: {flag: "--verbose", style: switch}}
timeout: 30                           # seconds, max 600
```

How it executes: url-sourced binaries are downloaded once (cached by
sha256, hash-verified before install); path-sourced binaries run directly
from the image. Either way, arguments are compiled into discrete argv
slots — agent input is never interpreted by the shell.

Manifests in this directory are loaded automatically when the run uses the
default tools mode; the directory being absent or empty is fine.
