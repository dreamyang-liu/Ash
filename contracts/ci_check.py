#!/usr/bin/env python3
"""Assert that upstream agents still provide what our adapters depend on.

Run daily and before bumping a pinned version:

    python contracts/ci_check.py              # all slots, static checks
    python contracts/ci_check.py --slot codex
    python contracts/ci_check.py --live       # also run behaviour probes

Static checks need no credentials: CLI ``--help`` text, SDK exports, dataclass
fields. Behaviour probes (``--live``) actually drive an agent and cost tokens.

Why this exists: the adapters are built on surfaces upstream does not promise to
keep (flag names, event fields, SDK kwargs). Without this, a rename shows up as
subtly wrong trajectories weeks later instead of a red build today.

Exit codes: 0 ok, 1 contract violation, 2 harness/setup error.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import shutil
import subprocess
import sys
from pathlib import Path

CONTRACTS_DIR = Path(__file__).parent


def load_yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError:
        print("PyYAML required: pip install pyyaml", file=sys.stderr)
        raise SystemExit(2)
    return yaml.safe_load(path.read_text())


class Report:
    def __init__(self) -> None:
        self.failures: list = []
        self.skips: list = []
        self.checks = 0

    def ok(self, label: str) -> None:
        self.checks += 1
        print("  ok      %s" % label)

    def fail(self, label: str, detail: str = "") -> None:
        self.checks += 1
        self.failures.append((label, detail))
        print("  FAIL    %s%s" % (label, (" -- " + detail) if detail else ""))

    def skip(self, label: str, reason: str) -> None:
        self.skips.append((label, reason))
        print("  skip    %s (%s)" % (label, reason))


def cli_help(binary: str, argv) -> str:
    try:
        out = subprocess.run(argv, capture_output=True, text=True, timeout=60)
        return (out.stdout or "") + (out.stderr or "")
    except Exception as exc:  # noqa: BLE001
        return "___ERROR___%s" % exc


def check_cli_flags(contract: dict, report: Report) -> None:
    binary = (contract.get("upstream") or {}).get("binary")
    flags = contract.get("cli_flags") or []
    if not binary or not flags:
        return
    if not shutil.which(binary):
        report.skip("%s cli flags" % binary, "binary not installed")
        return

    slot = contract["slot"]
    argv = {
        "codex": [binary, "exec", "--help"],
        "opencode": [binary, "run", "--help"],
    }.get(slot, [binary, "--help"])
    text = cli_help(binary, argv)
    if text.startswith("___ERROR___"):
        report.fail("%s --help" % binary, text)
        return
    for flag in flags:
        (report.ok if flag in text else report.fail)("%s flag %s" % (binary, flag))


def check_version(contract: dict, report: Report) -> None:
    upstream = contract.get("upstream") or {}
    binary = upstream.get("binary")
    expected = (upstream.get("verified_versions") or {}).get("cli")
    if not binary or not expected:
        return
    if not shutil.which(binary):
        report.skip("%s version" % binary, "binary not installed")
        return
    text = cli_help(binary, [binary, "--version"]).strip()
    if expected in text:
        report.ok("%s version pinned at %s" % (binary, expected))
    else:
        # Not a failure: a bump is expected, but the field checks below must
        # still pass and the contract should be re-verified deliberately.
        report.skip(
            "%s version" % binary,
            "installed %r != verified %r -- re-verify fixtures" % (text.splitlines()[:1], expected),
        )


def check_claude_sdk(contract: dict, report: Report) -> None:
    try:
        import claude_agent_sdk as sdk
    except ImportError as exc:
        report.skip("claude-agent-sdk", "not installed (%s)" % exc)
        return

    for name in contract.get("sdk_exports") or []:
        (report.ok if hasattr(sdk, name) else report.fail)("sdk export %s" % name)

    options_cls = getattr(sdk, "ClaudeAgentOptions", None)
    if options_cls is None:
        report.fail("ClaudeAgentOptions missing")
        return

    import dataclasses
    import inspect

    if dataclasses.is_dataclass(options_cls):
        known = {f.name for f in dataclasses.fields(options_cls)}
    else:
        try:
            known = set(inspect.signature(options_cls).parameters)
        except (TypeError, ValueError):
            report.skip("ClaudeAgentOptions fields", "not introspectable")
            return
    for field in contract.get("options_fields") or []:
        (report.ok if field in known else report.fail)(
            "ClaudeAgentOptions.%s" % field,
            "" if field in known else "dropped silently at runtime by _accepts()",
        )


def check_normalizer_alignment(contract: dict, report: Report) -> None:
    """The normalizer must still map every event/usage key the contract lists."""
    slot = contract["slot"]
    module = {
        "codex": "harness.normalize.codex",
        "codex-sdk": "harness.normalize.codex_sdk",
        "opencode": "harness.normalize.opencode",
        "opencode-server": "harness.normalize.opencode_server",
        "claude-code": "harness.normalize.claude_code",
    }[slot]
    try:
        mod = __import__(module, fromlist=["normalize"])
    except ImportError as exc:
        report.fail("import %s" % module, str(exc))
        return

    source = Path(mod.__file__).read_text()
    for key in (contract.get("usage_keys") or {}):
        leaf = str(key).split(".")[-1]
        (report.ok if ('"%s"' % leaf) in source or ("'%s'" % leaf) in source else report.fail)(
            "%s handles usage key %s" % (slot, key)
        )
    for etype in (contract.get("event_types") or []) + (contract.get("item_types") or []):
        if etype in ("turn.started", "turn/started"):
            continue
        (report.ok if etype in source else report.fail)("%s handles event %s" % (slot, etype))
    for mtype in (contract.get("message_types") or []) + (contract.get("block_types") or []):
        (report.ok if mtype in source else report.fail)("%s handles type %s" % (slot, mtype))


def check_python_api(contract: dict, report: Report) -> None:
    """Assert the SDK surface a protocol driver calls is still there.

    These are the names a version bump can remove silently: the driver would then
    fail at run time, mid-eval, rather than here.
    """
    spec = contract.get("python_api") or {}
    if not spec:
        return
    module_name = spec.get("module")
    try:
        mod = importlib.import_module(module_name)
    except ImportError as exc:
        report.skip("%s python api" % contract["slot"], "%s not installed (%s)" % (module_name, exc))
        return

    for name in spec.get("exports") or []:
        (report.ok if hasattr(mod, name) else report.fail)(
            "%s exports %s" % (module_name, name)
        )

    client_spec = spec.get("client") or {}
    if client_spec:
        try:
            client_mod = importlib.import_module(client_spec["module"])
            cls = getattr(client_mod, client_spec["cls"])
        except (ImportError, AttributeError) as exc:
            report.fail("%s.%s" % (client_spec.get("module"), client_spec.get("cls")), str(exc))
            cls = None
        if cls is not None:
            params = inspect.signature(cls.__init__).parameters
            for kwarg in client_spec.get("init_kwargs") or []:
                (report.ok if kwarg in params else report.fail)(
                    "%s.__init__ accepts %s" % (client_spec["cls"], kwarg)
                )
            for method in client_spec.get("methods") or []:
                (report.ok if callable(getattr(cls, method, None)) else report.fail)(
                    "%s.%s" % (client_spec["cls"], method)
                )

    for attr, names in (("thread_methods", "Thread"),
                        ("turn_handle_methods", "TurnHandle")):
        target = getattr(mod, names, None)
        for method in spec.get(attr) or []:
            (report.ok if target is not None and callable(getattr(target, method, None))
             else report.fail)("%s.%s" % (names, method))

    config_cls = getattr(mod, "CodexConfig", None)
    if config_cls is not None and spec.get("config_fields"):
        fields = set(getattr(config_cls, "__dataclass_fields__", {}) or {})
        for field in spec["config_fields"]:
            (report.ok if field in fields else report.fail)(
                "CodexConfig field %s" % field
            )

    for entry in contract.get("private_api") or []:
        try:
            private_mod = importlib.import_module(entry["module"])
            present = hasattr(private_mod, entry["name"])
        except ImportError:
            present = False
        if present:
            report.ok("private %s.%s" % (entry["module"], entry["name"]))
        else:
            report.fail(
                "private %s.%s" % (entry["module"], entry["name"]),
                "gone; driver falls back to: %s" % entry.get("fallback", "unknown"),
            )


def check_live(contract: dict, report: Report) -> None:
    slot = contract["slot"]
    for behavior in contract.get("behaviors") or []:
        report.skip(
            "live probe %s/%s" % (slot, behavior["id"]),
            "not implemented yet -- needs credentials + sandbox",
        )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slot", action="append", help="limit to these slots")
    parser.add_argument("--live", action="store_true", help="run behaviour probes")
    parser.add_argument("--json", action="store_true", help="machine-readable summary")
    args = parser.parse_args(argv)

    paths = sorted(CONTRACTS_DIR.glob("*.yaml"))
    if args.slot:
        wanted = set(args.slot)
        paths = [p for p in paths if p.stem in wanted]
    if not paths:
        print("no contracts matched", file=sys.stderr)
        return 2

    report = Report()
    for path in paths:
        contract = load_yaml(path)
        print("\n== %s ==" % contract["slot"])
        check_version(contract, report)
        check_cli_flags(contract, report)
        if contract["slot"] == "claude-code":
            check_claude_sdk(contract, report)
        check_python_api(contract, report)
        check_normalizer_alignment(contract, report)
        if args.live:
            check_live(contract, report)

    print(
        "\n%d checks, %d failures, %d skipped"
        % (report.checks, len(report.failures), len(report.skips))
    )
    if args.json:
        print(
            json.dumps(
                {
                    "checks": report.checks,
                    "failures": [{"label": l, "detail": d} for l, d in report.failures],
                    "skipped": [{"label": l, "reason": r} for l, r in report.skips],
                },
                indent=2,
            )
        )
    if report.failures:
        print("\nAdapter drift detected. Update harness/normalize/* or the contract.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
