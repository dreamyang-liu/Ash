"""SWE-rebench V2 grading on an isolated restored snapshot.

The parser file is supplied by the deployment and pinned by its SHA-256.
Task data is already checksum-validated by runstore.grading.dataset_rows.
"""

import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shlex

PATH = ("/opt/miniconda3/envs/testbed/bin:/opt/miniconda3/bin:"
        "/opt/conda/envs/testbed/bin:/opt/conda/bin:/usr/local/cargo/bin:"
        "/usr/local/go/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin")


def verify_guest_file(session, path, content):
    """A missing/corrupted verifier input is infrastructure failure, not reward 0."""
    from swebench_pro.grade import checked

    digest = hashlib.sha256(content.encode()).hexdigest()
    checked(session, "printf '%s\\n' " + shlex.quote(f"{digest}  {path}") + " | sha256sum -c -")


def load_parsers(spec):
    path = Path(spec.parser_path or "")
    if not path.is_file() or "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest() != spec.grader_revision:
        raise ValueError("SWE-rebench parser checksum differs from grader_revision")
    module_spec = importlib.util.spec_from_file_location("ash_swerebench_parsers", path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def resolved(status_map, task):
    expected = list(task.get("FAIL_TO_PASS") or []) + list(task.get("PASS_TO_PASS") or [])
    def normalize(name):
        for pattern in (
            r"\s*\[\s*\d+(?:\.\d+)?\s*(?:ms|s)\s*\]\s*$",
            r"\s+in\s+\d+(?:\.\d+)?\s+(?:msec|sec)\b",
            r"\s*\(\s*\d+(?:\.\d+)?\s*(?:ms|s)\s*\)\s*$",
        ):
            name = re.sub(pattern, "", name, flags=re.IGNORECASE)
        return name.strip()
    actual = {normalize(name): value for name, value in status_map.items()}
    return bool(expected) and all(actual.get(normalize(name)) == "PASSED" for name in expected)


def grade_snapshot(spec, row, backend, directory, *, session_factory):
    from swebench_pro.grade import checked, shell

    parsers = load_parsers(spec)
    config = row["install_config"]
    if isinstance(config, str):
        config = json.loads(config)
    parser = parsers.NAME_TO_PARSER.get(config["log_parser"]) or getattr(parsers, config["log_parser"], None)
    if parser is None:
        raise ValueError("Unknown SWE-rebench log parser")
    commands = config["test_cmd"]
    if isinstance(commands, str):
        commands = [commands]
    if not commands or any(not isinstance(command, str) or not command.strip() for command in commands):
        raise ValueError("SWE-rebench test_cmd must contain commands")
    if not row["test_patch"].strip() or not row["base_commit"]:
        raise ValueError("SWE-rebench needs test_patch and base_commit")
    workdir = "/" + row["repo"].split("/", 1)[1]
    patch_b64 = base64.b64encode(row["test_patch"].encode()).decode()
    session = session_factory(quiet=True, backend=backend)
    try:
        if not session.create(spec.snapshot_id, spec.resources):
            raise RuntimeError("Could not restore SWE-rebench grading snapshot")
        prefix = f"cd {shlex.quote(workdir)} && "
        checked(session, f"printf %s {shlex.quote(patch_b64)} | base64 -d > /tmp/ash-test.patch")
        reset = (
            f"base={shlex.quote(row['base_commit'])}; "
            "git apply --numstat /tmp/ash-test.patch | cut -f3- | while IFS= read -r path; do "
            'path=${path#\\"}; path=${path%\\"}; '
            'if git cat-file -e "$base:$path" 2>/dev/null; '
            'then git -c core.hooksPath=/dev/null checkout "$base" -- "$path"; '
            'else rm -f -- "$path"; fi; done'
        )
        checked(session, prefix + "bash -euo pipefail -c " + shlex.quote(reset))
        verify_guest_file(session, "/tmp/ash-test.patch", row["test_patch"])
        apply = ("git apply -v --3way --recount --ignore-space-change --whitespace=nowarn /tmp/ash-test.patch"
                 " || patch --fuzz=5 -p1 -i /tmp/ash-test.patch")
        _, stderr, code = shell(session, prefix + apply, 120)
        if code:
            return {"status": "completed", "resolved": False, "reason": "test_patch_apply_failed",
                    "error_output": stderr, "grader_revision": spec.grader_revision}
        script = ("set -uo pipefail\nexport PATH=" + shlex.quote(PATH) + "\n"
                  "export CI=1 PAGER=cat MANPAGER=cat PIP_PROGRESS_BAR=off TQDM_DISABLE=1\n"
                  "export _JAVA_OPTIONS=-Djava.net.preferIPv6Addresses=false\nFAIL=0\n"
                  + "\n".join(f"{command} || FAIL=1" for command in commands) + '\nexit "$FAIL"\n')
        encoded = base64.b64encode(script.encode()).decode()
        checked(session, f"printf %s {shlex.quote(encoded)} | base64 -d > /tmp/ash-eval.sh")
        verify_guest_file(session, "/tmp/ash-eval.sh", script)
        stdout, stderr, code = shell(session, prefix + "bash /tmp/ash-eval.sh 2>&1", int(spec.timeout_s))
        log = directory / "swerebench-tests.log"
        log.write_text(stdout + stderr)
        if code not in (0, 1):
            raise RuntimeError(f"SWE-rebench test runner failed with exit code {code}")
        status_map = parser(stdout) or {}
        return {"status": "completed", "resolved": resolved(status_map, row),
                "test_statuses": status_map, "log": str(log), "grader_revision": spec.grader_revision}
    finally:
        session.destroy()
