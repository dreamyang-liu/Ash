"""Admission of prepared images against their dataset source baseline."""

import json
import re
import shlex


class BaselineAdmissionError(ValueError):
    def __init__(self, message: str, report: dict):
        super().__init__(message)
        self.report = report

PROBE = r'''import json, pathlib, subprocess, sys
def git(*args):
    return subprocess.check_output(["git", *args]).decode().strip()
def tree(revision):
    result = {}
    for entry in subprocess.check_output(["git", "ls-tree", "-rz", revision]).split(b"\0"):
        if entry:
            metadata, path = entry.split(b"\t", 1)
            mode, kind, blob = metadata.decode().split()
            result[path.decode()] = [mode, kind, blob]
    return result
base = sys.argv[1]
original, prepared = tree(base), tree("HEAD")
content = [path for path in sorted(original.keys() & prepared.keys()) if original[path][1:] != prepared[path][1:]]
modes = [{"path":path, "before":original[path][0], "after":prepared[path][0]} for path in sorted(original.keys() & prepared.keys()) if original[path][0] != prepared[path][0]]
report = {"base":base, "head":git("rev-parse", "HEAD"), "tree":git("rev-parse", "HEAD^{tree}"),
          "parents":git("show", "-s", "--format=%P", "HEAD").split(),
          "status":git("--no-optional-locks", "status", "--porcelain=v1"),
          "added":sorted(prepared.keys()-original.keys()), "removed":sorted(original.keys()-prepared.keys()),
          "content_changes":content, "mode_changes":modes}
if content == ["pyproject.toml"]:
    before = subprocess.check_output(["git", "show", base+":pyproject.toml"])
    after = subprocess.check_output(["git", "show", "HEAD:pyproject.toml"])
    report["setuptools_pin_only"] = after == before.replace(b'requires = ["setuptools",', b'requires = ["setuptools==68.0.0",', 1)
if content and set(content).issubset({"setup.py", "tox.ini"}):
    replacements = [
        (b"'sphinxcontrib-applehelp'", b"'sphinxcontrib-applehelp<=1.0.7'"),
        (b"'sphinxcontrib-devhelp'", b"'sphinxcontrib-devhelp<=1.0.5'"),
        (b"'sphinxcontrib-htmlhelp'", b"'sphinxcontrib-htmlhelp<=2.0.4'"),
        (b"'sphinxcontrib-htmlhelp>=2.0.0'", b"'sphinxcontrib-htmlhelp>=2.0.0,<=2.0.4'"),
        (b"'sphinxcontrib-serializinghtml'", b"'sphinxcontrib-serializinghtml<=1.1.9'"),
        (b"'sphinxcontrib-serializinghtml>=1.1.5'", b"'sphinxcontrib-serializinghtml>=1.1.5,<=1.1.9'"),
        (b"'sphinxcontrib-qthelp'", b"'sphinxcontrib-qthelp<=1.0.6'"),
        (b"'Jinja2>=2.3'", b"'Jinja2<3.0'"),
        (b"'alabaster>=0.7,<0.8'", b"'alabaster>=0.7,<0.7.12'"),
        (b"'packaging',", b"'packaging', 'markupsafe<=2.0.1',"),
    ]
    checks = {}
    for name in content:
        before = subprocess.check_output(["git", "show", base+":"+name])
        after = subprocess.check_output(["git", "show", "HEAD:"+name])
        expected = before
        if name == "tox.ini":
            expected = before.replace(b"pytest", b"pytest -rA")
            report["sphinx_py311_url_rewrite"] = b"github.com/pytest -rA-dev/py" in after
        else:
            for old, new in replacements:
                expected = expected.replace(old, new)
        checks[name] = expected == after
    report["sphinx_preparation_checks"] = checks
untracked = [name.decode() for name in subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard", "-z"]).split(b"\0") if name]
report["untracked_paths"] = untracked
if untracked:
    copies = []
    for name in untracked:
        source = name[len("build/lib/"):] if name.startswith("build/lib/") else ""
        candidate = pathlib.Path(name)
        verified = False
        if source in prepared and prepared[source][1] == "blob" and candidate.is_file() and not candidate.is_symlink():
            original_bytes = subprocess.check_output(["git", "cat-file", "blob", prepared[source][2]])
            verified = candidate.read_bytes() == original_bytes
        copies.append({"path": name, "source": source, "identical": verified})
    report["build_copy_checks"] = copies
    report["build_copies_verified"] = all(row["identical"] for row in copies)
print(json.dumps(report))
'''


def validate_baseline(report: dict, instance: dict) -> str:
    expected = instance["base_commit"]
    copies = report.get("build_copy_checks") or []
    verified_build = (instance["instance_id"] == "psf__requests-1142"
                      and report.get("build_copies_verified") is True and bool(copies)
                      and all(row["identical"] and row["path"].startswith("build/lib/") for row in copies)
                      and sorted(row["path"] for row in copies) == sorted(report.get("untracked_paths", []))
                      and report["status"] == "?? build/")
    if (report["base"] != expected or ((report["status"] or report.get("untracked_paths")) and not verified_build)
            or report["added"] or report["removed"]):
        raise ValueError("dirty or structurally different baseline")
    if report["head"] == expected:
        if report["content_changes"] or report["mode_changes"]:
            raise ValueError("inconsistent baseline evidence")
        return "exact-base"
    if report["parents"] != [expected]:
        raise ValueError("prepared HEAD is not a direct child of dataset base")
    if any((change["before"], change["after"]) != ("100644", "100755") for change in report["mode_changes"]):
        raise ValueError("unexpected file type or mode changes in preparation")
    if not report["content_changes"] and not report["mode_changes"]:
        return "content-identical-preparation"
    if not report["content_changes"]:
        return "mode-only-preparation"
    if instance["instance_id"].startswith("sphinx-doc__sphinx-"):
        checks = report.get("sphinx_preparation_checks") or {}
        if (set(checks) == set(report["content_changes"])
                and set(checks).issubset({"setup.py", "tox.ini"}) and all(checks.values())):
            return "known-sphinx-preparation"
        raise ValueError("unexpected Sphinx preparation content")
    if not instance["instance_id"].startswith("astropy__astropy-"):
        raise ValueError("unreviewed image preparation delta")
    if report["content_changes"] != ["pyproject.toml"] or not report.get("setuptools_pin_only"):
        raise ValueError("unexpected content changes in Astropy preparation")
    return "known-astropy-preparation"


def probe_output(result) -> tuple[str, str]:
    outcome = getattr(result, "outcome", None)
    if outcome is not None:
        if outcome.exit_code != 0 or outcome.running or outcome.timed_out or outcome.truncated:
            raise ValueError("baseline probe did not complete: " + str(result.output))
        return outcome.stdout, outcome.stderr
    if not result.success:
        raise ValueError("baseline probe failed: " + str(getattr(result, "error", None) or result.output))
    text = result.output or ""
    try:
        payload = json.loads(text)
    except (ValueError, TypeError):
        payload = None
    if isinstance(payload, dict) and "stdout" in payload and "exit_code" in payload:
        if (payload["exit_code"] != 0 or payload.get("running") or payload.get("timed_out")
                or payload.get("stdout_truncated") or payload.get("stderr_truncated")):
            raise ValueError("baseline probe did not complete: " + text)
        return str(payload["stdout"]), str(payload.get("stderr") or "")
    return text, ""


def inspect_baseline(session, instance: dict) -> dict:
    expected = instance["base_commit"]
    if not re.fullmatch(r"[0-9a-f]{40}", expected):
        raise ValueError("invalid dataset base commit")
    result = session.execute("shell", {
        "command": "cd /testbed && GIT_OPTIONAL_LOCKS=0 python3 - " + shlex.quote(expected) + " <<'PY'\n" + PROBE + "\nPY",
        "timeout": 60,
    })
    stdout, stderr = probe_output(result)
    report = json.loads(stdout)
    report["probe_stderr"] = stderr
    try:
        report["admission"] = validate_baseline(report, instance)
    except ValueError as error:
        raise BaselineAdmissionError(str(error), report) from error
    return report
