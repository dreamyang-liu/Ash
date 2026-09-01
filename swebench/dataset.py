"""Dataset loading and instance utilities for SWE-bench / SWE-Gym."""

import json
import re
import shlex


_DATASET_MAP = {
    "lite": "princeton-nlp/SWE-bench_Lite",
    "verified": "princeton-nlp/SWE-bench_Verified",
    "full": "princeton-nlp/SWE-bench",
    "gym": "SWE-Gym/SWE-Gym",
    "gym-lite": "SWE-Gym/SWE-Gym-Lite",
}

_IMAGE_REGISTRIES = {
    "swebench": ("swebench", "_1776_"),
    "xingyaoww": ("xingyaoww", "_s_"),
}


def load_instances(
    subset: str = "lite",
    split: str = "",
    slice_spec: str = "",
    filter_regex: str = "",
) -> list[dict]:
    """Load SWE-bench instances from HuggingFace."""
    dataset_name = _DATASET_MAP.get(subset, subset)

    if not split:
        split = "train" if subset.startswith("gym") else "test"

    # Imported here, not at module scope. `datasets` is a heavy dependency needed
    # only to fetch the benchmark, and requiring it at import time meant every module
    # that touches this file -- and so every test in the package -- could not be
    # imported without it.
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Install datasets: pip install datasets") from exc

    dataset = load_dataset(dataset_name, split=split)
    instances = list(dataset)

    if filter_regex:
        pattern = re.compile(filter_regex)
        instances = [i for i in instances if pattern.search(i.get("instance_id", ""))]

    if slice_spec:
        parts = slice_spec.split(":")
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1]) if len(parts) > 1 and parts[1] else len(instances)
        instances = instances[start:end]

    return instances


def resolve_image(instance: dict, template: str = "", registry: str = "swebench") -> str:
    """Resolve Docker image name for a SWE-bench/SWE-Gym instance."""
    image_name = instance.get("image_name") or instance.get("env_image_key")
    if image_name:
        return image_name

    instance_id = instance.get("instance_id", "")

    if template:
        repo = instance.get("repo", "").replace("/", "__").lower()
        commit = instance.get("base_commit", "")[:12]
        return template.format(instance_id=instance_id, repo=repo, commit=commit)

    prefix, separator = _IMAGE_REGISTRIES.get(registry, ("swebench", "_1776_"))
    id_docker = instance_id.replace("__", separator)
    return f"{prefix}/sweb.eval.x86_64.{id_docker}:latest".lower()


def format_task_prompt(instance: dict) -> str:
    """Format a SWE-bench instance into a task prompt."""
    return f"""<issue>
{instance.get("problem_statement", "")}
</issue>

Repository: {instance.get("repo", "")}
You are working in /testbed which contains the repository at commit {instance.get("base_commit", "")}.

Fix the issue described above. Make minimal changes to the source code.
Do NOT modify test files. After making your changes, verify them by running relevant tests.
"""


def image_registry_for_subset(subset: str) -> str:
    """Return the image registry key for a dataset subset."""
    return "xingyaoww" if subset.startswith("gym") else "swebench"


#: Django test ids read ``test_name (module.Class)``; its runner wants
#: ``module.Class.test_name``.
_DJANGO_TEST_ID = re.compile(r"^(\S+)\s+\(([^)]+)\)$")


def malformed_test_ids(test_ids: list) -> list:
    """Ids whose brackets do not balance -- dataset damage, not agent behaviour.

    SWE-bench's own test lists split parametrised ids on the comma INSIDE the
    parameters, so ``test_stem[png-w/ orientation, bottom]`` arrives as two
    fragments: ``test_stem[png-w/ orientation`` and ``bottom]``. 65 of the 500
    Verified instances carry at least one.

    pytest cannot collect a fragment, so it answers "ERROR: not found" and exits
    non-zero -- and a grader reading only the exit code then reports a broken
    regression suite for every attempt, whatever the agent did. Measured: seven of
    a 32-instance batch's fourteen failures were this, three of them showing
    "target passes, regressions fail" across all eight attempts.

    The second kind is django's: its runner announces a test by the FIRST LINE OF
    ITS DOCSTRING when it has one, and the dataset harvested those display
    strings, so 165 of the 231 django instances carry ids like
    ``Tests creating/deleting CHECK constraints``. Handed back to
    ``runtests.py`` as a label, that is a fatal "test label path does not exist"
    at COLLECTION time -- the batch dies before running anything, and every
    verdict downstream says "regressions FAIL" about a suite that never ran.
    Measured on the first full 500: 77 of 94 "broke regressions" and 28 of 133
    "target failed" verdicts were this. (``test_x (module.Class)`` ids are fine;
    ``build_batch_test_command`` rewrites them into real labels.)
    """
    def inexpressible(test_id: str) -> bool:
        if test_id.count("[") != test_id.count("]"):
            return True
        # A space without either a pytest ``::`` path or a trailing
        # ``(module.Class)`` label is a docstring, not a test id.
        if " " in test_id and "::" not in test_id \
                and not _DJANGO_LABEL.search(test_id):
            return True
        return False

    return [t for t in test_ids if inexpressible(t)]


#: ``test_name (module.Class)`` -- django's display form that still carries the
#: real label. Anything space-separated WITHOUT this is prose.
_DJANGO_LABEL = re.compile(r"\(\w[\w.]*\)\s*$")


def parse_test_list(raw: object) -> list[str]:
    """Parse FAIL_TO_PASS / PASS_TO_PASS — a JSON list *string* in SWE-bench."""
    if isinstance(raw, list):
        return [str(t) for t in raw]
    if not isinstance(raw, str) or not raw.strip():
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []
    return [str(t) for t in parsed] if isinstance(parsed, list) else []


def test_files_of(instance: dict) -> list:
    """Test files the instance's own test_patch touches.

    Needed by runners whose test ids are bare function names (sympy): without
    the files, a keyword filter sweeps the entire library and matches nothing.
    """
    patch = instance.get("test_patch") or ""
    return re.findall(r"diff --git a/(\S+)", patch)


def build_test_command(repo: str, test_id: str) -> str:
    """Shell command whose exit code says whether one test passes.

    django ids need django's own runner; the other SWE-bench Verified repos are
    pytest-collectable. A test id the runner cannot collect simply fails, so a
    caller using this as a signal degrades rather than crashing.

    Lives here rather than with a harness because it is a fact about the dataset's
    repos, not about any one topology -- it was in the best-of-n harness, which is
    why deleting that harness would have taken the rollout server's grading with
    it.
    """
    return build_batch_test_command(repo, [test_id])


#: Runner script for repos whose test ids are bare function names. Written into
#: the sandbox as a FILE by the caller, never interpolated into a shell command:
#: nesting python inside a heredoc inside a JSON tool argument mangles every
#: backslash on the way, and an hour went into a regex that arrived as literal
#: "d+". The only thing the shell sees now is `python <path>`.
SYMPY_RUNNER = r"""
import importlib, json, sys, traceback

# Bare-name test ids (sympy reports these for all 75 of its Verified instances)
# are just module-level functions. Importing the module and CALLING them is the
# most direct reading available, and the only one that distinguishes "ran and
# passed" from "matched nothing" -- which matters because sympy's own runner
# returns truthy for a zero-match run, so a grader trusting it scores an
# UNTOUCHED repository as resolved. Two earlier attempts at this (bin/test -k,
# then sympy.test + stdout capture) both produced that false pass.
spec = json.load(open(sys.argv[1]))
wanted = set(spec["kw"])

found, failed, missing = 0, [], set(wanted)
for path in spec["files"]:
    name = path.replace("/", ".").rsplit(".py", 1)[0]
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        print("GRADER: cannot import %s: %s" % (name, exc))
        continue
    for test in sorted(wanted):
        function = getattr(module, test, None)
        if function is None or not callable(function):
            continue
        missing.discard(test)
        found += 1
        try:
            function()
        except Exception as exc:
            # sympy marks skips with an exception type named Skipped; a skipped
            # test is not a failure of the change under test.
            if type(exc).__name__ in ("Skipped", "SkipTest"):
                print("SKIP %s.%s" % (name, test))
                continue
            failed.append("%s.%s: %s" % (name, test, exc))
            print("FAIL %s.%s" % (name, test))
            traceback.print_exc(limit=3)
        else:
            print("PASS %s.%s" % (name, test))

print("GRADER: ran %d, failed %d, not found %s" % (found, len(failed), sorted(missing)))
if found == 0:
    sys.exit(2)          # nothing ran: a broken grader, not a pass
sys.exit(1 if failed else 0)
"""


def sympy_runner_spec(names: list, test_files: "list | None") -> dict:
    """The JSON the runner script reads: which files, which keywords."""
    return {"files": list(test_files or ["sympy"]), "kw": list(names)}


def needs_file_runner(repo: str, test_ids: list) -> bool:
    """Whether this repo's ids are bare function names needing SYMPY_RUNNER.

    sympy reports bare names for all 75 of its Verified instances; handed to
    pytest as paths they collect nothing and the run fails whatever the agent
    did -- a live 7-attempt experiment scored every attempt zero on exactly that.
    These images also ship no pytest at all.
    """
    if repo != "sympy/sympy":
        return False
    return any("::" not in t and "/" not in t for t in test_ids)


def build_batch_test_command(repo: str, test_ids: list[str],
                             test_files: "list[str] | None" = None) -> str:
    """Shell command whose exit code says whether ALL of ``test_ids`` pass.

    One invocation, not one per test: PASS_TO_PASS lists run to hundreds of
    ids, and per-test runner startup (5-20s each) would turn a regression
    check into an hour. Both runners accept multiple test specs and exit
    non-zero if any fails, which is exactly the all-or-nothing answer a
    regression gate needs.

    ``test_files`` narrows a runner that needs it (sympy reports bare test-function
    names, so without the files its keyword filter sweeps the whole library).
    """
    if repo == "django/django":
        specs = []
        for test_id in test_ids:
            m = _DJANGO_TEST_ID.match(test_id.strip())
            specs.append(f"{m.group(2)}.{m.group(1)}" if m else test_id)
        joined = " ".join(shlex.quote(spec) for spec in specs)
        return ("./tests/runtests.py --verbosity 0 --settings=test_sqlite "
                f"--parallel 1 {joined}")
    if needs_file_runner(repo, test_ids):
        # No shell command can express these. Emitting a pytest line anyway is
        # what produced seven false zeros; the caller routes them to
        # SYMPY_RUNNER instead.
        raise ValueError(
            "%s test ids are bare function names and need the file-based "
            "runner (SYMPY_RUNNER + sympy_runner_spec), not a shell command"
            % repo)
    joined = " ".join(shlex.quote(test_id) for test_id in test_ids)
    return f"python -m pytest -x -q {joined}"
