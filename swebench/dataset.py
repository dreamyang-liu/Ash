"""Dataset loading and instance utilities for SWE-bench / SWE-Gym."""

import json
import re
import shlex

try:
    from datasets import load_dataset
except ImportError:
    raise ImportError("Install datasets: pip install datasets")


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
    if repo == "django/django":
        m = _DJANGO_TEST_ID.match(test_id.strip())
        spec = f"{m.group(2)}.{m.group(1)}" if m else test_id
        return ("./tests/runtests.py --verbosity 0 --settings=test_sqlite "
                f"--parallel 1 {shlex.quote(spec)}")
    return f"python -m pytest -x -q {shlex.quote(test_id)}"
