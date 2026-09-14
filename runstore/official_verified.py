"""Isolated official-package entrypoint; never import Ash's namesake swebench."""

from importlib.metadata import version
import json
import os
from pathlib import Path
import runpy
import sys


def main() -> None:
    request = json.loads(Path(sys.argv[1]).read_text())
    describe = "--describe" in sys.argv
    repo = Path(__file__).resolve().parents[1]
    sys.path[:] = [entry for entry in sys.path if Path(entry or os.getcwd()).resolve() != repo]
    spec = request["spec"]
    if version("swebench") != spec["grader_revision"]:
        raise ValueError("Installed official SWE-bench version differs from the submitted revision")
    from swebench.harness.test_spec.test_spec import make_test_spec

    task = make_test_spec(request["task"], namespace="swebench")
    directory = Path(request["directory"])
    if describe:
        (directory / "verified-description.json").write_text(json.dumps({"image": task.instance_image_key}))
        return
    import docker

    original = docker.models.containers.ContainerCollection.create

    def create(collection, *args, **kwargs):
        kwargs.update(mem_limit=int(spec["resources"]["memory_mb"]) * 1024 * 1024,
                      nano_cpus=int(float(spec["resources"]["cpu"]) * 1_000_000_000),
                      network_disabled=spec["verifier_network"] == "deny")
        kwargs["labels"] = {**kwargs.get("labels", {}), "ash.runstore.attempt": request["run_id"]}
        return original(collection, *args, **kwargs)

    docker.models.containers.ContainerCollection.create = create
    predictions = directory / "predictions.jsonl"
    predictions.write_text(json.dumps({"instance_id": spec["instance_id"], "model_name_or_path": "runstore",
                                       "model_patch": (directory / "submission.patch").read_text()}) + "\n")
    dataset = directory / "official-task.json"
    dataset.write_text(json.dumps([request["task"]]))
    sys.argv = [sys.argv[0], "--dataset_name", str(dataset), "--predictions_path", str(predictions),
                "--run_id", request["run_id"], "--max_workers", "1", "--timeout", str(int(spec["timeout_s"])),
                "--namespace", "swebench", "--cache_level", "instance", "--clean", "false"]
    runpy.run_module("swebench.harness.run_evaluation", run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
