"""Launch the installed official harness without importing Ash's swebench package."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import runpy
import sys
from threading import BoundedSemaphore
from typing import Any, Callable


def limit_container_creation(build: Callable, workers: int) -> Callable:
    semaphore = BoundedSemaphore(workers)

    def limited(*args: Any, **kwargs: Any) -> Any:
        with semaphore:
            return build(*args, **kwargs)

    return limited


def prepare_images(specs: list, client: Any, build: Callable,
                   image_not_found: type[Exception], workers: int, output: Path) -> bool:
    def prepare(spec: Any) -> dict:
        record = {"instance_id": spec.instance_id, "image": spec.instance_image_key}
        try:
            try:
                client.images.get(spec.instance_image_key)
            except image_not_found:
                if spec.is_remote_image:
                    client.images.pull(spec.instance_image_key, platform=spec.platform)
                else:
                    build(spec, client, None, False)
            client.images.get(spec.instance_image_key)
            return {**record, "ok": True}
        except Exception as exc:
            return {**record, "ok": False, "error": f"{type(exc).__name__}: {exc}"}

    unique = {spec.instance_image_key: spec for spec in specs}
    output.parent.mkdir(parents=True, exist_ok=True)
    success = True
    with output.open("w", encoding="utf-8") as stream, ThreadPoolExecutor(max_workers=workers) as pool:
        for record in pool.map(prepare, unique.values()):
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            success = success and record["ok"]
    return success


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--prepare-workers", type=int, default=4)
    parser.add_argument("--startup-workers", type=int, default=4)
    parser.add_argument("--prepare-report", type=Path)
    options, remaining = parser.parse_known_args()
    if options.prepare_workers < 1 or options.startup_workers < 1:
        parser.error("prepare/startup workers must be positive")
    repo = Path(__file__).resolve().parents[1]
    sys.path[:] = [entry for entry in sys.path if Path(entry or os.getcwd()).resolve() != repo]
    import docker

    original_from_env = docker.from_env

    def from_env_with_timeout(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("timeout", int(os.environ.get("ASH_DOCKER_CLIENT_TIMEOUT", "600")))
        kwargs.setdefault("max_pool_size", int(os.environ.get("ASH_DOCKER_MAX_POOL_SIZE", "128")))
        return original_from_env(*args, **kwargs)

    docker.from_env = from_env_with_timeout
    if not options.prepare_only:
        from swebench.harness import docker_build

        docker_build.build_container = limit_container_creation(
            docker_build.build_container, options.startup_workers)
        sys.argv = [sys.argv[0], *remaining]
        runpy.run_module("swebench.harness.run_evaluation", run_name="__main__", alter_sys=True)
        return 0

    prepare_parser = argparse.ArgumentParser()
    prepare_parser.add_argument("--dataset_name", required=True)
    prepare_parser.add_argument("--split", default="test")
    prepare_parser.add_argument("--predictions_path", required=True)
    prepare_parser.add_argument("--run_id", required=True)
    prepare_parser.add_argument("--namespace", default="swebench")
    args, unused = prepare_parser.parse_known_args(remaining)
    if options.prepare_report is None:
        parser.error("--prepare-only requires --prepare-report")
    from swebench.harness.docker_build import build_env_images, build_instance_image
    from swebench.harness.run_evaluation import get_dataset_from_preds
    from swebench.harness.test_spec.test_spec import make_test_spec

    predictions = [json.loads(line) for line in Path(args.predictions_path).read_text().splitlines()
                   if line.strip()]
    dataset = get_dataset_from_preds(
        args.dataset_name, args.split, None,
        {item["instance_id"]: item for item in predictions}, args.run_id, False,
    )
    specs = [make_test_spec(instance, namespace=args.namespace or None) for instance in dataset]
    client = docker.from_env()
    try:
        if not args.namespace:
            build_env_images(client, dataset, False, options.prepare_workers)
        success = prepare_images(specs, client, build_instance_image, docker.errors.ImageNotFound,
                                 options.prepare_workers, options.prepare_report)
    finally:
        client.close()
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
