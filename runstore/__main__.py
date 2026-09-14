"""Run Store service and concurrent worker manager; no historical-data import command."""

import argparse
import os
from pathlib import Path
import signal
from threading import Event

from harness.core.journal import volatile_reason
from runstore.config import read_config, snapshot_validator
from runstore.index import Index
from runstore.store import Store


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["init", "serve", "worker", "reconcile"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18110)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--job-id")
    parser.add_argument("--concurrency", type=int, default=1, help="Maximum concurrent attempts per worker (default: 1)")
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error("--concurrency must be positive")
    config = read_config(args.config)
    reason = volatile_reason(config["artifact_root"])
    if reason:
        parser.error(reason)
    store = Store(os.environ["ASH_RUNSTORE_DSN"])
    if args.command == "init":
        store.initialize()
    elif args.command == "serve":
        import uvicorn
        from runstore.api import create_app

        index = Index(store, snapshot_validator(store, config))
        uvicorn.run(create_app(store, os.environ["ASH_RUNSTORE_TOKEN"], index=index,
                              profiles=config["profiles"]), host=args.host, port=args.port)
    else:
        from runstore.worker import Worker

        worker = Worker(store, config)
        if args.command == "reconcile":
            if not args.job_id:
                parser.error("reconcile requires --job-id")
            print("reconciled" if worker.reconcile(args.job_id) else "still quarantined")
            return
        from runstore.manager import WorkerManager

        stop = Event()
        signal.signal(signal.SIGTERM, lambda *_: stop.set())
        signal.signal(signal.SIGINT, lambda *_: stop.set())
        WorkerManager(worker, concurrency=args.concurrency).run(stop=stop, once=args.once)


if __name__ == "__main__":
    main()
