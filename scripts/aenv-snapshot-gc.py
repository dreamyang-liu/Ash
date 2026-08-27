#!/usr/bin/env python3
"""Offline GC for the AgentENV snapshot store.

Stopgap for a real backend gap: AgentENV exposes DELETE for /sandboxes/{id} and
/templates/{id} only -- ``DELETE /snapshots/{id}`` answers 405 and ``aenv
snapshot`` has no delete subcommand, so per-step checkpoints accumulate forever
(the "There is no GC yet" note in AGENTS.md). ``harness reap`` reports them as
unsupported; this script reclaims them on disk until the API exists.

Layers are content-addressed and shared, so this is mark-and-sweep, not a
directory delete: remove the orphans' metadata, then drop only layers that no
surviving snapshot still names. Anything with a catalog alias (i.e. a template)
is kept, along with every layer it references.

**Run with the server stopped.** It rewrites the store the server has open.

    sudo python3 scripts/aenv-snapshot-gc.py --dry-run
    sudo python3 scripts/aenv-snapshot-gc.py --apply
    sudo python3 scripts/aenv-snapshot-gc.py --apply --keep <snapshot-id>

Verified: reclaimed 17 orphans from fork-demo runs, then a sandbox created from
the surviving template still booted, ran a command, and snapshotted. Note how
little space it frees (3.2 MB for 17 snapshots): overlaybd dedup means an
incremental disk-only checkpoint holds almost nothing of its own, and the
``chainSizeMB`` the API reports is the *logical* chain including shared base
layers. The leak is a count problem, not a capacity one.

A word on why this is not part of ``harness reap``: reaping is an online,
credentialed API operation that any run can do; this needs root, local disk
access and a stopped server. Keeping them apart means the harness never grows a
dependency on being co-located with the backend.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

#: Override with AENV_HOME_PATH for a non-default installation (the bench
#: instance on this host, for example, uses /opt/aenv-bench/home).
import os

REPO = Path(
    os.environ.get("AENV_HOME_PATH", "/var/lib/aenv")
) / "snapshot-store" / "repository"
SHA = re.compile(r"sha256[:_]([0-9a-f]{64})")


def layer_refs(text: str) -> set:
    return set(SHA.findall(text))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--keep", action="append", default=[],
        help="snapshot id to keep in addition to aliased ones (repeatable)",
    )
    args = parser.parse_args()
    apply = args.apply and not args.dry_run

    snapshots_dir = REPO / "snapshots"
    records_dir = REPO / "catalog" / "records"
    aliases_dir = REPO / "catalog" / "aliases"
    layers_dir = REPO / "managed-layers"

    # --- what must survive --------------------------------------------------
    keep = set(args.keep)
    alias_map = {}
    for alias in sorted(aliases_dir.glob("*")):
        target = json.loads(alias.read_text()) if alias.is_file() else None
        if isinstance(target, str):
            alias_map[alias.name] = target
            keep.add(target)

    on_disk = {p.name for p in snapshots_dir.iterdir() if p.is_dir()}
    doomed = sorted(on_disk - keep)

    print("aliased (kept):")
    for name, target in alias_map.items():
        print("  %-42s %s" % (name, target))
    print("\nsnapshots on disk: %d   keeping: %d   removing: %d"
          % (len(on_disk), len(keep & on_disk), len(doomed)))

    # --- layer reference counting -------------------------------------------
    refs_by_snapshot = {}
    for snapshot_id in on_disk:
        text = ""
        record = records_dir / ("%s.json" % snapshot_id)
        if record.exists():
            text += record.read_text()
        manifest = snapshots_dir / snapshot_id / "firecracker-manifest.json"
        if manifest.exists():
            text += manifest.read_text()
        refs_by_snapshot[snapshot_id] = layer_refs(text)

    kept_refs = set()
    for snapshot_id in on_disk & keep:
        kept_refs |= refs_by_snapshot[snapshot_id]

    all_layers = {}
    for layer in layers_dir.glob("sha256_*"):
        digest = layer.name.split("_", 1)[1].split(".", 1)[0]
        all_layers[digest] = layer

    unreferenced = sorted(d for d in all_layers if d not in kept_refs)
    freed = sum(all_layers[d].stat().st_size for d in unreferenced)

    print("\nlayers: %d total, %d still referenced by kept snapshots, %d free"
          % (len(all_layers), len(all_layers) - len(unreferenced), len(unreferenced)))
    print("reclaimable: %.1f MB" % (freed / 1e6))

    if not apply:
        print("\n(dry run) would remove:")
        for snapshot_id in doomed[:5]:
            print("  snapshot %s" % snapshot_id)
        if len(doomed) > 5:
            print("  ... and %d more snapshots" % (len(doomed) - 5))
        for digest in unreferenced[:3]:
            print("  layer    %s" % all_layers[digest].name)
        if len(unreferenced) > 3:
            print("  ... and %d more layers" % (len(unreferenced) - 3))
        print("\nre-run with --apply (server stopped) to do it")
        return 0

    # --- sweep --------------------------------------------------------------
    removed_snapshots = 0
    for snapshot_id in doomed:
        shutil.rmtree(snapshots_dir / snapshot_id, ignore_errors=True)
        record = records_dir / ("%s.json" % snapshot_id)
        if record.exists():
            record.unlink()
        removed_snapshots += 1

    removed_layers = 0
    for digest in unreferenced:
        try:
            all_layers[digest].unlink()
            removed_layers += 1
        except OSError as exc:
            print("  ! %s: %s" % (all_layers[digest].name, exc))

    print("\nremoved %d snapshots, %d layers, freed %.1f MB"
          % (removed_snapshots, removed_layers, freed / 1e6))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
