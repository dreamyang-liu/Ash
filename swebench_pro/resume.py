"""Prepare every Pro image, then resume a reconciled, frozen single-pass batch."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

from swebench_pro.recovery import durable_json, read_json, reconcile

STORE = Path('/opt/aenv-bench/home/snapshot-store/repository')


def disk_manifest_valid(identifier: str) -> bool:
    path = STORE / 'snapshots' / identifier / 'firecracker-manifest.json'
    try:
        data = json.loads(path.read_text())
        if data.get('version') != 1 or not data.get('rootfs', {}).get('virtualSize'):
            return False
        if 'vmState' in data and not (path.parent / 'vm_state.bin').stat().st_size:
            return False
        return (path.parent / 'commit').is_file()
    except (OSError, ValueError, TypeError):
        return False


def install_template_health() -> None:
    from harness.execution.templates import TemplateBuilder

    original = TemplateBuilder._build_succeeded

    def checked(self: Any, client: Any, template_id: str) -> bool:
        return original(self, client, template_id) and disk_manifest_valid(template_id)

    TemplateBuilder._build_succeeded = checked


def choose_continuation(item: dict, actor_seconds: int, interrupted_at: str) -> dict:
    from harness.rollback import load_checkpoints
    from harness.slots.claude_history import find_prefix_source
    from swebench.fork_eval import CLAUDE_PROJECTS_DIR, conversation_restore

    journal = Path(item['journal'])
    events = [json.loads(line) for line in journal.read_text().splitlines() if line.strip()]
    started = next(event['ts'] for event in events if event.get('type') == 'run.started')
    elapsed = (datetime.fromisoformat(interrupted_at) - datetime.fromisoformat(started.replace('Z', '+00:00'))).total_seconds()
    remaining = max(0, actor_seconds - max(0, elapsed))
    budgets = [event for event in events if event.get('type') == 'pro.tool_budget' and 'tool_seconds' in event]
    points = [point for point in load_checkpoints(journal)
              if point.reason == 'captured' and point.snapshot_id and point.session_ckpt]
    for point in reversed(points):
        if not disk_manifest_valid(point.snapshot_id):
            continue
        restored = conversation_restore(journal, point.step, point.session_ckpt)
        if not restored:
            continue
        cut, prefix = restored
        prefix = prefix or find_prefix_source(CLAUDE_PROJECTS_DIR, point.session_ckpt, cut)
        if prefix is None:
            continue
        return {'snapshot_id': point.snapshot_id, 'step': point.step, 'cut': cut,
                'source_session': point.session_ckpt, 'source_transcript': str(prefix.transcript),
                'transcript_sha256': prefix.sha256, 'source_journal': str(journal),
                'journal_sha256': hashlib.sha256(journal.read_bytes()).hexdigest(),
                'actor_seconds_remaining': remaining, 'actor_seconds_charged': max(0, elapsed),
                'tool_seconds_charged': budgets[-1]['tool_seconds'] if budgets else 0,
                'consecutive_timeouts': budgets[-1].get('consecutive_timeouts', 0) if budgets else 0,
                'latest_recorded_step': points[-1].step}
    raise RuntimeError(f"No intact snapshot/native-prefix pair for {item['id']}")


def image_receipts(root: Path, manifest: dict) -> dict[str, dict]:
    receipts = {}
    for item in manifest['tasks']:
        row = read_json(root / 'image-ready' / f"{item['index']:03d}.json") or {}
        if row.get('runtime_port', 3000) != manifest.get('runtime_port', 3000):
            continue
        if manifest.get('require_resource_receipts') and (
            row.get('resources') != {'cpu': manifest['cpu'], 'memory_mb': manifest['memory_mb']}
            or not row.get('guest_memory_kib')
        ):
            continue
        if row and row.get('ok') and row.get('image') == item['image'] and disk_manifest_valid(row['snapshot_id']):
            receipts[item['id']] = row
    return receipts


def require_images_ready(root: Path, manifest: dict) -> dict[str, dict]:
    receipts = image_receipts(root, manifest)
    missing = [item['id'] for item in manifest['tasks'] if item['id'] not in receipts]
    if missing:
        raise RuntimeError(f"Image barrier not satisfied: {len(missing)} images missing")
    return receipts


def prepare_one(root: Path, index: int) -> int:
    from harness.execution.session import SandboxSession
    from swebench.fork_eval import backend_for
    from swebench_pro.bench import SWEbenchPro
    from swebench_pro.grade import checked, reset_base

    manifest = read_json(root / 'manifest.json')
    item = manifest['tasks'][index]
    install_template_health()
    args = SimpleNamespace(pro_repo=str(root / 'upstream'), pro_data=str(root / item['data']),
                           runtime_bin=str(root / 'source/runtime/ash-runtime'), timeout=3600,
                           pro_cpus=manifest['cpu'], pro_memory_mb=manifest['memory_mb'],
                           pro_runtime_port=manifest.get('runtime_port'))
    bench = SWEbenchPro(args)
    task = bench.catalogue(args)[item['id']]
    backend = backend_for(args, bench)
    backend['microvm']['request_timeout'] = 900
    session = SandboxSession(quiet=True, backend=backend)
    row = {'task': item['id'], 'image': item['image'], 'ok': False,
           'runtime_port': manifest.get('runtime_port', 3000)}
    try:
        if shutil.disk_usage(root).free < 160 * 1024**3:
            raise RuntimeError('Image preparation paused: disk below 160GiB reserve')
        if not session.create(task.image, bench.resources(bench.instance(task))):
            raise RuntimeError(session.create_error)
        guest_memory = int(checked(session, "awk '/^MemTotal:/ {print $2}' /proc/meminfo").strip())
        guest_cpus = int(checked(session, 'nproc').strip())
        expected_memory = manifest['memory_mb'] * 1024
        if not expected_memory * 0.9 <= guest_memory <= expected_memory or guest_cpus != manifest['cpu']:
            raise RuntimeError(f'Guest resources mismatch: {guest_cpus} CPUs, {guest_memory} KiB memory')
        reset_base(session, task)
        actual = checked(session, 'sha256sum /usr/local/bin/ash-runtime').split()[0]
        expected = manifest['sha256']['source/runtime/ash-runtime']
        if actual != expected:
            raise RuntimeError('Prepared image runtime checksum mismatch')
        snapshot = session.snapshot(disk_only=True)
        if snapshot is None or not disk_manifest_valid(snapshot.id):
            raise RuntimeError('Prepared image has no durable snapshot manifest')
        row.update(ok=True, snapshot_id=snapshot.id, runtime_sha256=actual,
                   resources={'cpu': manifest['cpu'], 'memory_mb': manifest['memory_mb']},
                   guest_memory_kib=guest_memory, guest_cpus=guest_cpus,
                   template_id=session._base_image,
                   finished_at=datetime.now(timezone.utc).isoformat())
    except Exception as exc:
        row['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        session.destroy()
    durable_json(root / 'image-ready' / f'{index:03d}.json', row)
    print(json.dumps(row), flush=True)
    return 0 if row['ok'] else 1


def probe_continuations(root: Path, manifest: dict) -> None:
    from harness.execution.session import SandboxSession
    from swebench.fork_eval import backend_for
    from swebench_pro.bench import SWEbenchPro
    from swebench_pro.grade import checked

    args = SimpleNamespace(pro_repo=str(root / 'upstream'), runtime_bin=str(root / 'source/runtime/ash-runtime'), timeout=600)
    backend = backend_for(args, SWEbenchPro(args))
    backend['microvm']['request_timeout'] = 900
    for item in manifest['tasks']:
        continuation = item.get('continuation')
        if not continuation:
            continue
        session = SandboxSession(quiet=True, backend=backend)
        try:
            if not session.create(continuation['snapshot_id']):
                raise RuntimeError(f"Continuation restore failed for {item['id']}: {session.create_error}")
            if checked(session, 'cd /app && git rev-parse --is-inside-work-tree').strip() != 'true':
                raise RuntimeError('Continuation repository probe failed')
            durable_json(root / 'continuation-probes' / f"{item['index']:03d}.json",
                         {'ok': True, 'task': item['id'], 'snapshot_id': continuation['snapshot_id']})
        finally:
            session.destroy()


def stage(root: Path) -> int:
    from swebench_pro.batch import controller

    lock = (root / 'recovery.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = read_json(root / 'manifest.json')
    image_workers = manifest.get('image_prepare_workers', 2)
    if type(image_workers) is not int or image_workers < 1:
        raise ValueError('image_prepare_workers must be a positive integer')
    (root / 'image-ready').mkdir(exist_ok=True)
    (root / 'image-logs').mkdir(exist_ok=True)

    def run_image(item: dict) -> dict:
        if (root / 'STOP_REQUEST.json').exists():
            return {'task': item['id'], 'stopped': True}
        if shutil.disk_usage(root).free < 160 * 1024**3:
            return {'task': item['id'], 'error': 'disk below 160GiB reserve'}
        log_path = root / 'image-logs' / f"{item['index']:03d}.log"
        with log_path.open('a') as log:
            subprocess.run([sys.executable, '-u', '-m', 'swebench_pro.resume', 'prepare-one',
                            '--out', str(root), '--index', str(item['index'])],
                           cwd=root / 'source', stdout=log, stderr=subprocess.STDOUT, check=False)
        return read_json(root / 'image-ready' / f"{item['index']:03d}.json") or {'task': item['id'], 'error': 'image worker produced no receipt'}

    try:
        for round_no in range(2):
            ready = image_receipts(root, manifest)
            pending = [item for item in manifest['tasks'] if item['id'] not in ready]
            if round_no == 0:
                deferred = set(manifest.get('deferred_image_indices', []))
                pending = [item for item in pending if item['index'] not in deferred]
            durable_json(root / 'recovery-status.json', {'phase': 'preparing_images', 'pid': os.getpid(),
                         'ready_images': len(ready), 'total_images': len(manifest['tasks']),
                         'image_prepare_workers': image_workers, 'actor_admission': False})
            with ThreadPoolExecutor(max_workers=image_workers) as pool:
                futures = [pool.submit(run_image, item) for item in pending]
                for future in as_completed(futures):
                    row = future.result()
                    if row.get('ok'):
                        ready[row['task']] = row
                    durable_json(root / 'recovery-status.json', {'phase': 'preparing_images', 'pid': os.getpid(),
                                 'ready_images': len(ready), 'total_images': len(manifest['tasks']),
                                 'image_prepare_workers': image_workers,
                                 'round': round_no, 'last_image': row, 'actor_admission': False,
                                 'free_bytes': shutil.disk_usage(root).free, 'updated_at': datetime.now(timezone.utc).isoformat()})
            if (root / 'STOP_REQUEST.json').exists():
                raise RuntimeError('Recovery stop requested')
        ready = require_images_ready(root, manifest)
        probe_continuations(root, manifest)
        original_plan = read_json(root / 'recovery-plan.json')
        for filename, expected in original_plan['preserved_sha256'].items():
            if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != expected:
                raise RuntimeError('Preserved result/journal changed: ' + filename)
        durable_json(root / 'all-images-ready.json', {'ready': True, 'image_count': len(ready),
                                                     'at': datetime.now(timezone.utc).isoformat()})
        if manifest.get('prepare_only'):
            durable_json(root / 'recovery-status.json', {'phase': 'images_prepared', 'pid': os.getpid(),
                         'ready_images': len(ready), 'total_images': len(manifest['tasks']),
                         'actor_admission': False, 'memory_mb': manifest['memory_mb']})
            return 0
        durable_json(root / 'recovery-status.json', {'phase': 'rollouts', 'pid': os.getpid(),
                     'ready_images': len(ready), 'total_images': len(manifest['tasks']), 'actor_admission': True})
        return controller(root)
    except Exception as exc:
        durable_json(root / 'recovery-status.json', {'phase': 'held', 'pid': os.getpid(),
                     'ready_images': len(image_receipts(root, manifest)), 'total_images': len(manifest['tasks']),
                     'actor_admission': False, 'error': f'{type(exc).__name__}: {exc}'})
        return 2


def launch(source: Path, root: Path, interrupted_at: str) -> dict:
    from swebench_pro.batch import configure_credentials

    configure_credentials()
    old = read_json(source / 'manifest.json')
    for relative, expected in old['sha256'].items():
        if hashlib.sha256((source / relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Original frozen file changed: ' + relative)
    root.mkdir(parents=True, exist_ok=False)
    plan = reconcile(source, root)
    ignore = shutil.ignore_patterns('__pycache__', '*.pyc')
    shutil.copytree(source / 'source', root / 'source', ignore=ignore)
    for name in ('tasks', 'upstream', 'official-config'):
        (root / name).symlink_to(source / name, target_is_directory=True)
    for name in ('resume.py', 'recovery.py', 'worker.py'):
        shutil.copy2(Path(__file__).with_name(name), root / 'source/swebench_pro' / name)
    rows = []
    for original, decision in zip(old['tasks'], plan['tasks'], strict=True):
        item = {**original, 'recovery_action': decision['action']}
        if decision['action'] == 'resume':
            item['continuation'] = choose_continuation(decision, old['actor_timeout_s'], interrupted_at)
        elif decision['action'] == 'preserve':
            grade = read_json(Path(decision['grade_path']))
            previous = read_json(Path(decision['source_directory']).parent / 'worker.json') or {}
            previous.update(task=item['id'], status='completed', evidence_valid=True,
                            resolved=grade['grade']['resolved'], finished_at=previous.get('finished_at') or 'preserved',
                            preserved_from=decision['source_directory'])
            durable_json(root / f"shard-{item['index']:03d}/worker.json", previous)
        elif decision['action'] != 'fresh':
            raise RuntimeError('Unsupported recovery action: ' + decision['action'])
        rows.append(item)
    manifest = {**old, 'tasks': rows, 'recovery': True, 'original_batch': str(source),
                'recovery_created_at': datetime.now(timezone.utc).isoformat(),
                'interrupted_at': interrupted_at, 'all_images_required_before_actor': True}
    canaries = [row['id'] for row in rows if row['recovery_action'] != 'preserve'][:2]
    for row in rows:
        row['canary'] = row['id'] in canaries
    manifest['canaries'] = canaries
    manifest['sha256'] = {key: value for key, value in old['sha256'].items() if not key.startswith('source/')}
    for path in (root / 'source').rglob('*'):
        if path.is_file():
            manifest['sha256'][str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    durable_json(root / 'manifest.json', manifest)
    environment = {**os.environ, 'PYTHONPATH': f"{root / 'source'}:{root / 'source/sdk'}"}
    with (root / 'recovery.log').open('x') as log:
        process = subprocess.Popen([sys.executable, '-u', '-m', 'swebench_pro.resume', 'stage', '--out', str(root)],
                                   cwd=root / 'source', env=environment, stdout=log, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, start_new_session=True)
    return {'pid': process.pid, 'root': str(root), 'actions': plan['actions'], 'model_admission': 'after_all_images_ready'}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['launch', 'stage', 'prepare-one'])
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--index', type=int)
    parser.add_argument('--interrupted-at', default='2026-09-09T09:14:48+00:00')
    args = parser.parse_args()
    if args.phase == 'launch':
        if args.source is None:
            parser.error('launch requires --source')
        print(json.dumps(launch(args.source.resolve(), args.out.resolve(), args.interrupted_at)), flush=True)
        return 0
    if args.phase == 'stage':
        return stage(args.out.resolve())
    if args.index is None:
        parser.error('prepare-one requires --index')
    return prepare_one(args.out.resolve(), args.index)


if __name__ == '__main__':
    raise SystemExit(main())
