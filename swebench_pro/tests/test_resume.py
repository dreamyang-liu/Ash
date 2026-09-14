from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import sys

import pytest

from swebench_pro import resume, worker
from swebench_pro.recovery import durable_json


def test_stage_uses_configured_parallelism_and_skips_ready_images(tmp_path, monkeypatch):
    import swebench_pro.batch
    from concurrent.futures import ThreadPoolExecutor

    durable_json(tmp_path / 'manifest.json', {'image_prepare_workers': 8,
                 'tasks': [{'index': 0, 'id': 'task', 'image': 'image'}]})
    durable_json(tmp_path / 'recovery-plan.json', {'preserved_sha256': {}})
    durable_json(tmp_path / 'image-ready/000.json', {'ok': True, 'image': 'image', 'snapshot_id': 'ready'})
    monkeypatch.setattr(resume, 'disk_manifest_valid', lambda identifier: True)
    monkeypatch.setattr(resume, 'probe_continuations', lambda *args: None)
    monkeypatch.setattr(swebench_pro.batch, 'controller', lambda root: 0)
    monkeypatch.setattr(resume.subprocess, 'run', lambda *args, **kwargs: pytest.fail('ready image was rerun'))
    workers = []

    def pool(*, max_workers):
        workers.append(max_workers)
        return ThreadPoolExecutor(max_workers=max_workers)

    monkeypatch.setattr(resume, 'ThreadPoolExecutor', pool)
    assert resume.stage(tmp_path) == 0
    assert workers == [8, 8]
    assert json.loads((tmp_path / 'all-images-ready.json').read_text())['image_count'] == 1


def test_no_actor_admission_until_every_image_has_valid_receipt(tmp_path, monkeypatch):
    manifest = {'tasks': [{'index': 0, 'id': 'a', 'image': 'image-a'},
                          {'index': 1, 'id': 'b', 'image': 'image-b'}]}
    monkeypatch.setattr(resume, 'disk_manifest_valid', lambda identifier: identifier != 'broken')
    durable_json(tmp_path / 'image-ready/000.json', {'ok': True, 'image': 'image-a', 'snapshot_id': 'ready'})
    with pytest.raises(RuntimeError, match='1 images missing'):
        resume.require_images_ready(tmp_path, manifest)
    durable_json(tmp_path / 'image-ready/001.json', {'ok': True, 'image': 'image-b', 'snapshot_id': 'broken'})
    with pytest.raises(RuntimeError):
        resume.require_images_ready(tmp_path, manifest)
    durable_json(tmp_path / 'image-ready/001.json', {'ok': True, 'image': 'image-b', 'snapshot_id': 'ready-b'})
    assert set(resume.require_images_ready(tmp_path, manifest)) == {'a', 'b'}


def test_resource_receipts_reject_old_shape(tmp_path, monkeypatch):
    manifest = {'tasks': [{'index': 0, 'id': 'task', 'image': 'image'}],
                'require_resource_receipts': True, 'cpu': 4, 'memory_mb': 12288}
    monkeypatch.setattr(resume, 'disk_manifest_valid', lambda identifier: True)
    assert resume.image_receipts(tmp_path, manifest) == {}
    receipt = {'ok': True, 'image': 'image', 'snapshot_id': 'snapshot',
               'resources': {'cpu': 4, 'memory_mb': 16384}, 'guest_memory_kib': 16000000}
    durable_json(tmp_path / 'image-ready/000.json', receipt)
    assert resume.image_receipts(tmp_path, manifest) == {}
    receipt.update(resources={'cpu': 4, 'memory_mb': 12288}, guest_memory_kib=12000000)
    durable_json(tmp_path / 'image-ready/000.json', receipt)
    assert list(resume.image_receipts(tmp_path, manifest)) == ['task']
    manifest['runtime_port'] = 34122
    assert resume.image_receipts(tmp_path, manifest) == {}
    receipt['runtime_port'] = 34122
    durable_json(tmp_path / 'image-ready/000.json', receipt)
    assert list(resume.image_receipts(tmp_path, manifest)) == ['task']


def test_preparation_defers_failed_images_and_does_not_launch_actor(tmp_path, monkeypatch):
    import swebench_pro.batch

    manifest = {'tasks': [{'index': index, 'id': f'task-{index}', 'image': f'image-{index}'}
                          for index in range(2)], 'image_prepare_workers': 1,
                'deferred_image_indices': [0], 'prepare_only': True, 'memory_mb': 12288}
    durable_json(tmp_path / 'manifest.json', manifest)
    durable_json(tmp_path / 'recovery-plan.json', {'preserved_sha256': {}})
    monkeypatch.setattr(resume, 'disk_manifest_valid', lambda identifier: True)
    monkeypatch.setattr(resume, 'probe_continuations', lambda *args: None)
    monkeypatch.setattr(resume.shutil, 'disk_usage', lambda root: SimpleNamespace(free=1024**4))
    monkeypatch.setattr(swebench_pro.batch, 'controller', lambda root: pytest.fail('actor admitted'))
    seen = []

    def prepare(arguments, **kwargs):
        index = int(arguments[-1])
        seen.append(index)
        durable_json(tmp_path / 'image-ready' / f'{index:03d}.json',
                     {'task': f'task-{index}', 'image': f'image-{index}', 'ok': True, 'snapshot_id': 'snapshot'})

    monkeypatch.setattr(resume.subprocess, 'run', prepare)
    assert resume.stage(tmp_path) == 0
    assert seen == [1, 0]
    assert json.loads((tmp_path / 'recovery-status.json').read_text())['actor_admission'] is False


def test_selection_requires_both_durable_disk_and_native_prefix(tmp_path, monkeypatch):
    import harness.rollback
    import harness.slots.claude_history
    import swebench.fork_eval

    journal = tmp_path / 'old.jsonl'
    journal.write_text(json.dumps({'type': 'run.started', 'ts': '2026-09-09T09:00:00Z'}) + '\n' +
                       json.dumps({'type': 'pro.tool_budget', 'tool_seconds': 200, 'consecutive_timeouts': 1}) + '\n')
    points = [SimpleNamespace(reason='captured', snapshot_id=f'snapshot-{step}', session_ckpt='session', step=step)
              for step in [1, 2, 3]]
    monkeypatch.setattr(harness.rollback, 'load_checkpoints', lambda path: points)
    monkeypatch.setattr(resume, 'disk_manifest_valid', lambda identifier: identifier != 'snapshot-3')
    monkeypatch.setattr(swebench.fork_eval, 'conversation_restore', lambda journal, step, session: ('cut', None) if step == 1 else None)
    monkeypatch.setattr(harness.slots.claude_history, 'find_prefix_source', lambda *args: SimpleNamespace(transcript='native', sha256='hash'))
    selected = resume.choose_continuation({'id': 'task', 'journal': str(journal)}, 3600, '2026-09-09T09:10:00+00:00')
    assert selected['step'] == 1 and selected['latest_recorded_step'] == 3
    assert selected['actor_seconds_remaining'] == 3000
    assert selected['tool_seconds_charged'] == 200


@pytest.mark.parametrize('mode,isolated,actor_fault,grade_kind', [
    ('resume', False, False, None), ('resume', True, False, None),
    ('regrade', False, False, None), ('regrade', True, False, None),
    ('resume', True, True, None), ('regrade', True, False, 'infra'),
    ('regrade', True, False, 'unresolved')])
def test_worker_uses_registered_prefix_and_remaining_budget(
        tmp_path, monkeypatch, mode, isolated, actor_fault, grade_kind):
    from harness.core.slot import SlotResult
    from harness.orchestrator import run as orchestrator_module
    from harness.slots import claude_code
    import harness.slots.claude_history as history
    import harness.execution.interceptors as interceptors
    import swebench.fork_eval as loop
    import swebench_pro.grade as grading
    from swebench_pro.bench import SWEbenchPro

    old = tmp_path / 'old.jsonl'
    old.write_text('old history')
    binary = tmp_path / 'cli'
    binary.write_text('binary')
    continuation = {'snapshot_id': 'saved-disk', 'step': 12, 'cut': 'cut-uuid', 'source_session': 'old-session',
                    'source_journal': str(old), 'journal_sha256': hashlib.sha256(old.read_bytes()).hexdigest(),
                    'transcript_sha256': 'native-hash', 'actor_seconds_remaining': 2700,
                    'actor_seconds_charged': 900, 'tool_seconds_charged': 123, 'consecutive_timeouts': 0}
    item = {'index': 0, 'id': 'task', 'image': 'image', 'data': 'data.jsonl',
            'recovery_action': 'resume', 'continuation': continuation}
    if mode == 'regrade':
        old.write_text(json.dumps({'type': 'checkpoint.captured', 'reason': 'captured',
                                  'snapshot_id': 'saved-disk', 'step': 12}) + '\n')
        execution = tmp_path / 'old-execution.json'
        execution.write_text(json.dumps({'run_id': 'parent', 'journal_path': str(old),
                                         'status': 'completed', 'usage': {'cost_usd': 1.5}}))
        item.pop('continuation')
        item.update(recovery_action='regrade', regrade={
            'source_journal': str(old), 'journal_sha256': hashlib.sha256(old.read_bytes()).hexdigest(),
            'source_execution': str(execution), 'execution_sha256': hashlib.sha256(execution.read_bytes()).hexdigest(),
            'snapshot_id': 'saved-disk'})
    original_history = old.read_text()
    manifest = {'tasks': [item], 'recovery': True, 'tool_timeout_s': 450, 'total_tool_seconds': 1800,
                'consecutive_timeouts': 3, 'sdk_version': 'test', 'bundled_cli': str(binary),
                'bundled_cli_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(), 'sha256': {},
                'slot': 'claude-code', 'model': 'model', 'actor_timeout_s': 3600, 'verifier_timeout_s': 3600,
                'cpu': 4, 'memory_mb': 12288,
                'prepare_workers': 1, 'grade_workers': 1}
    if mode == 'regrade':
        manifest['runtime_port'] = 34122
        manifest['agent_network'] = 'deny'
        manifest['verifier_network'] = 'deny'
        item['verifier_network'] = 'allow'
        item['collector_runtime_port'] = 3000
    if isolated:
        manifest['failure_policy'] = 'isolated'
    durable_json(tmp_path / 'manifest.json', manifest)
    durable_json(tmp_path / 'all-images-ready.json', {'ready': True})
    durable_json(tmp_path / 'image-ready/000.json', {'ok': True, 'image': 'image', 'snapshot_id': 'ready-base',
                                                   'runtime_port': manifest.get('runtime_port', 3000)})
    monkeypatch.setattr(resume, 'disk_manifest_valid', lambda identifier: True)
    monkeypatch.setattr(resume, 'install_template_health', lambda: None)
    monkeypatch.setattr(worker, 'version', lambda package: 'test')
    monkeypatch.setattr(worker, 'api', lambda route: [])
    arguments = ['worker', str(tmp_path), '0']
    shard = tmp_path / 'shard-000'
    if isolated:
        from swebench_pro.retry_queue import RetryQueue

        request = RetryQueue(tmp_path, manifest).claim(0)
        arguments.append(str(request))
        shard = request.parent
    monkeypatch.setattr(sys, 'argv', arguments)
    monkeypatch.setattr(history, 'find_prefix_source', lambda *args: SimpleNamespace(sha256='native-hash'))
    monkeypatch.setattr(history, 'prepare_prefix', lambda *args: {'resume_session_id': 'registered-prefix', 'manifest_path': 'prefix.json'})
    seen = []

    class FakeOrchestrator:
        def _wire_checkpoints(self, spec, journal, provisioned=None):
            return SimpleNamespace(exact_mode=True)

        def run(self, spec):
            assert mode != 'regrade', 'Regrade-only recovery called the actor'
            seen.append(spec)
            from harness.core.journal import JournalWriter

            provisioned = SimpleNamespace(tracker=interceptors.MutationTracker(),
                                          session=SimpleNamespace(sandbox_id='fake-sandbox', on_swap=[]))
            with JournalWriter(spec.journal_path, run_id='parent') as journal:
                self._wire_checkpoints(spec, journal, provisioned)
                if actor_fault:
                    journal.emit('checkpoint.captured', reason='execution_uncertain', step=1)
            return orchestrator_module.RunOutcome(run_id='parent', journal_path=spec.journal_path,
                                                  status='completed', usage={})

    monkeypatch.setattr(orchestrator_module, 'Orchestrator', FakeOrchestrator)
    if actor_fault:
        monkeypatch.setattr(loop, 'grade_attempt', lambda *args: pytest.fail('Uncertain actor was graded'))
    if grade_kind:
        artifacts = tmp_path / 'verifier'
        artifacts.mkdir()
        for name in ('metadata.json', 'grade.json', 'verifier-logs.tar.gz'):
            (artifacts / name).write_text('fixture')
        monkeypatch.setattr(loop, 'grade_attempt', lambda *args: loop.Grade(
            error='grader transport failure' if grade_kind == 'infra' else None,
            resolved=False, verifier_artifacts=str(artifacts)))
    for module, name in [(loop, 'Orchestrator'), (loop, 'grade_attempt'), (claude_code, 'ClaudeCodeSlot'),
                         (interceptors, 'MutationTracker'), (grading, 'SandboxSession'), (SWEbenchPro, 'prepare_image')]:
        monkeypatch.setattr(module, name, getattr(module, name))

    def fake_main(arguments):
        assert arguments[arguments.index('--pro-cpus') + 1] == '4'
        assert arguments[arguments.index('--pro-memory-mb') + 1] == '12288'
        if mode == 'regrade':
            assert arguments[arguments.index('--pro-runtime-port') + 1] == '34122'
            assert arguments[arguments.index('--pro-collector-runtime-port') + 1] == '3000'
            assert arguments[arguments.index('--agent-network') + 1] == 'deny'
            assert arguments[arguments.index('--verifier-network') + 1] == 'allow'
        else:
            assert '--pro-runtime-port' not in arguments
            assert '--agent-network' not in arguments and '--verifier-network' not in arguments
        out = Path(arguments[arguments.index('-o') + 1]) / 'task'
        out.mkdir()
        spec = orchestrator_module.RunSpec(prompt='fresh prompt', run_id='parent', sandbox_image='saved-disk',
                                           journal_path=out / 'parent.jsonl', backend={'microvm': {}}, timeout_s=3600)
        outcome = loop.Orchestrator().run(spec)
        if actor_fault:
            result = loop.grade_attempt(outcome)
            assert result.error
        elif grade_kind:
            loop.grade_attempt(outcome)
        return 0

    monkeypatch.setattr(loop, 'main', fake_main)
    worker.main()
    if isolated:
        assert not (tmp_path / 'STOP_REQUEST.json').exists()
        assert json.loads((shard / 'worker.json').read_text())['attempt_id']
        assert not json.loads((tmp_path / 'shard-000/worker.json').read_text()).get('finished_at')
        if actor_fault:
            state = json.loads((shard / 'worker.json').read_text())
            assert state['retryable'] is True and state['retry_kind'] == 'actor'
            assert state['execution_uncertain'] is True
        if grade_kind:
            state = json.loads((shard / 'worker.json').read_text())
            assert state['retryable'] is (grade_kind == 'infra')
            assert state['evidence_valid'] is (grade_kind == 'unresolved')
            if grade_kind == 'infra':
                assert state['retry_kind'] == 'grading'
    if mode == 'regrade':
        assert seen == []
        assert (shard / 'task/parent.jsonl').read_text() == original_history
        assert json.loads((shard / 'task/regrade-origin.json').read_text())['actor_reexecuted'] is False
        assert old.read_text() == original_history
        assert not (tmp_path / 'actor-workspaces').exists()
        return
    assert len(seen) == 1
    assert seen[0].resume_session_id == 'registered-prefix'
    assert seen[0].extra['resume_session_at'] == 'cut-uuid'
    assert seen[0].sandbox_image == 'saved-disk' and seen[0].timeout_s == 2700
    assert seen[0].origin['tool_seconds_charged'] == 123
    events = [json.loads(line) for line in Path(seen[0].journal_path).read_text().splitlines()]
    restored = next(event for event in events if event.get('restored'))
    assert restored['snapshot_id'] == 'saved-disk' and restored['step'] == 0 and restored['source_step'] == 12
    assert old.read_text() == 'old history'
