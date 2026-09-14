import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from harness.execution.templates import env_salt, template_name
from swebench_pro.recovery import durable_json, read_json


@pytest.fixture
def migration(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[3] / 'artifacts/pro-template-memory12-20260909/migrate.py'
    if not path.is_file():
        pytest.skip('Local migration tooling is not bundled with the standalone Ash repository')
    spec = importlib.util.spec_from_file_location('pro_template_migration', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ('OUTPUT', 'BACKUP', 'RUN', 'STORE'):
        directory = tmp_path / name.lower()
        directory.mkdir()
        monkeypatch.setattr(module, name, directory)
    return module


def test_template_inventory_matches_exact_pro_identity_and_shape(migration):
    environment = ['PATH=/bin']
    shape = {'cpu': 4, 'memory_mb': 16384}
    alias = template_name('pro-image', 'runtime', 3000, shape, salt=env_salt(environment))
    durable_json(migration.BACKUP / 'template-identity.json',
                 {'fingerprint': 'runtime', 'port': 3000, 'resources': shape})
    durable_json(migration.BACKUP / 'manifest.json',
                 {'tasks': [{'index': 0, 'id': 'task', 'image': 'pro-image',
                             'continuation': {'snapshot_id': 'saved-trajectory'}}]})
    for identifier, name, memory, source in [
        ('old', alias, 16384, 'Template'),
        ('old-retry', alias + '-r1', 16384, 'Template'),
        ('other', 'ash-swebench-unrelated', 16384, 'Template'),
        ('new', alias, 12288, 'Template'),
        ('saved-trajectory', alias, 16384, 'Sandbox'),
    ]:
        durable_json(migration.STORE / 'catalog/records' / f'{identifier}.json',
                     {'id': identifier, 'alias': name, 'source': {source: {}},
                      'resources': {'memory_mib': memory, 'cpu_count': 4},
                      'committed': {'image_configs': [{'mountPath': '/', 'config': {'Env': environment}}]}})
    migration.inventory()
    plan = read_json(migration.OUTPUT / 'old-templates.json')
    assert {row['record']['id'] for row in plan['targets']} == {'old', 'old-retry'}
    assert plan['protected_snapshots'] == ['saved-trajectory']


def test_retirement_refuses_incomplete_replacements(migration, monkeypatch):
    import swebench_pro.resume

    monkeypatch.setattr(migration, 'load_cleanup', lambda: SimpleNamespace(api=lambda route: []))
    monkeypatch.setattr(swebench_pro.resume, 'require_images_ready', lambda *args: {'task': {}})
    durable_json(migration.RUN / 'manifest.json', {'memory_mb': 12288, 'prepare_only': True})
    with pytest.raises(RuntimeError, match='All 731 replacements'):
        migration.retire()
    assert not (migration.OUTPUT / 'retirement').exists()


def test_retirement_refuses_active_sandboxes(migration, monkeypatch):
    monkeypatch.setattr(migration, 'load_cleanup', lambda: SimpleNamespace(api=lambda route: [{}]))
    with pytest.raises(RuntimeError, match='Active sandboxes'):
        migration.retire()


def test_memory_cleanup_refuses_active_sandboxes(migration):
    with pytest.raises(RuntimeError, match='Active sandboxes'):
        migration.reclaim_memory(SimpleNamespace(api=lambda route: [{}]))


def test_retirement_checks_replacement_catalog_resources(migration, monkeypatch):
    import swebench_pro.resume

    monkeypatch.setattr(migration, 'load_cleanup', lambda: SimpleNamespace(api=lambda route: []))
    monkeypatch.setattr(migration, 'verify_preserved', lambda: None)
    ready = {f'task-{index}': {'snapshot_id': 'replacement', 'template_id': 'template'} for index in range(731)}
    monkeypatch.setattr(swebench_pro.resume, 'require_images_ready', lambda *args: ready)
    durable_json(migration.RUN / 'manifest.json', {'memory_mb': 12288, 'prepare_only': True})
    durable_json(migration.OUTPUT / 'old-templates.json',
                 {'targets': [{'task': 'task-0', 'record': {'id': 'old'}}]})
    durable_json(migration.STORE / 'catalog/records/replacement.json', {'resources': {'memory_mib': 16384}})
    with pytest.raises(RuntimeError, match='Replacement catalog memory mismatch'):
        migration.retire()
    assert not (migration.OUTPUT / 'retirement').exists()
