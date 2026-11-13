"""Actual regression code through the isolated Git/subprocess protocol."""
import json
import shutil

from crude_production_autoresearch.preprocess import prepare
from crude_production_autoresearch.research import ROOT, EDITABLE, git, run


def test_real_regression_loop_freezes_data_and_supplies_contract(tmp_path, production_data):
    root = tmp_path / 'repo'
    root.mkdir()
    shutil.copytree(ROOT / 'src', root / 'src', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(ROOT / 'configs', root / 'configs')
    shutil.copyfile(ROOT / 'program.md', root / 'program.md')
    (root / '.gitignore').write_text('data/\n__pycache__/\n')
    d = production_data
    d['anio'], d['mes'], d['fechaingreso'] = d.date.dt.year, d.date.dt.month, d.loaded
    raw = root / 'data/raw.csv'
    raw.parent.mkdir()
    d.to_csv(raw, index=False)
    prepare({'raw_path': str(raw), 'output_dir': str(root / 'data/processed_oil'),
             'train_start': '2022-01-01', 'validation_start': '2023-01-01',
             'test_start': '2025-01-01', 'max_history_months': 13})
    git(root, 'init')
    git(root, 'config', 'user.name', 'Oil test')
    git(root, 'config', 'user.email', 'oil@example.invalid')
    git(root, 'add', '.')
    git(root, 'commit', '-m', 'Oil regression fixture')
    seen = []
    def proposer(context):
        seen.append(context)
        return {'hypothesis': 'Verify identical features tie',
                'files': {EDITABLE[0]: context['files'][EDITABLE[0]] + '\n# tie experiment\n'}}
    output = tmp_path / 'run'
    summary = run(root, output, proposer, experiments=1, timeout=60)
    entries = [json.loads(line) for line in (output / 'history.jsonl').read_text().splitlines()]
    assert [e['status'] for e in entries] == ['baseline', 'discard']
    assert 'persistence' in entries[0]['metrics']
    assert seen[0]['data']['task'] == 'monthly_oil_forecast_v1'
    assert 'test' not in seen[0]['data']['splits']
    assert not ((output / 'data/test_X.parquet').exists())
    assert summary['best_mae'] >= 0.
