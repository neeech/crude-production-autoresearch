import pandas as pd
import pytest

from crude_production_autoresearch.history import CONTEXT, add_history
from crude_production_autoresearch.preprocess import load_production, prepare, temporal_split


def test_calendar_gap_and_late_record_are_not_visible(production_data):
    d = production_data
    origin = pd.Timestamp('2023-04-01')
    d.loc[d.date.eq('2023-01-01'), 'loaded'] = pd.Timestamp('2023-05-01')
    d = d[~d.date.eq('2022-12-01')]
    row = add_history(d).set_index('date').loc[origin]
    assert row.history[0]['months_ago'] == 2
    assert 3 not in [e['months_ago'] for e in row.history]
    assert 4 not in [e['months_ago'] for e in row.history]
    assert all(2 <= e['months_ago'] <= 13 for e in row.history)


def test_current_and_lag1_outcomes_do_not_change_forecast_input(production_data):
    d = production_data
    before = add_history(d).set_index('date').loc['2023-04-01']
    d.loc[d.date.ge('2023-03-01'), ['prod_pet', 'tipoestado']] = [999., 'future']
    after = add_history(d).set_index('date').loc['2023-04-01']
    assert before.history == after.history
    for col in CONTEXT:
        assert before['lag2_' + col] == after['lag2_' + col]


def test_eligibility_uses_past_and_keeps_future_zero(production_data):
    d = production_data
    d.loc[d.date.eq('2023-04-01'), 'prod_pet'] = 0.
    rows = add_history(d).set_index('date')
    assert rows.loc['2023-04-01', 'target'] == 0.
    assert pd.Timestamp('2023-06-01') not in rows.index


def test_late_labels_are_purged(production_data):
    rows = add_history(production_data)
    rows.loc[rows.date.eq('2022-11-01'), 'target_loaded'] = pd.Timestamp('2023-02-01')
    parts = temporal_split(rows, '2023-01-01', '2025-01-01')
    assert not (parts['train'].date.eq('2022-11-01').any())
    assert parts['train'].target_loaded.lt('2023-01-01').all()


def test_ingestion_duplicates_and_export_no_current_values(tmp_path, production_data):
    p = tmp_path
    d = production_data
    d['anio'], d['mes'], d['fechaingreso'] = d.date.dt.year, d.date.dt.month, d.loaded
    raw = p / 'raw.csv'
    d.to_csv(raw, index=False)
    loaded = load_production(raw)
    assert loaded.idpozo.iloc[0] == '001'
    config = {'raw_path': str(raw), 'output_dir': str(p / 'processed'), 'max_history_months': 13,
              'train_start': '2022-01-01', 'validation_start': '2023-01-01', 'test_start': '2025-01-01'}
    manifest = prepare(config)
    x = pd.read_parquet(p / 'processed/validation_X.parquet')
    assert {'prod_pet', 'idpozo', 'loaded', 'target', 'date'}.isdisjoint(x)
    assert manifest['primary_metric'] == 'mae'
    pd.concat([d, d.iloc[:1]]).to_csv(raw, index=False)
    with pytest.raises(ValueError, match='Duplicate'):
        load_production(raw)
