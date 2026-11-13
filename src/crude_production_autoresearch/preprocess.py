"""Protected production ingestion, availability and temporal partitions."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
import numpy as np
import pandas as pd
from .history import CONTEXT, HISTORY_FIELDS, NUMERIC, add_history

SOURCE_URL = 'http://datos.energia.gob.ar/dataset/c846e79c-026c-4040-897f-1ad3543b407c/resource/b5b58cdc-9e07-41f9-b392-fb9ec68b0725/download/produccin-de-pozos-de-gas-y-petrleo-no-convencional.csv'
REQUIRED = ['idpozo', 'anio', 'mes', 'fechaingreso'] + NUMERIC + CONTEXT


def load_production(path):
    data = pd.read_csv(path, dtype={'idpozo': 'string'}, low_memory=False)
    if not set(REQUIRED).issubset(data):
        raise ValueError('Missing required production columns: ' + str(set(REQUIRED) - set(data)))
    if data.idpozo.isna().any() or data.idpozo.str.strip().eq('').any():
        raise ValueError('Missing well identifiers')
    data['date'] = pd.to_datetime(dict(year=data.anio, month=data.mes, day=1), errors='raise')
    data['loaded'] = pd.to_datetime(data.fechaingreso, format='mixed', errors='raise')
    if data[['date', 'loaded']].isna().any().any():
        raise ValueError('Missing observation or load dates')
    if data.duplicated(['idpozo', 'date']).any():
        raise ValueError('Duplicate well-month keys: resolve revisions outside the experiment')
    for col in NUMERIC + ['profundidad', 'coordenadax', 'coordenaday']:
        data[col] = pd.to_numeric(data[col], errors='raise').astype(float)
        if not np.isfinite(data[col].dropna()).all():
            raise ValueError('Non-finite source values: ' + col)
    for col in NUMERIC:
        data.loc[data[col].between(-1e-9, 0), col] = 0.
        data.loc[data[col].lt(0), col] = np.nan
    data.loc[data.profundidad.le(0), 'profundidad'] = np.nan
    data.loc[data.tef.gt(data.date.dt.days_in_month + .01), 'tef'] = np.nan
    valid_coordinates = data.coordenadax.between(-76, -52) & data.coordenaday.between(-56, -20)
    data.loc[~valid_coordinates, ['coordenadax', 'coordenaday']] = np.nan
    for col in set(CONTEXT) - {'profundidad', 'coordenadax', 'coordenaday'}:
        data[col] = data[col].astype('string').str.strip().replace('', pd.NA)
    # A pre-close timestamp cannot establish availability of a complete month.
    data.loc[data.loaded.lt(data.date + pd.offsets.MonthBegin(1)), 'loaded'] = pd.NaT
    return data


def temporal_split(data, validation_start, test_start, train_start='2015-01-01'):
    val, test, start = map(pd.Timestamp, [validation_start, test_start, train_start])
    if not start < val < test:
        raise ValueError('Require train_start < validation_start < test_start')
    parts = {'train': data[data.date.ge(start) & data.date.lt(val) & data.target_loaded.lt(val)].copy(),
             'validation': data[data.date.ge(val) & data.date.lt(test) & data.target_loaded.lt(test)].copy(),
             'test': data[data.date.ge(test)].copy()}
    if any(part.empty for part in parts.values()):
        raise ValueError('Every partition must contain rows')
    return parts


def prepare(config):
    raw = Path(config['raw_path'])
    source = load_production(raw)
    rows = add_history(source, config['max_history_months'])
    rows['forecast_month'] = rows.date.dt.month.astype('int8')
    rows['forecast_year'] = rows.date.dt.year.astype('int16')
    for name in CONTEXT:
        rows[name] = rows['lag2_' + name]
    rows['last_prod_pet'] = rows.lag2_prod_pet
    features = CONTEXT + ['forecast_month', 'forecast_year', 'last_prod_pet', 'history']
    parts = temporal_split(rows, config['validation_start'], config['test_start'], config['train_start'])
    out = Path(config['output_dir'])
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, part in parts.items():
        part = part.sort_values(['date', 'idpozo'])
        part[features].to_parquet(out / f'{name}_X.parquet', index=False)
        part[['row_id', 'target']].to_parquet(out / f'{name}_y.parquet', index=False)
        part[['row_id', 'idpozo', 'date', 'target_loaded']].to_parquet(out / f'{name}_metadata.parquet', index=False)
        summary[name] = {'rows': len(part), 'wells': int(part.idpozo.nunique()),
                         'zero_rate': float(part.target.eq(0).mean()),
                         'start': str(part.date.min().date()), 'end': str(part.date.max().date())}
    manifest = {'task': 'monthly_oil_forecast_v1', 'config': config,
                'source_url': SOURCE_URL, 'raw_sha256': hashlib.sha256(raw.read_bytes()).hexdigest(),
                'features': features, 'history_fields': HISTORY_FIELDS, 'splits': summary,
                'primary_metric': 'mae', 'unit': 'm3',
                'availability': 'start of target month; calendar lags >=2; loaded <= origin; bounded history',
                'eligibility': 't-2 oil >0, petroliferous, Neuquina, loaded by origin; observed nonnegative target',
                'limitations': 'revised snapshot; load not certified public release; context may reflect current registry; absent targets excluded, never zero-filled',
                'versions': {'pandas': pd.__version__}}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/data.json')
    parser.add_argument('--download', action='store_true', help='Download only if raw_path does not exist')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    raw = Path(config['raw_path'])
    if args.download and not raw.exists():
        raw.parent.mkdir(parents=True, exist_ok=True)
        temporary = raw.with_suffix('.part')
        with urllib.request.urlopen(SOURCE_URL, timeout=120) as response, temporary.open('wb') as output:
            while block := response.read(1024 * 1024):
                output.write(block)
        temporary.replace(raw)
    print(json.dumps(prepare(config)['splits'], indent=2))


if __name__ == '__main__':
    main()
