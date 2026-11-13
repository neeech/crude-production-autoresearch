"""Two baselines: train on train, evaluate validation, leave test reserved."""
import argparse
from datetime import datetime, timezone
import json
import time
from pathlib import Path
import shutil
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from .evaluate import metrics
from .modeling import build_model


def progress(message):
    print(f'[train {datetime.now().astimezone().isoformat(timespec="seconds")}] {message}', flush=True)


def load_data(directory, split):
    x = pd.read_parquet(directory / f'{split}_X.parquet')
    y = pd.read_parquet(directory / f'{split}_y.parquet')
    meta = pd.read_parquet(directory / f'{split}_metadata.parquet')
    if len(x) != len(y) or len(x) != len(meta) or not y.row_id.equals(meta.row_id):
        raise ValueError('Partition files do not align')
    forbidden = {'target', 'prod_pet', 'prod_gas', 'prod_agua', 'tef', 'idpozo', 'date', 'row_id', 'target_loaded', 'loaded'}
    if forbidden.intersection(x.columns):
        raise ValueError('Outcome or identifiers included in features')
    if 'history' in x:
        # Parquet may deserialize nested lists as NumPy arrays. Keep the prompt contract.
        x['history'] = x.history.map(lambda events: [] if events is None else list(events))
    return x, y.target.to_numpy()


def run(config):
    started = time.perf_counter()
    progress('Cargando train y validación')
    data = Path(config['data_dir'])
    train_x, train_y = load_data(data, 'train')
    val_x, val_y = load_data(data, 'validation')
    progress(f'Datos cargados: train={len(train_y)} filas; validación={len(val_y)} filas')
    out = Path(config['output_root']) / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out.mkdir(parents=True)
    shutil.copytree(Path(__file__).parent, out / 'code',
                    ignore=shutil.ignore_patterns('__pycache__'))
    (out / 'config.json').write_text(json.dumps(config, indent=2))
    shutil.copyfile(data / 'manifest.json', out / 'data_manifest.json')
    constant = float(np.median(train_y))
    model = build_model(train_x.columns.tolist(), config)
    progress(f'Iniciando model.fit: {type(model).__name__}')
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        model.fit(train_x, train_y)
    progress('model.fit terminado; prediciendo validación y calculando métricas')
    results = pd.DataFrame([
        {'model': 'constant', **metrics(val_y, np.full(len(val_y), constant))},
        {'model': 'persistence', **metrics(val_y, val_x.last_prod_pet.to_numpy())},
        {'model': 'candidate', **metrics(val_y, model.predict(val_x))},
    ])
    progress('Métricas calculadas; guardando modelo y artefactos')
    joblib.dump(model, out / 'model.joblib')
    (out / 'constant.json').write_text(json.dumps({'prediction_m3': constant}))
    results.to_csv(out / 'results.csv', index=False)
    (out.parent / 'latest.json').write_text(json.dumps({'run_dir': str(out)}, indent=2))
    print(results.to_string(index=False))
    progress(f'Finalizado en {time.perf_counter() - started:.1f}s; artefactos: {out}')
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/baselines.json')
    args = parser.parse_args()
    run(json.loads(Path(args.config).read_text()))


if __name__ == '__main__':
    main()
