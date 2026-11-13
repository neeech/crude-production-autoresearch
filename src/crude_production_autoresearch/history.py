"""Protected monthly history and recorded-availability policy."""
import pandas as pd

NUMERIC = ['prod_pet', 'prod_gas', 'prod_agua', 'tef']
CONTEXT = ['tipoestado', 'tipoextraccion', 'tipopozo', 'empresa', 'formacion',
           'cuenca', 'provincia', 'sub_tipo_recurso', 'profundidad',
           'coordenadax', 'coordenaday', 'areayacimiento']
HISTORY_FIELDS = NUMERIC + ['tipoestado', 'tipoextraccion', 'months_ago']


def add_history(data, max_months=13):
    """Only t-2..t-max_months, loaded by origin. Eligibility uses t-2 alone."""
    if max_months < 2:
        raise ValueError('History must include lag 2')
    targets = data[['idpozo', 'date', 'loaded', 'prod_pet']].rename(
        columns={'prod_pet': 'target', 'loaded': 'target_loaded'}).copy()
    targets['month_id'] = targets.date.dt.year * 12 + targets.date.dt.month
    past = data.copy()
    past['month_id'] = past.date.dt.year * 12 + past.date.dt.month
    previous = past[['idpozo', 'month_id', 'loaded'] + NUMERIC + CONTEXT].copy()
    previous['month_id'] += 2
    previous = previous.rename(columns={c: 'lag2_' + c for c in previous if c not in ['idpozo', 'month_id']})
    rows = targets.merge(previous, on=['idpozo', 'month_id'], how='inner', validate='one_to_one')
    eligible = (rows.lag2_loaded.le(rows.date) & rows.lag2_prod_pet.gt(0)
                & rows.lag2_tipopozo.str.lower().str.contains('petrol', na=False)
                & rows.lag2_cuenca.str.upper().eq('NEUQUINA') & rows.target.ge(0)
                & rows.target_loaded.notna())
    rows = rows[eligible].reset_index(drop=True)
    rows['row_id'] = range(len(rows))
    histories = [[] for _ in range(len(rows))]
    for lag in range(2, max_months + 1):
        source = past[['idpozo', 'month_id', 'loaded'] + NUMERIC + ['tipoestado', 'tipoextraccion']].copy()
        source['month_id'] += lag
        available = rows[['idpozo', 'month_id', 'date', 'row_id']].merge(
            source, on=['idpozo', 'month_id'], how='inner', validate='one_to_one')
        available = available[available.loaded.le(available.date)]
        for record in available.to_dict('records'):
            event = {c: None if pd.isna(record[c]) else record[c]
                     for c in NUMERIC + ['tipoestado', 'tipoextraccion']}
            event['months_ago'] = lag
            histories[record['row_id']].append(event)
    rows['history'] = histories
    return rows
