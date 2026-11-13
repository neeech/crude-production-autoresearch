import pandas as pd
import pytest

from crude_production_autoresearch.history import CONTEXT, NUMERIC


@pytest.fixture
def production_data():
    """Monthly well history shared by preprocessing and regression integration tests."""
    dates = pd.date_range('2022-01-01', '2025-06-01', freq='MS')
    data = pd.DataFrame({'idpozo': ['001'] * len(dates), 'date': dates,
                         'loaded': dates + pd.offsets.MonthBegin(1) + pd.Timedelta(days=10)})
    for col in NUMERIC:
        data[col] = 10.
    for col in CONTEXT:
        data[col] = 'context'
    data['profundidad'] = 2000.
    data['coordenadax'], data['coordenaday'] = -69., -39.
    data['cuenca'], data['tipopozo'] = 'NEUQUINA', 'Petrolífero'
    return data
