from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from crude_production_autoresearch.evaluate import metrics
from crude_production_autoresearch.modeling import build_model
from crude_production_autoresearch.features import build_features
from crude_production_autoresearch.train import load_data


def test_regression_metrics_and_fixed_nonnegative_projection():
    assert metrics([0, 10], [-10, 8])['mae'] == 1.


@pytest.mark.parametrize('y, pred', [([1], [float('nan')]), ([1], [1, 2]), ([-1], [1])],
                         ids=['nonfinite-prediction', 'length-mismatch', 'negative-target'])
def test_metrics_reject_invalid_inputs(y, pred):
    with pytest.raises(ValueError):
        metrics(y, pred)


def test_new_features_used_and_unseen_category_no_fit():
    frame = pd.DataFrame({'last_prod_pet': [1., 2., 3., 4.], 'empresa': ['a', 'a', 'b', 'b']})
    def engineered(x):
        out = build_features(x)
        out['square'] = out.last_prod_pet ** 2
        return out
    with patch('crude_production_autoresearch.modeling.build_features', engineered):
        model = build_model(list(frame), {'seed': 42, 'min_category_frequency': 1})
    model.fit(frame, [10., 20., 30., 40.])
    assert 'numeric__square' in model.named_steps['features'].get_feature_names_out()
    before = model.named_steps['features'].named_transformers_['numeric'].named_steps['impute'].statistics_.copy()
    assert np.isfinite(model.predict(pd.DataFrame({'last_prod_pet': [100.], 'empresa': ['new']}))).all()
    np.testing.assert_equal(before, model.named_steps['features'].named_transformers_['numeric'].named_steps['impute'].statistics_)


def test_parquet_history_and_alignment(tmp_path):
    p = tmp_path
    pd.DataFrame({'last_prod_pet': [1.], 'history': [[{'prod_pet': 1., 'months_ago': 2}]]}).to_parquet(p / 'train_X.parquet')
    pd.DataFrame({'row_id': [7], 'target': [2.]}).to_parquet(p / 'train_y.parquet')
    pd.DataFrame({'row_id': [7]}).to_parquet(p / 'train_metadata.parquet')
    x, y = load_data(p, 'train')
    assert isinstance(x.history.iloc[0], list)
    assert y.tolist() == [2.]
    pd.DataFrame({'row_id': [8]}).to_parquet(p / 'train_metadata.parquet')
    with pytest.raises(ValueError, match='align'):
        load_data(p, 'train')
