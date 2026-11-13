import pandas as pd

from crude_production_autoresearch.features import build_features


def test_features_preserve_input_rows_and_history():
    frame = pd.DataFrame({'history': [[{'prod_pet': 10., 'months_ago': 2}], []],
                          'last_prod_pet': [10., 20.], 'empresa': ['a', None]}, index=[8, 3])
    before = frame.copy(deep=True)
    out = build_features(frame)
    pd.testing.assert_frame_equal(frame, before)
    assert out.index.equals(frame.index)
    assert 'history' not in out
    assert out.last_prod_pet.tolist() == [10., 20.]
