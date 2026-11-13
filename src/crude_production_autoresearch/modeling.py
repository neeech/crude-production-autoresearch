"""Editable regression factory. Initial model: Ridge with train-fitted preprocessing."""
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from .features import build_features


def build_model(columns, config):
    """Choose columns after feature creation, including new experimental columns."""
    encoder = Pipeline([
        ('impute', SimpleImputer(strategy='constant', fill_value='__MISSING__')),
        ('encode', OneHotEncoder(handle_unknown='infrequent_if_exist',
                                 min_frequency=config['min_category_frequency'])),
    ])
    numbers = Pipeline([
        ('impute', SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)),
        ('scale', StandardScaler()),
    ])
    transform = ColumnTransformer([
        ('categorical', encoder, make_column_selector(dtype_exclude='number')),
        ('numeric', numbers, make_column_selector(dtype_include='number')),
    ], sparse_threshold=1.0)
    return Pipeline([
        ('prepare', FunctionTransformer(build_features, validate=False)),
        ('features', transform),
        ('model', Ridge(alpha=config.get('ridge_alpha', 100.), solver='lsqr')),
    ])
