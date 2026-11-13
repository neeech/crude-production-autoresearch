"""Protected regression evaluation in cubic meters; minimize MAE."""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def metrics(y, prediction):
    y, p = np.asarray(y, dtype=float), np.asarray(prediction, dtype=float)
    if y.ndim != 1 or not len(y) or p.shape != y.shape or not np.isfinite(p).all() or not np.isfinite(y).all() or (y < 0).any():
        raise ValueError('Aligned, nonempty, finite regression vectors required')
    p = np.maximum(p, 0.)
    return {'mae': float(mean_absolute_error(y, p)),
            'rmse': float(np.sqrt(mean_squared_error(y, p)))}
