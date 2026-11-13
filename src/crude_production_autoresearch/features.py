"""Editable baseline: scalar context; experiments may aggregate supplied history."""
import numpy as np


def build_features(frame):
    out = frame.drop(columns='history', errors='ignore').copy()
    for col in out.select_dtypes(exclude='number'):
        values = out[col].astype('string')
        out[col] = values.astype(object).where(values.notna(), np.nan)
    return out
