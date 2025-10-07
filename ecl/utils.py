import numpy as np
import pandas as pd

def sigmoid(x):
    return 1/(1+np.exp(-x))

def binarize(series, threshold=0.5):
    return (series >= threshold).astype(int)

def group_indices(groups):
    mapping = {}
    for i, g in enumerate(np.unique(groups)):
        mapping[g] = i
    return mapping

def safe_div(a, b, default=0.0):
    try:
        return a / b if b != 0 else default
    except Exception:
        return default
