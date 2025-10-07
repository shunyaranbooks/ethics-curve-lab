import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

def _make_synthetic_adult_like(n=5000, seed=42, shift=0.0):
    rng = np.random.default_rng(seed)
    X, y = make_classification(n_samples=n, n_features=8, n_informative=5, n_redundant=1,
                               n_clusters_per_class=2, flip_y=0.03, random_state=seed)
    df = pd.DataFrame(X, columns=[f'f{i}' for i in range(X.shape[1])])
    df['label'] = y
    # Protected attributes (synthetic)
    # sex: 0/1 ; race: 0/1 (binary simplification for demo)
    df['sex'] = (rng.random(n) > 0.5 + shift).astype(int)  # drift sex proportion with shift
    df['race'] = (rng.random(n) > 0.6 - shift/2).astype(int)
    # Slight correlation drift: feature shift
    for c in ['f0','f1']:
        df[c] = df[c] + shift
    return df

def load_synthetic_timeslices(k=4, n_per=4000, base_seed=7):
    """Yield k time slices with gradual drift."""
    for t in range(k):
        shift = 0.0 + 0.15*t  # increase drift each slice
        df = _make_synthetic_adult_like(n=n_per, seed=base_seed+t, shift=shift)
        # simple chronological split train/test
        split = int(len(df)*0.7)
        train = df.iloc[:split].reset_index(drop=True)
        test  = df.iloc[split:].reset_index(drop=True)
        meta = {
            'protected': ['sex','race'],
            'time_index': t,
            'shift': shift
        }
        yield t, train, test, meta

def load_csv_timeslices(csv_path, protected, time_col=None, k=4):
    """Optional CSV loader. Splits by time_col quantiles or rows if no time_col."""
    df = pd.read_csv(csv_path)
    if time_col and time_col in df.columns:
        df = df.sort_values(time_col)
    chunks = np.array_split(df, k)
    for t, chunk in enumerate(chunks):
        split = int(len(chunk)*0.7)
        train = chunk.iloc[:split].reset_index(drop=True)
        test  = chunk.iloc[split:].reset_index(drop=True)
        meta = {'protected': protected, 'time_index': t}
        yield t, train, test, meta
