import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

FEATURE_EXCLUDE = set(['label','sex','race'])

def _XY(df):
    y = df['label'].astype(int).values
    X = df[[c for c in df.columns if c not in FEATURE_EXCLUDE]].values
    return X, y

def train_eval_model(train_df, test_df, protected=('sex','race')):
    Xtr, ytr = _XY(train_df)
    Xte, yte = _XY(test_df)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(Xtr, ytr)
    prob = clf.predict_proba(Xte)[:,1]
    pred = (prob >= 0.5).astype(int)
    acc = accuracy_score(yte, pred)
    try:
        auc = roc_auc_score(yte, prob)
    except Exception:
        auc = float('nan')
    return {
        'pred': pred,
        'prob': prob,
        'y_true': yte,
        'acc': acc,
        'auc': auc,
        'protected': {p: test_df[p].values for p in protected},
        'features': [c for c in test_df.columns if c not in FEATURE_EXCLUDE]
    }
