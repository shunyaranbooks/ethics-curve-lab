import numpy as np
import pandas as pd

def psi(expected, actual, bins=10, eps=1e-9):
    # Population Stability Index per feature, averaged
    try:
        e_hist, e_edges = np.histogram(expected, bins=bins, density=True)
        a_hist, _ = np.histogram(actual, bins=e_edges, density=True)
        e_pct = e_hist / (e_hist.sum()+eps)
        a_pct = a_hist / (a_hist.sum()+eps)
        vals = (a_pct - e_pct) * np.log((a_pct + eps)/(e_pct + eps))
        return float(max(0.0, np.sum(vals)))
    except Exception:
        return 0.0

def kl_divergence(p, q, eps=1e-9):
    p = p/(p.sum()+eps); q = q/(q.sum()+eps)
    return float(np.sum(p*np.log((p+eps)/(q+eps))))

def drift_stats(X_train, X_test):
    # PSI averaged over columns
    psis = []
    for i in range(X_train.shape[1]):
        psis.append(psi(X_train[:,i], X_test[:,i]))
    psi_avg = float(np.mean(psis)) if psis else 0.0
    # KL on a random feature as proxy (for demo)
    if X_train.shape[1] > 0:
        tr_hist, edges = np.histogram(X_train[:,0], bins=20, density=True)
        te_hist, _ = np.histogram(X_test[:,0], bins=edges, density=True)
        kl = kl_divergence(tr_hist, te_hist)
    else:
        kl = 0.0
    # Normalize drift to 0..1 (coarse: 0..1 after clipping at thresholds)
    drift_norm = float(np.clip((psi_avg+kl)/2.0, 0, 1))
    return {'psi': drift_norm, 'psi_raw': psi_avg, 'kl_raw': kl}
