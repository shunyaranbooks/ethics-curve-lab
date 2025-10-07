import numpy as np

def curved_ethics_score(fair, harm, transp, gov, drift, weights=(0.35,0.20,0.20,0.15), lam=0.10, prev=None, smooth=0.05):
    """Compute CES for a time slice.

    fair, harm, transp, gov, drift in [0,1] (harm and drift are 'bad' => we invert harm)
    weights: (w1,w2,w3,w4) for (fair, (1-harm), transp, gov)
    lam: penalty for drift
    prev: previous base score before smoothing (or None for first slice)
    smooth: penalty weight for absolute change vs previous base
    
    Returns float in [0,1].
    """
    w1, w2, w3, w4 = weights
    base = w1*fair + w2*(1-harm) + w3*transp + w4*gov - lam*drift
    if prev is None:
        return float(np.clip(base, 0, 1))
    penalty = smooth * abs(base - prev)
    return float(np.clip(base - penalty, 0, 1))
