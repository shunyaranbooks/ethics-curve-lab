import numpy as np
import pandas as pd
from .utils import safe_div

def _group_mask(values, group_value):
    return (values == group_value)

def demographic_parity_diff(pred, group):
    # P(ŷ=1 | A=0) - P(ŷ=1 | A=1) absolute
    gvals = np.unique(group)
    if len(gvals) < 2: return 0.0
    rates = []
    for gv in gvals[:2]:
        m = _group_mask(group, gv)
        rates.append(safe_div(pred[m].sum(), m.sum(), default=0.0))
    return float(abs(rates[0] - rates[1]))

def tpr_tnr_by_group(y_true, pred, group):
    gvals = np.unique(group)
    out = {}
    for gv in gvals[:2]:
        m = _group_mask(group, gv)
        tp = ((y_true[m]==1) & (pred[m]==1)).sum()
        fn = ((y_true[m]==1) & (pred[m]==0)).sum()
        tn = ((y_true[m]==0) & (pred[m]==0)).sum()
        fp = ((y_true[m]==0) & (pred[m]==1)).sum()
        tpr = safe_div(tp, tp+fn, 0.0)
        tnr = safe_div(tn, tn+fp, 0.0)
        fpr = safe_div(fp, fp+tn, 0.0)
        out[gv] = dict(tpr=tpr, tnr=tnr, fpr=fpr)
    return out

def equalized_odds_diff(y_true, pred, group):
    stats = tpr_tnr_by_group(y_true, pred, group)
    keys = list(stats.keys())
    if len(keys) < 2: return 0.0
    g0, g1 = keys[0], keys[1]
    tpr_diff = abs(stats[g0]['tpr'] - stats[g1]['tpr'])
    tnr_diff = abs(stats[g0]['tnr'] - stats[g1]['tnr'])
    return float(max(tpr_diff, tnr_diff))

def fpr_gap(y_true, pred, group):
    stats = tpr_tnr_by_group(y_true, pred, group)
    keys = list(stats.keys())
    if len(keys) < 2: return 0.0
    g0, g1 = keys[0], keys[1]
    return float(abs(stats[g0]['fpr'] - stats[g1]['fpr']))

def worst_group_accuracy(y_true, pred, group):
    gvals = np.unique(group)
    accs = []
    for gv in gvals[:2]:
        m = _group_mask(group, gv)
        acc = (y_true[m] == pred[m]).mean() if m.sum()>0 else 0.0
        accs.append(acc)
    return float(min(accs)) if accs else 0.0

def composite_fairness(y_true, pred, protected_dict):
    # simple 0..1 fairness: 1 - normalized max disparity
    dps, eods, fprs, wga = [], [], [], []
    for k, group in protected_dict.items():
        dps.append(demographic_parity_diff(pred, group))
        eods.append(equalized_odds_diff(y_true, pred, group))
        fprs.append(fpr_gap(y_true, pred, group))
        wga.append(worst_group_accuracy(y_true, pred, group))
    # Normalize disparities (assuming thresholds of 0.3 are "very bad")
    disparity = max(dps + eods + fprs) if (dps or eods or fprs) else 0.0
    fairness = max(0.0, 1.0 - (disparity/0.3))
    # Blend with worst-group-accuracy
    fairness = 0.7*fairness + 0.3*min(wga) if wga else fairness
    return float(np.clip(fairness, 0, 1)), {
        'dp_max': max(dps) if dps else 0.0,
        'eo_max': max(eods) if eods else 0.0,
        'fpr_gap_max': max(fprs) if fprs else 0.0,
        'wga_min': min(wga) if wga else 0.0
    }

def harm_proxy(y_true, pred, protected_dict):
    # proxy: disparity-weighted false positive rate
    harms = []
    for k, group in protected_dict.items():
        stats = tpr_tnr_by_group(y_true, pred, group)
        keys = list(stats.keys())
        if len(keys) < 2: continue
        g0, g1 = keys[0], keys[1]
        harms.append(abs(stats[g0]['fpr'] - stats[g1]['fpr']))
    harm = float(np.mean(harms)) if harms else 0.0
    return float(np.clip(harm, 0, 1))

def transparency_proxy(eval_out, meta):
    # proxy: fraction of features documented (heuristic)
    nfeat = len(eval_out['features'])
    documented = max(int(0.8*nfeat), 1)  # assume 80% documented in demo
    return float(documented / max(nfeat, 1))

def governance_proxy():
    # simple 0.8 baseline in demo; in real, read org signals
    return 0.8
