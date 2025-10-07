import os, json
import numpy as np
from ecl.data_ingestor import load_synthetic_timeslices
from ecl.model_runner import train_eval_model
from ecl.metrics_engine import composite_fairness, harm_proxy, transparency_proxy, governance_proxy
from ecl.drift import drift_stats
from ecl.ces import curved_ethics_score
from ecl.governance import enforce_policy
from ecl.report import save_ces_series, save_summary

def run_demo(policy_path='policies/default_policy.yaml', outdir='reports'):
    os.makedirs(outdir, exist_ok=True)
    prev_base = None
    ces_series = []
    details = []

    for t, train, test, meta in load_synthetic_timeslices(k=4, n_per=4000, base_seed=13):
        # Train & evaluate per slice
        eval_out = train_eval_model(train, test, protected=tuple(meta['protected']))
        fairness, fair_dbg = composite_fairness(eval_out['y_true'], eval_out['pred'], eval_out['protected'])
        harm = harm_proxy(eval_out['y_true'], eval_out['pred'], eval_out['protected'])
        transp = transparency_proxy(eval_out, meta)
        gov = governance_proxy()

        # Drift between train and test
        from ecl.model_runner import _XY
        Xtr, _ = _XY(train); Xte, _ = _XY(test)
        dstat = drift_stats(Xtr, Xte)

        # CES
        base_before_smooth = 0.35*fairness + 0.20*(1-harm) + 0.20*transp + 0.15*gov - 0.10*dstat['psi']
        ces = curved_ethics_score(fairness, harm, transp, gov, dstat['psi'], prev=prev_base, smooth=0.05)
        ces_series.append(ces)

        # Governance actions
        record = enforce_policy(ces, fairness, harm, transp, policy_path, t, report_dir=outdir)

        details.append({
            'time_index': t,
            'fairness': fairness,
            'fair_debug': fair_dbg,
            'harm': harm,
            'transparency': transp,
            'gov': gov,
            'drift': dstat,
            'ces': ces,
            'base_no_smooth': base_before_smooth,
            'actions': record['actions'],
        })
        prev_base = base_before_smooth

    img_path = save_ces_series(ces_series, outdir=outdir, fname='ces_trend.png')
    sum_path = save_summary({'series': ces_series, 'details': details}, outdir=outdir, fname='summary.json')
    print(f"Saved CES plot: {img_path}\nSaved summary: {sum_path}")

if __name__ == '__main__':
    run_demo()
