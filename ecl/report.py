import os, json
import numpy as np
import matplotlib.pyplot as plt

def save_ces_series(ces_series, outdir="reports", fname="ces_trend.png"):
    os.makedirs(outdir, exist_ok=True)
    plt.figure()
    plt.plot(range(len(ces_series)), ces_series, marker='o')
    plt.title("Curved Ethics Score (CES) over Time")
    plt.xlabel("Time Slice")
    plt.ylabel("CES")
    plt.grid(True)
    plt.tight_layout()
    path = os.path.join(outdir, fname)
    plt.savefig(path, dpi=180)
    return path

def save_summary(summary, outdir="reports", fname="summary.json"):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, fname)
    with open(path, 'w') as f:
        json.dump(summary, f, indent=2)
    return path
