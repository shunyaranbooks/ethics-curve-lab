# 🧩 Ethics Curve Lab (ECL) — Open Research Toolkit

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17288003.svg)](https://doi.org/10.5281/zenodo.17288003)

**Version:** v0.1.1  
**Author:** Dr. Mohammad Amir Khusru Akhtar  
**Affiliation:** Usha Martin University, Ranchi, India  
**Contact:** shunya.ran.books@gmail.com  
**Homepage:** [https://www.shunyapublications.com](https://www.shunyapublications.com)

---

## 🔍 Overview

The **Ethics Curve Lab (ECL)** operationalizes the philosophical framework from  
_**The Ethics Curve: Curved Ethics for Future Values (PhiloMind™ Book 5)**_ —  
transforming moral philosophy into a computational ethics research toolkit.

ECL measures, simulates, and governs **ethical drift** in AI systems.  
It computes a **Curved Ethics Score (CES)** across model time-slices, detects **Moral Drift**,  
and enforces **Recursive Governance** whenever CES falls below a policy threshold.

> “**Ethics is not a code. It is a trajectory.**” — *The Ethics Curve*

---

## ✨ What ECL Provides

- **Ethics-as-Trajectory:** compute CES(t) across time instead of one-off checks.  
- **Fairness Metrics:** demographic parity diff, equalized odds diff, FPR gap, worst-group accuracy.  
- **Drift Metrics:** PSI & KL divergence approximations to detect population/label shifts.  
- **Governance:** YAML-based policy triggers re-audits when CES < threshold.  
- **Reports:** JSON + PNG outputs for publication-ready ethics trends.  
- **Dashboard:** Streamlit visualization for CES and audit cycles.  

---

## 🚀 Quickstart

### 1️⃣ Installation
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
```

Or minimal:
```bash
pip install -r requirements.txt
```

### 2️⃣ Run Demo (Synthetic Adult-like Data)
```bash
python examples/adult_income_demo.py
```

This will:
- Simulate 4 time-slices with controlled drift  
- Train a simple model on each slice  
- Compute fairness, harm, transparency, and drift  
- Compute **CES(t)** and save to `reports/ces_trend.png`  
- Log governance actions to `reports/audit_log.json`  

### 3️⃣ Launch Dashboard
```bash
streamlit run app/dashboard.py
```
Then open the URL shown in your terminal.

---

## 📂 Project Structure
```
ethics-curve-lab/
├─ ecl/
│  ├─ ces.py
│  ├─ data_ingestor.py
│  ├─ drift.py
│  ├─ governance.py
│  ├─ metrics_engine.py
│  ├─ model_runner.py
│  ├─ report.py
│  └─ utils.py
├─ app/
│  └─ dashboard.py
├─ examples/
│  └─ adult_income_demo.py
├─ policies/
│  └─ default_policy.yaml
├─ reports/
├─ data/
├─ notebooks/
├─ README.md
├─ LICENSE
├─ pyproject.toml
└─ requirements.txt
```

---

## 🧮 Curved Ethics Score (CES)
\[
CES(t) = w_1·Fair(t) + w_2(1−Harm(t)) + w_3·Transp(t) + w_4·Gov(t) − λ·Drift(t) − s·|Base(t)−Base(t−1)|
\]
*s* represents a smoothness penalty discouraging sharp regressions over time.

---

## 📘 Reference Book

This toolkit is based on the concepts introduced in  
**[The Ethics Curve: Curved Ethics for Future Values (PhiloMind™ Book 5)](https://a.co/d/bSrI427)**  
by *Dr. Mohammad Amir Khusru Akhtar (Shunya Publications)*.

---

## 🧪 Re-run with your Own CSV
Place your dataset in `data/` and modify  
`examples/adult_income_demo.py` → `load_csv_timeslices(...)`.  
Document protected attributes to audit fairness and drift.

---

## 📝 Citation
Akhtar, M. A. K. (2025). *Ethics Curve Lab (ECL) v0.1.1 — Open Research Toolkit (Enhanced Release).*  
Usha Martin University, Ranchi, India. DOI: [10.5281/zenodo.17288003](https://doi.org/10.5281/zenodo.17288003)

---

## 📄 License
MIT License © 2025 Shunya Publications
