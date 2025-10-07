import os, json
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Ethics Curve Lab — Dashboard", layout="centered")
st.title("Ethics Curve Lab — CES Dashboard")

reports_dir = st.text_input("Reports directory", "reports")
log_path = os.path.join(reports_dir, "audit_log.json")

if os.path.exists(log_path):
    data = json.load(open(log_path,'r'))
    st.success(f"Loaded {len(data)} records")
    ces = [d['ces'] for d in data]
    times = [d['time_index'] for d in data]
    fig, ax = plt.subplots()
    ax.plot(times, ces, marker='o')
    ax.set_title("Curved Ethics Score (CES) over Time")
    ax.set_xlabel("Time Slice")
    ax.set_ylabel("CES")
    ax.grid(True)
    st.pyplot(fig)

    st.subheader("Audit Actions")
    for d in data:
        st.markdown(f"""
**Slice {d['time_index']}** — CES: **{d['ces']:.3f}**  
- Fairness: {d['fairness']:.3f} · Harm: {d['harm']:.3f} · Transparency: {d['transparency']:.3f}  
**Actions:** {', '.join(d['actions']) if d['actions'] else '—'}
        """)
else:
    st.info("No audit_log.json found yet. Run the demo first (examples/adult_income_demo.py).")
