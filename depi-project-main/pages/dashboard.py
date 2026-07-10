import streamlit as st
import sys, os
from datetime import datetime, timedelta
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from db import load_alerts, check_connection

logo_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "logo.png")

with st.sidebar:
    if os.path.exists(logo_path):
        st.image(logo_path, use_container_width=True)
        st.divider()
    st.subheader("Filters")
    days_back = st.slider("Last N days", 1, 30, 7)
    st.divider()
    col_r, col_rs = st.columns(2)
    with col_r:
        if st.button("Refresh"):
            st.cache_resource.clear()
            st.rerun()
    with col_rs:
        if st.button("Reset"):
            st.rerun()
    st.divider()
    if st.button("Logout"):
        st.session_state["authenticated"] = False
        st.rerun()

db_ok = check_connection()
if db_ok:
    st.success("MongoDB Connected")
else:
    st.error("MongoDB Offline")

st.title("Alerts Dashboard")
st.caption("Analysis of driver safety alerts stored in MongoDB")

with st.spinner("Loading data from MongoDB..."):
    docs = load_alerts(limit=1000)

if not docs:
    st.warning("No alerts found in MongoDB yet. Run some detections first.")
    st.stop()

import pandas as pd

df = pd.DataFrame(docs)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp", ascending=False).reset_index(drop=True)

sources   = ["All"] + sorted(df["source"].unique().tolist())
sel_src   = st.sidebar.selectbox("Source", sources)
sel_state = st.sidebar.multiselect(
    "Risk State",
    options=["SAFE", "WARNING", "HIGH_RISK"],
    default=["WARNING", "HIGH_RISK"]
)

cutoff = datetime.utcnow() - timedelta(days=days_back)
fdf = df[df["timestamp"] >= cutoff]
if sel_src != "All":
    fdf = fdf[fdf["source"] == sel_src]
if sel_state:
    fdf = fdf[fdf["risk_state"].isin(sel_state)]

if fdf.empty:
    st.info("No data matches the selected filters.")
    st.stop()

total     = len(fdf)
high_risk = (fdf["risk_state"] == "HIGH_RISK").sum()
warning   = (fdf["risk_state"] == "WARNING").sum()
avg_risk  = fdf["risk_score"].mean() if "risk_score" in fdf.columns else 0.0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Alerts",   total)
c2.metric("High Risk",      int(high_risk))
c3.metric("Warnings",       int(warning))
c4.metric("Avg Risk Score", f"{avg_risk:.2f}")

st.divider()

col_left, col_right = st.columns(2)

with col_left:
    st.markdown("#### Alerts Over Time")
    fdf_time        = fdf.copy()
    fdf_time["date"] = fdf_time["timestamp"].dt.date
    daily           = fdf_time.groupby("date").size().reset_index(name="count")
    daily["date"]   = daily["date"].astype(str)
    st.bar_chart(daily.set_index("date")["count"], color="#ef4444", height=220)

with col_right:
    st.markdown("#### Most Common Violations")
    all_alerts_flat = []
    for row in fdf["alerts"]:
        if isinstance(row, list):
            all_alerts_flat.extend(row)
    if all_alerts_flat:
        counts   = Counter(all_alerts_flat).most_common(6)
        labels   = [c[0] for c in counts]
        values   = [c[1] for c in counts]
        chart_df = pd.DataFrame({"count": values}, index=labels)
        st.bar_chart(chart_df, color="#f97316", height=220)
    else:
        st.caption("No violation data available.")

st.divider()

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("#### Risk State Breakdown")
    state_counts = fdf["risk_state"].value_counts()
    color_map    = {"HIGH_RISK": "#ef4444", "WARNING": "#f97316", "SAFE": "#22c55e"}
    for state, count in state_counts.items():
        pct   = count / total * 100
        color = color_map.get(state, "#94a3b8")
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:10px;margin-bottom:8px;'>"
            f"<div style='width:14px;height:14px;background:{color};border-radius:3px;'></div>"
            f"<span style='font-weight:600;min-width:110px;'>{state}</span>"
            f"<div style='flex:1;background:#2a2a2a;border-radius:4px;height:16px;'>"
            f"<div style='width:{pct:.0f}%;background:{color};height:16px;border-radius:4px;'></div></div>"
            f"<span style='min-width:50px;text-align:right;'>{count} ({pct:.0f}%)</span>"
            f"</div>",
            unsafe_allow_html=True
        )

with col_b:
    st.markdown("#### Source Breakdown")
    src_counts = fdf["source"].value_counts()
    src_colors = {"realtime": "#06b6d4", "video": "#8b5cf6", "image": "#eab308"}
    for src, count in src_counts.items():
        pct   = count / total * 100
        color = src_colors.get(src, "#94a3b8")
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:10px;margin-bottom:8px;'>"
            f"<div style='width:14px;height:14px;background:{color};border-radius:3px;'></div>"
            f"<span style='font-weight:600;min-width:110px;'>{src}</span>"
            f"<div style='flex:1;background:#2a2a2a;border-radius:4px;height:16px;'>"
            f"<div style='width:{pct:.0f}%;background:{color};height:16px;border-radius:4px;'></div></div>"
            f"<span style='min-width:50px;text-align:right;'>{count} ({pct:.0f}%)</span>"
            f"</div>",
            unsafe_allow_html=True
        )

st.divider()

st.markdown("#### Recent Alerts")
table_df = fdf[["timestamp", "source", "risk_state", "risk_score", "alerts"]].head(20).copy()
table_df["timestamp"] = table_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
table_df["alerts"]    = table_df["alerts"].apply(lambda x: ", ".join(x) if isinstance(x, list) else x)
table_df.columns      = ["Time", "Source", "State", "Risk Score", "Alerts"]

def color_state(val):
    colors = {
        "HIGH_RISK": "color:#ef4444;font-weight:bold",
        "WARNING":   "color:#f97316;font-weight:bold",
        "SAFE":      "color:#22c55e",
    }
    return colors.get(val, "")

st.dataframe(
    table_df.style.map(color_state, subset=["State"]),
    use_container_width=True,
    hide_index=True,
)