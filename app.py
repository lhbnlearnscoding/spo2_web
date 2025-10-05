# app.py
import io
import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="SpO₂ Exercise Monitor", layout="wide")

# ---------- Helpers ----------
def normalize_columns(df: pd.DataFrame):
    cols = [c.lower().strip() for c in df.columns]
    df.columns = cols
    # chấp nhận nhiều tên cột thời gian phổ biến
    time_cols = [c for c in ["time", "t", "timestamp", "second", "seconds"] if c in df.columns]
    spo2_cols = [c for c in ["spo2", "sao2", "spo2_pct"] if c in df.columns]

    if not time_cols or not spo2_cols:
        raise ValueError("File CSV cần có cột thời gian (time/t/second/...) và cột SpO₂ (spo2/SpO2/sao2).")

    tcol = time_cols[0]
    scol = spo2_cols[0]
    df = df[[tcol, scol]].rename(columns={tcol: "time", scol: "spo2"})
    # ép kiểu
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df["spo2"] = pd.to_numeric(df["spo2"], errors="coerce")
    df = df.dropna().sort_values("time")
    return df.reset_index(drop=True)

def moving_avg(x: np.ndarray, w: int):
    if w <= 1:
        return x
    cumsum = np.cumsum(np.insert(x, 0, 0))
    out = (cumsum[w:] - cumsum[:-w]) / float(w)
    # pad để chiều dài khớp
    pad = np.concatenate([np.full(w-1, out[0]), out])
    return pad

def detect_events(df: pd.DataFrame, thr: float, min_dur: int):
    """
    Phát hiện các khoảng thời gian spo2 < thr kéo dài >= min_dur (tính theo 'time' đơn vị giây).
    Trả về list dict: start, end, duration, min_spo2, mean_spo2
    """
    events = []
    below = df["spo2"].values < thr
    t = df["time"].values.astype(float)

    start_idx = None
    for i, b in enumerate(below):
        if b and start_idx is None:
            start_idx = i
        if (not b or i == len(below)-1) and start_idx is not None:
            end_idx = i if not b else i  # inclusive
            dur = t[end_idx] - t[start_idx] + (t[1]-t[0] if len(t) > 1 else 1.0)
            if dur >= min_dur:
                seg = df.iloc[start_idx:end_idx+1]
                events.append({
                    "start": float(seg["time"].iloc[0]),
                    "end": float(seg["time"].iloc[-1]),
                    "duration_s": float(round(dur, 2)),
                    "min_spo2": int(seg["spo2"].min()),
                    "mean_spo2": float(round(seg["spo2"].mean(), 2)),
                })
            start_idx = None
    return events

def make_report(df, events):
    out = {
        "total_samples": [len(df)],
        "duration_s": [float(df["time"].iloc[-1] - df["time"].iloc[0]) if len(df) > 1 else 0.0],
        "min_spo2": [int(df["spo2"].min()) if not df.empty else None],
        "mean_spo2": [float(round(df["spo2"].mean(), 2)) if not df.empty else None],
        "n_events": [len(events)]
    }
    return pd.DataFrame(out)

# ---------- Sidebar controls ----------
st.sidebar.title("⚙️ Analysis Settings")
thr = st.sidebar.slider("SpO₂ alert threshold (%)", min_value=80, max_value=98, value=90, step=1)
min_dur = st.sidebar.slider("Minimum event duration (seconds)", min_value=1, max_value=60, value=10, step=1)
smooth_w = st.sidebar.slider("Smoothing (moving average window)", min_value=1, max_value=15, value=3, step=1)
st.sidebar.markdown("---")

mode = st.sidebar.radio("Data source", ["Upload CSV", "Simulated exercise data"], index=0)

# ---------- Data input ----------
st.title("🏃 SpO₂ Exercise Monitor")
st.caption("Theo dõi SpO₂ khi tập thể dục, phát hiện các đợt tụt SpO₂ dưới ngưỡng.")

if mode == "Upload CSV":
    st.markdown("**Example CSV format:**")
    st.code("time,spo2\n0,98\n1,97\n2,95\n3,92\n4,88\n5,85\n6,89\n7,91\n8,94\n9,96\n", language="csv")
    up = st.file_uploader("Upload CSV file (must contain time & spo2 columns)", type=["csv"])
    if up is not None:
        df_raw = pd.read_csv(up)
        try:
            df = normalize_columns(df_raw)
        except Exception as e:
            st.error(f"CSV read error: {e}")
            st.stop()
else:
    st.info("Simulating 15 minutes of SpO₂ data with a few exercise-induced drops.")
    rng = np.random.default_rng(42)
    T = 900
    time_s = np.arange(T)
    base = rng.normal(97.5, 0.6, size=T)
    df = pd.DataFrame({"time": time_s, "spo2": np.clip(base, 90, 100).round()})
    for start in [120, 420, 700]:
        dur = rng.integers(12, 28)
        drop = rng.normal(87, 1.2, size=dur)
        df.loc[start:start+dur-1, "spo2"] = np.clip(drop, 80, 95).round()
    df["spo2"] = np.clip(df["spo2"] + rng.normal(0, 0.5, size=T), 80, 100).round()

if 'df' not in locals():
    st.stop()

if df.empty or df["time"].nunique() < 2:
    st.warning("Dataset is too short or invalid.")
    st.stop()

# ---------- Smoothing ----------
df_plot = df.copy()
df_plot["spo2_smooth"] = moving_avg(df_plot["spo2"].to_numpy(), smooth_w)

# ---------- Detect events ----------
events = detect_events(df_plot[["time", "spo2_smooth"]].rename(columns={"spo2_smooth": "spo2"}), thr, min_dur)
rep = make_report(df_plot, events)

# ---------- Charts ----------
col_a, col_b = st.columns([2, 1])

with col_a:
    st.subheader("📈 SpO₂ over time")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df_plot["time"], df_plot["spo2"], alpha=0.3, label="SpO₂ (raw)")
    ax.plot(df_plot["time"], df_plot["spo2_smooth"], label=f"SpO₂ (smoothed, w={smooth_w})")
    ax.axhline(thr, linestyle="--", label=f"Threshold {thr}%", linewidth=1)
    for ev in events:
        ax.axvspan(ev["start"], ev["end"], alpha=0.15, color="red")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("SpO₂ (%)")
    ax.set_ylim(80, 100)
    ax.legend(loc="lower right")
    st.pyplot(fig)

with col_b:
    st.subheader("📊 Session summary")
    st.table(rep)

st.markdown("### ⚠️ Detected desaturation events")
if not events:
    st.success("No SpO₂ desaturation events below threshold were found.")
else:
    st.error(f"Found **{len(events)}** desaturation events:")
    st.dataframe(pd.DataFrame(events))

# ---------- Export ----------
st.markdown("### ⬇️ Export report")
csv_buf = io.StringIO()
pd.DataFrame(events).to_csv(csv_buf, index=False)
st.download_button("Download events as CSV", data=csv_buf.getvalue(),
                   file_name="spo2_events.csv", mime="text/csv")
