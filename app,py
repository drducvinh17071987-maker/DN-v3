# app_v3.py
# DN v3 demo: 3 tabs (RR / SpO2 / HR), 30-point input, output table (Level10 + Trend10 + note)
# No charts. Deterministic computation. Illustrative only.

import re
import numpy as np
import pandas as pd
import streamlit as st


# -----------------------------
# Core helpers
# -----------------------------
def parse_series(text: str, default_start: int = 0) -> pd.DataFrame:
    """
    Accepts:
      - One column: values only (one per line)
      - Two columns CSV: time,value (header optional)
    Returns dataframe with columns: time_min, value
    """
    text = (text or "").strip()
    if not text:
        return pd.DataFrame(columns=["time_min", "value"])

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    # If commas appear, try CSV parsing.
    has_comma = any("," in ln for ln in lines)

    if has_comma:
        # Remove possible header if it contains letters
        # Accept "time_min,value" or "time,value" etc.
        rows = []
        for ln in lines:
            parts = [p.strip() for p in ln.split(",")]
            if len(parts) < 2:
                continue
            # Skip header-like rows
            if re.search(r"[A-Za-z]", parts[0] + parts[1]):
                continue
            try:
                t = float(parts[0])
                v = float(parts[1])
                rows.append((t, v))
            except ValueError:
                continue
        df = pd.DataFrame(rows, columns=["time_min", "value"])
        if df.empty:
            return df
        # Ensure sorted by time
        df = df.sort_values("time_min").reset_index(drop=True)
        return df

    # One value per line
    vals = []
    for ln in lines:
        # Skip header-like
        if re.search(r"[A-Za-z]", ln):
            continue
        try:
            vals.append(float(ln))
        except ValueError:
            pass

    df = pd.DataFrame({"time_min": np.arange(default_start, default_start + len(vals)), "value": vals})
    return df


def compute_confirmed_v3(
    df: pd.DataFrame,
    system: str,
    N: int,
    eps: float,
    K: float,
    level_good: float,
    level_bad: float,
    high_is_bad: bool,
) -> pd.DataFrame:
    """
    Computes:
      Level10: based on rolling mean of value over N samples (inclusive).
      Trend10: two-point long-range compare i vs i-N.
    Returns columns:
      time_min, raw, T_levelN, E_levelN, T_trendN, E_trendN, note
    """
    out = df.copy()
    out = out.rename(columns={"value": "raw"})
    out["system"] = system

    # Level window mean
    out["mean_N"] = out["raw"].rolling(window=N, min_periods=N).mean()

    # Level T
    # For SpO2 (low is bad): T = (good - mean)/ (good - bad)
    # For RR/HR (high is bad): T = (mean - good)/ (bad - good)
    denom = (level_good - level_bad) if not high_is_bad else (level_bad - level_good)
    denom = float(denom)

    if not high_is_bad:
        t_level = (level_good - out["mean_N"]) / denom
    else:
        t_level = (out["mean_N"] - level_good) / denom

    out["T_levelN"] = t_level.clip(lower=0.0, upper=1.0)
    out["E_levelN"] = 1.0 - out["T_levelN"] ** 2

    # Trend (two-point long-range): compare i vs i-N
    out["raw_prevN"] = out["raw"].shift(N)
    out["pct_dN"] = 100.0 * (out["raw"] - out["raw_prevN"]) / out["raw_prevN"]

    out["T_raw"] = out["pct_dN"] / float(K)
    out["T_trendN"] = (1.0 - float(eps)) * np.tanh(out["T_raw"])
    out["E_trendN"] = 1.0 - out["T_trendN"] ** 2

    # Clean unavailable rows
    out.loc[out["mean_N"].isna(), ["T_levelN", "E_levelN"]] = np.nan
    out.loc[out["raw_prevN"].isna(), ["T_trendN", "E_trendN", "pct_dN"]] = np.nan

    # -----------------------------
    # Notes (simple, 1 main label per row)
    # -----------------------------
    # Thresholds are illustrative for tagging only (not clinical).
    # Use magnitude of T_trend as a proxy for drift strength.
    trend_mag = out["T_trendN"].abs()
    trend_sign = np.sign(out["T_trendN"])

    # Trend labels
    out["flag_pronounced"] = trend_mag >= 0.85
    out["flag_trend_active"] = (trend_mag >= 0.45) & (~out["flag_pronounced"])

    # Sustained drift: 3 consecutive TREND_ACTIVE or PRONOUNCED with same direction
    drift_flag = (trend_mag >= 0.45).astype(int)  # includes pronounced
    same_dir = (trend_sign == trend_sign.shift(1)) & (trend_sign == trend_sign.shift(2))
    out["flag_sustained"] = (drift_flag.rolling(3).sum() == 3) & same_dir

    # Recovery trend: sign flips from negative to positive with meaningful magnitude
    out["flag_recovery_trend"] = (out["T_trendN"] >= 0.25) & (out["T_trendN"].shift(1) <= -0.25)

    # Low reserve (Level): T_level >= 0.50 (E<=0.75)
    out["flag_low_reserve"] = out["T_levelN"] >= 0.50

    # Level recovery: E_level increasing 2 consecutive steps (when available)
    out["flag_level_recovery"] = (out["E_levelN"] > out["E_levelN"].shift(1)) & (out["E_levelN"].shift(1) > out["E_levelN"].shift(2))

    # Priority: one label per row
    def pick_label(row) -> str:
        if pd.isna(row["raw"]):
            return ""
        # Keep INIT/LEVEL_AVAILABLE implicit by blanks; optional if you want
        if row.get("flag_recovery_trend", False):
            return "RECOVERY_TREND"
        if row.get("flag_pronounced", False):
            return "PRONOUNCED_TREND"
        if row.get("flag_sustained", False):
            return "SUSTAINED_DRIFT"
        if row.get("flag_low_reserve", False):
            return "LOW_RESERVE"
        if row.get("flag_trend_active", False):
            return "TREND_ACTIVE"
        if row.get("flag_level_recovery", False):
            return "LEVEL_RECOVERY"
        return ""

    out["note"] = out.apply(pick_label, axis=1)

    # Final display columns
    display = out[["time_min", "system", "raw", "T_levelN", "E_levelN", "T_trendN", "E_trendN", "note"]].copy()

    # Rounding for readability
    for c in ["T_levelN", "E_levelN", "T_trendN", "E_trendN"]:
        display[c] = display[c].round(3)
    display["raw"] = display["raw"].round(3)
    display["time_min"] = display["time_min"].astype(int, errors="ignore")

    return display


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="DN v3 Demo (Level + Trend)", layout="wide")
st.title("DN v3 Demo — Level + Trend (RR / SpO₂ / HR)")
st.caption("Illustrative, deterministic computation. No alarms, no clinical decisions.")

N = 10
eps = 0.05

tabs = st.tabs(["RR", "SpO₂", "HR"])

# Default example snippets (30 points each)
default_rr = "\n".join(map(str, [14,14,15,15,16,16,17,17,18,19,20,21,22,23,24,25,26,27,28,29,30,30,29,29,28,28,27,27,26,26]))
default_spo2 = "\n".join(map(str, [98,98,97,97,96,96,95,94,93,92,90,90,91,92,93,94,95,96,96,96,97,97,98,98,98,99,99,99,99,99]))
default_hr = "\n".join(map(str, [78,78,79,80,81,82,83,84,85,86,94,95,96,97,98,99,100,101,102,103,104,104,105,106,107,108,108,109,110,110]))


def tab_block(
    system_name: str,
    K: float,
    good: float,
    bad: float,
    high_is_bad: bool,
    default_text: str,
):
    colL, colR = st.columns([1, 1])

    with colL:
        st.subheader(f"Input ({system_name}) — paste 30 values or time,value CSV")
        st.write(f"Parameters: **N={N}**, **ε={eps}**, **K={K}**")
        if not high_is_bad:
            st.write(f"Level anchors: good={good}, bad={bad} (low is worse)")
        else:
            st.write(f"Level anchors: good={good}, bad={bad} (high is worse)")

        txt = st.text_area(
            "Input data",
            value=default_text,
            height=260,
            help="Option A: one value per line.\nOption B: CSV lines 'time_min,value' (header optional).",
        )
        run = st.button(f"Run {system_name}", type="primary")

    with colR:
        st.subheader("Output table")
        if run:
            df = parse_series(txt)
            if df.empty or len(df) < N + 1:
                st.error(f"Need at least {N+1} points to compute Trend{N}, and at least {N} points for Level{N}.")
                st.stop()

            result = compute_confirmed_v3(
                df=df,
                system=system_name,
                N=N,
                eps=eps,
                K=K,
                level_good=good,
                level_bad=bad,
                high_is_bad=high_is_bad,
            )
            st.dataframe(result, use_container_width=True)

            st.caption(
                "Columns: T_levelN/E_levelN use rolling mean over N points. "
                "T_trendN/E_trendN use long-range two-point comparison (i ↔ i−N). "
                "Notes are illustrative tags (single label per row)."
            )
        else:
            st.info("Paste data and press Run to compute DN Level + DN Trend.")


with tabs[0]:
    # RR: high is bad, good=12, bad=30, K=25
    tab_block(system_name="RR", K=25, good=12, bad=30, high_is_bad=True, default_text=default_rr)

with tabs[1]:
    # SpO2: low is bad, good=100, bad=85, K=5
    tab_block(system_name="SpO₂", K=5, good=100, bad=85, high_is_bad=False, default_text=default_spo2)

with tabs[2]:
    # HR: high is bad, good=60, bad=120, K=20
    tab_block(system_name="HR", K=20, good=60, bad=120, high_is_bad=True, default_text=default_hr)
