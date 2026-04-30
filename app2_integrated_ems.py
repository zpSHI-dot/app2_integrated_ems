import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# =========================================================
# Substation 2.0 — EMS Buffer Dashboard using Duck Curve data
# Inputs: real aggregated demand data from Excel, not synthetic loads
# EMS: same 15-minute flat-buffer logic as ems_buffer_from_duck_curve notebook
# =========================================================

st.set_page_config(page_title="Residential Energy Hub", layout="wide", page_icon="⚡")
st.title("⚡ Substation 2.0: EMS Buffer Dashboard")
st.caption("Uses Duck Curve aggregated demand data and EMS buffer analysis, with 15-minute resolution.")

DEFAULT_DUCK_FILE = "Duck curve.xlsx"
START_TIME = "2023-01-01 00:00:00"  # Jan 1, Sunday
TIME_RESOLUTION = "15min"

# ---------------------------------------------------------
# Data loading and cleaning
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def read_excel_clean(file_or_path):
    """Read Duck Curve Excel even if the first row is stored as data instead of header."""
    raw = pd.read_excel(file_or_path, sheet_name=0, header=None)

    header_row = None
    for i in range(min(10, len(raw))):
        row_values = raw.iloc[i].astype(str).str.strip().tolist()
        if "Aggregated Power [kW]" in row_values or "Aggregated Power [W]" in row_values:
            header_row = i
            break

    if header_row is not None:
        df = raw.iloc[header_row + 1:].copy()
        df.columns = raw.iloc[header_row].astype(str).str.strip()
        df = df.reset_index(drop=True)
    else:
        df = pd.read_excel(file_or_path, sheet_name=0)
        df.columns = [str(c).strip() for c in df.columns]

    # Drop fully empty columns and rows
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
    return df


def find_demand_column(df):
    preferred = [
        "Aggregated Power [kW]",
        "Aggregated Power kW",
        "Aggregated_Power_kW",
        "demand_kw",
        "Demand_kW",
    ]
    for col in preferred:
        if col in df.columns:
            return col

    numeric_cols = []
    for col in df.columns:
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().sum() > 0:
            numeric_cols.append(col)

    keywords = ["kw", "power", "demand", "load", "aggregated", "duck"]
    scored = []
    for col in numeric_cols:
        name = str(col).lower()
        score = sum(k in name for k in keywords)
        valid = pd.to_numeric(df[col], errors="coerce").notna().sum()
        scored.append((score, valid, col))

    if not scored:
        raise ValueError("No numeric demand column found. Please check the Excel file.")

    scored.sort(reverse=True, key=lambda x: (x[0], x[1]))
    return scored[0][2]


def find_time_column(df, demand_col):
    candidates = ["timestamp", "Timestamp", "DateTime", "Datetime", "Time", "Interval Start"]
    for col in candidates:
        if col in df.columns and col != demand_col:
            return col

    for col in df.columns:
        if col == demand_col:
            continue
        parsed = pd.to_datetime(df[col], errors="coerce")
        if parsed.notna().sum() >= max(5, int(0.8 * len(df))):
            return col
    return None


@st.cache_data(show_spinner=False)
def prepare_timeseries_from_excel(file_or_path, start=START_TIME, freq=TIME_RESOLUTION):
    df = read_excel_clean(file_or_path)
    demand_col = find_demand_column(df)
    time_col = find_time_column(df, demand_col)

    demand_kw = pd.to_numeric(df[demand_col], errors="coerce")

    # The Duck Curve file usually has Day + Interval Start, not a full timestamp.
    # We preserve 15-minute resolution by constructing the full index from Jan 1.
    timestamp = None
    if "Day" in df.columns and "Interval Start" in df.columns:
        # This creates one continuous 15-minute time series starting Jan 1 Sunday.
        timestamp = pd.date_range(start=start, periods=len(df), freq=freq)
    elif time_col is not None and time_col != "Interval Start":
        timestamp = pd.to_datetime(df[time_col], errors="coerce")
        if timestamp.notna().sum() == 0 and pd.api.types.is_numeric_dtype(df[time_col]):
            timestamp = pd.to_datetime(df[time_col], unit="D", origin="1899-12-30", errors="coerce")
    else:
        timestamp = pd.date_range(start=start, periods=len(df), freq=freq)

    ts = pd.DataFrame({"timestamp": timestamp, "demand_kw": demand_kw})
    ts = ts.dropna(subset=["timestamp", "demand_kw"]).sort_values("timestamp")
    ts = ts.drop_duplicates(subset=["timestamp"], keep="first").reset_index(drop=True)
    return ts, demand_col


# ---------------------------------------------------------
# EMS buffer logic from the notebook
# ---------------------------------------------------------
def add_time_step_hours(ts):
    ts = ts.copy()
    next_time = ts["timestamp"].shift(-1)
    dt = (next_time - ts["timestamp"]).dt.total_seconds() / 3600.0
    fallback = dt[(dt > 0) & dt.notna()].median()
    if not np.isfinite(fallback) or fallback <= 0:
        fallback = 0.25
    ts["dt_hours"] = dt.fillna(fallback)
    ts.loc[ts["dt_hours"] <= 0, "dt_hours"] = fallback
    return ts


def flat_buffer_ems_one_day(day_df):
    day_df = day_df.copy()
    demand = day_df["demand_kw"].to_numpy(dtype=float)
    dt = day_df["dt_hours"].to_numpy(dtype=float)

    daily_energy_kwh = float(np.sum(demand * dt))
    day_duration_hours = float(np.sum(dt))
    mv_import_kw = daily_energy_kwh / day_duration_hours if day_duration_hours > 0 else 0.0

    day_df["mv_import_ems_kw"] = mv_import_kw
    day_df["battery_kw"] = demand - mv_import_kw  # +ve discharge, -ve charge

    soc_delta_kwh = np.cumsum((mv_import_kw - demand) * dt)
    day_df["soc_kwh"] = soc_delta_kwh - np.min(soc_delta_kwh)

    day_df["daily_energy_kwh"] = daily_energy_kwh
    day_df["required_battery_capacity_kwh"] = float(day_df["soc_kwh"].max() - day_df["soc_kwh"].min())
    day_df["required_battery_power_kw"] = float(np.max(np.abs(day_df["battery_kw"])))
    day_df["battery_mode"] = np.where(
        day_df["battery_kw"] > 0, "Discharging",
        np.where(day_df["battery_kw"] < 0, "Charging", "Idle")
    )
    return day_df


@st.cache_data(show_spinner=False)
def schedule_ems_buffer(ts):
    ts = add_time_step_hours(ts)
    ts["date"] = ts["timestamp"].dt.date

    pieces = []
    for _, group in ts.groupby("date", sort=True):
        # Use full or partial day as supplied; full day is normally 96 rows.
        pieces.append(flat_buffer_ems_one_day(group))

    result = pd.concat(pieces, ignore_index=True)
    result["time_of_day"] = result["timestamp"].dt.strftime("%H:%M")
    result["hour_decimal"] = result["timestamp"].dt.hour + result["timestamp"].dt.minute / 60
    return result


# ---------------------------------------------------------
# Sidebar inputs
# ---------------------------------------------------------
st.sidebar.header("Input data")
uploaded = st.sidebar.file_uploader("Upload Duck Curve Excel", type=["xlsx", "xls"])

try:
    source = uploaded if uploaded is not None else DEFAULT_DUCK_FILE
    ts, demand_col = prepare_timeseries_from_excel(source)
    result = schedule_ems_buffer(ts)
except Exception as exc:
    st.error(f"Could not load/analyze the Duck Curve file: {exc}")
    st.stop()

available_dates = sorted(result["date"].unique())
selected_date = st.sidebar.selectbox("Select day", available_dates, index=0)
day_df = result[result["date"] == selected_date].copy()

current_idx = st.sidebar.slider(
    "15-minute interval",
    min_value=0,
    max_value=max(len(day_df) - 1, 0),
    value=min(72, max(len(day_df) - 1, 0)),
)
current = day_df.iloc[current_idx]

st.sidebar.success(f"Loaded {len(result):,} rows from demand column: {demand_col}")
st.sidebar.download_button(
    "Download EMS analysis CSV",
    result.to_csv(index=False).encode("utf-8"),
    file_name="ems_buffer_analysis_from_duck_curve.csv",
    mime="text/csv",
)

# ---------------------------------------------------------
# Tabs
# ---------------------------------------------------------
tab1, tab2 = st.tabs(["📊 EMS Buffer Dashboard", "🔎 Final Analysis Data"])

with tab1:
    st.markdown("### 🚦 Live EMS Status")

    actual = float(current["demand_kw"])
    mv_import = float(current["mv_import_ems_kw"])
    battery = float(current["battery_kw"])
    soc = float(current["soc_kwh"])
    batt_cap = float(day_df["required_battery_capacity_kwh"].iloc[0])
    batt_power = float(day_df["required_battery_power_kw"].iloc[0])
    daily_energy = float(day_df["daily_energy_kwh"].iloc[0])

    if abs(battery) < 1e-6:
        signal = "🟢 Idle"
        signal_msg = "Demand equals flat MV import target."
    elif battery > 0:
        signal = "🔴 Battery Discharging"
        signal_msg = "Demand is above the flat MV import target; battery supplies the difference."
    else:
        signal = "🟢 Battery Charging"
        signal_msg = "Demand is below the flat MV import target; battery charges from MV import surplus."

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Actual demand", f"{actual:.2f} kW")
    m2.metric("MV import with EMS", f"{mv_import:.2f} kW")
    m3.metric("Battery action", f"{battery:.2f} kW", help="Positive = discharge, negative = charge")
    m4.metric("SoC", f"{soc:.2f} kWh")

    st.info(f"**EMS signal:** {signal} — {signal_msg}")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 📉 MV Side: EMS Buffer Concept")
        fig_mv = go.Figure()
        fig_mv.add_trace(go.Scatter(
            x=day_df["hour_decimal"], y=day_df["demand_kw"],
            mode="lines", name="Duck Curve demand", line=dict(color="royalblue", width=3)
        ))
        fig_mv.add_trace(go.Scatter(
            x=day_df["hour_decimal"], y=day_df["mv_import_ems_kw"],
            mode="lines", name="MV import with EMS", line=dict(color="darkorange", width=4)
        ))
        fig_mv.add_vline(x=float(current["hour_decimal"]), line_width=2, line_dash="dash", line_color="gray")
        fig_mv.update_layout(xaxis_title="Hour of day", yaxis_title="Power (kW)", height=380, hovermode="x unified")
        st.plotly_chart(fig_mv, use_container_width=True)

    with c2:
        st.markdown("#### 🦆 LV Side: Real Duck Curve Demand")
        fig_lv = go.Figure()
        fig_lv.add_trace(go.Scatter(
            x=day_df["hour_decimal"], y=day_df["demand_kw"],
            mode="lines", name="Aggregated Power [kW]", line=dict(color="royalblue", width=3)
        ))
        fig_lv.add_vline(x=float(current["hour_decimal"]), line_width=2, line_dash="dash", line_color="gray")
        fig_lv.update_layout(xaxis_title="Hour of day", yaxis_title="Power (kW)", height=380, hovermode="x unified")
        st.plotly_chart(fig_lv, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("#### ⚡ Battery Schedule")
        colors = ["crimson" if v > 0 else "mediumseagreen" for v in day_df["battery_kw"]]
        fig_batt = go.Figure()
        fig_batt.add_trace(go.Bar(
            x=day_df["hour_decimal"], y=day_df["battery_kw"],
            name="Battery kW", marker_color=colors, opacity=0.85
        ))
        fig_batt.add_hline(y=0, line_color="black", line_width=1)
        fig_batt.add_vline(x=float(current["hour_decimal"]), line_width=2, line_dash="dash", line_color="gray")
        fig_batt.update_layout(xaxis_title="Hour of day", yaxis_title="Battery power (kW)", height=330, hovermode="x unified")
        st.plotly_chart(fig_batt, use_container_width=True)

    with c4:
        st.markdown("#### 🔋 Battery Energy Reservoir")
        fig_soc = go.Figure()
        fig_soc.add_trace(go.Scatter(
            x=day_df["hour_decimal"], y=day_df["soc_kwh"],
            fill="tozeroy", name="SoC", mode="lines", line=dict(color="purple", width=3)
        ))
        fig_soc.add_hline(y=batt_cap, line_dash="dash", line_color="red", annotation_text=f"Required capacity {batt_cap:.1f} kWh")
        fig_soc.add_vline(x=float(current["hour_decimal"]), line_width=2, line_dash="dash", line_color="gray")
        fig_soc.update_layout(xaxis_title="Hour of day", yaxis_title="Energy (kWh)", height=330, hovermode="x unified")
        st.plotly_chart(fig_soc, use_container_width=True)

    st.markdown("### Daily EMS sizing from actual Duck Curve data")
    s1, s2, s3 = st.columns(3)
    s1.metric("Daily energy", f"{daily_energy:.2f} kWh")
    s2.metric("Required battery capacity", f"{batt_cap:.2f} kWh")
    s3.metric("Required battery power", f"{batt_power:.2f} kW")

with tab2:
    st.markdown("### Final analysis data used by the app")
    st.dataframe(
        day_df[[
            "timestamp", "demand_kw", "mv_import_ems_kw", "battery_kw", "battery_mode",
            "soc_kwh", "daily_energy_kwh", "required_battery_capacity_kwh", "required_battery_power_kw"
        ]],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Year-level summary")
    daily_summary = result.groupby("date", as_index=False).agg(
        daily_energy_kwh=("daily_energy_kwh", "first"),
        mv_import_ems_kw=("mv_import_ems_kw", "first"),
        required_battery_capacity_kwh=("required_battery_capacity_kwh", "first"),
        required_battery_power_kw=("required_battery_power_kw", "first"),
        peak_actual_demand_kw=("demand_kw", "max"),
    )
    st.dataframe(daily_summary, use_container_width=True, hide_index=True)
