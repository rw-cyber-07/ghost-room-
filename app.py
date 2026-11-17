"""Ghost Room Detector dashboard.   Run with:  streamlit run app.py
All data is SIMULATED (except the optional video people-count you upload)."""
import numpy as np
import pandas as pd
import streamlit as st

from core import (ACTIONS, FLOOR_OVERHEAD_KW, INTERVAL_H, STANDBY_KW, consolidation_plan,
                  label, make_data, run_agent, train_ai)

st.set_page_config(page_title="Ghost Room Detector", layout="wide")
st.title("Ghost Room Detector")
st.caption("An agent that compares the timetable, real room occupancy and electricity use, "
           "recommends or performs switch-offs after verification, and suggests class consolidation.")
st.warning("SIMULATED DATA - DEMONSTRATION ONLY. Control actions are simulated on screen.", icon="⚠️")

# ---------------- sidebar: every assumption is visible and adjustable ----------------
st.sidebar.header("Agent settings")
mode = st.sidebar.radio("Mode", ["auto", "advisory"], format_func=lambda m: {
    "auto": "Auto (agent acts, logs everything)", "advisory": "Advisory (person approves)"}[m])
confirm = st.sidebar.slider("Wait before acting (readings of 15 min)", 1, 6, 2)
extra = st.sidebar.slider("Extra wait if a class is booked", 0, 4, 2)
thr = st.sidebar.slider("Excess-power threshold (kW)", 0.3, 2.0, 0.8, 0.1)
st.sidebar.header("Impact assumptions")
tariff = st.sidebar.number_input("Electricity tariff (AED/kWh)", 0.05, 1.0, 0.30, 0.01)
st.sidebar.caption(f"Standby power per room: {STANDBY_KW} kW. Floor overhead if open: "
                   f"{FLOOR_OVERHEAD_KW} kW. These are assumptions - replace with real, sourced values.")


@st.cache_data
def load():
    df = make_data()
    _, df = train_ai(df)
    return df


@st.cache_data
def run(thr, mode, confirm, extra, tariff):
    d = label(load(), threshold_kw=thr)
    log, summary, daily = run_agent(d, mode=mode, confirm=confirm, extra_for_class=extra,
                                    tariff=tariff)
    return d, log, summary, daily


d, log, summary, daily = run(thr, mode, confirm, extra, tariff)
days = d["time"].dt.normalize().nunique()

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Room monitor", "Agent log", "Impact", "Class consolidation", "Live room (video)"])

# ---------------- tab 1: monitor ----------------
with tab1:
    c1, c2 = st.columns(2)
    day = c1.selectbox("Day", sorted(d["time"].dt.date.unique()), index=1)
    times = d[d["time"].dt.date == day]["time"].drop_duplicates().sort_values()
    t = c2.select_slider("Time", options=list(times), value=list(times)[40],
                         format_func=lambda x: x.strftime("%H:%M"))
    now = d[d["time"] == t].copy()
    k = st.columns(5)
    k[0].metric("Rooms monitored", len(now))
    k[1].metric("Occupied", int(now["occupied"].sum()))
    k[2].metric("Ghost class / waste", int(now["status"].isin(["Ghost class", "Waste"]).sum()))
    k[3].metric("Unscheduled use", int((now["status"] == "Unscheduled use").sum()))
    k[4].metric("Excess power now", f"{now.loc[now['status'].isin(['Ghost class', 'Waste']), 'excess_kw'].sum():.1f} kW")
    view = now[["room", "scheduled", "headcount", "capacity", "kw", "expected_kw", "status", "action"]].copy()
    view["scheduled"] = view["scheduled"].map({1: "Class booked", 0: "No class"})
    view = view.rename(columns={"room": "Room", "scheduled": "Timetable", "headcount": "People",
                                "capacity": "Seats", "kw": "Power (kW)", "expected_kw": "AI expected (kW)",
                                "status": "Status", "action": "Recommended action"})
    st.dataframe(view.round(2), use_container_width=True, hide_index=True)
    st.caption("Status = timetable x people present x power compared with what the AI model expects "
               "for that room, hour and occupancy (no weather input).")
    st.bar_chart(now.set_index("room")[["kw", "expected_kw"]])

# ---------------- tab 2: agent log ----------------
with tab2:
    st.subheader("What the agent did, and why")
    st.write(f"Mode: **{mode}**. Safeguards: waits {confirm * 15} min "
             f"(+{extra * 15} min if a class is booked), never switches off right before a booked class, "
             "switches back on when people appear, logs every action.")
    st.dataframe(log.assign(time=log["time"].dt.strftime("%a %d %b %H:%M")).tail(300),
                 use_container_width=True, hide_index=True)

# ---------------- tab 3: impact ----------------
with tab3:
    lab = "saved" if mode == "auto" else "saved if recommendations are approved"
    m = st.columns(3)
    m[0].metric("Energy " + lab, f"{summary['kwh_saved']:.0f} kWh")
    m[1].metric("Cost", f"AED {summary['aed_saved']:.0f}")
    m[2].metric("Share of wasted energy recovered", f"{summary['share_recovered'] * 100:.0f}%")
    st.write(f"Over **{days} simulated days** for 8 rooms. Per month this is about "
             f"**{summary['kwh_saved'] / days * 30:.0f} kWh**.")
    st.bar_chart(daily.rename("kWh saved per day"))
    st.info(f"Safety check: the agent switched equipment off {summary['switch_offs']} times, and "
            f"{summary['restored_by_people']} of those were reversed because people showed up. "
            "Those are moments a real room would be uncomfortable - tune the wait settings to trade "
            "savings against comfort. In a real deployment this number must be measured, not simulated.")
    st.caption("Savings = (power above standby) x time, counted only after a switch-off. "
               "Everything depends on the assumptions in the sidebar. Estimates are illustrative.")

# ---------------- tab 4: consolidation ----------------
with tab4:
    st.subheader("Suggested room plan for one time slot")
    c1, c2 = st.columns(2)
    day2 = c1.selectbox("Day ", [x for x in sorted(d["time"].dt.date.unique())
                                 if pd.Timestamp(x).dayofweek < 5], index=1)
    hours = sorted(d[(d["time"].dt.date == day2) & (d["time"].dt.hour.between(8, 16))]["time"].unique())
    t2 = c2.select_slider("Time ", options=hours, value=hours[8], format_func=lambda x: pd.Timestamp(x).strftime("%H:%M"))
    plan, sm = consolidation_plan(d[d["time"] == t2])
    if plan.empty:
        st.write("No classes with people at this time.")
    else:
        a = st.columns(4)
        a[0].metric("Classes with people", sm["classes_with_people"])
        a[1].metric("Ghost bookings", sm["ghost_bookings"])
        a[2].metric("Moves needed", sm["moves_needed"])
        a[3].metric("Floors that can power down", len(sm["floors_freed"]))
        st.dataframe(plan, use_container_width=True, hide_index=True)
        st.write(f"Floors in use now: {sm['floors_before']} -> after plan: {sm['floors_after']}. "
                 f"Estimated saving at this moment: **{sm['kw_saved_now']:.1f} kW** "
                 f"(ghost rooms x room load + freed floors x {FLOOR_OVERHEAD_KW} kW assumed overhead).")
    st.caption("This is a planning suggestion for timetabling, not a live control. The floor overhead "
               "value is an assumption. Do not add this saving to the agent's saving without care - "
               "ghost rooms are counted in both.")

# ---------------- tab 5: live room from video ----------------
with tab5:
    st.subheader("Live room: people counted from your own demo video")
    st.write("Run `people_counter.py` on your clip, then upload the CSV it creates. "
             "Nothing is uploaded anywhere - the file is read by this app only.")
    up = st.file_uploader("people_counts.csv (columns: second, people)", type="csv")
    c1, c2 = st.columns(2)
    booked = c1.checkbox("A class is booked for this room", value=True)
    power = c2.slider("Room power right now (kW) - from a meter or your assumption", 1.0, 5.0, 3.5, 0.1)
    if up is not None:
        v = pd.read_csv(up)
        v["smooth"] = v["people"].rolling(5, center=True, min_periods=1).median()
        v["occupied"] = v["smooth"] >= 1
        v["excess"] = power - STANDBY_KW
        v["status"] = np.select(
            [~v["occupied"] & (v["excess"] > thr) & booked, ~v["occupied"] & (v["excess"] > thr),
             v["occupied"] & ~booked],
            ["Ghost class", "Waste", "Unscheduled use"], default="Normal")
        st.line_chart(v.set_index("second")[["people", "smooth"]])
        st.dataframe(v["status"].value_counts().rename("seconds"), use_container_width=True)
        last = v.iloc[-1]["status"]
        st.success(f"Latest status: {last} -> {ACTIONS[last]}")
        st.caption("Short gaps are smoothed (rolling median) so one missed detection does not "
                   "trigger an action. A real system would count heads only and store no images.")
    else:
        st.info("Upload a CSV to see it here. No video? Skip this tab; the other tabs use simulated data.")
