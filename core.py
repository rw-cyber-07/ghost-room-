"""Ghost Room Detector - core logic (no weather/climate inputs).

ALL DATA IS SIMULATED. Say so clearly in the report and the pitch.

Pieces
  make_data()           simulated 15-minute readings for 8 rooms (timetable, headcount, power)
  train_ai()            learns "expected efficient power" for each room/time, measures excess
  label()               timetable x occupancy x excess  ->  status + recommended action
  run_agent()           look -> decide -> act -> explain loop with safeguards
  consolidation_plan()  suggests moving classes so whole floors can be powered down
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

# ---- Assumptions (state these in your report; replace with real spec-sheet values) ----
INTERVAL_H = 0.25          # one reading = 15 minutes
STANDBY_KW = 1.2           # power drawn by an empty room that is properly switched off
LOAD_KW = 3.0              # extra power when AC + lights run at full
ROOMS = {                  # room: (capacity, floor/zone)  -> replace with real Liwa rooms
    "A101": (40, 1), "A102": (60, 1), "A103": (30, 1),
    "A201": (40, 2), "A202": (80, 2), "A203": (30, 2),
    "A301": (50, 3), "A302": (25, 3),
}
FEATURES = ["hour_f", "dow", "fill", "capacity"]   # NOTE: no temperature / climate input

ACTIONS = {
    "Ghost class": "Class booked but room empty: switch off after verification",
    "Waste": "No class, nobody there, equipment on: switch off after verification",
    "Unscheduled use": "People present without a booking: keep on, log it",
    "Normal": "No action",
}


# ------------------------------------------------------------------ 1. data
def make_data(days=30, seed=7):
    """Simulate one reading per room every 15 minutes."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-10-05", periods=days * 96, freq="15min")  # starts on a Monday
    n = len(idx)
    hour_f = (idx.hour + idx.minute / 60).to_numpy()
    dow = idx.dayofweek.to_numpy()
    dates = idx.normalize()
    day_list = pd.unique(dates)
    # six 90-minute teaching blocks between 08:00 and 17:00
    block = np.where((hour_f >= 8) & (hour_f < 17), np.floor((hour_f - 8) / 1.5), -1).astype(int)

    frames = []
    for room, (cap, zone) in ROOMS.items():
        sched = np.zeros(n, int)
        frac = np.zeros(n)        # fraction of seats filled during a class
        study = np.zeros(n)       # unscheduled self-study
        left_on = np.zeros(n, bool)   # equipment accidentally left running
        for d in day_list:
            dm = np.asarray(dates == d)
            if d.dayofweek >= 5:          # Saturday/Sunday: no classes
                continue
            for b in range(6):
                m = dm & (block == b)
                if rng.random() < 0.6:    # class booked in this block
                    sched[m] = 1
                    r = rng.random()
                    if r < 0.12:          # ghost class: booked but nobody comes
                        frac[m] = 0.0
                    elif r < 0.27:        # low attendance: consolidation candidate
                        frac[m] = rng.uniform(0.12, 0.35)
                    else:
                        frac[m] = rng.uniform(0.5, 0.95)
                elif rng.random() < 0.08:  # a few students use the room without a booking
                    study[m] = rng.uniform(0.05, 0.25)
            if rng.random() < 0.55:       # equipment left on for a while (lunch, evening)
                pos = np.where(dm)[0]
                start = rng.integers(9 * 4, 20 * 4)
                left_on[pos[start:start + rng.integers(4, 17)]] = True

        fill = np.clip(np.clip(frac + study, 0, 1) * rng.normal(1, 0.04, n), 0, 1)
        head = np.round(fill * cap).astype(int)
        fill = head / cap
        on = (sched == 1) | (head > 0) | left_on
        kw = STANDBY_KW + on * LOAD_KW * (0.6 + 0.4 * fill) + rng.normal(0, 0.08, n)
        frames.append(pd.DataFrame({
            "time": idx, "room": room, "zone": zone, "capacity": cap,
            "hour_f": hour_f, "dow": dow, "scheduled": sched,
            "headcount": head, "fill": fill, "kw": kw,
            # ground truth, used ONLY to check the detector in tests (a real campus has no such column)
            "truth_waste": on & (head == 0),
        }))
    return pd.concat(frames).sort_values(["time", "room"]).reset_index(drop=True)


# ------------------------------------------------------------------ 2. AI
def train_ai(df, train_frac=0.7, quantile=0.3):
    """Learn the power a room NORMALLY needs at a given time and occupancy.

    A quantile Gradient Boosting model predicts the lower-typical (30th percentile)
    power for similar situations, i.e. 'efficient' consumption. Readings far above it
    are excess. Climate/weather is deliberately not an input."""
    cut = df["time"].quantile(train_frac)
    train = df[df["time"] <= cut]
    model = GradientBoostingRegressor(
        loss="quantile", alpha=quantile, n_estimators=120, max_depth=3, random_state=0)
    model.fit(train[FEATURES], train["kw"])
    out = df.copy()
    out["expected_kw"] = model.predict(out[FEATURES])
    out["excess_kw"] = out["kw"] - out["expected_kw"]
    return model, out


# ------------------------------------------------------------------ 3. rules
def label(df, threshold_kw=0.8, min_present=1):
    """Combine timetable + occupancy + AI excess into a status for every reading."""
    d = df.copy()
    d["occupied"] = d["headcount"] >= min_present
    d["high"] = d["excess_kw"] > threshold_kw
    waste = d["high"] & ~d["occupied"]
    d["status"] = np.select(
        [waste & (d["scheduled"] == 1), waste & (d["scheduled"] == 0),
         d["occupied"] & (d["scheduled"] == 0)],
        ["Ghost class", "Waste", "Unscheduled use"], default="Normal")
    d["action"] = d["status"].map(ACTIONS)
    return d


# ------------------------------------------------------------------ 4. agent
def run_agent(d, mode="auto", confirm=2, extra_for_class=2, grace=2,
              tariff=0.30):
    """Look -> decide -> act -> explain, one room at a time, in time order.

    Safeguards
      * wait `confirm` consecutive flagged readings (15 min each) before acting
      * wait `extra_for_class` more readings if a class is booked (late students)
      * never switch off just before a booked class starts (`grace` readings)
      * switch back ON as soon as someone appears or a class is about to start
      * every action is logged with a reason

    mode='auto'     -> actions are executed
    mode='advisory' -> actions are recommendations; savings are 'if approved'
    """
    verb_off, verb_on = (("SWITCH OFF", "SWITCH ON") if mode == "auto"
                         else ("RECOMMEND OFF", "RECOMMEND ON"))
    log, saved_by_day = [], {}
    saved_kwh, false_off = 0.0, 0

    for room, g in d.groupby("room", sort=False):
        g = g.sort_values("time")
        t = g["time"].to_numpy()
        sched = g["scheduled"].to_numpy()
        occ = g["occupied"].to_numpy()
        kw = g["kw"].to_numpy()
        flag = g["status"].isin(["Ghost class", "Waste"]).to_numpy()
        prev = np.r_[0, sched[:-1]]
        starts = (sched == 1) & (prev == 0)          # a booked class begins here
        off, streak = False, 0
        for i in range(len(g)):
            upcoming = starts[i:i + 1 + grace].any()  # class begins now or very soon
            if not off:
                streak = streak + 1 if flag[i] else 0
                need = confirm + (extra_for_class if sched[i] else 0)
                if streak >= need and not starts[i + 1:i + 1 + grace].any():
                    kind = g["status"].iloc[i]
                    why = (f"{kind}: {'class booked' if sched[i] else 'no class'}, "
                           f"nobody present for {streak * 15} min, "
                           f"{g['excess_kw'].iloc[i]:.1f} kW above expected")
                    log.append((t[i], room, verb_off, why))
                    off, streak = True, 0
            else:
                if upcoming or occ[i]:
                    if upcoming:
                        why = "A booked class is about to start"
                    else:
                        why = "People detected again"
                        false_off += 1
                    log.append((t[i], room, verb_on, why))
                    off, streak = False, 0
                else:
                    kwh = max(0.0, kw[i] - STANDBY_KW) * INTERVAL_H
                    saved_kwh += kwh
                    day = pd.Timestamp(t[i]).normalize()
                    saved_by_day[day] = saved_by_day.get(day, 0.0) + kwh

    log_df = pd.DataFrame(log, columns=["time", "room", "action", "reason"]).sort_values("time")
    truth_kwh = float((np.maximum(d["kw"] - STANDBY_KW, 0) * d["truth_waste"]).sum() * INTERVAL_H)
    summary = {
        "kwh_saved": saved_kwh,
        "aed_saved": saved_kwh * tariff,
        "switch_offs": int((log_df["action"] == verb_off).sum()) if len(log_df) else 0,
        "restored_by_people": false_off,
        "waste_kwh_total": truth_kwh,
        "share_recovered": saved_kwh / truth_kwh if truth_kwh else 0.0,
        "mode": mode,
    }
    daily = pd.Series(saved_by_day, dtype=float).sort_index()
    return log_df, summary, daily


# ------------------------------------------------------------------ 5. consolidation
FLOOR_OVERHEAD_KW = 2.5   # ASSUMPTION: corridor lights + ventilation + lift standby per open floor


def _assign(classes, rooms_df, floors, buffer, keep_in_place=True):
    """Place classes only into rooms on `floors`. Returns {class_room: new_room} or None."""
    allowed = set(rooms_df.index[rooms_df["zone"].isin(floors)])
    need = {r: int(np.ceil(c["headcount"] * buffer)) for r, c in classes.iterrows()}
    taken, result = set(), {}
    if keep_in_place:                       # classes already on a kept floor stay put
        for r in classes.index:
            if r in allowed and rooms_df.loc[r, "capacity"] >= need[r]:
                taken.add(r)
                result[r] = r
    for r in sorted((r for r in classes.index if r not in result), key=lambda x: -need[x]):
        cands = [(rooms_df.loc[x, "capacity"], x) for x in allowed
                 if x not in taken and rooms_df.loc[x, "capacity"] >= need[r]]
        if not cands:
            return None
        best = min(cands)[1]
        taken.add(best)
        result[r] = best
    return result


def consolidation_plan(slot, buffer=1.1):
    """slot = the readings of ONE timestamp (one row per room).

    Idea: find the smallest set of floors that can hold every class that actually has
    people in it, keep classes where they are when possible, and move the rest into the
    smallest room that fits. Floors left empty can be powered down. This is a PLANNING
    suggestion (for timetabling), not something done live."""
    from itertools import combinations
    s = slot.set_index("room")
    classes = s[(s["scheduled"] == 1) & (s["headcount"] >= 1)]
    booked = s[s["scheduled"] == 1]
    zones = sorted(int(z) for z in s["zone"].unique())
    best_plan = None
    for k in range(1, len(zones) + 1):
        options = []
        for fl in combinations(zones, k):
            res = _assign(classes, s, fl, buffer) or _assign(classes, s, fl, buffer, False)
            if res is not None:
                moves = sum(1 for a, b in res.items() if a != b)
                options.append((moves, fl, res))
        if options:
            best_plan = min(options, key=lambda o: o[0])
            break
    if best_plan is None:
        return pd.DataFrame(), {"note": "No feasible plan"}
    _, floors_kept, res = best_plan
    rows = [{"class_now_in": a, "people": int(classes.loc[a, "headcount"]), "move_to": b,
             "room_size_now": int(s.loc[a, "capacity"]), "room_size_new": int(s.loc[b, "capacity"]),
             "moves": a != b} for a, b in res.items()]
    plan = pd.DataFrame(rows).sort_values("moves", ascending=False).reset_index(drop=True)
    floors_before = sorted(int(z) for z in booked["zone"].unique())
    freed = [z for z in floors_before if z not in floors_kept]
    ghost = len(booked) - len(classes)
    summary = {
        "classes_with_people": len(classes),
        "ghost_bookings": ghost,
        "floors_before": floors_before,
        "floors_after": sorted(floors_kept),
        "floors_freed": freed,
        "moves_needed": int(plan["moves"].sum()) if len(plan) else 0,
        "kw_saved_now": ghost * LOAD_KW + len(freed) * FLOOR_OVERHEAD_KW,
    }
    return plan, summary
