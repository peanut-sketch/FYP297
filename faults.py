"""Fault detection: compare measured power with what the twin says the panels SHOULD produce.
  python faults.py            analyse data/measured.csv (timestamp, power_w, [irradiance_wm2], [module_temp_c])
  python faults.py --selftest inject known faults into synthetic data and check they are found
Best results need an on-site plane-of-array irradiance sensor (irradiance_wm2). Without it the expected power comes from
satellite/reanalysis irradiance, which is noisier, so use the looser thresholds (T['loose'])."""
import sys, json
import numpy as np, pandas as pd
import twin
ROOT, CFG = twin.ROOT, twin.CFG
T = dict(daylight=.15, outage_ratio=.05, outage_hours=2, step_drop=.08, soil_slope=-.0025, soil_decline=.03, shade_dev=.12, base_days=14)

def expected_kw(df, kwp):
    """Expected AC power (kW) from measured POA irradiance (+ module temp if present)."""
    tc = df["module_temp_c"] if "module_temp_c" in df else 30 + df["irradiance_wm2"] * .03
    return (df["irradiance_wm2"] / 1000 * (1 + CFG["temp_coeff"] * (tc - 25)) * CFG["system_loss"] * kwp).clip(lower=0)

def analyse(df, kwp):
    """df: hourly, tz-aware, columns power_w, exp_kw. Returns list of fault events."""
    d = pd.DataFrame({"m": df["power_w"] / 1000, "e": df["exp_kw"]}).dropna()
    d = d[d.e / kwp > T["daylight"]]; d["r"] = d.m / d.e; ev = []
    # 1. outage: output ~0 while the sun is up
    z = (d.r < T["outage_ratio"]).astype(int); g = (z.diff().fillna(0) != 0).cumsum()
    for _, s in d[z == 1].groupby(g[z == 1]):
        if len(s) >= T["outage_hours"]: ev.append(dict(type="outage", start=str(s.index[0]), end=str(s.index[-1]), loss_pct=100,
                      note="Zero output in daylight: inverter trip, breaker, disconnection or logger failure."))
    d = d[d.r >= T["outage_ratio"]]
    day = d.groupby(d.index.date).agg(m=("m", "sum"), e=("e", "sum"), n=("m", "size")); day = day[day.n >= 4]
    rel = (day.m / day.e); idx = pd.to_datetime(rel.index)
    if len(rel) < T["base_days"] + 7: return ev, "not enough history"
    base = rel.iloc[:T["base_days"]].median(); rel = rel / base
    # 2. step loss (persistent): bypass diode, string or connector failure
    step_days = []; i = 7
    while i < len(rel) - 4:
        before, after = rel.iloc[i - 7:i].median(), rel.iloc[i:i + 5].median()
        if before - after > T["step_drop"] and after < 1 - T["step_drop"]:
            j = i
            loss = (before - after) * 100
            ev.append(dict(type="step_loss", start=str(idx[j].date()), end="ongoing", loss_pct=round(loss, 1),
                           note=f"Persistent ~{loss:.0f}% drop. Around 33% suggests one bypass-diode substring (3 per module) or a lost string; check I-V curve and connectors."))
            step_days.append(j); i += 14
        else: i += 1
    # 3. soiling / degradation: slow decline outside step regimes
    sm = rel.rolling(5, min_periods=3, center=True).median(); w = 14; inside = False
    for i in range(0, len(sm) - w):
        if any(abs(i - s) < w + 3 for s in step_days): continue
        seg = sm.iloc[i:i + w].values
        if np.isnan(seg).any(): continue
        slope = np.polyfit(np.arange(w), seg, 1)[0]
        if slope < T["soil_slope"] and seg[0] - seg[-1] > T["soil_decline"]:
            if inside: ev[-1]["end"] = str(idx[i + w - 1].date()); ev[-1]["loss_pct"] = max(ev[-1]["loss_pct"], round((1 - seg[-1]) * 100, 1))
            else: ev.append(dict(type="soiling_or_degradation", start=str(idx[i + int(np.argmax(seg))].date()), end=str(idx[i + w - 1].date()), loss_pct=round((1 - seg[-1]) * 100, 1),
                                 note="Gradual loss. If it recovers after rain or cleaning: soiling. If it never recovers: degradation, hot spots or delamination.")); inside = True
        else: inside = False
    # 4. shading: loss concentrated in particular hours of the day (recurs daily)
    h0 = d[d.index < d.index[0] + pd.Timedelta(days=T["base_days"])]; bprof = (h0.r).groupby(h0.index.hour).median()
    for st in range(T["base_days"], int((d.index[-1] - d.index[0]).days) - 7, 7):
        a = d.index[0] + pd.Timedelta(days=st); s = d[(d.index >= a) & (d.index < a + pd.Timedelta(days=14))]
        if len(s) < 40: continue
        rr = (s.r.groupby(s.index.hour).median() / bprof).dropna(); lvl = rr.median(); bad = sorted(rr.index[rr < lvl - T["shade_dev"]])
        if len(bad) >= 2 and not any(e["type"] == "shading" for e in ev):
            ev.append(dict(type="shading", start=str(a.date()), end="ongoing", loss_pct=round((1 - rr[bad].mean() / lvl) * 100, 1), hours=[int(b) for b in bad],
                           note=f"Repeating loss around {bad[0]:02d}:00-{bad[-1] + 1:02d}:00: new obstruction (tree, structure) or dirt pattern."))
    return ev, "ok"

def selftest():
    import forecast as F
    w = F.synth(120, seed=3); w = w[w.index < w.index[-1] - pd.Timedelta(days=2)]
    kwp = 1.0; e = F.power(w) * kwp; rng = np.random.default_rng(0)
    m = e * (1 + rng.normal(0, .02, len(e))); t = np.arange(len(e)) / 24; hr = w.index.hour
    m = m * np.where((t > 20) & (t < 40), 1 - .004 * (t - 20), 1) * np.where((t >= 40) & (t < 41), 0.92, 1)       # soiling 20-40d, rain clears it
    m = m * np.where(t > 62, 1 - .33, 1)                                                                          # bypass-diode substring from day 62
    m = np.where((t > 85) & (t < 86) & (hr >= 10), 0, m)                                                        # outage day 85
    m = m * np.where((t > 95) & (hr >= 15) & (hr <= 17), .75, 1)                                                  # shading 15-18h from day 95
    df = pd.DataFrame({"power_w": m * 1000, "exp_kw": e}, index=w.index)
    ev, st = analyse(df, kwp); d0 = w.index[0].normalize()
    print("truth: soiling d20-40 | step 33% d62 | outage d85 | shading 15-17h d95\n")
    for x in ev: print(f"{x['type']:24s} day {(pd.Timestamp(x['start']).tz_localize(None) - d0.tz_localize(None)).days:>3}  loss {x['loss_pct']}%" + (f"  hours {x['hours']}" if 'hours' in x else ""))

def main():
    f = ROOT / CFG["measured_csv"]
    if not f.exists(): print("No", f, "- add telemetry (timestamp, power_w, irradiance_wm2)"); return
    df = pd.read_csv(f, parse_dates=["timestamp"]).set_index("timestamp")
    df.index = df.index.tz_localize(CFG["tz"]) if df.index.tz is None else df.index.tz_convert(CFG["tz"])
    df = df.resample("h").mean()
    if "irradiance_wm2" in df: df["exp_kw"] = expected_kw(df, CFG["bench_kwp"])
    else:
        import forecast as F; w = F.om("https://archive-api.open-meteo.com/v1/archive", start_date=str(df.index[0].date()), end_date=str(df.index[-1].date()))
        df["exp_kw"] = (F.power(w) * CFG["bench_kwp"]).reindex(df.index, method="nearest", tolerance=pd.Timedelta("40min"))
        print("No irradiance sensor column: using satellite irradiance, expect more false alarms.")
    ev, st = analyse(df, CFG["bench_kwp"])
    out = {"generated": pd.Timestamp.now(CFG["tz"]).isoformat(timespec="minutes"), "status": "ALERT" if ev else "OK", "detail": st, "events": ev}
    (ROOT / "docs/data/faults.json").write_text(json.dumps(out, indent=1)); print(out["status"], len(ev), "event(s)")

if __name__ == "__main__": selftest() if "--selftest" in sys.argv else main()
