"""Forecasting: learn from history, forecast the next ~48 h, and benchmark vs baselines (MAPE/RMSE).
  python forecast.py              live: Open-Meteo archive (history) + forecast API
  python forecast.py --synthetic  offline test data
Target = data/measured.csv (bench power) if it has >=14 days, else the physics twin driven by archived irradiance.
Features are only things a day-ahead weather forecast provides (cloud layers, temp, humidity, clear-sky, sun position),
so the model learns to turn forecast weather into PV output."""
import sys, json
import numpy as np, pandas as pd, pvlib
from sklearn.ensemble import HistGradientBoostingRegressor
import twin
ROOT, CFG, SYN = twin.ROOT, twin.CFG, "--synthetic" in sys.argv
V = "shortwave_radiation,direct_normal_irradiance,diffuse_radiation,temperature_2m,wind_speed_10m,cloud_cover,cloud_cover_low,cloud_cover_mid,cloud_cover_high,relative_humidity_2m"
C = ["ghi", "dni", "dhi", "temp_air", "wind_speed", "cc", "cc_low", "cc_mid", "cc_high", "rh"]
LOC = pvlib.location.Location(CFG["lat"], CFG["lon"], CFG["tz"], CFG["altitude"])

def om(url, **p):
    import requests
    r = requests.get(url, timeout=90, params={"latitude": CFG["lat"], "longitude": CFG["lon"], "timezone": CFG["tz"],
                     "hourly": V, "wind_speed_unit": "ms", **p}); r.raise_for_status(); h = r.json()["hourly"]
    df = pd.DataFrame({c: h[k] for c, k in zip(C, V.split(","))}, index=pd.to_datetime(h["time"]).tz_localize(CFG["tz"]))
    df.index -= pd.Timedelta(minutes=30)          # radiation is the mean of the preceding hour: use mid-interval
    return df.dropna()

def synth(days, seed=1):
    rng = np.random.default_rng(seed); n = days * 24 + 48
    t = pd.date_range(end=pd.Timestamp.now(CFG["tz"]).floor("h") + pd.Timedelta(hours=47), periods=n, freq="h") - pd.Timedelta(minutes=30)
    cs, sp = LOC.get_clearsky(t), LOC.get_solarposition(t)
    cc = np.clip(pd.Series(rng.normal(55, 28, n)).rolling(8, min_periods=1).mean().values + 18 * np.exp(-((t.hour - 15) / 3) ** 2), 0, 100)
    k = np.clip(1 - .75 * (cc / 100) ** 1.4 + rng.normal(0, .05, n), .05, 1.05)
    ghi = cs.ghi * k; dni = cs.dni * np.clip(k ** 2.5, 0, 1); dhi = np.clip(ghi - dni * np.cos(np.radians(sp.zenith)), 0, None)
    f = rng.uniform(.2, .6, n)
    return pd.DataFrame({"ghi": ghi, "dni": dni, "dhi": dhi, "temp_air": 28 + 4 * np.sin((t.hour - 9) / 24 * 2 * np.pi) - 3 * cc / 100,
                         "wind_speed": 2 + rng.random(n), "cc": cc, "cc_low": cc * f, "cc_mid": cc * (1 - f) * .5, "cc_high": cc * (1 - f) * .5,
                         "rh": 70 + .2 * cc + rng.normal(0, 3, n)}, index=t)

def power(w):  # physics twin, kW per kWp (fixed fleet tilt)
    sp = LOC.get_solarposition(w.index)
    return twin.kw_per_kwp(twin.poa(CFG["fleet_tilt"], CFG["fleet_azimuth"], sp, w), w.temp_air, w.wind_speed)

def feats(w):
    sp, cs = LOC.get_solarposition(w.index), LOC.get_clearsky(w.index)
    return pd.DataFrame({"cs_ghi": cs.ghi, "zen": sp.apparent_zenith, "hour": w.index.hour, "doy": w.index.dayofyear, "temp": w.temp_air,
                         "wind": w.wind_speed, "cc": w.cc, "cc_low": w.cc_low, "cc_mid": w.cc_mid, "cc_high": w.cc_high, "rh": w.rh}, index=w.index)

def metrics(y, p):
    y, p = np.asarray(y), np.asarray(p); m = y > .05           # daylight only (MAPE is undefined near zero)
    return {"rmse_kw_per_kwp": round(float(np.sqrt(np.mean((y - p) ** 2))), 4), "nmae_pct": round(float(np.mean(np.abs(y - p)) / y[m].mean() * 100), 1),
            "mape_daylight_pct": round(float(np.mean(np.abs(y[m] - p[m]) / y[m]) * 100), 1)}

def target(w):
    f = ROOT / CFG["measured_csv"]
    if f.exists() and not SYN:
        m = pd.read_csv(f, parse_dates=["timestamp"]).set_index("timestamp")["power_w"]
        m.index = m.index.tz_localize(CFG["tz"]) if m.index.tz is None else m.index.tz_convert(CFG["tz"])
        y = (m.resample("h").mean() / 1000 / CFG["bench_kwp"]); y.index += pd.Timedelta(minutes=30)
        y = y.reindex(w.index, method="nearest", tolerance=pd.Timedelta("20min")).dropna()
        if len(y) >= 14 * 24: return y, "measured bench data"
    return power(w), "physics twin on archived irradiance (proxy)"

def main():
    now = pd.Timestamp.now(CFG["tz"])
    if SYN: allw = synth(500)
    else:
        hist = om("https://archive-api.open-meteo.com/v1/archive", start_date=str((now - pd.Timedelta(days=CFG["train_days"])).date()),
                  end_date=str((now - pd.Timedelta(days=3)).date()))
        fut = om("https://api.open-meteo.com/v1/forecast", past_days=3, forecast_days=3)
        allw = pd.concat([hist, fut[fut.index > hist.index[-1]]])
    past, fut = allw[allw.index < now - pd.Timedelta(hours=1)], allw[allw.index >= now - pd.Timedelta(minutes=30)]
    y, tsrc = target(past); X = feats(past).loc[y.index]
    cut = y.index[-1] - pd.Timedelta(days=CFG["holdout_days"])
    tr, te = y.index <= cut, y.index > cut
    mk = lambda: HistGradientBoostingRegressor(max_iter=300, learning_rate=.06, max_leaf_nodes=24, random_state=0)
    m = mk().fit(X[tr], y[tr]); pred = np.clip(m.predict(X[te]), 0, None)
    # baselines: yesterday's output; yesterday's clear-sky index x today's clear-sky output
    cs = LOC.get_clearsky(past.index); csw = past.assign(ghi=cs.ghi, dni=cs.dni, dhi=cs.dhi); csp = power(csw).reindex(y.index)
    pers = y.shift(24).reindex(y.index)[te].fillna(0); kcs = (y / csp.clip(lower=.05)).clip(0, 1.2).shift(24); smart = (kcs * csp)[te].fillna(0)
    res = {"model": metrics(y[te], pred), "persistence_24h": metrics(y[te], pers), "smart_persistence_24h": metrics(y[te], smart)}
    m = mk().fit(X, y)                                     # refit on everything, then forecast
    fp = np.clip(m.predict(feats(fut)), 0, None)
    out = {"generated": now.isoformat(timespec="minutes"), "target": tsrc, "holdout_days": CFG["holdout_days"], "metrics": res,
           "times": [t.isoformat(timespec="minutes") for t in fut.index], "kw_per_kwp": np.round(fp, 3).tolist(),
           "fleet_mw": np.round(fp * CFG["installed_capacity_mwp"] * CFG["fleet_factor"]).tolist()}
    (ROOT / "docs/data/forecast.json").write_text(json.dumps(out, separators=(",", ":")))
    print("target:", tsrc); [print(f"{k:24s}", v) for k, v in res.items()]
    print("next 48h GWh (fleet):", round(sum(out["fleet_mw"][:48]) / 1000, 1))

if __name__ == "__main__": main()
