"""Singapore solar PV digital twin: physics model shared by forecast.py, faults.py and loads.py.
  python twin.py            live: Open-Meteo forecast (no API key) for today + tomorrow
  python twin.py --offline  clear-sky with a synthetic cloud pattern, no network
Writes docs/data/latest.json: 30-min kW/kWp for a fixed fleet-tilt array and a single-axis tracker,
national fleet MW (installed_capacity_mwp x fleet_factor), system demand and solar share.
Shared API:
  ROOT, CFG                               project folder and config.json
  poa(tilt, azimuth, solpos, weather)     plane-of-array irradiance, W/m2 (weather: ghi, dni, dhi)
  kw_per_kwp(poa, temp_air, wind_speed)   AC output per kWp, temperature- and loss-corrected"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd, pvlib

ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / "config.json").read_text())
(ROOT / "docs/data").mkdir(parents=True, exist_ok=True)
LOC = pvlib.location.Location(CFG["lat"], CFG["lon"], CFG["tz"], CFG["altitude"])


def poa(tilt, azimuth, sp, w):
    """Plane-of-array global irradiance (W/m2), Hay-Davies sky model. tilt/azimuth may be scalars or series (tracker)."""
    tilt = pd.Series(tilt, index=w.index) if np.ndim(tilt) else tilt
    azimuth = pd.Series(azimuth, index=w.index) if np.ndim(azimuth) else azimuth
    irr = pvlib.irradiance.get_total_irradiance(
        tilt, azimuth, sp.apparent_zenith, sp.azimuth, w.dni, w.ghi, w.dhi,
        dni_extra=pvlib.irradiance.get_extra_radiation(w.index), model="haydavies", albedo=CFG["albedo"])
    return irr["poa_global"].fillna(0).clip(lower=0)


def kw_per_kwp(g, temp_air, wind_speed):
    """AC kW per kWp: STC scaling, Faiman cell temperature, linear temperature coefficient, lumped system losses."""
    tc = pvlib.temperature.faiman(g, temp_air, wind_speed)
    return (g / 1000 * (1 + CFG["temp_coeff"] * (tc - 25)) * CFG["system_loss"]).clip(lower=0)


def tracker_poa(sp, w):
    """Single-axis N-S tracker with backtracking (utility-scale / floating-style comparison)."""
    tr = pvlib.tracking.singleaxis(sp.apparent_zenith, sp.azimuth, axis_tilt=0, axis_azimuth=180,
                                   max_angle=CFG["tracker_max_angle"], backtrack=True, gcr=CFG["tracker_gcr"])
    return poa(tr.surface_tilt.fillna(0), tr.surface_azimuth.fillna(180), sp, w)


def grid(days=2):
    """30-min timestamps (mid-interval) from today 00:00 SGT for `days` days."""
    start = pd.Timestamp.now(CFG["tz"]).normalize()
    return pd.date_range(start, periods=days * 48, freq="30min") + pd.Timedelta(minutes=15)


def weather_live(t):
    import requests
    v = "shortwave_radiation,direct_normal_irradiance,diffuse_radiation,temperature_2m,wind_speed_10m"
    r = requests.get("https://api.open-meteo.com/v1/forecast", timeout=60, params={
        "latitude": CFG["lat"], "longitude": CFG["lon"], "timezone": CFG["tz"], "hourly": v,
        "wind_speed_unit": "ms", "past_days": 1, "forecast_days": 3})
    r.raise_for_status(); h = r.json()["hourly"]
    df = pd.DataFrame({"ghi": h["shortwave_radiation"], "dni": h["direct_normal_irradiance"], "dhi": h["diffuse_radiation"],
                       "temp_air": h["temperature_2m"], "wind_speed": h["wind_speed_10m"]},
                      index=pd.to_datetime(h["time"]).tz_localize(CFG["tz"]) - pd.Timedelta(minutes=30))  # hour-mean -> mid-interval
    df = df.dropna()
    # interpolate the clear-sky index (not raw irradiance) so the sun's shape is kept at 30-min resolution
    cs_h, cs_t = LOC.get_clearsky(df.index), LOC.get_clearsky(t)
    out = pd.DataFrame(index=t)
    u = df.index.union(t)
    for c in ("ghi", "dni", "dhi"):
        k = (df[c] / cs_h[c].where(cs_h[c] > 10)).clip(0, 1.5)
        out[c] = (k.reindex(u).interpolate("time").ffill().bfill().reindex(t).fillna(0) * cs_t[c])
    for c in ("temp_air", "wind_speed"):
        out[c] = df[c].reindex(u).interpolate("time").ffill().bfill().reindex(t)
    return out


def weather_offline(t, seed=7):
    """Clear-sky scaled by a typical Singapore pattern: clearer mornings, convective cloud mid-afternoon."""
    rng = np.random.default_rng(seed); h = t.hour + t.minute / 60
    cs = LOC.get_clearsky(t); sp = LOC.get_solarposition(t)
    k = np.clip(.85 - .4 * np.exp(-((h - 15) / 2) ** 2) + rng.normal(0, .06, len(t)), .15, 1)
    ghi = cs.ghi * k; dni = cs.dni * k ** 2.5
    dhi = np.clip(ghi - dni * np.cos(np.radians(sp.zenith)).clip(lower=0), 0, None)
    return pd.DataFrame({"ghi": ghi, "dni": dni, "dhi": dhi, "temp_air": 27.5 + 3.5 * np.sin((h - 9) / 24 * 2 * np.pi),
                         "wind_speed": 2 + rng.random(len(t))}, index=t)


def demand(t):
    """System demand (MW): data/demand.csv if present, else an illustrative Singapore weekday shape."""
    f = ROOT / CFG["demand_csv"]
    if f.exists():
        d = pd.read_csv(f, parse_dates=["timestamp"]).set_index("timestamp")["demand_mw"]
        d.index = d.index.tz_localize(CFG["tz"]) if d.index.tz is None else d.index.tz_convert(CFG["tz"])
        prof = d.groupby(d.index.hour * 2 + d.index.minute // 30).mean()       # typical day from the CSV
        return prof.reindex(t.hour * 2 + t.minute // 30).values, "EMA half-hourly demand (typical day from " + CFG["demand_csv"] + ")"
    h = t.hour + t.minute / 60
    mw = 5900 + 1450 * np.exp(-((h - 14.5) / 4.2) ** 2) + 350 * np.exp(-((h - 20.5) / 1.8) ** 2) - 250 * np.exp(-((h - 5) / 2) ** 2)
    return np.asarray(mw), "illustrative shape (add data/demand.csv for real EMA data)"


def main():
    offline = "--offline" in sys.argv
    t = grid(2)
    if offline: w, src = weather_offline(t), "offline clear-sky + synthetic cloud"
    else:
        try: w, src = weather_live(t), "Open-Meteo forecast"
        except Exception as e:
            print("Open-Meteo unavailable (", e, ") - falling back to offline weather"); w, src = weather_offline(t), "offline clear-sky + synthetic cloud"
    sp = LOC.get_solarposition(t)
    fixed = kw_per_kwp(poa(CFG["fleet_tilt"], CFG["fleet_azimuth"], sp, w), w.temp_air, w.wind_speed)
    track = kw_per_kwp(tracker_poa(sp, w), w.temp_air, w.wind_speed)
    fleet = fixed * CFG["installed_capacity_mwp"] * CFG["fleet_factor"]
    dem, dsrc = demand(t)
    out = {"generated": pd.Timestamp.now(CFG["tz"]).isoformat(timespec="minutes"), "source": src, "demand_source": dsrc,
           "installed_capacity_mwp": CFG["installed_capacity_mwp"], "capacity_note": CFG.get("capacity_note", ""),
           "times": [x.isoformat(timespec="minutes") for x in t],
           "kw_per_kwp": {"fixed": np.round(fixed.values, 4).tolist(), "tracker": np.round(track.values, 4).tolist()},
           "weather": {"ghi": np.round(w.ghi.values, 1).tolist(), "temp_air": np.round(w.temp_air.values, 1).tolist()},
           "fleet_mw": np.round(fleet.values).tolist(), "demand_mw": np.round(dem).tolist(),
           "solar_share_pct": np.round(fleet.values / dem * 100, 2).tolist()}
    days = {}
    for i, d in enumerate(("today", "tomorrow")):
        s = slice(i * 48, (i + 1) * 48)
        days[d] = {"fleet_gwh": round(float(fleet.values[s].sum() * .5 / 1000), 2),
                   "peak_mw": round(float(fleet.values[s].max())), "peak_share_pct": round(float(max(out["solar_share_pct"][s])), 1),
                   "fixed_kwh_per_kwp": round(float(fixed.values[s].sum() * .5), 2), "tracker_kwh_per_kwp": round(float(track.values[s].sum() * .5), 2)}
    out["daily"] = days
    (ROOT / "docs/data/latest.json").write_text(json.dumps(out, separators=(",", ":")))
    print("source:", src); print("demand:", dsrc)
    for d, v in days.items(): print(f"{d:9s}", json.dumps(v))


if __name__ == "__main__": main()
