"""Singapore solar PV digital twin: fetch weather, simulate with PVLIB, write docs/data/latest.json.
Usage: python twin.py            (live, Open-Meteo)
       python twin.py --offline  (clear-sky x constant clearness; for testing without network)"""
import json, sys, pathlib
import numpy as np, pandas as pd, pvlib

ROOT = pathlib.Path(__file__).parent
CFG = json.loads((ROOT / "config.json").read_text())
OFFLINE = "--offline" in sys.argv

def weather(loc):
    """Return 30-min DataFrame: ghi, dni, dhi (W/m2), temp_air (C), wind_speed (m/s), tz-aware SGT index."""
    if OFFLINE:
        t = pd.date_range(pd.Timestamp.now(tz=CFG["tz"]).normalize() - pd.Timedelta(days=1),
                          periods=48 * 3, freq="30min")
        cs = loc.get_clearsky(t)
        return pd.DataFrame({"ghi": cs.ghi * .6, "dni": cs.dni * .45, "dhi": cs.dhi * 1.6,
                             "temp_air": 30.0, "wind_speed": 2.0}, index=t)
    import requests
    r = requests.get("https://api.open-meteo.com/v1/forecast", timeout=30, params={
        "latitude": CFG["lat"], "longitude": CFG["lon"], "timezone": CFG["tz"],
        "hourly": "shortwave_radiation,direct_normal_irradiance,diffuse_radiation,temperature_2m,wind_speed_10m",
        "wind_speed_unit": "ms", "past_days": 1, "forecast_days": 2})
    r.raise_for_status()
    h = r.json()["hourly"]
    df = pd.DataFrame({"ghi": h["shortwave_radiation"], "dni": h["direct_normal_irradiance"],
                       "dhi": h["diffuse_radiation"], "temp_air": h["temperature_2m"],
                       "wind_speed": h["wind_speed_10m"]},
                      index=pd.to_datetime(h["time"]).tz_localize(CFG["tz"]))
    # Open-Meteo radiation is the mean of the PRECEDING hour: centre it, then go to 30-min steps
    df.index = df.index - pd.Timedelta(minutes=30)
    return df.resample("30min").interpolate("time").ffill().dropna()

def kw_per_kwp(poa, temp_air, wind):
    tc = pvlib.temperature.sapm_cell(poa, temp_air, wind,
          **pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"]["open_rack_glass_glass"])
    return (poa / 1000 * (1 + CFG["temp_coeff"] * (tc - 25)) * CFG["system_loss"]).clip(lower=0).fillna(0)

def poa(tilt, az, sp, w):
    p = pvlib.irradiance.get_total_irradiance(tilt, az, sp.apparent_zenith, sp.azimuth,
                                              w.dni, w.ghi, w.dhi, albedo=CFG["albedo"])
    return p.poa_global.fillna(0)

def demand(index):
    """Typical-day demand from data/demand.csv (columns: timestamp, demand_mw), else a fallback shape."""
    f = ROOT / CFG["demand_csv"]
    if f.exists():
        d = pd.read_csv(f, parse_dates=["timestamp"]).set_index("timestamp")["demand_mw"].tail(48 * 30)
        prof = d.groupby(d.index.hour * 2 + d.index.minute // 30).mean()
        return [round(float(prof.get(i.hour * 2 + i.minute // 30, np.nan)), 0) for i in index], "EMA CSV (30-day typical day)"
    h = index.hour + index.minute / 60
    v = 6300 + 1000 * np.exp(-((h - 14) / 4.5) ** 2) - 300 * np.exp(-((h - 4) / 2.5) ** 2) + 175 * np.exp(-((h - 20) / 2) ** 2)
    return [round(float(x)) for x in v], "illustrative (add data/demand.csv)"

def main():
    loc = pvlib.location.Location(CFG["lat"], CFG["lon"], CFG["tz"], CFG["altitude"])
    w = weather(loc)
    sp = loc.get_solarposition(w.index)
    trk = pvlib.tracking.singleaxis(sp.apparent_zenith, sp.azimuth, axis_tilt=0, axis_azimuth=0,
                                    max_angle=CFG["tracker_max_angle"], backtrack=True, gcr=CFG["tracker_gcr"])
    cfgs = {
        "fixed": poa(CFG["fleet_tilt"], CFG["fleet_azimuth"], sp, w),
        "flat": poa(0, 180, sp, w),
        "tracker": pvlib.irradiance.get_total_irradiance(trk.surface_tilt.fillna(0), trk.surface_azimuth.fillna(0),
                     sp.apparent_zenith, sp.azimuth, w.dni, w.ghi, w.dhi, albedo=CFG["albedo"]).poa_global.fillna(0),
    }
    out = {k: kw_per_kwp(v, w.temp_air, w.wind_speed) for k, v in cfgs.items()}
    fleet = out["fixed"] * CFG["installed_capacity_mwp"] * CFG["fleet_factor"]
    dem, src = demand(w.index)
    doc = {"generated": pd.Timestamp.now(tz=CFG["tz"]).isoformat(timespec="minutes"),
           "source": "offline test" if OFFLINE else "Open-Meteo (satellite-derived) + PVLIB",
           "capacity_mwp": CFG["installed_capacity_mwp"], "demand_source": src,
           "times": [t.isoformat(timespec="minutes") for t in w.index],
           "ghi": w.ghi.round(0).tolist(),
           "kw_per_kwp": {k: v.round(3).tolist() for k, v in out.items()},
           "fleet_mw": fleet.round(0).tolist(), "demand_mw": dem}
    p = ROOT / "docs/data/latest.json"; p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, separators=(",", ":")))
    print("wrote", p, "| peak fleet MW:", int(fleet.max()), "| kWh/kWp (fixed, 3 days):", round(out["fixed"].sum() / 2, 2))

if __name__ == "__main__":
    main()
