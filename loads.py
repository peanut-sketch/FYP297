"""Residential vs commercial load profiles normalised to 1 kWh/day, plus a bench PV(+1 kWh battery) self-consumption test.
Shapes follow the brief: residential = low midday standby (~25 W) with a heavy evening peak 18:30-23:00 (~175 W) and a small
morning bump; commercial = ramp from 07:30, plateau 09:00-18:00, minimal night load. These are PARAMETRIC shapes, not
measured data: replace them with EMA/HDB or OpenEI profiles when you have them (see PROFILES below)."""
import json
import numpy as np
import twin
ROOT, CFG = twin.ROOT, twin.CFG
t = np.arange(48) / 2
sig = lambda x, c, w: 1 / (1 + np.exp(-(x - c) / w))
PROFILES = {
  "residential": 25 + 35 * np.exp(-((t - 7.25) / .9) ** 2) + 150 * (sig(t, 18.5, .35) - sig(t, 23, .4)) * (.6 + .4 * np.exp(-((t - 20.5) / 1.5) ** 2)),
  "commercial": .08 + .92 * (sig(t, 8.25, .45) - sig(t, 18.5, .4)),
}
norm = lambda s, kwh=1.0: s / (s.sum() * .5) * kwh        # kW per half-hour so that sum(kW*0.5h) = kwh

def dispatch(pv, load, cap=1.0, pmax=.5, eff=.92):
    """Greedy self-consumption with a battery (kWh). pv/load in kW at 30-min steps. Returns kWh totals."""
    soc = 0.0; imp = exp = 0.0; r = np.sqrt(eff)
    for p, l in zip(pv, load):
        n = (p - l) * .5
        if n > 0:
            c = min(n, (cap - soc) / r, pmax * .5); soc += c * r; exp += n - c
        else:
            d = min(-n, soc * r, pmax * .5); soc -= d / r; imp += -n - d
    return imp, exp

def main():
    d = json.load(open(ROOT / "docs/data/latest.json")); kwp = CFG["bench_kwp"]
    pv = np.array(d["kw_per_kwp"]["fixed"]) * kwp; days = len(pv) / 48; out = {"bench_kwp": kwp, "profiles_kw": {}, "results": {}, "pv_source": d["source"]}
    for name, sh in PROFILES.items():
        kw = norm(sh); out["profiles_kw"][name] = np.round(kw, 4).tolist(); load = np.tile(kw, int(days)); pvd = pv[:len(load)]
        pe = pvd.sum() * .5; le = load.sum() * .5; res = {}
        for lab, cap in (("no_battery", 0.0), ("battery_1kwh", 1.0)):
            imp, exp = dispatch(pvd, load, cap=cap) if cap else dispatch(pvd, load, cap=1e-9)
            res[lab] = {"self_consumption_pct": round((pe - exp) / pe * 100, 1), "self_sufficiency_pct": round((le - imp) / le * 100, 1),
                        "grid_import_kwh_day": round(imp / days, 2), "export_kwh_day": round(exp / days, 2)}
        out["results"][name] = res | {"pv_kwh_day": round(pe / days, 2), "load_kwh_day": round(le / days, 2)}
    (ROOT / "docs/data/loads.json").write_text(json.dumps(out, separators=(",", ":")))
    for k, v in out["results"].items(): print(k, json.dumps(v))
if __name__ == "__main__": main()
