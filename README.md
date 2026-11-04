# Singapore Solar PV Digital Twin
Adapted from cindyiskandar/digital-twin-solar-pv (Easter Bush, Edinburgh).

## Go live (about 10 minutes)
1. Create a new GitHub repo, push this folder to it.
2. Settings > Pages > Deploy from branch > `main` / `/docs`.
3. Settings > Actions > General > Workflow permissions > Read and write.
4. Actions tab > "Update twin" > Run workflow. Then open `https://<you>.github.io/<repo>/`.
No API keys or SSH keys: Open-Meteo needs none and the workflow pushes with the built-in token.

## Real EMA demand
Download a month from https://www.ema.gov.sg/resources/statistics/half-hourly-system-demand-data,
reshape to `data/demand.csv` with columns `timestamp,demand_mw` (30-min, SGT), commit. Until then demand is an illustrative shape.

## Update each quarter
`installed_capacity_mwp` in config.json from EMA's quarterly statistics.

Local test: `pip install -r requirements.txt && python twin.py` (or `--offline`).
View the dashboard locally: `python -m http.server -d docs`, then open http://localhost:8000 (browsers block the data files when opening `index.html` directly).

## Forecasting, faults and loads (added)
- `python forecast.py`: trains on up to 2 years of Open-Meteo history, forecasts 48 h, reports RMSE / nMAE / MAPE on a held-out final 30 days against persistence baselines. Writes `docs/data/forecast.json`. Add `--synthetic` to test offline.
- `python faults.py`: reads `data/measured.csv` (`timestamp,power_w,irradiance_wm2[,module_temp_c]`) and flags outage, step loss (bypass diode / string), soiling or degradation, and shading. `--selftest` injects known faults.
- `python loads.py`: residential and commercial profiles at 1 kWh/day, bench PV plus 1 kWh battery self-consumption. Parametric shapes until you swap in real profiles.
- `.github/workflows/forecast.yml` runs all of it daily at 03:00 SGT.
- Benchmark vs EMC: add the EMC/EMA forecast as a CSV and compare with `forecast.json` (not built yet, I could not verify a public forecast feed).
