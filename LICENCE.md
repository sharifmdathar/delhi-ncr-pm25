# Attribution and licensing

## The data this project uses

The measurements are **not** redistributed here. Everything under
`data/raw/` except the four small station-metadata tables is fetched by the
scripts in `scripts/` (see `DATA_CARD.md`).

| what | source | accessed via | licence |
|---|---|---|---|
| hourly PM2.5 and station meteorology (the target and the features) | Central Pollution Control Board (CPCB) continuous monitoring network, 15-minute records | the `Vonter/india-cpcb-aqi` release archive | Open Database Licence (ODbL) 1.0 |
| ERA5 reanalysis wind (one ablation only) | ERA5 via the Open-Meteo archive API | <https://archive-api.open-meteo.com/v1/archive> | Copernicus licence (free, non-commercial) |
| station metadata: identifier, state, city, name, latitude, longitude | CPCB station list; coordinates from the XKDR station registry | `data/raw/*.csv` (tracked) | ODbL 1.0, as below |

## The tracked station-metadata tables

`data/raw/stations.csv`, `data/raw/station_coords.csv`,
`data/raw/cpcb_stations.csv` and `data/raw/cpcb_station_catalog.csv` are small
factual tables: station identifier, state, city, station name, and
(where known) latitude and longitude.  They contain no measurements.  They are
tracked so that the distance and wind-aware graphs can be built without
downloading 1.3 GB of records, and they can be regenerated with:

```bash
python scripts/fetch_station_coords.py                 # from the tracked tables
python scripts/fetch_station_coords.py --xkdr_key aqi_...   # from XKDR
```

**These files are offered under the Open Database Licence (ODbL) 1.0**, the
same licence as the database they are derived from, so the share-alike
condition is satisfied: <http://opendatacommons.org/licenses/odbl/1.0/>.

Attribution: the station metadata is derived from the **Central Pollution
Control Board (CPCB)**, Ministry of Environment, Forest and Climate Change,
Government of India — <https://cpcb.gov.in/> — and from the **XKDR Forum**
station registry — <https://airquality.xkdr.org/>. The measurements themselves
are credited to CPCB; the ODbL packaging is by the `Vonter/india-cpcb-aqi`
project.

## A caveat we have flagged rather than hidden

The ODbL grant comes from the mirror project. CPCB's own website copyright
policy (updated 1 September 2019,
<https://cpcb.gov.in/copyright-policy/>) says:

> "Material on this site is subject to copyright protection unless otherwise
> indicated. The material may be downloaded to file or printer without
> requiring specific prior permission. Any other proposed use of the material
> is subject to the approval of competent authority of CPCB."

The `india-cpcb-aqi` README also notes that "some individual contents of the
database are under copyright by CPCB". We read these as a website-content
policy that the ODbL grant covers for the database, but the two are not
obviously consistent, and we would rather say so than assert a conclusion the
sources do not support. The practical effect of that reading is why this
repository redistributes **no measurements at all** — only the four small
metadata tables above, under the ODbL, with attribution, and regenerable in
one command. If you would rather the repository contained no CPCB-derived
material whatsoever, delete those four files and run
`scripts/fetch_station_coords.py` with an XKDR key.

## This repository's own code

MIT — see `LICENSE`. The code, the report and the analysis are ours; the
data is not, and is not included.
