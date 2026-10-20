# Attribution and licensing

## What this repository contains, precisely

| | |
|---|---|
| code, report, analysis | **ours**, MIT ([LICENSE](LICENSE)) |
| **one** Derived Database of the CPCB records | `data/sample/sample_dataset.npz` — 212 KB, 8 stations, 3 months (Nov 2024 – Jan 2025) |
| station metadata | `data/raw/{stations,station_coords,cpcb_stations,cpcb_station_catalog}.csv` — identifier, state, city, name, coordinates. No measurements. |
| **everything else** | **not included.** The full record (1.3 GB) and the ERA5 extract are fetched by the scripts. |

So: the repository redistributes **one small sample** of the observations, and
nothing else. It does not redistribute the CPCB record, the ERA5 extract, the
frozen full dataset, or the model checkpoints.

## The sources

| what | source | accessed via | licence |
|---|---|---|---|
| hourly PM2.5 and station meteorology (the target and the features) | Central Pollution Control Board (CPCB) continuous monitoring network, 15-minute records | the `Vonter/india-cpcb-aqi` release archive | Open Database Licence (ODbL) 1.0 |
| ERA5 reanalysis wind (one ablation only) | ERA5 via the Open-Meteo archive API | <https://archive-api.open-meteo.com/v1/archive> | Copernicus licence (free, non-commercial) |
| station metadata | CPCB station list; coordinates from the XKDR station registry | `data/raw/*.csv` (tracked) | ODbL 1.0, as below |

## The sample dataset

`data/sample/sample_dataset.npz` is a **Derived Database** of the CPCB
records: the same cleaning, the same normalisation, the same split logic as
the full build, over eight stations and three months. It is **offered under
the Open Database Licence 1.0** — <http://opendatacommons.org/licenses/odbl/1.0/> —
which is what that licence requires of an adapted version: attribution, and
the share-alike condition, both of which this repository satisfies by
offering the sample under the ODbL and by making the code that produced it
available.

Attribution: the sample is derived from the **Central Pollution Control
Board (CPCB)**, Ministry of Environment, Forest and Climate Change,
Government of India — <https://cpcb.gov.in/> — via the `Vonter/india-cpcb-aqi`
packaging. The measurements are credited to CPCB.

It can be rebuilt from source at any time:

```bash
python scripts/fetch_cpcb_years.py --years 2024 2025   # ~1.3 GB
python scripts/prepare_real_data.py pm25 --src data/raw/cpcb_station_csv
python scripts/make_sample_dataset.py
```

## The ambiguity we are flagging rather than resolving

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
sources do not support.

What we did about it:

- We chose the conservative option. The ODbL permits distributing a Derived
  Database, and we do so with attribution and under the same licence — but
  only a **212 KB** sample of a 1.3 GB record. The rest is fetched from
  source by the scripts in `scripts/`.
- We did not publish the full dataset, the raw records, the ERA5 extract, or
  the checkpoints.
- **If CPCB would prefer the sample not to be here, deleting
  `data/sample/` and `scripts/make_sample_dataset.py` removes it; the full
  pipeline still runs from the fetch scripts.** We would also be glad to
  remove it on request, and we would welcome clarification of their terms for
  redistributed derivatives.
