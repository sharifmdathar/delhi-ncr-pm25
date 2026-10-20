# Delhi-NCR Hourly PM2.5 Spatiotemporal Forecasting

**Research Question:** Does a wind-aware dynamic graph beat a static distance graph, and when?

**Answer:** Partly — and the detail matters. A wind-aware directed graph *does*
beat a static distance graph for Graph WaveNet (33.77 vs 34.60 µg/m³, paired
difference 0.78, 95% CI [0.40, 1.15] by bootstrap over 105 test weeks), but is
indistinguishable from it for STGCN (33.05 vs 33.18, CI [−0.11, 0.30]). More
importantly, **no graph at all beats both** in both architectures
(0.36–0.51 µg/m³, both CIs excluding zero) — and Graph WaveNet's *learned*
adaptive adjacency, which is trained end-to-end rather than designed by hand,
is the **worst** graph of the four (+1.27 µg/m³ vs no graph, CI [0.58, 2.01]).

Two follow-ups pin down *why*, and they change the story rather than confirm it:

- **The graph is not delivering the regional signal.** A no-graph model given
  the network-wide mean PM2.5 as an extra input feature scores 32.60 against
  32.66 without it (paired difference 0.05, CI [−0.11, −0.00]) — and is
  significantly better than every graph model. So handing the model the
  regional signal for free is worth nothing, and the graph's spatial mixing is
  a pure cost.
- **Nor is the null result caused by the noisy local wind.** Rebuilding the
  wind-aware graph with ERA5 reanalysis instead of the stations' own
  (noisy, and missing for 9 of 42) anemometers changes nothing. This tests
  one edge construction, not wind-aware graphs in general.

The differences are statistically clear but small: the largest gap between
graph structures is ~1.3 µg/m³ on a score of roughly 33, i.e. about 4%. The
headline margin is against the best *simple* baseline — 32.66 against
seasonal-naive's 38.63, **15.5% better** (paired CI [4.84, 7.38] µg/m³).

Every neural model here does beat two strong baselines, so the comparison is
meaningful rather than a broken-model artefact.

---

## Results (Test 2024–2025, 2000 windows, masked µg/m³, 8 seeds for the graph models)

| Config | MAE | RMSE | R² | 1h | 48h |
|--------|-----|------|-----|----|-----|
| **STGCN, no graph** | **32.66 ± 0.16** | 57.06 | 0.628 | 15.04 | 36.31 |
| LSTM Univariate | 32.87 ± 0.09 | 57.57 | 0.622 | 15.46 | 36.92 |
| STGCN, wind-aware | 33.05 ± 0.12 | 57.99 | 0.616 | 15.35 | 36.48 |
| STGCN, distance | 33.18 ± 0.15 | 57.84 | 0.618 | 15.75 | 36.58 |
| GraphWaveNet, no graph | 33.44 ± 0.13 | 58.49 | 0.610 | 15.22 | 37.58 |
| STGCN, correlation | 33.46 ± 0.43 | 57.88 | 0.618 | 15.56 | 36.33 |
| GraphWaveNet, wind-aware | 33.77 ± 0.24 | 59.42 | 0.597 | 15.56 | 37.65 |
| GraphWaveNet, distance | 34.60 ± 0.61 | 59.96 | 0.590 | 15.67 | 38.48 |
| GraphWaveNet, correlation | 35.43 ± 1.11 | 59.23 | 0.600 | 15.96 | 37.88 |
| GraphWaveNet, learned adaptive | 34.70 ± 0.47 | 59.51 | 0.596 | 16.08 | 37.65 |
| LSTM Multivariate | 38.20 ± 1.83 | 66.94 | 0.489 | 16.27 | 41.66 |
| Transformer | 38.88 ± 1.04 | 66.34 | 0.498 | 16.38 | 42.41 |
| Ridge (24 lags) | 44.03 | 73.69 | 0.380 | 15.83 | 42.34 |
| Persistence | 46.79 | 79.01 | 0.288 | 16.55 | 44.74 |

### Graph ablation

| Graph | STGCN MAE | GraphWaveNet MAE |
|-------|-----------|------------------|
| **None (identity)** | **32.66 ± 0.16** | **33.44 ± 0.13** |
| Wind-aware (directed) | 33.05 ± 0.12 | 33.77 ± 0.24 |
| Distance (static) | 33.18 ± 0.15 | 34.60 ± 0.61 |
| Correlation (static) | 33.46 ± 0.43 | 35.43 ± 1.11 |
| Learned adaptive | — | 34.70 ± 0.47 |

Differences this small are not decidable from a per-seed standard deviation, so
the comparison is made with a **paired bootstrap over the 105 ISO weeks of the
test period** (the same weeks resampled for both models), which respects the
temporal correlation in the errors:

| Comparison | ΔMAE | 95% CI | P(A better) | Verdict |
|------------|------|--------|-------------|---------|
| STGCN, distance vs wind-aware | +0.09 | [−0.11, 0.30] | 0.186 | not distinguishable |
| GraphWaveNet, distance vs wind-aware | +0.78 | [0.40, 1.15] | 0.000 | **wind-aware better** |
| STGCN, distance vs no graph | +0.51 | [0.15, 0.89] | 0.003 | **no graph better** |
| STGCN, wind-aware vs no graph | +0.42 | [0.12, 0.74] | 0.003 | **no graph better** |
| GraphWaveNet, distance vs no graph | +1.14 | [0.56, 1.75] | 0.000 | **no graph better** |
| GraphWaveNet, wind-aware vs no graph | +0.36 | [0.01, 0.75] | 0.023 | **no graph better** |
| GraphWaveNet, learned adaptive vs no graph | +1.27 | [0.58, 2.01] | 0.000 | **no graph better** |
| GraphWaveNet, learned adaptive vs wind-aware | +0.90 | [0.46, 1.39] | 0.000 | **wind-aware better** |
| STGCN, no graph vs ridge (24 lags) | −11.27 | [−13.02, −9.65] | 1.000 | **no graph better** |
| STGCN, no graph vs persistence | −14.02 | [−16.05, −12.08] | 1.000 | **no graph better** |

### Where the graph penalty is actually paid: severe episodes

Severe PM2.5 episodes in Delhi occur under calm, stable conditions: 78.2% of the
test windows in the lowest wind third contain a severe-episode hour, against
52.1% and 49.0% in the two windier thirds. Splitting the test set by episode
membership shows where the ablation comes from:

| Subset | STGCN no graph | STGCN wind | STGCN dist. | GWN no graph | GWN wind | GWN dist. |
|--------|----------------|------------|-------------|--------------|----------|-----------|
| all windows | **32.66** | 33.05 | 33.18 | **33.44** | 33.77 | 34.60 |
| episode windows | **42.31** | 42.92 | 43.16 | **43.34** | 43.84 | 45.06 |
| non-episode windows | **17.90** | 17.93 | 17.91 | **18.29** | 18.35 | 18.58 |

The graph penalty is paid almost entirely in the high-pollution regime, and
spatial mixing does not rescue the high-pollution case even where transport
should matter most.

### Spatial pattern: the periphery pays the most

If a graph carried information a station could not see, the benefit should be
largest at the edge of the network. The opposite holds. Using distance from
the network centroid as a proxy for peripherality, the penalty that spatial
mixing imposes relative to the no-graph model grows with that distance, for
every architecture and every graph type:

| model | wind-aware penalty near / far | distance penalty near / far | corr(radius, penalty) |
|-------|------------------------------|-----------------------------|------------------------|
| STGCN | +0.13 / +0.62 | +0.35 / +0.72 | +0.42 (wind), +0.34 (dist) |
| GraphWaveNet | +0.20 / +0.33 | +1.23 / +1.06 | +0.31 (wind), +0.11 (dist) |

### Adjacency introspection

The self-adaptive adjacency that Graph WaveNet learns is essentially
uncorrelated with pairwise distance (Spearman *r* = +0.09) and with the
prevailing-wind alignment (*r* = +0.02). The network learns a structure that
is neither the distance graph nor the wind-aware graph — and that structure
does not pay either.

### Case studies

| event | start | network peak (µg/m³) | persistence | ridge | no graph | wind-aware | GWN wind |
|-------|-------|---------------------|-----------|-------|----------|------------|----------|
| Diwali 2024 | 2024-10-30 | 221 | 53.8 | **59.5** | 65.9 | 81.7 | 92.3 |
| Winter inversion | 2024-01-13 | 575 | 108.6 | 86.3 | **20.1** | **20.0** | 27.2 |
| Dust episode | 2024-05-10 | 162 | **15.5** | 15.6 | 17.8 | 15.5 | **14.5** |

Diwali is the failure case: fireworks add a sharp anthropogenic spike with no
precursor in the inputs, so the neural models smooth over it and lose to ridge.

### The headline margin, per horizon

Plain persistence is the hard baseline at 1 hour; seasonal naive is the hard
one at 24 and 48 hours. The honest comparison at each horizon is against
whichever of the two is lower:

| horizon | best model | best simple baseline | its MAE | margin |
|---------|-----------|----------------------|---------|--------|
| 1h | 15.05 | persistence | 16.55 | 9.0% |
| 3h | 23.82 | persistence | 31.15 | 23.5% |
| 6h | 28.53 | seasonal naive | 38.39 | 25.7% |
| 12h | 31.05 | seasonal naive | 38.58 | 19.5% |
| 24h | 32.92 | persistence | 38.41 | 14.3% |
| 48h | 36.39 | seasonal naive | 38.60 | 5.7% |
| **overall** | **32.66** | **seasonal naive** | **38.63** | **15.5%** |

(Seed-averaged; the 24 h tie goes to persistence by 0.01 µg/m³.) The overall
margin is 6.04 µg/m³, paired bootstrap CI [4.84, 7.38].

### Rolling-origin check (is it just the split boundary?)

Train on 2019…Y, validate on Q4 of Y, test on Y+1. The ranking is stable as
the boundary moves, so the finding is not an artefact of one split:

| Model | Fold | Test year | No graph | Wind-aware | Distance | Best |
|-------|------|-----------|----------|------------|----------|------|
| STGCN | fold0 | 2021 | 36.65 | 36.98 | 37.34 | no graph |
| GraphWaveNet | fold0 | 2021 | 37.56 | 37.63 | 43.25 | no graph |
| STGCN | fold1 | 2022 | 32.96 | 33.09 | 32.61 | distance |
| STGCN | fold2 | 2023 | 33.03 | 33.01 | 33.58 | wind-aware |
| GraphWaveNet | fold2 | 2023 | 34.23 | 33.64 | 34.05 | wind-aware |
| STGCN | fold3 | 2024 | 33.82 | 34.57 | 34.22 | no graph |
| STGCN | fold4 | 2025 | 30.76 | 30.77 | 31.41 | no graph |
| GraphWaveNet | fold4 | 2025 | 31.32 | 31.23 | 32.87 | wind-aware |

### Is the null result caused by the noisy local wind?

Nine of 42 stations report no wind at all, and where it is reported the
per-station vectors agree only weakly (cross-station correlation 0.11). So we
rebuilt the wind-aware graph with **ERA5 reanalysis** instead — complete for
every station and spatially smooth (correlation 0.98) — changing nothing else:

| model | edge direction | MAE |
|-------|----------------|-----|
| STGCN | no graph | **32.66 ± 0.16** |
| STGCN | wind, station | 33.05 ± 0.12 |
| STGCN | wind, ERA5 | 33.10 ± 0.21 |
| GraphWaveNet | no graph | **33.44 ± 0.13** |
| GraphWaveNet | wind, station | 33.77 ± 0.24 |
| GraphWaveNet | wind, ERA5 | 34.01 ± 0.59 |

The two wind sources are indistinguishable (STGCN Δ 0.01, CI [−0.25, 0.29];
GraphWaveNet Δ 0.21, CI [−0.16, 0.57]).

**What this does and does not show.** The null result is not caused by the
noisy or missing local wind. It is *not* a claim that wind cannot help: this
tests one edge construction (an hourly, wind-direction-weighted kernel using a
period-mean prevailing direction) against one alternative wind source.
A time-lagged advection kernel, a wind field finer than ERA5's 25 km grid, or
an edge weight driven by the concentration gradient rather than the direction
are different constructions and are not tested here.

### Is the graph just delivering the regional signal?

A no-graph model with the **network-wide mean PM2.5** as an extra input
feature (computed from the dataset's own PM2.5, so nothing else changes):

| configuration | MAE |
|---------------|-----|
| STGCN, no graph | 32.66 ± 0.16 |
| **STGCN, no graph + network mean** | **32.60 ± 0.09** |
| STGCN, wind-aware graph | 33.05 ± 0.12 |
| GraphWaveNet, no graph | 33.44 ± 0.13 |
| GraphWaveNet, wind-aware graph | 33.77 ± 0.24 |

Telling the model the regional signal explicitly is worth **nothing**
(paired difference 0.05, CI [−0.11, −0.00]) — and that model is significantly
*better* than every graph model (0.47 better than STGCN with the wind-aware
graph, CI [0.18, 0.80]).

This rules out the benign reading. The graph is not delivering the regional
signal and being ignored for it; delivering that signal is worth nothing at
all, so the graph's spatial mixing is a pure cost.

### "And when?" — leave-station-out

Ten stations are held out entirely: their targets never enter the training
loss. Ridge is refitted on the remaining 32 stations. The spatial baselines
are the honest comparators in the second variant, because they use the same
information the graph has — the other stations' recent observations — and
nothing else.

| variant | STGCN no graph | STGCN wind | STGCN dist | GWN no graph | GWN wind | GWN dist | **IDW neighbours** | network mean | ridge |
|---------|----------------|------------|------------|--------------|----------|----------|-------------------|--------------|-------|
| targets unseen, own history visible | **35.35** | 35.73 | 35.98 | **35.67** | 36.27 | 36.70 | 41.24 | 47.01 | 47.48 |
| targets *and* own history unseen | 69.19 | 55.54 | 54.33 | 70.68 | 57.82 | 55.45 | **41.24** | 47.01 | 47.48 |

- **For an unseen station that still has its own history, the graph still
  hurts** (0.46–0.84 µg/m³, all intervals excluding zero) — but the models
  *do* generalise to new monitors, beating the best simple spatial baseline
  by 5.15 and the network-wide mean by 11.50.
- **With the station's own history removed as well, the graph loses badly.** A
  two-line inverse-distance-weighted average of the neighbours reaches 41.24
  (R² = 0.51); the best graph model reaches 53.45 (R² = 0.15). The graph is
  **14.3 µg/m³ worse**, CI [9.9, 19.1]. The network-wide mean alone (47.01)
  also beats every graph model.

> We first reported this as "the graph is essential, a 13.4 µg/m³ improvement,
> the largest effect in the study". That was against a strawman — the only
> no-graph comparator there was a model with its eyes closed, and the baselines
> that *did* have access to the station's own history were unbeaten anyway.
> With honest spatial baselines the sign flips, and the corrected version is
> in the report.

Read together: the neighbours do not carry as much information about a
station as that station's own recent observations, and a graph neural network
mixes them *worse* than an inverse-distance-weighted average does.

Full tables (all horizons, episode analysis, per-station errors) in
[`RESULTS_SUMMARY.md`](RESULTS_SUMMARY.md).

---

## Quick start

### Five minutes, no download

A tiny sample dataset is included (8 stations, 3 months of December 2024 and
January 2025, 215 KB) so the pipeline can be run end to end immediately:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/evaluate.py --split test --data data/sample/sample_dataset.npz
python scripts/train_all.py --data data/sample/sample_dataset.npz \
    --models stgcn --graph identity --seeds 42 --epochs 4
```

The sample is a Derived Database of the CPCB records, offered under the ODbL
with attribution — see `data/sample/README.md` and `LICENCE.md`.

### The full pipeline

```bash
# 1. Fetch real CPCB data (2019-2025, 15-min).  The station files also carry
#    temperature, humidity, wind speed/direction and rainfall measured at the
#    station, so no reanalysis download is needed.  ~745 MB per year.
python scripts/fetch_cpcb_years.py --years 2019 2020 2021 2022 2023 2024 2025
python scripts/prepare_real_data.py pm25 --src data/raw/cpcb_station_csv

# 1b. ERA5 wind for the ERA5-wind ablation (free, no API key, ~5 min)
python scripts/fetch_era5_wind.py --years 2019 2020 2021 2022 2023 2024 2025

# 1c. Station coordinates (only needed if you want to rebuild the graphs
#     without the tracked metadata tables; needs a free XKDR key)
python scripts/fetch_station_coords.py --xkdr_key aqi_...

# 2. Build frozen dataset (fails if any station is below 70% real coverage)
python scripts/build_dataset.py

# 3. Train (--graph selects the adjacency; --split_mode expanding runs a
#    rolling-origin fold instead of the fixed split; --holdout_k runs the
#    leave-station-out experiment)
python scripts/train_all.py --models lstm_univariate stgcn --seeds 42 123 456
python scripts/train_all.py --models graph_wavenet --graph wind_aware --seeds 42 123 456
python scripts/train_all.py --models stgcn --split_mode expanding --fold fold0
python scripts/train_all.py --models stgcn --graph wind_aware_era5 --seeds 42
python scripts/train_all.py --models stgcn --graph identity --holdout_k 10 --holdout_mode both
python scripts/train_netmean.py --models stgcn --graph identity --seeds 42 123 456

# 4. Evaluate, analyse, write the report
python scripts/evaluate.py --split test --max_windows 2000
python scripts/bootstrap.py --n_boot 50000
python scripts/analyze.py --max_windows 1200
python scripts/eval_lso.py                       # leave-station-out
python scripts/provenance.py                     # provenance + mask audit
python scripts/make_report.py
python scripts/make_results_summary.py

# 5. Checks
python -m pytest tests/ -q
```

---

## Data

| Item | Value |
|------|-------|
| Source | CPCB 15-min, via `Vonter/india-cpcb-aqi` releases (2009–2025) |
| Stations | **42** (34 Delhi + 8 Noida/Ghaziabad), ≥70% real coverage in every split |
| Period | **2019-01-01 → 2025-12-31** (61,345 hours, 7 years) |
| Target | Hourly PM2.5 (µg/m³) |
| Input | Past 72 hours |
| Output | Next 48 hours |
| Features | PM2.5 + station-measured T, RH, wind speed/direction, rainfall |
| Splits | Train 2019–2022 (34,922 windows) · Val 2023 (8,641) · Test 2024–2025 (17,402), chronological, no window crosses a boundary |
| Normalisation | log1p(PM2.5) → z-score (train-only stats) |
| Missing | masked (1 = really observed), honoured in loss and metrics |

Split distributions are matched (mean 102.0 / 98.0 / 97.5 µg/m³; fraction
> 300 µg/m³ = 4.82% / 4.23% / 4.15%), and the training split contains **four
full winters**, so winter dynamics are learnable.  Split boundaries are
contiguous (0 h train→val) with a 23 h gap before test; no window's input or
target ever crosses a boundary.

### Weather coverage caveat

The weather variables are the station-measured values in the CPCB files, not a
reanalysis.  That gives genuine local structure (mean cross-station correlation
of `wind_u` is 0.11; no two stations ever record the same wind in the same
hour) — but **9 of the 42 stations report no wind at all**, in any year.  For
those stations the wind-aware graph falls back to the network-mean wind
direction.  `scripts/wind_coverage_analysis.py` shows that the penalty any
graph imposes is *larger* at the stations that do have wind.  A natural
reading — and we want to be clear this is a **hypothesis consistent with the
data, not a demonstrated mechanism** — is that what a graph costs scales with
how much it mixes: a station-specific wind produces station-specific edges
that pull in neighbours carrying a different local signal, while the fallback
produces a coarser, more uniform graph that dilutes less.

---

## Models

| Model | Description | Key config |
|-------|-------------|------------|
| Persistence | ŷ = y(t−1) | baseline |
| Seasonal naive (24 h) | ŷ = y(t−24) | baseline — much stronger than persistence at 24 h and 48 h |
| Hour-of-week average | per-station mean PM2.5 for each hour of the week, train split only | baseline / climatological floor |
| Ridge (24 lags) | per-station linear map, last 24 h → 48 horizons | baseline / linear floor |
| LSTM Univariate | shared-weight LSTM applied per station | hidden=64, layers=2, dropout=0.3 |
| LSTM Multivariate | single LSTM, all stations as channels | hidden=64, layers=2 |
| Transformer | temporal attention, stations as channels | hidden=64, layers=2, heads=4 |
| STGCN | graph convolution + GRU | hidden=64, σ=5 km |
| GraphWaveNet | dilated causal convs + graph conv | blocks=2, layers=4, conv width 32 |

All models predict the **change from the last observed value** and carry a
**linear autoregressive skip**, so each starts at Persistence and is measured
relative to ridge. See `src/models/base.py`.

**Seasonal naive is the bar that matters at long horizons.** Delhi's PM2.5 has
a 51 µg/m³ diurnal cycle, so the value 24 hours earlier is far more
informative than the value 1 hour earlier:

| baseline | 1h | 24h | 48h | overall |
|---|---|---|---|---|
| Persistence | 16.55 | 38.41 | 44.74 | 46.79 |
| **Seasonal naive (24 h)** | 38.86 | 38.41 | **38.60** | **38.63** |
| Hour-of-week average | 60.15 | 59.71 | 59.86 | 59.86 |
| Ridge (24 lags) | 15.83 | **36.45** | 42.34 | 44.03 |

The best neural model beats seasonal naive by 6.04 µg/m³ overall (paired
bootstrap CI [4.83, 7.38]), but the LSTM Multivariate does **not** beat it
reliably (−0.59, CI [−2.58, 1.69]) — a result worth stating plainly.

### Sanity checks that were run, and what they caught

| check | expected | result |
|-------|----------|--------|
| Persistence R² at 1h (window/timestamp alignment) | > 0.8 | **0.889** — passes, but only after fixing a bug (below) |
| Persistence MAE at 1h vs raw data | ≈ mean \|1h change\| | 16.55 vs 16.6 — matches |
| Every neural model beats ridge at every horizon | yes | yes |
| Every window's inputs *and* targets inside its own split | yes | verified programmatically |
| Split distributions matched across train/val/test | yes | 102.0 / 98.0 / 97.5 µg/m³ |

### The persistence bug, and how it was caught

This is worth telling because it is the kind of thing an interviewer's
question tends to be about.

The alignment check in the table above is: *persistence's R² at a one-hour
horizon should be well above 0.8*, because a one-hour-ahead forecast that
merely repeats the last observation should explain almost all of the
variance. **It came out at 0.259.** Persistence was scoring MAE 46.6 µg/m³ at
one hour, while the mean absolute hourly change measured directly on the raw
CPCB records was 16.6.

The data was fine. The dataset's own `mean |y[t+1] − y[t]|` is 16.95, matching
the raw records (16.98), and the frozen hourly series aligns with a direct
resample of the raw 15-minute files at lag 0 (mean abs difference 8.59,
against 13.14 at lag +1 and 21.56 at lag −1).

The bug was in `scripts/evaluate.py::persistence_predictions`. The walk-back
that finds "the last observed value" for each station looked like this:

```python
last = y[m - 1]
for back in range(0, 72):
    cand = y[m - 1 - back]
    obs  = mask[m - 1 - back]
    last = np.where(obs > 0, cand, last)   # overwrites with OLDER values
```

Every observed hour in the window overwrote the previous one, so the value
that survived was the *oldest* one — the baseline was really "persistence from
up to 72 hours ago". That also explained the nonsensical non-monotonic
horizon curve.

The fix is to fill each station only once, scanning backwards, so the most
recent observation wins. Ridge was implemented correctly and was unaffected;
the models never took that code path, so **no retraining was needed** — but
every reported baseline number changed, and the earlier headline figures were
overstated. After the fix, persistence scores R² 0.889 and MAE 16.3 at one
hour, and 46.79 overall (was 54.83).

The lesson we took: a sanity check with a *predicted* value is worth more
than a sanity check with a computed one. We now run the set in the table
above, and `scripts/provenance.py` audits the frozen dataset for any value
that is not backed by a real observation.

### The per-horizon curve is not monotonic, and that is the data

Persistence's error is worse at 12 h than at 24 h. This is a property of
Delhi's PM2.5, not an artefact: the network-mean PM2.5 has a 51 µg/m³
diurnal cycle (peak 117.8 µg/m³ at 05:00, trough 66.4 µg/m³ at 22:00), so a
12-hour horizon lands on the opposite phase of that cycle while a 24-hour
horizon returns to the same one. The raw records show the same pattern (mean
|12 h diff| 50.30 against mean |24 h diff| 39.54).

---

## Data provenance

Full detail in [`DATA_CARD.md`](DATA_CARD.md); the machine-checked audit is
regenerated by `scripts/provenance.py` into `outputs/provenance.md`.

| what | source | accessed via | licence |
|---|---|---|---|
| hourly PM2.5 + station meteorology | CPCB continuous monitoring, 15-min | `Vonter/india-cpcb-aqi` release archive | ODbL |
| ERA5 reanalysis wind (one ablation) | ERA5 via Open-Meteo | `archive-api.open-meteo.com` | Copernicus |
| station metadata | CPCB station list, XKDR registry | committed in `data/raw/*.csv` | public |

**No raw data is committed.** The CPCB records are ODbL and the ERA5 files are
under the Copernicus licence; the fetch scripts regenerate them (no API keys
needed) and only the small public station-metadata tables are in the repo.

**No synthetic values, and that is checked mechanically.** The build fails if
any station falls below 70% real-observation coverage in any split, the target
is never imputed, and the audit reports zero values present while flagged
unobserved. The target's observed coverage is 90.80% (train), 92.32% (val) and
93.52% (test); every metric is computed on the observed remainder only.

---

## Graph structures

| Type | Definition |
|------|------------|
| Distance | Gaussian kernel exp(−d²/2σ²) on haversine distance, σ = 5 km |
| Correlation | pairwise-complete Pearson correlation of PM2.5 (train only), thresholded |
| Wind-aware | directed edge i→j only when the prevailing wind at i points at j: exp(−d²/2σ²)·max(0, cos(θ_ij − φ_i)) |
| Wind-aware (ERA5) | the same construction, but the edge directions come from ERA5 instead of the stations' anemometers — an ablation on the *source* of the wind |
| Learned adaptive | softmax(relu(E₁E₂ᵀ)), learned in-model |
| None (identity) | ablation baseline |

---

## Project layout

```
├── AGENTS.md                  locked scope
├── EXPERIMENTS.md             every run logged, including the failed study
├── RESULTS_SUMMARY.md         final tables
├── scripts/
│   ├── fetch_cpcb_years.py   download + extract CPCB 2019-2025
│   ├── prepare_real_data.py   OpenCity parsing + ERA5 weather
│   ├── build_dataset.py       frozen dataset, with the coverage gate
│   ├── train_all.py           training (batched, delta targets, graph arg)
│   ├── evaluate.py            baselines + every checkpoint
│   ├── analyze.py             episodes, spatial errors, figures
│   ├── make_report.py         LaTeX generated from the JSON artefacts
│   └── make_animated_map.py   GIF
├── src/
│   ├── constants.py           locked scope + MIN_REAL_DATA_FRACTION
│   ├── data/                  cleaning (no synthetic fallback), dataset, graphs
│   ├── models/                base (delta + skip), lstm, transformer, stgcn, graph_wavenet
│   ├── train/                 masked metrics
│   └── eval/                  episodes, plots
├── data/processed/            frozen dataset + coverage_report.csv
├── checkpoints/               33 trained models
├── outputs/                   JSON results, figures, aqi_map.gif
└── report/report.tex          generated report
```

---

## Limitations

- **Selection bias.** The modelling set keeps only stations with ≥70% real
  observation coverage in every split, which drops the poorly-maintained
  monitors. The results therefore describe a network of *well-maintained*
  stations, and the coverage of the dropped ones is not random: outages
  cluster at particular stations and periods, and they coincide with
  high-pollution episodes, so the observed remainder is not a random sample
  of the underlying process. All metrics are masked to observed values, which
  is the best available correction, but it does not remove the selection
  effect. Generalising to a network with intermittent stations is untested.
- Seven years of data (2019–2025), but the fixed test block covers only two
  winters, so generalisation across pollution regimes is only partly tested.
  The rolling-origin evaluation (5 folds, test years 2021–2025) is our partial
  answer to this.
- The wind-aware graph uses a period-mean prevailing direction and assumes a
  single direction per station. The per-station wind vectors agree only weakly
  (mean cross-station correlation of the u-component is 0.11), so that mean
  direction is itself a noisy estimate — though the ERA5 ablation shows this
  noise is not what decides the result.
- The ERA5 ablation tests **one** edge construction against **one**
  alternative wind source. It does not rule out time-lagged advection, finer
  wind fields, or gradient-driven edge weights.
- The graph comparison rests on 8 seeds for the two graph architectures, but on
  a single model family and a single hyperparameter setting per configuration.
  Each variant received the same tuning effort, but we did not sweep the
  graph-specific hyperparameters.
- The leave-station-out experiment uses 10 held-out stations, 2 seeds and 4
  epochs per run (the fixed-split runs use 8 seeds and 8 epochs), because it
  was time-boxed. It is enough to resolve a 14 µg/m³ effect; it is not enough
  to resolve a 0.4 µg/m³ one with the same confidence.
- In the leave-station-out "own history also hidden" variant, persistence,
  ridge and seasonal-naive are **not** blinded: they keep using the held-out
  stations' recent values, which the models were denied. The comparison
  between the models and the *spatial* baselines (IDW, network mean) is fair,
  because those also use only the other stations; the comparison against
  persistence/ridge/seasonal-naive is deliberately generous to the baselines.
- Static graphs except GraphWaveNet's learned variant.
- Severe episodes remain 3–5× worse than ordinary conditions for every model.
- The LSTM Multivariate does not beat the seasonal-naive baseline reliably
  (−0.59 µg/m³, CI [−2.58, 1.69]).

---

## License and attribution

**Code: MIT** ([LICENSE](LICENSE)).

**Data: not redistributed.** No measurements are included in this
repository — they are fetched by the scripts. Full attribution and the
licence position are in [LICENCE.md](LICENCE.md) and
[DATA_CARD.md](DATA_CARD.md). In short:

- PM2.5 and station meteorology: **Central Pollution Control Board (CPCB)**,
  Ministry of Environment, Forest and Climate Change, Government of India
  (<https://cpcb.gov.in/>), via the `Vonter/india-cpcb-aqi` release archive,
  under the **Open Database Licence 1.0**. The station-meteorology variables
  are the values measured at the stations, distributed in the same CPCB files.
- ERA5 reanalysis wind (one ablation only): **ERA5 via the Open-Meteo
  archive API**, under the **Copernicus licence**.
- Station coordinates: the **XKDR Forum** station registry
  (<https://airquality.xkdr.org/>).

The four small station-metadata tables that *are* tracked (identifier, city,
name, coordinates — no measurements) are offered under the ODbL with
attribution, and are regenerable with one command. CPCB's own website
copyright policy adds a condition beyond the ODbL grant; we flag that
ambiguity explicitly in `LICENCE.md` rather than resolving it in our own
favour.
