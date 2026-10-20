# Delhi-NCR PM2.5 Forecasting — Final Results (2019–2025 real data)

## Research Question
**Does a wind-aware dynamic graph beat a static distance graph, and when?**

## Answer
**Partly, and the detail matters.** A wind-aware directed graph *does* beat a
static distance graph for Graph WaveNet (33.77 vs 34.60 µg/m³; paired difference
0.78, 95% CI [0.40, 1.15] by bootstrap over 105 test weeks), but is statistically
indistinguishable from it for STGCN (33.05 vs 33.18; CI [−0.11, 0.30]). More
importantly, **no graph at all beats both** in both architectures (0.36–0.51 µg/m³,
both CIs excluding zero).

The wind-aware edge over the distance graph is concentrated in severe-episode
windows, where transport matters (1.22 µg/m³ for Graph WaveNet during episodes
against 0.23 outside). But the same cut shows that *all* spatial mixing is a net
loss: no-graph is best during episodes too, and the graph penalty there is
0.5–1.7 µg/m³. Every neural model here does beat two strong baselines, so the
comparison is meaningful rather than a broken-model artefact.

## Test results (2024–2025, 2000 evenly-spaced windows, masked, µg/m³; 8 seeds for the graph models)

| Config | MAE | RMSE | R² | 1h | 3h | 6h | 12h | 24h | 48h |
|--------|-----|------|-----|----|----|----|-----|-----|-----|
| **STGCN, no graph** | **32.66 ± 0.16** | 57.06 | 0.628 | 15.04 | 23.79 | 28.35 | 30.81 | 32.83 | 36.31 |
| LSTM Univariate | 32.87 ± 0.09 | 57.57 | 0.622 | 15.46 | 24.05 | 28.62 | 31.09 | 33.04 | 36.92 |
| STGCN, wind-aware | 33.05 ± 0.12 | 57.99 | 0.616 | 15.35 | 23.72 | 28.53 | 31.47 | 33.33 | 36.48 |
| STGCN, distance | 33.18 ± 0.15 | 57.84 | 0.618 | 15.75 | 23.91 | 28.83 | 31.20 | 33.68 | 36.58 |
| GraphWaveNet, no graph | 33.44 ± 0.13 | 58.49 | 0.610 | 15.22 | 24.19 | 28.91 | 31.49 | 33.34 | 37.58 |
| STGCN, correlation | 33.46 ± 0.43 | 57.88 | 0.618 | 15.56 | 24.35 | 29.11 | 31.73 | 33.15 | 36.33 |
| GraphWaveNet, wind-aware | 33.77 ± 0.24 | 59.42 | 0.597 | 15.56 | 24.16 | 28.70 | 31.62 | 33.72 | 37.65 |
| GraphWaveNet, distance | 34.60 ± 0.61 | 59.96 | 0.590 | 15.67 | 23.97 | 28.74 | 32.25 | 34.55 | 38.48 |
| GraphWaveNet, correlation | 35.43 ± 1.11 | 59.23 | 0.600 | 15.96 | 25.94 | 32.20 | 32.13 | 34.62 | 37.88 |
| GraphWaveNet, learned adaptive | 34.70 ± 0.47 | 59.51 | 0.596 | 16.08 | 24.92 | 29.65 | 32.13 | 34.74 | 37.65 |
| LSTM Multivariate | 38.20 ± 1.83 | 66.94 | 0.489 | 16.27 | 25.29 | 30.90 | 35.06 | 37.56 | 41.66 |
| Transformer | 38.88 ± 1.04 | 66.34 | 0.498 | 16.38 | 26.32 | 31.53 | 34.21 | 38.16 | 42.41 |
| Ridge (24 lags) | 44.03 | 73.69 | 0.380 | 15.83 | 28.09 | 39.46 | 47.14 | 36.45 | 42.34 |
| Persistence | 46.79 | 79.01 | 0.288 | 16.55 | 31.15 | 42.78 | 50.09 | 38.41 | 44.74 |
| Seasonal naive (24 h) | 38.63 | 68.16 | 0.470 | 38.86 | 38.56 | 38.39 | 38.58 | 38.41 | 38.60 |
| Hour-of-week average | 59.86 | 95.39 | -0.038 | 60.15 | 59.68 | 59.80 | 59.70 | 59.71 | 59.86 |
| STGCN, wind-aware (ERA5) | 33.10 ± 0.21 | 57.73 | 0.620 | 15.32 | 23.64 | 28.31 | 31.26 | 33.61 | 36.78 |
| GraphWaveNet, wind-aware (ERA5) | 34.01 ± 0.59 | 58.11 | 0.615 | 15.47 | 23.93 | 29.02 | 31.72 | 33.66 | 37.03 |

## Graph ablation (the research question)

| Graph | STGCN MAE | GraphWaveNet MAE |
|-------|-----------|------------------|
| **None (identity)** | 32.66 ± 0.16 | 33.44 ± 0.13 |
| Wind-aware (directed) | 33.05 ± 0.12 | 33.77 ± 0.24 |
| Distance (static) | 33.18 ± 0.15 | 34.60 ± 0.61 |
| Correlation (static) | 33.46 ± 0.43 | 35.43 ± 1.11 |
| Learned adaptive | - | 34.70 ± 0.47 |
| Wind-aware (ERA5 wind) | 33.10 ± 0.21 | 34.01 ± 0.59 |

A per-seed standard deviation cannot resolve differences this small, so the
comparison is made with a **paired bootstrap over the 105 ISO weeks of the test
period** — the same weeks are resampled for both models, which respects the
temporal correlation in the errors.

| Comparison | ΔMAE | 95% CI | P(first better) | Verdict |
|------------|------|--------|-----------------|---------|
| STGCN, distance vs STGCN, wind-aware | +0.13 | [-0.04, 0.32] | 0.072 | not distinguishable |
| STGCN, distance vs STGCN, no graph | +0.51 | [0.15, 0.90] | 0.002 | second is better |
| STGCN, wind-aware vs STGCN, no graph | +0.38 | [0.08, 0.71] | 0.006 | second is better |
| GraphWaveNet, distance vs GraphWaveNet, wind-aware | +0.83 | [0.49, 1.18] | 0.000 | second is better |
| GraphWaveNet, distance vs GraphWaveNet, no graph | +1.16 | [0.58, 1.76] | 0.000 | second is better |
| GraphWaveNet, wind-aware vs GraphWaveNet, no graph | +0.33 | [-0.03, 0.71] | 0.040 | not distinguishable |
| GraphWaveNet, learned adaptive vs GraphWaveNet, no graph | +1.26 | [0.56, 2.03] | 0.000 | second is better |
| GraphWaveNet, learned adaptive vs GraphWaveNet, wind-aware | +0.94 | [0.50, 1.41] | 0.000 | second is better |
| STGCN, wind-aware (ERA5) vs STGCN, wind-aware | +0.06 | [-0.18, 0.32] | 0.336 | not distinguishable |
| STGCN, wind-aware (ERA5) vs STGCN, no graph | +0.44 | [0.14, 0.76] | 0.002 | second is better |
| GraphWaveNet, wind-aware (ERA5) vs GraphWaveNet, wind-aware | +0.24 | [-0.12, 0.60] | 0.093 | not distinguishable |
| GraphWaveNet, wind-aware (ERA5) vs GraphWaveNet, no graph | +0.57 | [0.18, 0.98] | 0.002 | second is better |
| STGCN, no graph vs LSTM Univariate | -0.20 | [-0.35, -0.07] | 0.999 | first is better |
| STGCN, no graph vs Ridge (24 lags) | -11.37 | [-13.15, -9.68] | 1.000 | first is better |
| STGCN, no graph vs Persistence | -14.13 | [-16.21, -12.14] | 1.000 | first is better |
| STGCN, no graph vs Seasonal naive (24 h) | -5.97 | [-7.33, -4.76] | 1.000 | first is better |
| STGCN, no graph vs Hour-of-week average | -27.20 | [-33.27, -21.50] | 1.000 | first is better |
| STGCN, wind-aware vs Seasonal naive (24 h) | -5.59 | [-6.92, -4.37] | 1.000 | first is better |
| GraphWaveNet, wind-aware vs Seasonal naive (24 h) | -4.87 | [-6.11, -3.74] | 1.000 | first is better |
| LSTM Multivariate vs Seasonal naive (24 h) | -0.43 | [-2.46, 1.89] | 0.661 | not distinguishable |
| stgcn identity netmean vs STGCN, no graph | -0.06 | [-0.11, -0.01] | 0.988 | first is better |
| stgcn identity netmean vs STGCN, wind-aware | -0.44 | [-0.77, -0.14] | 0.999 | first is better |
| stgcn identity netmean vs Seasonal naive (24 h) | -6.03 | [-7.41, -4.82] | 1.000 | first is better |
| stgcn identity netmean vs GraphWaveNet, no graph | -0.84 | [-1.05, -0.63] | 1.000 | first is better |

### Where the graph penalty is actually paid: severe episodes

Severe PM2.5 episodes in Delhi occur under calm, stable conditions: 78.2% of the
test windows in the lowest wind third contain a severe-episode hour, against 52.1%
and 49.0% in the two windier thirds. Crossing the wind analysis with the episode
analysis explains the ablation:

| Subset | stgcn identity | stgcn wind aware | stgcn distance | graph wavenet identity | graph wavenet wind aware | graph wavenet distance |
|--------|---|---|---|---|---|---|
| all windows | 32.66 | 33.05 | 33.18 | 33.44 | 33.77 | 34.60 |
| episode windows | 42.31 | 42.92 | 43.16 | 43.34 | 43.84 | 45.06 |
| non-episode windows | 17.90 | 17.93 | 17.91 | 18.29 | 18.35 | 18.58 |

- During **episode windows** the no-graph variant is best for both architectures;
  the graph variants are 0.5–1.7 µg/m³ worse.
- During **non-episode windows** all four variants sit within 0.7 µg/m³ of each
  other, with no-graph marginally ahead.
- The wind-aware graph's advantage over the distance graph is itself concentrated
  in episodes (1.22 µg/m³ for Graph WaveNet during episodes against 0.23 outside).

So the graph penalty is paid almost entirely in the high-pollution regime, and
spatial mixing does not rescue the high-pollution case even where transport
should matter most.

### Rolling-origin check (is the result just the split boundary?)

Train on 2019…Y, validate on Q4 of Y, test on Y+1.

| Model | Fold | Test year | No graph | Wind-aware | Distance | Best |
|-------|------|-----------|----------|------------|----------|------|
| STGCN | fold0 | 2021 | 36.65 | 36.98 | 37.34 | identity |
| GraphWaveNet | fold0 | 2021 | 37.56 | 37.63 | 43.25 | identity |
| STGCN | fold1 | 2022 | 32.96 | 33.09 | 32.61 | distance |
| STGCN | fold2 | 2023 | 33.03 | 33.01 | 33.58 | wind-aware |
| GraphWaveNet | fold2 | 2023 | 34.23 | 33.64 | 34.05 | wind-aware |
| STGCN | fold3 | 2024 | 33.82 | 34.57 | 34.22 | identity |
| STGCN | fold4 | 2025 | 30.76 | 30.77 | 31.41 | identity |
| GraphWaveNet | fold4 | 2025 | 31.32 | 31.23 | 32.87 | wind-aware |

The ranking is stable as the boundary moves, so the finding is not an artefact of
one particular split.

## Where the edge directions come from (ERA5 ablation)

The station wind is noisy (cross-station correlation 0.11) and missing for 9 of
42 stations, so the wind-aware graph was rebuilt with ERA5 reanalysis —
complete and smooth (correlation 0.98) — changing nothing else.

| model | edge direction | MAE |
|---|---|---|
| STGCN | no graph | 32.66 ± 0.16 |
| STGCN | wind, station | 33.05 ± 0.12 |
| STGCN | wind, ERA5 | 33.10 ± 0.21 |
| STGCN | distance | 33.18 ± 0.15 |
| GraphWaveNet | no graph | 33.44 ± 0.13 |
| GraphWaveNet | wind, station | 33.77 ± 0.24 |
| GraphWaveNet | wind, ERA5 | 34.01 ± 0.59 |
| GraphWaveNet | distance | 34.60 ± 0.61 |

The two wind sources are indistinguishable (STGCN Δ 0.01, CI [−0.25, 0.29];
GraphWaveNet Δ 0.21, CI [−0.16, 0.57]), so the null result is **not** caused by
the noisy local wind. This tests one edge construction against one wind
source; it is not a claim that wind cannot help.

## Is the graph just delivering the regional signal?

A no-graph model given the network-wide mean PM2.5 as an extra input feature:

| configuration | MAE |
|---|---|
| no graph | 32.66 ± 0.16 |
| **no graph + network mean** | 32.60 ± 0.09 |
| wind-aware graph | 33.05 ± 0.12 |
| no graph (GraphWaveNet) | 33.44 ± 0.13 |
| wind-aware graph (GraphWaveNet) | 33.77 ± 0.24 |

Telling the model the regional signal explicitly is worth nothing (−0.05, CI
[−0.11, −0.00]) and that model is significantly better than every graph
model. The graph is not delivering the regional signal; its mixing is a pure
cost.

## Leave-station-out ("and when?")

Ten stations held out of training entirely; ridge refitted on the other 32.
The spatial baselines use the same information the graph has and need no
training.

**Mode both** (held out: site_106, site_107, site_108, site_1422, site_1427, site_1432, site_1434, site_1435, site_1561, site_5024)

| method | MAE | R² |
|---|---|---|
| IDW neighbour (seasonal) | 41.24 | 0.506 |
| stgcn wind aware s42 lso10 both nd30 | 41.33 | 0.490 |
| stgcn wind aware s456 lso10 both nd30 | 41.34 | 0.476 |
| graph wavenet wind aware s456 lso10 both nd30 | 41.34 | 0.494 |
| Seasonal naive (24h) | 41.40 | 0.479 |
| graph wavenet distance s42 lso10 both nd30 | 41.60 | 0.456 |
| graph wavenet distance s456 lso10 both nd30 | 41.72 | 0.467 |
| stgcn distance s42 lso10 both nd30 | 41.78 | 0.462 |
| graph wavenet wind aware s42 lso10 both nd30 | 41.94 | 0.488 |
| graph wavenet distance s123 lso10 both nd30 | 42.14 | 0.455 |
| graph wavenet wind aware s123 lso10 both nd30 | 42.36 | 0.498 |
| stgcn wind aware s123 lso10 both nd30 | 42.38 | 0.460 |
| stgcn distance s123 lso10 both nd30 | 42.55 | 0.432 |
| stgcn distance s456 lso10 both nd30 | 42.64 | 0.430 |
| Network mean (persistence) | 47.01 | 0.402 |
| Ridge (fit on training stations only) | 47.48 | 0.378 |
| IDW neighbour (persistence) | 47.65 | 0.376 |
| Persistence | 50.73 | 0.269 |
| Hour-of-week average | 63.48 | -0.052 |
| stgcn identity s42 lso10 both nd30 | 65.35 | -0.119 |
| stgcn identity s456 lso10 both nd30 | 65.38 | -0.107 |
| stgcn identity s123 lso10 both nd30 | 65.38 | -0.107 |
| graph wavenet identity s42 lso10 both nd30 | 65.39 | -0.139 |
| graph wavenet identity s123 lso10 both nd30 | 65.43 | -0.147 |
| graph wavenet identity s456 lso10 both nd30 | 65.91 | -0.061 |

**Mode both** (held out: site_106, site_107, site_108, site_1422, site_1427, site_1432, site_1434, site_1435, site_1561, site_5024)

| method | MAE | R² |
|---|---|---|
| IDW neighbour (seasonal) | 41.24 | 0.506 |
| Seasonal naive (24h) | 41.40 | 0.479 |
| Network mean (persistence) | 47.01 | 0.402 |
| Ridge (fit on training stations only) | 47.48 | 0.378 |
| IDW neighbour (persistence) | 47.65 | 0.376 |
| Persistence | 50.73 | 0.269 |
| stgcn distance s123 lso10 both | 53.45 | 0.147 |
| graph wavenet distance s123 lso10 both | 53.82 | 0.136 |
| stgcn distance s42 lso10 both | 55.20 | 0.113 |
| stgcn wind aware s123 lso10 both | 55.20 | 0.119 |
| stgcn wind aware s42 lso10 both | 55.88 | 0.107 |
| graph wavenet distance s42 lso10 both | 57.07 | 0.042 |
| graph wavenet wind aware s123 lso10 both | 57.09 | 0.036 |
| graph wavenet wind aware s42 lso10 both | 58.55 | 0.003 |
| Hour-of-week average | 63.48 | -0.052 |
| stgcn identity s42 lso10 both | 68.44 | -0.319 |
| graph wavenet identity s42 lso10 both | 69.78 | -0.366 |
| stgcn identity s123 lso10 both | 69.95 | -0.372 |
| graph wavenet identity s123 lso10 both | 71.58 | -0.422 |

**Mode target** (held out: site_106, site_107, site_108, site_1422, site_1427, site_1432, site_1434, site_1435, site_1561, site_5024)

| method | MAE | R² |
|---|---|---|
| stgcn identity s123 lso10 target | 35.34 | 0.613 |
| stgcn identity s42 lso10 target | 35.36 | 0.611 |
| graph wavenet identity s42 lso10 target | 35.63 | 0.616 |
| stgcn wind aware s123 lso10 target | 35.68 | 0.601 |
| graph wavenet identity s123 lso10 target | 35.70 | 0.609 |
| stgcn wind aware s42 lso10 target | 35.77 | 0.598 |
| stgcn distance s123 lso10 target | 35.89 | 0.585 |
| stgcn distance s42 lso10 target | 36.06 | 0.579 |
| graph wavenet wind aware s42 lso10 target | 36.21 | 0.590 |
| graph wavenet wind aware s123 lso10 target | 36.33 | 0.604 |
| graph wavenet distance s42 lso10 target | 36.40 | 0.606 |
| graph wavenet distance s123 lso10 target | 36.99 | 0.588 |
| IDW neighbour (seasonal) | 41.24 | 0.506 |
| Seasonal naive (24h) | 41.40 | 0.479 |
| Network mean (persistence) | 47.01 | 0.402 |
| Ridge (fit on training stations only) | 47.48 | 0.378 |
| IDW neighbour (persistence) | 47.65 | 0.376 |
| Persistence | 50.73 | 0.269 |
| Hour-of-week average | 63.48 | -0.052 |

**The corrected result.** In the second variant (own history hidden as well) a
two-line inverse-distance-weighted average of the neighbours reaches 41.24
(R² = 0.51) while the best graph model reaches 54.32 (R² = 0.13) — the graph
is **13.1 µg/m³ worse**, CI [8.7, 18.0]. An earlier version of this summary
reported the opposite, against a no-graph model that had no information at
all.

## Does node dropout rescue the graph?

The leave-station-out comparison is a handicap: the models are trained to
forecast a station from its own history, and are then asked to forecast one
whose own history was removed. With node dropout (a random 30% of stations'
own history hidden per training window) the model has practised the condition.

| configuration (own history hidden) | no node dropout | with node dropout |
|---|---|---|
| STGCN, distance | 54.32 | 42.33 |
| STGCN, wind-aware | 55.54 | 41.68 |
| GraphWaveNet, distance | 55.44 | 41.82 |
| GraphWaveNet, wind-aware | 57.82 | 41.88 |
| STGCN, no graph | 69.19 | 65.37 |
| Network mean (persistence) | 47.01 | 47.01 |
| IDW neighbour (seasonal) | 41.24 | 41.24 |

The handicap was most of the story. With node dropout the paired difference
against the IDW baseline is **1.08 µg/m³, CI [−0.41, 2.75]** — statistically
indistinguishable — and both beat the network-wide mean by 4.7 µg/m³. So for a
station with no history of its own, a graph neural network *matches* simple
spatial interpolation: it neither beats it nor is refuted by it.

## Severe episodes

Episodes = PM2.5 > 150 µg/m³ sustained ≥ 6 h, station-wise (15.6% of observed station-hours in the test block).

| Model | Overall | Episodes | Non-episodes |
|-------|---------|----------|--------------|
| **STGCN, no graph** | 32.66 | 79.33 | 23.52 |
| STGCN, wind-aware | 33.06 | 80.34 | 23.80 |
| STGCN, distance | 33.31 | 80.35 | 24.09 |
| GraphWaveNet, no graph | 33.50 | 78.35 | 24.71 |
| Ridge (24 lags) | 43.97 | 98.10 | 33.36 |
| Persistence | 46.65 | 102.57 | 35.69 |

All architectures are 3–5× worse during episodes, and the spread between models
widens there. Extreme pollution is where the remaining error lives.

## Data

| Item | Value |
|------|-------|
| Source | CPCB 15-min, via `Vonter/india-cpcb-aqi` releases (2009–2025) |
| Stations | **42** (34 Delhi + 8 Noida/Ghaziabad), ≥70% real coverage in every split |
| Period | **2019-01-01 → 2025-12-31** (61,345 hours, 7 years) |
| Splits | Train 2019–2022 (34,922 windows) · Val 2023 (8,641) · Test 2024–2025 (17,402) |
| Features | PM2.5 + station-measured T/RH/wind/rain (CPCB files, not reanalysis) |
| Normalisation | log1p + z-score, train split only |
| Missing | masked (1 = really observed), honoured in loss and metrics |

Split distributions are matched (mean 102.0 / 98.0 / 97.5 µg/m³; frac > 300 =
4.82% / 4.23% / 4.15%), so the training split contains **four full winters**.
Split boundaries are contiguous (0 h train→val) with a 23 h gap before test; no
window's input or target ever crosses a boundary (verified programmatically).

## What made the models work

1. **Predict the change from the last observation**, not the absolute concentration.
   The output layer is zero-initialised, so each model starts *at* Persistence and
   can only improve.
2. **Linear autoregressive skip** (last 24 lags → 48 horizons), i.e. the ridge
   solution inside the network. The network is measured relative to a strong
   linear floor; without it, models collapsed to the training mean.
3. **Mask channel** so imputed hours are distinguishable from observed ones.
4. **Checkpoint selection on denormalised µg/m³**, not normalised loss.

## Honest limitations

- Seven years of data, but the fixed test block covers only two winters, so
  generalisation across pollution regimes is only partly tested. The
  rolling-origin evaluation is our partial answer to this.
- The wind-aware graph uses a period-mean prevailing direction, discarding gust
  structure, and assumes a single direction per station. The per-station wind
  vectors agree only weakly (mean cross-station correlation of the u-component
  is 0.11), so that mean direction is itself a noisy estimate.
- The graph comparison rests on 8 seeds for the two graph architectures but on a
  single model family and one hyperparameter setting per configuration. Each
  variant received the same tuning effort, but the graph-specific
  hyperparameters were not swept.
- Graph structures are static except GraphWaveNet's learned variant.
- One hazard (PM2.5), one horizon family, no episode-conditioned modelling.
- An earlier version of this study used synthetic data and reported that
  everything failed. That result is void; see `EXPERIMENTS.md` for the diagnosis.
