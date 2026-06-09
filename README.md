# Florida Hurricane Catastrophe Model

**A location-level, stochastic Monte Carlo catastrophe model for a Florida coastal homeowners portfolio — from hazard simulation through vulnerability, financial terms, and a multi-layer reinsurance programme.**

Built on a compound-Poisson framework and run over **100,000 simulated years**, the model produces full exceedance-probability (EP) curves and prices the impact of an excess-of-loss (XoL) reinsurance tower on the portfolio's tail risk.

\---

## Headline results

A synthetic but realistic book: **1,000 coastal locations, USD 500M total insured value (TIV)**.

|Metric (USD M)|AEP Gross|AEP Net|OEP Gross|OEP Net|
|-|-:|-:|-:|-:|
|Average Annual Loss|10.67|9.37|9.57|8.30|
|PML 1-in-100|118.1|75.9|107.7|60.0|
|PML 1-in-250|149.4|94.3|133.7|60.0|

*AEP = Aggregate Exceedance Probability (full annual loss). OEP = Occurrence Exceedance Probability (largest single event in the year). Gross = before reinsurance; Net = after the XoL programme.*

**What the reinsurance tower buys:** the per-occurrence XoL programme cuts the 1-in-250 single-event loss by **55%** (from 133.7M to the 60M retention) and the 1-in-100 by **44%**. On an aggregate-annual basis the reduction is more modest (≈37%) — a direct, expected consequence of buying *per-occurrence* protection rather than aggregate stop-loss.

\---

## Why this model exists

A hurricane doesn't strike insured homes at random. It makes landfall on a stretch of coast and sweeps a coherent swath inland, so a single storm accumulates loss across many neighbouring policies at once. Pricing that concentration is the entire reason catastrophe models work at the **location level** rather than treating a portfolio as one aggregate number.

This project rebuilds a Florida hurricane book from individual exposures up: each of the 1,000 locations has its own coordinates, value, construction type, and policy terms, and each simulated storm produces a wind footprint that those locations experience together. The output is a defensible view of the loss distribution — the kind of curve a primary insurer uses to decide how much reinsurance to buy, and a reinsurer uses to price it.

> This is \*\*version 2\*\*. Version 1 modelled the same portfolio at the aggregate level with a single lognormal severity distribution. v2 replaces that with a spatial, peril-driven engine — and, reassuringly, the two arrive at nearly the same Average Annual Loss (10.7M vs 10.5M) by completely independent paths. That cross-check is one of the model's strongest validation anchors.

\---

## How it works — the six-step pipeline

Each step is an independent, validated module. Run them in order (or all at once via `run\_all.py`).

### 1 · Exposure — *the book of business*

`data/generate\_exposure.py` → `data/exposure.csv`

1,000 synthetic coastal homeowner locations across eight hurricane-exposed Florida counties, weighted toward the South-East metro (Miami-Dade, Broward, Palm Beach) where real exposure concentrates. Each location carries TIV (lognormal, scaled so the book sums to exactly **USD 500M**), construction class, occupancy, and Florida-style hurricane deductibles (2% / 5% / 10% of TIV).

*Validation:* TIV sums to 500M, unique IDs, no nulls, and a strict construction↔occupancy consistency check (every Mobile Home is Manufactured, and vice-versa).

### 2 · Hazard — *the storms*

`model/hazard.py` · `model/hazard\_diagnostics.py`

A stochastic event generator with **moving tracks**. Storm frequency follows a Poisson process (λ = 0.7 storms/year); intensity is drawn from a Saffir-Simpson category distribution calibrated to the shape of Florida landfalls. Each storm makes landfall along a coastline polyline (Atlantic, Keys, Gulf), then propagates inland and weakens. A modified Rankine vortex wind field gives every location the **maximum sustained wind** it experienced during the storm's passage.

*Validation (10,000-storm diagnostics):* 35.8% of landfalls fall in the South-East corridor (matching the sampling weights); Broward and Palm Beach receive wind on par with Miami-Dade (no down-coast bias); **75.5% of storms strike two or more counties**, confirming the spatial correlation that justifies the location-level approach.

### 3 · Vulnerability — *wind to damage*

`model/vulnerability.py`

HAZUS-anchored damage functions that map wind to a damage ratio (0–1), **differentiated by construction type**. Curves operate in 3-second peak gust (HAZUS's native unit), with sustained wind converted via a 1.3 gust factor (open-terrain, Exposure C). Damage caps reflect real structural behaviour — Manufactured homes reach total loss, while Reinforced Concrete saturates near 0.75 even in extreme wind.

*Validation:* curves are monotonic, bounded by their caps, and the fragility hierarchy (Manufactured > Wood Frame > Masonry > Reinforced Concrete) holds at every wind speed.

### 4 · Loss integration — *the three-level hierarchy*

`model/loss.py`

The full 100,000-year catalogue (69,483 events), vectorised over locations. For every location in every event:

```
ground\_up = damage\_ratio(gust, construction) × TIV
gross     = clip(ground\_up − deductible, 0, policy\_limit)
```

Deductibles apply **per occurrence** — each hurricane re-triggers them — and policy terms are applied per-location *before* aggregating.

*Results:* AAL ground-up **15.1M** (3.02% of TIV); AAL gross **10.7M** (2.13%); deductibles absorb **29.5%** of ground-up loss; **51.0%** of years are loss-free (≥ the e<sup>−λ</sup> = 49.66% Poisson floor). The fragility signal comes through cleanly: Manufactured homes hold 8.7% of TIV but contribute **32.4%** of AAL (3.7×), while Reinforced Concrete holds 15.2% of TIV and just 3.5% of loss (0.2×).

### 5 · Reinsurance — *the XoL tower*

`model/reinsurance.py` · `model/ep\_utils.py`

A three-layer excess-of-loss programme applied per occurrence, completing the hierarchy with `net = gross − recovery`:

|Layer|Structure|Covers per-event loss|Triggered in|
|-|-|-|-:|
|Working|40M xs 60M|60M – 100M|3.9% of years|
|Middle|50M xs 100M|100M – 150M|1.3% of years|
|Cat high|50M xs 150M|150M – 200M|0.2% of years|

The contiguous tower provides 140M of capacity above a 60M retention. Expected annual reinsurance recovery is **1.29M/year** — the technical floor of the programme's premium.

*Validation:* recovery is zero below the 60M attachment, capped at full capacity above 200M exhaustion, and net never exceeds gross. PML calculation lives in a single shared module (`ep\_utils.py`) so every figure in the project comes from one source of truth.

### 6 · Outputs — *the story*

`run\_all.py` · `model/summary.py` · `outputs/ep\_master.png`

A one-command orchestrator reproduces the entire pipeline from scratch; `summary.py` produces the headline metrics table; and the master EP plot tells the risk-transfer story in a single figure.

\---

## Key visualizations

|||
|-|-|
|`outputs/ep\_master.png`|**The headline.** AEP and OEP curves, gross vs net of reinsurance, with the recovery band and PML callouts.|
|`outputs/ep\_gross\_vs\_net.png`|Per-occurrence EP curve showing the XoL tower flattening the tail at the 60M retention.|
|`outputs/hazard\_footprint.png`|A single Cat-4 storm's wind field over the portfolio — a coherent regional strike, not scattered noise.|
|`outputs/vulnerability\_curves.png`|The four construction-type damage curves with Saffir-Simpson thresholds.|
|`outputs/landfall\_distribution.png`|10,000 sampled landfalls along the Florida coastline.|
|`outputs/counties\_hit\_per\_event.png`|Multi-county accumulation per storm — the spatial-correlation evidence.|

\---

## Why you can trust the numbers — validation anchors

Every step is checked against an independent reference, not just "does it run":

* **AAL convergence:** v2's spatial engine lands within 2% of v1's aggregate lognormal — independent mechanisms, same answer.
* **Poisson floor:** the share of loss-free years sits at or above e<sup>−λ</sup>, the analytic probability of zero storms.
* **AEP ≥ OEP** at every return period (an annual aggregate must be at least its largest single event).
* **Fragility hierarchy** reproduced from physics: the most vulnerable construction contributes loss far in excess of its share of value.
* **Spatial correlation** confirmed empirically: most storms accumulate across multiple stacked counties.
* **Reproducibility:** every result is seeded (`seed = 42`) and regenerable end-to-end with one command.

\---

## Methodology, assumptions \& limitations

Built to mirror industry practice; **parameters are illustrative, not calibrated to proprietary data.** Stated plainly so results are read in context:

* **Synthetic exposure.** The portfolio is generated, not sourced from real policy data.
* **HAZUS-anchored, not exact.** Vulnerability curves reproduce the *behaviour* and hierarchy of HAZUS, not FEMA's exact coefficients.
* **Single-terrain gust factor.** A constant 1.3 sustained-to-gust conversion (open terrain); marine or built-up exposure would differ.
* **Simplified track physics.** A parametric vortex with generic inland decay; no asymmetry, forward-speed effects, or east/west-coast distinction.
* **Wind only.** No storm surge, inland flood, or demand-surge sub-perils.
* **XoL without reinstatements or co-participation.** The net loss is flat at the retention by design — a real programme would typically include both.

These are deliberate scope choices for a transparent, auditable model — and each is a natural extension.

\---

## Future extensions

* A real stochastic event catalogue calibrated to NOAA HURDAT2.
* Secondary perils (storm surge, inland flood) and correlated demand surge.
* Reinstatements and co-participation in the reinsurance structure.
* Sensitivity and uncertainty analysis around key parameters.
* Climate-conditioned frequency/intensity scenarios.

\---

## Run it

```bash
# full pipeline from scratch (\~1 min)
python run\_all.py

# quick smoke run (1,000 years) to verify the pipeline end-to-end
python run\_all.py --quick

# or run any step on its own
python data/generate\_exposure.py
python model/hazard.py
python model/vulnerability.py
python model/loss.py
python model/reinsurance.py
python model/summary.py
```

**Stack:** Python · NumPy · pandas · Matplotlib. No external data dependencies — everything is generated and reproducible from seed.

\---

## Project structure

```
hurricane-cat-model/
├── data/
│   ├── generate\_exposure.py     # step 1 — synthetic FL exposure
│   └── exposure.csv
├── model/
│   ├── hazard.py                # step 2 — stochastic moving-track storms
│   ├── hazard\_diagnostics.py
│   ├── vulnerability.py         # step 3 — HAZUS-anchored damage curves
│   ├── loss.py                  # step 4 — ground-up / gross loss engine
│   ├── reinsurance.py           # step 5 — multi-layer XoL tower
│   ├── ep\_utils.py              # shared PML / EP-curve logic
│   └── summary.py               # step 6 — headline metrics
├── outputs/                     # plots
├── run\_all.py                   # orchestrator
└── results/                     # generated CSVs (gitignored, reproducible)
```

\---

*Built by Emiliano Gaston Lopez · www.linkedin.com/in/emiliano-gastón-lópez-b278753a1 ·  A learning project demonstrating end-to-end catastrophe-model construction; not for production pricing or risk-transfer decisions.*

