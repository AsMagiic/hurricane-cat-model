# Florida Hurricane Catastrophe Model

**A location-level, stochastic Monte Carlo catastrophe model for a Florida coastal homeowners portfolio — from individual exposures through hazard, vulnerability, policy terms, and a multi-layer reinsurance programme, producing gross and net exceedance-probability curves.**

The model builds a book of 1,000 insured locations, simulates 100,000 years of stochastic hurricanes as moving wind footprints, translates wind to damage through construction-specific vulnerability curves, applies per-policy financial terms, and prices the effect of an excess-of-loss (XoL) reinsurance tower on the portfolio's tail — reporting the loss distribution **gross and net of reinsurance**.

> **Note on scope.** A learning / portfolio project built to demonstrate the conceptual engine behind production catastrophe models (RMS, Verisk Touchstone, CoreLogic). The *method* follow the conceptual structure of a vendor-model pipeline — exposure, hazard, vulnerability, financial, and reinsurance — not just aggregate loss simulation. The *input parameters* are illustrative, not calibrated to proprietary data. See [Assumptions](#parameters-and-assumptions) and [Limitations](#limitations).

---

## The pipeline at a glance

```
1. Exposure        generate_exposure.py   1,000 FL coastal locations, USD 500M TIV
2. Hazard          hazard.py              stochastic moving-track storms -> wind per location
3. Vulnerability   vulnerability.py       HAZUS-anchored damage curves by construction
4. Financial       loss.py                ground-up -> gross (per-occurrence deductibles)
5. Reinsurance     reinsurance.py         per-occurrence XoL tower -> net
6. EP metrics      summary.py             AEP & OEP, gross & net, PMLs
                   run_all.py             runs the whole chain end-to-end
```

---

## Headline results

A synthetic but plausible book: **1,000 coastal locations, USD 500M total insured value (TIV)**, over 100,000 simulated years.

| Metric (USD M) | AEP Gross | AEP Net | OEP Gross | OEP Net |
|---|---:|---:|---:|---:|
| Average Annual Loss | 10.67 | 9.37 | 9.57 | 8.30 |
| PML 1-in-100 | 118.1 | 75.9 | 107.7 | 60.0 |
| PML 1-in-250 | 149.4 | 94.3 | 133.7 | 60.0 |

*AEP = Aggregate Exceedance Probability (full annual loss). OEP = Occurrence Exceedance Probability (largest single event of the year). Gross = before reinsurance; Net = after the XoL tower.*

**What the reinsurance tower buys:** gross-to-net PML reduction of **-44.3% / -55.1%** on a per-occurrence basis (1-in-100 / 1-in-250), and **-35.8% / -36.9%** on an aggregate-annual basis.

A detail worth reading off the numbers: the **OEP net flattens at exactly USD 60M at both return periods**. Because the contiguous tower covers every dollar from the 60M retention up to 200M exhaustion, any single event that pierces the programme leaves the insurer with exactly its 60M retention — a clean illustration of how the tower caps per-occurrence net loss. The **AEP net does *not* flatten**, because a bad year can stack several retentions from multiple events, none of which individually triggers the full tower.

---

## The problem

An insurer can comfortably pay for the *average* year; it is bankrupted by the *bad* year. For a hurricane book, average annual losses are modest, but there is a real chance of a single catastrophic season costing an order of magnitude more. Pricing that risk — and deciding how much loss to retain versus transfer to reinsurers — requires the full loss distribution, not just its mean, and it requires that distribution **both gross and net of reinsurance**. This model produces both, and explicitly prices the retention decision the XoL tower represents.

---

## Methodology

The model executes a six-step pipeline (orchestrated by `run_all.py`), each step an independent, validated module:

**1 - Exposure** — `data/generate_exposure.py` builds 1,000 synthetic coastal homeowner locations across eight hurricane-exposed Florida counties, weighted toward the South-East metro where real exposure concentrates. Each location carries coordinates, TIV, construction class, occupancy, and a Florida-style hurricane deductible tier.

**2 - Hazard** — `model/hazard.py` generates a stochastic catalogue of **moving-track** storms. Frequency is Poisson; intensity follows a Saffir-Simpson distribution; each storm makes landfall along a coastline polyline and propagates inland under a modified Rankine vortex wind field, giving every location the maximum sustained wind it experienced during the storm's passage.

**3 - Vulnerability** — `model/vulnerability.py` maps wind to a damage ratio through HAZUS-anchored curves, **differentiated by construction type**, operating in 3-second peak gust (with a sustained-to-gust conversion).

**4 - Financial** — `model/loss.py` runs the full 100,000-year catalogue and applies the three-level loss hierarchy: `ground_up = damage_ratio x TIV`, then `gross = clip(ground_up - deductible, 0, policy_limit)`, with deductibles applied **per location, per occurrence, before aggregation**.

**5 - Reinsurance** — `model/reinsurance.py` applies a per-occurrence XoL tower to each event's portfolio loss, completing the hierarchy with `net = gross - recovery`.

**6 - EP metrics** — `model/summary.py` reads the loss tables and produces the AEP and OEP curves and PMLs, gross and net, all through a single shared PML routine (`model/ep_utils.py`) so every figure traces to one source of truth.

### Reference parametric model (v1)

The repository also retains the original **portfolio-aggregate** model — `model/aggregate_loss.py`, `model/ep_curve.py`, `model/sanity_check.py` — which modelled the same portfolio with a single compound-Poisson process: `N ~ Poisson(lambda)` events per year, each drawing a portfolio loss from a Lognormal severity. This is **not** the primary model; it is kept as an independent validation anchor. Reassuringly, the v2 spatial engine reproduces v1's Average Annual Loss to within ~2% (10.7M vs 10.5M) by an entirely different mechanism — strong evidence the location-level chain is assembled correctly.

---

## Parameters and assumptions

All parameters are **illustrative** — plausible and chosen to make the dynamics visible, not estimated from data.

**Exposure**

| Parameter | Value |
|---|---|
| Locations | 1,000 |
| Total insured value | USD 500,000,000 (book sums exactly) |
| Counties | 8 FL coastal, weighted to the South-East (Miami-Dade, Broward, Palm Beach heaviest) |
| Construction mix | Masonry 509 - Wood Frame 240 - Reinforced Concrete 172 - Manufactured 79 |
| Hurricane deductibles | 2% / 5% / 10% of TIV (per occurrence) |

**Hazard**

| Parameter | Value |
|---|---|
| Frequency | `N ~ Poisson(lambda = 0.7)` storms/year |
| Intensity | Saffir-Simpson category weights (Cat1 0.40 -> Cat5 0.04) |
| Landfall | sampled along a FL coastline polyline, SE/SW segment weighting |
| Radius of max winds | 30-55 km |
| Track heading | +/-45 deg of north, inland filling with ~120 km e-folding decay |
| Wind field | modified Rankine vortex; per-event max sustained wind per location |

**Vulnerability** (3-second gust, midpoint = gust at 50% damage)

| Construction | Midpoint (gust mph) | Damage cap |
|---|---:|---:|
| Manufactured | 110 | 1.00 |
| Wood Frame | 145 | 1.00 |
| Masonry | 165 | 0.90 |
| Reinforced Concrete | 185 | 0.75 |

Gust factor 1.3 (open terrain, Exposure C); damage forced to zero below a 65 mph gust threshold.

**Reinsurance** — per-occurrence XoL tower

| Layer | Structure | Covers per-event loss |
|---|---|---|
| Working | 40M xs 60M | 60M - 100M |
| Middle | 50M xs 100M | 100M - 150M |
| Cat high | 50M xs 150M | 150M - 200M |

60M retention, 140M total capacity, 200M exhaustion.

<details>
<summary>v1 reference: Lognormal severity parameterization</summary>

The v1 model parameterized its Lognormal severity from an arithmetic mean (USD 15M) and coefficient of variation (1.5):

```
sigma = sqrt( ln(1 + CV^2) ) = sqrt(ln(3.25)) ~ 1.0857
mu    = ln(mean) - sigma^2/2 = ln(15e6) - 0.5894 ~ 15.9342
```

The `-sigma^2/2` correction offsets the upward pull of the tail so the arithmetic mean lands exactly on USD 15M. Omitting it silently inflates the mean by orders of magnitude — a classic modeling error the AAL validation catches.
</details>

---

## Results

The headline table above is the model's output. Three things the numbers tell:

**The value of reinsurance.** The gap between the gross and net curves is what the XoL programme buys. On a per-occurrence basis it removes 48M from the 1-in-100 single-event loss (107.7M -> 60.0M) and 73.7M from the 1-in-250 (133.7M -> 60.0M). The expected annual recovery — the technical floor of the programme's premium — is about USD 1.29M/year, concentrated in the Working layer (which triggers in ~3.9% of years) and tapering to the Cat-high layer (~0.2% of years).

**Vulnerability drives loss concentration.** Average Annual Loss by construction departs sharply from the share of value: Manufactured homes hold 8.7% of TIV but contribute **32.4%** of AAL (3.7x), while Reinforced Concrete holds 15.2% of TIV and just 3.5% of loss (0.2x). This is the construction-differentiated vulnerability working as intended.

**Geography drives the tail.** Because a single storm sweeps a coherent swath of coast, losses accumulate across the stacked South-East counties (Miami-Dade, Broward, Palm Beach) — the spatial correlation that fattens the tail and justifies modelling at the location level rather than in aggregate.

Generated figures (in `outputs/`):

| File | Shows |
|---|---|
| `ep_master.png` | AEP and OEP curves, gross vs net, with PML callouts — the headline |
| `ep_gross_vs_net.png` | Per-occurrence EP curve, tower flattening the net at the 60M retention |
| `hazard_footprint.png` | A single storm's wind field over the portfolio |
| `vulnerability_curves.png` | The four construction-type damage curves |
| `landfall_distribution.png` | 10,000 sampled landfalls along the coastline |
| `counties_hit_per_event.png` | Multi-county accumulation per storm |

---

## Validation

Each module is checked against independent references, not just "does it run":

- **Hazard** — eye-of-calm at the storm centre, wind peaks at the radius of max winds, monotonic decay outward; mean storms/year converges to lambda; Cat-3-plus share ~ 0.35; high-wind locations form coherent geographic clusters (10,000-storm diagnostics: 35.8% of landfalls in the SE corridor, 75.5% of storms strike 2+ counties).
- **Vulnerability** — curves monotonic, bounded by their caps, fragility hierarchy preserved at every wind speed, damage zero below threshold, gust conversion sanity-checked.
- **Financial loss** — gross <= ground-up everywhere; the annual table holds exactly N years (including loss-free years); each year's max single event <= its aggregate; share of loss-free years >= the e^(-lambda) = 49.66% Poisson floor (51.0% simulated).
- **Reinsurance** — net <= gross always; recovery bounded in [0, tower capacity]; zero recovery below the first attachment; full recovery at exhaustion.

The **v1 reference model** carries its own three classical anchors (frequency -> lambda, AAL -> lambda*E[X], zero-loss years -> e^(-lambda)), now serving as the cross-check on the aggregate baseline.

---

## Limitations

Stated plainly so results are read in context. Several limitations of the v1 aggregate model are *resolved* in v2 (location-level resolution, modelled deductibles, modelled reinsurance, multi-county correlation). What remains:

- **Single lambda, no clustering.** Storm frequency has no over-dispersion or seasonal clustering, which understates the aggregate (AEP) tail.
- **No secondary uncertainty.** Damage ratios are deterministic given wind and construction — no variance around the mean curve.
- **Single peril.** Wind only; no storm surge, inland flood, or demand surge.
- **Simplified track physics.** Generic +/-45 deg heading, constant Rmax, no forward-speed asymmetry.
- **Uncalibrated parameters.** All inputs are synthetic and illustrative, not fit to data.
- **Reinsurance structure.** No reinstatements, no co-participation, no quota share — the net loss is flat at the retention by design.

Each is a natural extension rather than a flaw.

---

## Future extensions

- A stochastic event catalogue with over-dispersed frequency, calibrated to NOAA HURDAT2.
- Generalized Pareto (peaks-over-threshold) tail modelling.
- Secondary uncertainty (variance around vulnerability curves).
- Multi-peril (storm surge, inland flood, demand surge).
- Reinstatements and quota-share in the reinsurance structure.
- Reproduction in the open-source [Oasis LMF](https://oasislmf.org) framework.

---

## Run it

```bash
pip install numpy pandas matplotlib

# full pipeline from scratch (~1 min)
python run_all.py

# quick smoke run (1,000 years) to verify the pipeline end-to-end
python run_all.py --quick
```

Dependency order is `loss.py` -> `reinsurance.py` -> `summary.py` (the orchestrator handles it). Each module can also be run on its own and prints its own demo plus validation asserts. A fixed seed (`42`) makes every result reproducible.

**Stack:** Python - NumPy - pandas - Matplotlib. No external data dependencies — everything is generated from seed.

---

## Repository structure

```
hurricane-cat-model/
|-- run_all.py                   # orchestrates the full pipeline
|-- data/
|   |-- generate_exposure.py     # step 1 - synthetic FL exposure
|   `-- exposure.csv
|-- model/
|   |-- hazard.py                # step 2 - stochastic moving-track storms
|   |-- hazard_diagnostics.py    #          spatial validation
|   |-- vulnerability.py         # step 3 - HAZUS-anchored damage curves
|   |-- loss.py                  # step 4 - ground-up / gross loss engine
|   |-- reinsurance.py           # step 5 - multi-layer XoL tower -> net
|   |-- ep_utils.py              #          shared PML / EP-curve logic
|   |-- summary.py               # step 6 - headline metrics
|   |-- aggregate_loss.py        # v1 reference - compound-Poisson aggregate
|   |-- ep_curve.py              # v1 reference - AEP/OEP curves
|   `-- sanity_check.py          # v1 reference - Poisson validation
|-- outputs/                     # generated plots (7 PNGs)
`-- results/                     # generated CSVs (gitignored, reproducible)
    `-- summary_metrics.csv      # headline metrics table (versioned)
```

---

*Built by Emiliano Gaston Lopez - www.linkedin.com/in/emiliano-gaston-lopez-actuarial - A learning project demonstrating end-to-end catastrophe-model construction; not for production pricing or risk-transfer decisions.*
