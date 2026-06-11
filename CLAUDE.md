# Hurricane Cat Model — Florida Homeowners (v3)

## Status
v2.0 is frozen at tag `v2.0` (location-level model, fully functional).
This branch targets v3 — do not break v2 behaviour without a clear decision log entry.

## v2 baseline (frozen)
- 1,000 synthetic FL coastal locations | TIV USD 500M
- Hazard: stochastic moving-track, illustrative parameters (lambda=0.7/yr)
- Vulnerability: HAZUS-anchored logistic damage curves, 4 construction types
- Finance: three-layer per-occurrence XoL tower, 140M capacity, retention 60M, exhaustion 200M
  (Layer 1: 40M xs 60M / Layer 2: 50M xs 100M / Layer 3: 50M xs 150M);
  attachments illustrative, to be re-anchored to OEP return periods in Phase 4
- Catalog: 100,000 simulated years, seed=42
- EP utils: `model/ep_utils.py` — single source of truth (convention: p_k = k/N)
- Outputs: AEP/OEP gross+net EP curves, PML 1-in-100/250, AAL, ep_master.png

## v3 goals (in order)
1. **Hazard calibration** — fit frequency/track parameters to HURDAT2; replace
   illustrative lambda with basin-season empirical distribution
2. **Wind model** — Holland B profile with translation asymmetry;
   Kaplan-DeMaria inland decay applied along track
3. **Secondary uncertainty** — Beta-distributed damage ratios per event;
   propagate epistemic uncertainty into EP bands
4. **Exposure & financials** — OED exposure format (LocPerilsCovered, etc.);
   ELT and YLT outputs; reinstatements on XoL layers
5. **Backtesting** — reproduce Andrew 1992 and Ian 2022 industry loss estimates
   as validation benchmarks

## Conventions (mandatory for all v3 code)
- **Config over code**: every tuneable parameter lives in `config/*.yaml`;
  nothing hardcoded in model files
- **Units in every docstring**: wind speed functions must declare units
  (kt / mph / m/s) — silent conversion bugs are the #1 wind-model failure mode
- **Seeds explicit everywhere**: `np.random.default_rng(seed)` only;
  no module-level RNG state
- **Tests required**: every statistical module in `model/` gets a matching
  `tests/test_<module>.py` using pytest; CI must stay green

## Stack
Python 3.11+, numpy, pandas, matplotlib, scipy, pyyaml, pytest

## Decisions log
<!-- Add entries as: `YYYY-MM-DD — decision — rationale` -->
- 2026-06-10 — renamed tower layers Working/Middle/Cat High → Layer 1/2/3 — "Working" implies a high-frequency attachment; this tower attaches at 60M (~1-in-25yr trigger), so the label was technically inaccurate
- 2026-06-10 — v2 tower attachments (60/100/150M) are illustrative round numbers — to be re-anchored to OEP return periods in Phase 4 (Paso 4.1), with per-layer expected loss, ROL and reinstatements

## Phase 1 calibration outcomes (HURDAT2) — full rationale in config/model_v3.yaml `source:` fields
- **Frequency**: λ=0.6576/yr — Poisson GLM (log link), single covariate standardized AMO
  (TNA dropped, collinearity r=0.892), satellite era 1966-2022, evaluated at current
  climate (2013-2022 mean). NB/overdispersion comparison NOT run — Poisson assumed
  (defensible for sub-unity annual landfall counts). `calibration/frequency_glm.py`.
- **Intensity**: continuous truncated-lognormal at landfall (loc=64 kt, mu_log=4.4362,
  sigma_log=0.2518), inverse-CDF sampled. Selected over Shifted Weibull (upper-tail QQ
  MAD 1.49 vs 1.81 kt, dAIC=-2.30). Full record 1851-2024 (n=112) — pre-satellite Vmax
  bias is non-directional, unlike count-detection bias driving the satellite-era choice
  for frequency. Continuous Vmax is source of truth; category is a derived label.
  `calibration/intensity.py`.
- **Landfall geography**: KDE over arc-length of simplified TIGER coastline; replaces
  v2 fixed coastal segments. s_cut at Cape Sable = 1057.43 km splits Atlantic vs Gulf.
- **Heading & translation**: regime-conditioned von Mises (Atlantic approach NW /
  Gulf approach NE). Translation speed stored PER EVENT in catalogue metadata as
  `translation_speed_kmh`.

## Phase 2 contract (read before touching wind physics)
- `translation_speed_kmh` is ALREADY in catalogue metadata per event — Paso 2.2
  (asymmetry) READS it, does not recompute it.
- Current wind placeholders are intentionally illustrative, replaced ONLY in their
  designated Phase 2 step — do not touch outside it:
    - modified Rankine vortex (hazard.py)         → Holland (Paso 2.1)
    - Rmax uniform(30,55) km, constant on track    → Vickery-Wadhera (Paso 2.3)
    - e-folding decay 120 km                        → Kaplan-DeMaria (Paso 2.3)
- **RNG discipline (Phase 2 onward)**: the legacy per-storm RNG stream is FROZEN. All
  new stochastic physics (Holland B, asymmetry alpha, Rmax error term) draw from a
  substream spawned UNCONDITIONALLY per storm (`rng.spawn(1)[0]`), so toggling any
  physics switch off reproduces the baseline stream bit-for-bit.
