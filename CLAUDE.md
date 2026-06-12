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
1. ~~**Hazard calibration**~~ — DONE (Phase 1): GLM frequency, trunc-lognormal
   intensity, KDE geography, regime headings — see Phase 1 outcomes below
2. ~~**Wind model**~~ — DONE (Phase 2): Holland B, asymmetry, K-D decay,
   V&W Rmax — see Phase 2 outcomes below
3. **Secondary uncertainty** — Beta-distributed damage ratios per event;
   propagate epistemic uncertainty into EP bands. CURRENT PHASE — Step 3.0a
   (MPI intensity cap) DONE; next: 3.0b stochastic WPR residual, then 3.1+.
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
- **Complete commits**: every step's commit must include ALL regenerated
  versioned artifacts (outputs/*.png, results/summary_metrics.csv) from the
  final production run — partial commits left the repo inconsistent twice
  on 2026-06-12

## Stack
Python 3.11+, numpy, pandas, matplotlib, scipy, pyyaml, pytest

## Decisions log
<!-- Add entries as: `YYYY-MM-DD — decision — rationale` -->
- 2026-06-10 — renamed tower layers Working/Middle/Cat High → Layer 1/2/3 — "Working" implies a high-frequency attachment; this tower attaches at 60M (~1-in-25yr trigger), so the label was technically inaccurate
- 2026-06-10 — v2 tower attachments (60/100/150M) are illustrative round numbers — to be re-anchored to OEP return periods in Phase 4 (Paso 4.1), with per-layer expected loss, ROL and reinstatements
- 2026-06-12 — results/summary_metrics.csv was committed with stale numbers (waterfall Config-2 intermediate state, AAL 7.58M) — waterfall subprocesses write to the same production file as run_all.py; fixed in f5b378c by regenerating from clean HEAD (reproduced v3 baseline exactly)
- 2026-06-12 — waterfall analysis runs must write to an isolated directory (results/waterfall/), never production summary_metrics.csv — implemented in Step 3.0a: `--results-dir results/waterfall` passed to run_all.py subprocesses; regression guard asserts prod mtime unchanged after sweep
- 2026-06-12 — .gitignore `results/` changed to `results/*` — directory-level ignore made the `!results/summary_metrics.csv` negation dead letter (file was tracked only by legacy status)
- 2026-06-12 — Step 3.0a DoD NOT fully closed — pending next session: (1) report OEP/AEP gross at 1-in-500 and 1-in-1000, cap=off vs cap=on, from real runs (rank check N=100k: 0-indexed ranks 199/99) and add the before/after tail table to the README — this is the headline evidence of the step (deep-tail correction, TVaR prerequisite); (2) verify the README subsection exists (why truncation, MPI vs historical record, renormalization vs clipping) and matches final numbers. Then Step 3.0b (stochastic WPR residual).

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

## Phase 2 outcomes — CLOSED 2026-06-12 (exhaustive 14-module review, no bugs found)
All Phase 2 physics implemented behind config switches in `hazard.physics`
(config/model_v3.yaml), each with an `off`/legacy branch that is bit-identical
to the prior baseline:
- `wind_profile`: rankine | **holland** (Holland 1980 gradient-balance profile)
- `rmax_method`: uniform | **vickery_wadhera** (V&W 2008 eq. 13, lognormal error)
- `b_method`: constant | **vickery_wadhera** (V&W 2008 eq. 14)
- `translation_asymmetry`: off | **on** (a=0.5, Powell 1980 / HAZUS-MH)
- `decay_method`: efold | **kaplan_demaria** (K-D 1995, DeMaria et al. 2006 coeffs)
(bold = production default). Env-var override layer `CATMODEL_*` in
model_config.py (`_PHYSICS_OVERRIDES`) lets analysis runs toggle switches per
subprocess — used by `analysis/waterfall.py`.

**Pre-3.0a v3 baseline (Phase 2 final, seed 42, 100k years, all Phase 2 switches on, cap off):**
AAL gross 9,171,353 | OEP-100 113.44M | OEP-250 147.15M | AEP-100 122.69M |
AEP-250 158.74M. Waterfall: v2 → +Rmax V&W (−0.45M) → +Holland&B (+4.45M) →
+asymmetry (+0.74M) → +decay K-D (+0.86M) → v3 pre-cap (9.17M); tail interaction
sub-additive (−19.8M at OEP-250), measured not assumed.

## Step 3.0a outcomes — DONE 2026-06-12
Upper truncation of landfall intensity distribution at MPI = 165 kt (DeMaria &
Kaplan 1994, ~163 kt at SST 30°C rounded up 2 kt). Switch: `intensity_cap` off|**on**
(off = bit-identical to pre-3.0a; on = renormalized inverse-CDF truncation).

**v3+3.0a baseline (seed 42, 100k years, all switches incl. cap=on):**
AAL gross 9,151,137 | OEP-100 113.23M | OEP-250 146.88M | AEP-100 122.39M |
AEP-250 158.69M. Anchored in `analysis/waterfall.py::_V3_ANCHORS` (Config 5,
self-check diff=0.0000). Cap effect: ~0.45% of events affected; OEP-100 −0.21M,
OEP-250 −0.27M vs pre-cap (gentle); "Max wind 243 kt / 280 mph" artifact gone.
Also delivered: waterfall subprocess isolation (results/waterfall/ dir, regression
guard in main()); `run_all.py --results-dir` and `model/summary.py --results-dir`.

**Deferred backlog (documented limitations, verified zero loss impact):**
- Asymmetry term `a·Vt` has no radial decay (clip verified sub-damage-threshold;
  refine by scaling with V_sym/Vmax if ever needed).
- Coriolis latitude frozen at landfall along track (~10% f variation over
  300 km; trivial fix next time wind_field.py is touched — track carries lat_c).

## RNG discipline (Phase 2 onward — MANDATORY for all new stochastic physics)
The legacy per-storm RNG stream is FROZEN. All new stochastic components
(Holland B, Rmax error term, and anything added in Phase 3+) draw from a
substream spawned UNCONDITIONALLY per storm (`rng.spawn(1)[0]`), so toggling
any physics switch off reproduces the baseline stream bit-for-bit. Each
sampler must consume a FIXED number of draws per call regardless of branch.