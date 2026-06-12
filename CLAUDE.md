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
   (MPI intensity cap) DONE; Step 3.0b (stochastic WPR residual) DONE (ships off
   by default, blocked on 3.0c Rmax floor); next: 3.0c physical Rmax floor (~8 km),
   then 3.1 Beta-distributed damage ratios.
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
- 2026-06-12 — Step 3.0a DoD CLOSED — deep-tail audit complete (seed 42, 100k years, real runs, cap=off bit-identical to pre-3.0a baseline). OEP deltas at 1-in-100/250/500/1000: −0.21M/−0.27M/+0.25M/+0.54M; AEP deltas: −0.30M/−0.05M/−0.95M/−0.85M. OEP sign reverses at 1-in-500/1000 via V&W Rmax coupling (capped storms have 5–18× larger Rmax, replacing sub-physical compact artifacts; 1,619/100k years have cap=on > cap=off, max excess 150M; +0.25M/+0.54M within bootstrap MC noise, CIs overlap). README subsection corrected (correct mechanism: V&W Rmax, not probability-mass redistribution). Next: Step 3.0b (stochastic WPR residual).
- 2026-06-12 — Step 3.0b: nested spawn architecture is the canonical pattern for adding a new stochastic physics component without perturbing existing streams. `vw_rng = rng.spawn(1)[0]` (same slot per storm as before); `wpr_rng = vw_rng.spawn(1)[0]` (nested child — vw_rng.spawn() does not consume vw_rng's bitgenerator variates). spawn(2) from the parent RNG does NOT work — it increments the parent's n_children_spawned by 2 per storm, so storm N's children use different slots than before. Nested spawn from a child is the correct pattern.
- 2026-06-12 — Step 3.0b DoD CLOSED — stochastic WPR residual implemented and tested. Ships off by default (wpr=off bit-identical to 3.0a baseline; _V3_ANCHORS unchanged). wpr=on effect: AAL −0.259M (−2.8%), OEP-100 −3.1M, OEP-250 −0.4M, OEP-500 +4.8M, OEP-1000 +10.4M. Sign reverses between 250 and 500 — variance dominates at deep tail. Sub-physical Rmax (< 8 km): 153/100k storms (0.153%, min 1.2 km) under wpr=on. wpr=on NOT production-ready: deferred to Step 3.0c physical Rmax floor. Config 6 added to waterfall.
- 2026-06-12 — Step 3.0c: physical Rmax floor (~8 km observational limit, no observed TC has Rmax < 8 km) — V&W extrapolates to sub-physical Rmax at high Δp; the WPR residual (Step 3.0b) makes this material for high-ε draws near the 165 kt cap (153/100k storms, min Rmax 1.2 km). MEDIUM priority (upgrade from low). Required before wpr=on is production-ready.

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
self-check diff=0.0000). Cap effect (cap=off → cap=on, real runs):
OEP: −0.21M/−0.27M/+0.25M/+0.54M at 1-in-100/250/500/1000;
AEP: −0.30M/−0.05M/−0.95M/−0.85M. Max event gross cap=off 323.36M / cap=on
326.79M. OEP sign reverses at 1-in-500/1000 via V&W Rmax coupling: uncapped 200–240 kt
storms have Rmax 1–6 km (sub-physical, near-zero-loss artifacts); capping at
165 kt expands Rmax 5–18× (lower Vmax → lower Δp → larger Rmax), turning
compact near-misses into broad direct hits. 1,619/100k years have cap=on >
cap=off (max excess 150 M). +0.25M/+0.54M diffs at 1-in-500/1000 are within
bootstrap MC noise (5–10% of CI half-width; CIs overlap). "Max wind 243 kt /
280 mph" artifact gone. Also delivered: waterfall subprocess isolation (results/waterfall/
dir, regression guard in main()); `run_all.py --results-dir` and
`model/summary.py --results-dir`.

**Deferred backlog (documented limitations, verified zero loss impact):**
- Asymmetry term `a·Vt` has no radial decay (clip verified sub-damage-threshold;
  refine by scaling with V_sym/Vmax if ever needed).
- Coriolis latitude frozen at landfall along track (~10% f variation over
  300 km; trivial fix next time wind_field.py is touched — track carries lat_c).
- MPI cap bounds the uncapped lognormal Δp tail but does NOT bound Δp when the WPR
  residual (wpr=on) amplifies it via high-ε draws. Under wpr=on: 153/100k storms have
  Rmax < 8 km (min 1.2 km) — the sub-physical Rmax artefact is reintroduced. DEFERRED
  to Step 3.0c: physical Rmax floor (~8 km). MEDIUM priority (was low for 3.0a, elevated
  by 3.0b making it material).

## Step 3.0b outcomes — DONE 2026-06-12
Stochastic WPR residual: Δp = a·Vmax^b · exp(ε), ε ~ N(0, σ²), σ=0.2458.
Switch: `wpr_residual` off|**off-default** (off=bit-identical to 3.0a; on=lognormal ε activated).
wpr=off is production default; _V3_ANCHORS unchanged (Config 5, cap=on, wpr absent → defaults off).

**v3+3.0b baseline (seed 42, 100k years, wpr=off = 3.0a baseline unchanged):**
Same as 3.0a: AAL gross 9,151,137 | OEP-100 113.23M | OEP-250 146.88M | AEP-100 122.39M | AEP-250 158.69M.

**wpr=on diagnostic (seed 42, 100k years):**
AAL gross 8.892M (−0.259M, −2.8%); OEP deltas: −3.1M/−0.4M/+4.8M/+10.4M at 1-in-100/250/500/1000;
AEP deltas: −3.0M/+0.1M/+4.0M/+11.7M. Jensen bias drives AAL down (mean Δp +3.1%, mean Rmax decreases);
variance inflation dominates at 1-in-500/1000 (low-ε draws → very large Rmax → large footprint losses).
Sub-physical Rmax: 153/100k storms (0.153%) have Rmax < 8 km under wpr=on (min 1.2 km).
Config 6 (wpr=on) added to waterfall; self-check still passes for Config 5 (diff=0.0000).
RNG: nested spawn (`wpr_rng = vw_rng.spawn(1)[0]`) — see RNG discipline section above.
tests/test_wpr_residual.py: 6 tests (bit-identical sequence, ε stats, Jensen bias, substream independence, draw discipline).

## RNG discipline (Phase 2 onward — MANDATORY for all new stochastic physics)
The legacy per-storm RNG stream is FROZEN. All new stochastic components draw
from substreams spawned UNCONDITIONALLY per storm. Spawn architecture (Step 3.0b):

```
vw_rng  = rng.spawn(1)[0]          # V&W Rmax + B draws; same slot per storm as legacy sub_rng
wpr_rng = vw_rng.spawn(1)[0]       # nested child of vw_rng; dedicated to WPR ε draw
```

**Critical constraint:** Do NOT call `rng.spawn(N)` for N > 1 within a single storm. Each
`rng.spawn(1)` call increments `rng`'s SeedSequence `n_children_spawned` by 1. Calling
`rng.spawn(2)` in one storm shifts ALL subsequent storm spawn slots by +1, breaking bit-
identity. The nested-spawn pattern (`child.spawn(1)[0]`) uses `child`'s SeedSequence counter,
not `rng`'s — so `rng`'s counter stays at the pre-3.0b rate (exactly 1 spawn per storm).

**For the next new stochastic component (3.1+):** use `vw_rng.spawn(2)` to split into
`vw_rng.children[0]` (existing V&W draws) and `vw_rng.children[1]` (new component) —
OR add another level of nesting from `wpr_rng`. One child per noise source; each
spawned unconditionally; each consuming a FIXED number of draws per call.