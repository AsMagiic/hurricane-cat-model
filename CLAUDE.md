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
   by default); Step 3.0c (physical Rmax floor) DONE (on by default, unblocks
   wpr=on sub-physical concern); next: 3.1 Beta-distributed damage ratios.
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
- 2026-06-12 — Step 3.0c DoD CLOSED — physical Rmax floor (8 km, on by default) implemented and tested. floor=off bit-identical to post-3.0b baseline; identity confirmed by reconciliation run (B raw<8km=99 == C floored=99, same 65,759-storm event count across all three scenarios, pure max() with zero RNG draws). Production baseline shift: AAL +$82,612 (+0.001%); all OEP/AEP quantiles unchanged (4 floored storms too compact to register at any return period). Under wpr=on+floor=on: min Rmax=8.000 km exactly, 0 storms < 8 km — 3.0c unblocks wpr=on for sub-physical concern. Remaining wpr=on question: mean vs median Δp centering (open design decision from 3.0b). Also corrected stale CLAUDE.md counts: 153→99 and min 1.2 km→0.920 km (the 153 was a 3.0a backlog projection transcribed unverified into 3.0b outcomes; real seed=42 100k-year run gives 99). Next: Step 3.1 Beta-distributed damage ratios.
- 2026-06-12 — Vulnerability re-architecture: migrating from v2 logistic damage-ratio curves to the industry-standard 5-state damage-state framework (HAZUS Hurricane TM Table 5-44). Source verification finding: HAZUS, FPHLM, and Pinelli et al. (2004) publish methodology but NOT calibrated parameters — Pinelli inputs are explicitly "hypothetical". All fragility parameters are own-derived by triangulation from public sources. Task 2a (calibration script) DONE. Task 2b (model integration, new config block, vulnerability mode switch) DONE — see Task 2b DoD entry and outcomes below.
- 2026-06-12 — Task 2a DoD CLOSED — fragility_thetas.py derives DS1-DS4 theta/beta parameters by (1) fixing theta3=v2 midpoint, theta1=88*(midpoint/145), (2) grid-searching beta in [0.10,0.24] (0.25 infeasible: exp(0.5)=1.6487 > 145/88=1.6477), (3) Nelder-Mead with 3 fixed multistarts for theta2/theta4 minimising MSE vs v2 logistic over g=linspace(70,220,80). 19/19 tests pass; 128/128 suite green. See Task 2a outcomes section below.
- 2026-06-12 — Task 2b DoD CLOSED — damage_state_mean vulnerability mode added behind config switch (CATMODEL_VULN_METHOD env-var; parallel to physics overrides). logistic_deterministic mode bit-identical to 3.0c production baseline (AAL $9,151,220 confirmed). B−A AAL delta: −$2,155,712 (−23.5%); WF dominates (54% of delta, −35.2% class drop), not Manufactured (20%, −13.6%) as initially predicted — WF shoulder band [100–130 mph] has larger DS-logistic DR gap because theta3=145 puts DS3 onset deep into Cat-3 winds; Manufactured (theta3=110) closes the gap sooner. Masonry 26% of delta (consistent with 50.7% TIV share); RC negligible. C−B gross delta exactly $0.00 — sub-threshold Manufactured damage ($15,109/year ground-up) fully deductible-absorbed; threshold decision RESOLVED: threshold=on (default). Masonry-RC sub-71 mph crossover: g*=71.0 mph, max violation 2.2e-5 DR, band [65, 69.8] mph; bound < 2e-4 enforced by updated test. Kernel-builder pattern (`build_event_kernel`) pre-computes per-location parameter arrays in closure; `scipy.special.ndtr` for ~65M kernel evals per run. 141/141 tests pass. YAML-CSV weld test enforces bit-equality. Next: Step 3.1 Beta-distributed damage ratios.
- 2026-06-12 — 3.0b Jensen question RESOLVED: wind_pressure.py fits via np.polyfit on ln(Δp)~ln(Vmax), no bias correction → coefficient `a` = exp(E[ln Δp]) = MEDIAN of conditional Δp. Therefore Δp = a·Vmax^b · exp(ε) is correct and needs NO −σ²/2 centering: the deterministic line was the median, and exp(ε) recovers the full distribution whose mean is median·exp(σ²/2). The +3% Jensen shift is not a bug — it corrects a pre-existing UNDER-estimate of mean Δp by the deterministic (median-based) implementation. Residual is statistically honest as-is. OPEN DESIGN DECISION (deferred to after 3.0c): should production (wpr=on) use the mean (current behavior, statistically unbiased for expected loss) or be re-centered to preserve the old median? Defer until the Rmax floor exists, since part of the +3% Δp mass lands in sub-physical Rmax that the floor will reshape — deciding the Δp center before the floor would be deciding on numbers that will change.

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
  residual (wpr=on) amplifies it via high-ε draws. Under wpr=on: 99/100k storms have
  Rmax < 8 km (min 0.920 km) — the sub-physical Rmax artefact is reintroduced. RESOLVED
  by Step 3.0c: physical Rmax floor (8 km, on by default). (Note: the original entry
  cited 153/min 1.2 km — those were unverified projections; real run corrected in 3.0c.)

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
Sub-physical Rmax: 99/100k storms (0.099%) have Rmax < 8 km under wpr=on (min 0.920 km).
(Corrected in Step 3.0c: original 153/1.2 km were unverified projections from 3.0a backlog.)
Config 6 (wpr=on) added to waterfall; self-check still passes for Config 5 (diff=0.0000).
RNG: nested spawn (`wpr_rng = vw_rng.spawn(1)[0]`) — see RNG discipline section above.
tests/test_wpr_residual.py: 6 tests (bit-identical sequence, ε stats, Jensen bias, substream independence, draw discipline).

## Step 3.0c outcomes — DONE 2026-06-12
Physical Rmax floor: Rmax = max(Rmax_vw, 8 km), applied after `_vw_rmax_sample` and
before `_vw_b_sample` so B couples to the floored radius. Switch: `rmax_floor` off|**on**
(off = bit-identical to post-3.0b baseline; on = censoring floor applied).
rmax_floor=on is production default (correct physics; sub-floor storms are near-zero-loss
compact near-misses — atom at 8 km introduces no material metric distortion).
No new RNG draws; floor=off identity proven by recon_30c.py (B raw<8km=99 == C floored=99).

**v3+3.0c baseline (seed 42, 100k years, all switches incl. floor=on, wpr=off):**
AAL gross 9,151,220 | OEP-100 113.23M | OEP-250 146.88M | AEP-100 122.39M |
AEP-250 158.69M. Anchored in `analysis/waterfall.py::_V3_ANCHORS` (Config 7,
self-check diff=0.0000). Floor effect (floor=off → floor=on, wpr=off): AAL +$82,612
(+0.001%); OEP/AEP: 0.0000M change at all return periods. 4 storms floored (raw Rmax
6.658–7.9 km → 8.0 km); none large enough to move any return-period rank.

**wpr=on+floor=on validation (seed 42, 100k years):**
Min Rmax = 8.000 km; 99 storms floored; 0 storms < 8 km — 3.0c unblocks wpr=on for
the sub-physical Rmax concern. Metrics: AAL 8.8943M (−0.257M vs floor=on+wpr=off);
OEP-100 110.10M (−3.12M); OEP-500 174.08M (+4.81M); OEP-1000 201.44M (+10.36M).
Deep-tail variance dominates, consistent with 3.0b findings. Remaining wpr=on question:
mean vs median Δp centering (open design decision deferred to 3.1).

**3.0b stale count corrected:**
153/100k (min 1.2 km) in 3.0a backlog and 3.0b outcomes were an unverified projection
from 3.0a carried forward without a real run. recon_30c.py (seed=42, 100k years, wpr=on,
floor=off) confirms: 99 storms, min Rmax 0.920 km. Both entries corrected in this commit.

tests/test_rmax_floor.py: 6 tests (floor=off bit-identity, floor switch, AAL direction,
B coupling to floored Rmax — deterministic at _FLOOR_KM_TEST=40 km, 3/10 storms trigger,
no conditional skip). Config 7 (floor=on, wpr=off) added to waterfall.

## Task 2a outcomes — DONE 2026-06-12 (vulnerability re-architecture, calibration step)
Damage-state fragility parameters derived in `calibration/fragility_thetas.py`.
Framework: 5 states DS0-DS4, lognormal fragility P(DS≥k|g) = Φ(ln(g/θ_k)/β),
consequence [0, 0.02, 0.10, 0.50, 1.00] (HAZUS TM §8.1.4.3).
E[DR|g] = Σ_{k=1}^{4} Δlr_k · Φ(ln(g/θ_k)/β), Δlr=[0.02,0.08,0.40,0.50].
All wind speeds 3-s peak gust (mph).

**Derived parameters (seed-free, deterministic):**

| Class              |  β   |  θ₁   |  θ₂   |  θ₃   |  θ₄   |  RMSE  |
|--------------------|------|--------|--------|--------|--------|--------|
| Manufactured       | 0.11 |  66.8  |  74.5  | 110.0  | 122.8  | 0.0495 |
| Wood Frame         | 0.13 |  88.0  | 100.2  | 145.0  | 165.1  | 0.0666 |
| Masonry            | 0.15 | 100.1  | 116.3  | 165.0  | 191.7  | 0.0393 |
| Reinforced Concrete| 0.20 | 112.3  | 143.7  | 185.0  | 226.8  | 0.0062 |

**E[DR] at gust_threshold = 65 mph (DS scheme has no hard threshold):**
Manufactured 0.01664 | Wood Frame 0.00023 | Masonry 0.00004 | RC 0.00007.
Manufactured is the most consequential signal: θ₁=66.8 mph sits 1.8 mph above
the v2 threshold, so the DS scheme assigns real damage (E[DR]~1.7% at 65 mph,
~0.4% at 60 mph) exactly where the v2 logistic hard-cuts to zero. Storm
peripheries cover large areas at every event, and Manufactured has the highest
damage ratio — small ΔDR × large area × high frequency = potentially material
AAL delta. Task 2b threshold decision RESOLVED: production uses threshold=on (default).
C−B gross AAL = $0.00 exactly — sub-threshold Manufactured damage ($15,109/year
ground-up, gust 65–66.76 mph) fully absorbed by per-location deductible before
gross. Switch engaged confirmed by direct probe (kernel at gust=60: off→0.527%,
on→0.0). WF/Masonry/RC: moot (E[DR]@65 < 0.025%). See Task 2b outcomes below.

The E[DR] curves for Mfg/WF show a 'shoulder' (~0.10 plateau around
85-105/100-120 mph respectively): DS1+DS2 saturate (envelope damage complete,
2%+8%) before DS3 (structural damage) onset. This plateau-then-rise shape is an
EXPECTED structural feature of the damage-state scheme — physically meaningful
and impossible to represent with a logistic. Not a calibration artifact.

At extreme winds the DS-scheme E[DR] exceeds the old logistic caps
(Masonry ~0.93 vs cap 0.90; RC ~0.71 and rising vs cap 0.75): the intended
consequence of the free-cap design decision — destruction probability emerges
from the DS4 fragility (θ₄) instead of being forbidden by an artificial
ceiling. Task 3 validation should expect this difference vs the logistic.

**Feasibility:** θ₃/θ₁ = 145/88 = 1.6477 (identical ratio for all classes by
construction). β=0.25 infeasible (exp(0.5)=1.6487 > 1.6477); effective grid
[0.10, 0.24]. No beta pinned at either edge.

**Source verification:** HAZUS Hurricane TM (Table 5-44, §8.1.4.3), FPHLM,
Pinelli et al. (2004) — all publish methodology only, not calibrated parameters.
Own-derived by triangulation; derivation rationale inline in fragility_thetas.py.

**Artifacts (versioned):** outputs/fragility_calibration.png (2×2 grid, v2
logistic vs DS E[DR] + 4 fragility curves per class); outputs/fragility_thetas.csv.
tests/test_fragility_calibration.py: 19 tests (pure functions, feasibility,
determinism, sanity: thetas↑, SEP, cross-class hierarchy, beta bounds, E[DR]↑).

**Task 2b DONE.** See Task 2b outcomes section below. Next: Step 3.1 Beta-distributed damage ratios.

## Task 2b outcomes — DONE 2026-06-12 (damage_state_mean vulnerability mode, kernel-builder integration)
`damage_state_mean` mode added behind config switch. `logistic_deterministic` (production
default) is bit-identical to 3.0c baseline; `damage_state_mean` is a full model-shape change.

**Architecture:**
- `config/model_v3.yaml`: new `method`, `ds_gust_threshold`, `damage_states` blocks under `vulnerability:`
- `model_config.py`: `_materialize_damage_states` + `_VULN_OVERRIDES` + `_apply_vulnerability_overrides`
  (parallel to `_apply_physics_overrides`; targets `tree.vulnerability`; env vars
  `CATMODEL_VULN_METHOD`, `CATMODEL_DS_GUST_THRESHOLD`)
- `model/vulnerability.py`: `build_event_kernel(constructions_array)` → vectorized closure built
  once at module setup; `scipy.special.ndtr` (raw C-level CDF, ~65M evals/run);
  switch-aware `damage_ratio()`
- `model/loss.py`: `_vuln_kernel = build_event_kernel(constructions)` replaces inline
  `midpoints/caps/ks` arrays; `_event_loss` delegates to kernel unchanged structurally
- `tests/test_vulnerability.py`: 5 test classes, 13 tests: bit-identical logistic, YAML-CSV weld
  (float parse of same string → deterministic bit-equality), cross-implementation ≤1e-12,
  DS-mean properties (monotone, bounded, hierarchy, crossover bounds), kernel-vs-scalar

**Three-config diagnostic (seed 42, 100k years):**

| Config | Mode | Threshold | AAL gross ($) | OEP-100 (M) | OEP-250 (M) | AEP-100 (M) | AEP-250 (M) |
|--------|------|-----------|---------------|-------------|-------------|-------------|-------------|
| A: logistic | logistic_deterministic | — | 9,151,220 | 113.23 | 146.88 | 122.39 | 158.69 |
| B: DS-mean, on | damage_state_mean | on | 6,995,508 | 96.75 | 130.08 | 102.98 | 138.02 |
| C: DS-mean, off | damage_state_mean | off | 6,995,508 | 96.75 | 130.08 | 102.98 | 138.02 |

Config A bit-identical to 3.0c production baseline. ✓

**B−A: −$2,155,712 gross AAL (−23.5%)** — paradigm shift, not a small recalibration.
The DS scheme has a qualitatively different curve shape (shoulder plateau + steeper rise)
vs. the smooth logistic, with RMSE 0.050–0.067 across classes.

**B−A per-construction decomposition (gross AAL, seed 42, 100k years):**

| Class | A gross AAL ($) | B gross AAL ($) | Delta ($) | Δ% class | Share of Δ |
|-------|-----------------|-----------------|-----------|----------|------------|
| Wood Frame | 3,314,116 | 2,148,461 | −1,165,655 | −35.2% | 54.1% |
| Masonry | 2,459,070 | 1,895,453 | −563,617 | −22.9% | 26.2% |
| Manufactured | 3,122,662 | 2,696,610 | −426,052 | −13.6% | 19.8% |
| RC | 255,373 | 254,984 | −389 | −0.2% | 0.02% |

WF dominates at 54% of the delta — not Manufactured (20%) as the TIV-ratio argument
suggested. Mechanism: WF's DS1+DS2 shoulder saturates at ~10% DR for gusts in [100–130 mph]
before DS3 onset at theta3=145 mph. The logistic is at 15–30% in that band. The gap
(10–20 pp) hits in the most frequent Cat-2/3 damage wind range, and WF carries 2.9×
more TIV than Manufactured ($126.8M vs $43.7M). Manufactured (theta3=110) transitions
into DS3 at lower gusts, so the shoulder gap closes sooner — smaller per-% drop (13.6%
vs 35.2% for WF). Masonry (26% of delta) is proportional to its 50.7% TIV share with a
22.9% class drop — no anomaly. RC negligible as expected.

**C−B threshold decision — RESOLVED (deductible-absorbed, not truly zero):**
- Probe: threshold=off kernel returns E[DR]=0.527% at gust=60 mph for Manufactured;
  threshold=on returns 0.0 — switch engaged correctly
- C ground-up AAL: $9,806,471 | B ground-up: $9,791,362 → **C−B ground-up = +$15,109/year**
  (measured positive; sub-threshold Manufactured damage exists)
- C gross AAL = B gross AAL = **$6,995,508.24 exactly → C−B gross = $0.00**
- Mechanism: gust 65–66.76 mph → Manufactured E[DR] ≈ 0.3–0.8% → ground-up
  ≈ $500–1,600 per location → fully absorbed by per-location deductible before gross
- DECISION: threshold=on (production default). Revisit only if deductible structure
  changes materially.

**Masonry-RC crossover (DS-mean, g < 71 mph):**
RC beta=0.20 fatter DS1 tail exceeds Masonry beta=0.15 for g ∈ [65, 69.8] mph (threshold=on
active range). Analytical g* = 71.0 mph; max violation 2.2e-5 DR at gust=65.71 mph.
Well below 2e-4 bound. Test `test_cross_class_hierarchy` now asserts: strict hierarchy
g > 72 mph AND max sub-72 violation < 2e-4 (both with explicit assert, not silently weakened).
Revisit if Task 3 validation probes the 65–70 mph gust band.

**141/141 tests pass.** YAML-CSV weld test (`TestYamlCsvWeld`) enforces bit-equality of
YAML thetas/betas to `outputs/fragility_thetas.csv` — future hand-edit of either file
fails loudly.

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