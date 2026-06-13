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
3. ~~**Secondary uncertainty**~~ — DONE: Step 3.0a (MPI intensity cap); Step 3.0b
   (stochastic WPR residual, off by default); Step 3.0c (physical Rmax floor, on
   by default); Step 3.1 (Gaussian copula common-shock Beta damage draws, off by
   default, sensitivity capability). Production baseline unchanged throughout.
4. **Industry output formats** — ~~OED exposure format (Location + Account,
   LocPerilsCovered, ConstructionCode, OccupancyCode, WTC peril, read adapter)~~
   DONE (Step 4.1); ~~standard YLT as single source for EP metrics; stable EventId;
   full RP set [5,10,25,50,100,250,500,1000]~~ DONE (Step 4.2); ~~ELT outputs~~
   DONE (Step 4.3); reinstatements on XoL layers.
   (NOTE: real per-location site-exposure / terrain modelling is v4, not here —
   see vulnerability closure entry.)
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
- 2026-06-12 — Task 2b DoD CLOSED — damage_state_mean vulnerability mode added behind config switch (CATMODEL_VULN_METHOD env-var; parallel to physics overrides). logistic_deterministic mode bit-identical to 3.0c production baseline (AAL $9,151,220 confirmed; validates loss.py kernel-builder refactor). B−A AAL delta: −$2,155,712 (−23.5%, CHARACTERIZED NOT ADJUDICATED — Task 3 validates against HAZUS Table 5-46); PREDICTION CORRECTION: WF dominates (54% of delta, −35.2% class drop), not Manufactured (20%, −13.6%) as predicted — WF deficit window wide (theta3=145, RMSE 0.067, gap spans ~95–160 mph); Masonry 26% of delta (sub-proportional to 50.7% TIV, consistent with RMSE 0.039); RC negligible. C−B threshold: PORTFOLIO-CONTINGENT — C−B ground-up +$15,109/yr (switch engaged, probe confirmed at gust=60 E[DR]=0.527%); C−B gross $0.00 deductible-absorbed-exactly; deferred to Task 3, re-evaluate if Phase 4 OED deductible structure changes. Masonry-RC sub-71 mph crossover: g*=71.0 mph, max violation 2.2e-5 DR, band [65, 69.8] mph; mechanism: per-class betas → theta ordering ≠ E[DR] ordering; bound < 2e-4 enforced by two explicit test asserts. 141/141 tests pass. Next: Step 3.1 Beta-distributed damage ratios.
- 2026-06-12 — Task 3 DoD CLOSED — DS-mean fragility validated against HAZUS Elena 1985 field data (Table 5-43, 8 Manufactured Housing parks). Ex-ante criterion: MdAE<=16.5 pts AND MAE<=19.6 pts (=1.5× HAZUS certified errors MdAE=10.97, MAE=13.04). RESULT: FAIL. Our MdAE=68.4 pts, MAE=67.3 pts (~4× criterion). MECHANISM: conceptual anchoring error — theta3 was anchored to the logistic midpoint (50% MEAN DR), but in the DS framework theta3 is the MEDIAN OF DS3 (50% PROBABILITY of extensive/structural damage). These are different physical quantities. Internal validation passed (bit-identical, RMSE vs logistic, cross-impl ≤1e-12) because the reference shared the same anchoring error — internal consistency cannot detect a flawed reference. Elena field data exposes it: 0–40% observed major damage at 109–126 mph vs our 47–89%. Terrain check CONFIRMED (rough-site mean error 69.9 pts > Dauphin 49.2 pts; secondary signal, dwarfed by anchoring bias). Discrimination verdict: data CAN discriminate — DS-mean is measurably worse than HAZUS on this metric, NOT "both consistent with noise." Scope: Manufactured only (8 parks, 1 storm); WF/Masonry/RC unvalidated. Production default logistic_deterministic unchanged. Re-anchoring strategy is a separate decision. 165/165 tests pass. See Task 3 outcomes section.
- 2026-06-12 — 3.0b Jensen question RESOLVED: wind_pressure.py fits via np.polyfit on ln(Δp)~ln(Vmax), no bias correction → coefficient `a` = exp(E[ln Δp]) = MEDIAN of conditional Δp. Therefore Δp = a·Vmax^b · exp(ε) is correct and needs NO −σ²/2 centering: the deterministic line was the median, and exp(ε) recovers the full distribution whose mean is median·exp(σ²/2). The +3% Jensen shift is not a bug — it corrects a pre-existing UNDER-estimate of mean Δp by the deterministic (median-based) implementation. Residual is statistically honest as-is. OPEN DESIGN DECISION (deferred to after 3.0c): should production (wpr=on) use the mean (current behavior, statistically unbiased for expected loss) or be re-centered to preserve the old median? Defer until the Rmax floor exists, since part of the +3% Δp mass lands in sub-physical Rmax that the floor will reshape — deciding the Δp center before the floor would be deciding on numbers that will change.
- 2026-06-13 — Vulnerability re-architecture CLOSED: explored, field-validated, archived as non-production. Damage-state framework (Tasks 2a/2b/3) built and internally validated, but field validation vs Hurricane Elena failed ~4× the ex-ante bar. Two compounding causes: conceptual anchoring error (theta3 tied to 50% mean-DR, not 50% P(extensive)) AND terrain-exposure mismatch (model is open-terrain, survey sites suburban). KEY FINDING: no version of the vulnerability — logistic or DS — is field-validatable without a per-location site-exposure layer the v3 lacks. v2 midpoints confirmed (via project history) to be heuristic (HAZUS-behaviour-anchored, never calibrated to published values — which don't exist in extractable form). Production stays logistic_deterministic. v4 path: site-exposure layer → field-validated fragilities.
- 2026-06-13 — Step 3.1 DoD CLOSED — Gaussian copula common-shock Beta damage draws implemented as a sensitivity capability (off by default; bit-identical to 3.0c baseline when off). damage_rng = np.random.default_rng([seed, 1]) — two-integer entropy SeedSequence, independent of the hazard spawn tree by both structural design (SeedSequence entropy and spawn_key are distinct struct fields with separate inputs per NumPy docs; zero state collision across slots 0-19 confirmed by exhaustive check; no explicit non-collision theorem published for this cross-entropy-vs-spawn-key case — guarantee rests on the measured zero-collision check) and empirical probe (max |r|=0.00267 at N=1M, 3.9x reduction vs N=100k, confirmed sampling noise). Smoke: AAL=$9,151,220.11 (diff $0.11 = display rounding). rho-sweep (CV=0.40 placeholder): AAL flat across all rho (delta=0.12M); OEP-1000 monotone: 191.1->192.5->200.6->213.4->222.7M for rho=0/0/0.3/0.7/1.0 (rho=1 is +31.6M/+16.5% vs deterministic). Parameters uncalibrated (CV=0.40, rho=0.5 illustrative). CV calibration deferred to v4 (MDR-dependent function requires per-event damage data); rho calibration deferred to v4. Canonical seeding method for future simulation-level RNG chains that must not perturb hazard substreams. 18 tests pass (5 classes). Next: Phase 4 (industry output formats: OED, ELT/YLT, reinstatements). Site-exposure layer is v4.
- 2026-06-13 — Step 4.1.1 CLOSED — OED financial field names corrected to spec: DedCode1Building→LocDedCode1Building, DedType1Building→LocDedType1Building, LimitType1Building→LocLimitType1Building. Value fields (LocDed1Building, LocLimit1Building) were already correct. data/oed/location.csv regenerated; data/exposure.csv removed (git rm — demoted to tests/fixtures/exposure_reference.csv in 4.1). Test blind-spot closed: test_financial_spec_columns_present asserts all 5 Loc-prefixed financial columns present; test_unprefixed_financial_columns_absent asserts the 3 wrong names are absent. loss.py docstring updated (stale data/exposure.csv reference → model.exposure_io). 206/206 tests pass.
- 2026-06-13 — Step 4.2 DoD CLOSED — Standard YLT adopted as single source of truth for all EP metrics. model/outputs.py::build_ylt() constructs results/ylt.csv (7 columns: Year, NumEvents, AggGroundUp, AggGross, AggNet, MaxOccGross, MaxOccNet) from annual_losses.csv + events_net.csv using the same arithmetic as summary.py's former _load() net reconstruction (groupby sum of recovery_total → AggNet; groupby max of portfolio_net → MaxOccNet). model/summary.py refactored to read ylt.csv instead of reconstructing net series — internal consistency guaranteed by identical arithmetic. RETURN_PERIODS now from config/model_v3.yaml::summary.return_periods = [5,10,25,50,100,250,500,1000]; ep_utils remains the sole EP kernel (convention p_k=k/N). EventId: global monotonic 1-based integer assigned in loss.py (no RNG), first column of events.csv and carried through to events_net.csv. Bit-identity confirmed: AAL_M 9.151/7.709/8.471/7.04; PML_1in100_M 122.39/70.32/113.23/60.0; PML_1in250_M 158.69/90.61/146.88/60.0 — all unchanged. New RP rows (5,10,25,50,500,1000) sourced from real run. run_all.py: Step 5.5 (YLT build) inserted between reinsurance and summary. 235/235 tests pass (29 new in test_outputs.py: 12 equivalence, 11 integrity, 6 EventId).
- 2026-06-13 — Step 4.3 DoD CLOSED — Sampled Event Loss Table (SELT) added as results/elt.csv, emitted by model/outputs.py alongside the YLT. SELT is a sampled ELT (NOT a rated catalog): AnnualRate = 1/N_years = 0.00001 for every event (uniform), reflecting that each event is drawn once from a 100,000-year simulation. MeanLossGross and MeanLossNet sourced from portfolio_gross / portfolio_net in events_net.csv (no RNG, no simulation re-run). StdDevIndependent and StdDevCorrelated are NaN — deferred to v4 (require calibrated CV/ρ and a moment treatment of the deductible/limit nonlinearity). ExposureValue = reference portfolio TIV ($500M, constant per row, sourced from load_portfolio()["tiv"].sum()). AAL reconciliation identity confirmed: sum(AnnualRate × MeanLossGross) = YLT AEP-gross AAL (events partition years → summation commutes; relative error < 1e-6 enforced by test). Load-bearing guards: TestEltReconciliation (gross + net AAL, rate uniformity, rate sum = n_events/n_years); TestEltIntegrity (column order, StdDev null, ExposureValue constant, net≤gross, 8 synthetic tests); TestEltEventIdSchema (EventId unique, matches events_net.csv set). summary_metrics.csv unchanged. 249/249 tests pass (14 new: 4 reconciliation + 8 integrity + 2 EventId-schema).
- 2026-06-13 — Step 4.1 DoD CLOSED — OED v4 exposure format adopted as canonical representation. Portfolio serialised to data/oed/location.csv + data/oed/account.csv; consumed by all model readers via model/exposure_io.py::load_oed_exposure(). data/exposure.csv demoted to tests/fixtures/exposure_reference.csv (frozen bit-identity reference, not read by live pipeline). OED mapping: ConstructionCode Wood Frame=5050, Masonry=5100, RC=5150, Manufactured=5350; OccupancyCode Single Family=1051, Condo=1055, Mobile Home=1051; LocPerilsCovered=PolPerilsCovered="WTC"; DedType=LimitType=0 (Amount). DedType=Amount required for bit-identity: deductible is pre-rounded integer stored directly; %TIV storage would require float reconstruction and could silently differ by ±1 dollar. Lossless round-trip via OrgConstructionCode/OrgOccupancyCode provenance fields (scheme="MODEL"); OccupancyCode alone is NOT invertible (Single Family and Mobile Home both → 1051). Terrain invariant preserved: OED carries exposure only; Exposure C gust factor 1.3 remains a model assumption in config, not wired from any OED field. Bit-identity confirmed by real run_all.py: AAL 9.151M, OEP-100 113.23M, OEP-250 146.88M, AEP-100 122.39M, AEP-250 158.69M — unchanged. 204/204 tests pass (29 new in test_exposure_io.py: 12 round-trip, 15 OED validity, 2 error-path). Next: ELT/YLT output formats.

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

## Step 3.1 outcomes — DONE 2026-06-13 (secondary uncertainty — sensitivity capability, off by default)
Gaussian copula common-shock Beta draws per event. Switch: `damage_uncertainty` off|**off-default**
(off=bit-identical to 3.0c baseline; on=Beta-distributed damage ratios activated).
Production baseline and `results/summary_metrics.csv` unchanged.

**Mechanism:**
For each event, draw one common shock Z_event ~ N(0,1) and per-location noise ε_i ~ N(0,1).
Copula quantile: U_i = Φ(√ρ · Z_event + √(1−ρ) · ε_i). Realized DR_i = Beta.ppf(Uᵢ; αᵢ, βᵢ)
where (αᵢ, βᵢ) are derived from the deterministic mean DR (logistic curve) and a global CV.
At ρ=0 noise is independent across locations — LLN washes it out over 1,000 locations and the
tail barely moves. At ρ=1 all locations share the same quantile (common shock) — diversification
collapses, the tail fattens significantly. This is the mechanism through which ρ inflates OEP.

**RNG architecture:** `damage_rng = np.random.default_rng([seed, 1])` — two-integer entropy
seeds a SeedSequence distinct from the hazard root `np.random.default_rng(seed)`.
Independence rests on two lines of evidence:
- STRUCTURAL: NumPy's SeedSequence stores user entropy and spawn_key in distinct
  struct fields; docs state spawn_key "users will not" set (so `SeedSequence([42,1])`
  has `entropy=[42,1]` and `spawn_key=()`, while any `rng.spawn(k)[0]` child has
  `entropy=42` and `spawn_key=(k,)` — separate, unshared inputs). Exhaustively
  confirmed: zero state collision between `SeedSequence([42, 1])` initial state and
  any of slots 0-19 of `SeedSequence(42)`'s spawn tree (checked via `generate_state(4)`
  and `PCG64.state`). No explicit non-collision theorem is published for this
  cross-entropy-vs-spawn-key case; the guarantee rests on the measured zero-collision
  check, which the structural field disjointness makes implausible to violate
  accidentally.
- EMPIRICAL: Correlation probe across all 10 hazard spawn slots + root stream.
  N=100k: max |r| = 0.01055 at slot 5 (marginally above 3sigma=0.009487, flagged for
  targeted follow-up). N=1M targeted at slot 5: |r| = 0.002674 < 3sigma=0.003000.
  Reduction 3.9x ~= sqrt(10) confirmed sampling noise (structural correlation is
  N-invariant; empirical |r| decays as 1/sqrt(N) for independent streams).
The structural argument is the guarantee; the probe is the verification.
Hazard stream uncontaminated confirmed by bit-identical smoke: uncertainty=off
-> AAL = $9,151,220.11 (diff from 3.0c integer anchor: $0.11 = display rounding only). PASS.

**Parameters (NOT calibrated):**
- `damage_cv = 0.40` — placeholder, order-of-magnitude; pointwise DR CV is larger than
  aggregate severity CV (LLN effect). To be replaced with an MDR-dependent function in v4
  when per-event damage data is available.
- `damage_rho = 0.5` — sensitivity knob, not calibrated. Swept in diagnostics. Industry
  spatial correlations vary widely; v4 should calibrate from per-event loss data.
- `damage_uncertainty = "off"` — production default. Sensitivity analysis only.

**ρ-sweep table (seed 42, 100k years, CV=0.40):**

| Config | uncertainty | ρ | AAL (M) | OEP-100 | OEP-250 | OEP-500 | OEP-1000 | AEP-100 | AEP-250 |
|---|---|---|---|---|---|---|---|---|---|
| Baseline (det.) | off | — | 9.15 | 113.23 | 146.88 | 169.27 | 191.08 | 122.39 | 158.69 |
| ρ=0.0 | on | 0.0 | 9.27 | 113.55 | 148.24 | 169.66 | 192.49 | 123.11 | 159.44 |
| ρ=0.3 | on | 0.3 | 9.26 | 115.37 | 154.31 | 180.43 | 200.55 | 124.98 | 164.69 |
| ρ=0.7 | on | 0.7 | 9.26 | 119.40 | 160.53 | 191.41 | 213.41 | 129.51 | 172.04 |
| ρ=1.0 | on | 1.0 | 9.26 | 123.07 | 166.28 | 197.00 | 222.72 | 132.51 | 177.23 |

Predicted signs CONFIRMED: AAL flat (Δ=0.12M across all ρ, within MC noise);
OEP-1000 monotone rising: 191.1 < 192.5 < 200.6 < 213.4 < 222.7.
At ρ=1.0 the 1-in-1000 gross OEP is +31.6M (+16.5%) vs the deterministic baseline.
At ρ=0.0 the tail barely moves (+1.4M at OEP-1000), confirming LLN diversification.
**These are sensitivity bounds, not precision claims.** Absolute tail levels depend on the
heuristic logistic vulnerability and uncalibrated CV=0.40.

**Deferrals:**
- CV calibration: deferred to v4 (requires per-event damage data → MDR-dependent CV);
  current CV=0.40 is illustrative only — do NOT cite the ρ-sweep absolute OEP levels as
  production estimates.
- ρ calibration: deferred to v4 (calibrate from multi-event loss runs with observed damage);
  current ρ=0.5 default in config is illustrative.
- wpr=on interaction: wpr=on + uncertainty=on untested; wpr=off remains production default.

**Artifacts:** `outputs/secondary_uncertainty_ep.png` (OEP star figure, 5 configs);
`results/waterfall/sensitivity_secondary.csv` (ρ-sweep table, not versioned).
**tests/test_damage_uncertainty.py:** 5 test classes, 18 tests — TestBitIdentical (bit-identical
10-storm baseline + 5k-year AAL smoke), TestRngDiscipline (hazard stream uncontaminated),
TestCommonShock (shared quantile U at ρ=1, heterogeneous portfolio; portfolio std rises with ρ),
TestBetaMoments (mean unbiased, std≈CV×mean at ρ=0), TestEdgeCases (m=0→dr=0, bounds, clamp).

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
AAL delta. Task 2b threshold decision PORTFOLIO-CONTINGENT: threshold=on (production default with
current deductibles). C−B gross AAL = $0.00 exactly (deductible-absorbed-exactly) —
sub-threshold Manufactured damage ($15,109/year ground-up, gust 65–66.76 mph) fully
absorbed by per-location deductible before gross; re-evaluate when Phase 4 introduces
real OED contracts. Switch engaged confirmed by direct probe (kernel at gust=60:
off→0.527%, on→0.0). WF/Masonry/RC: moot (E[DR]@65 < 0.025%). See Task 2b outcomes below.

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

**Task 2b DONE.** See Task 2b outcomes section below. Task 3 (field validation) DONE — see Task 3 outcomes section. Next step (re-anchoring vs paradigm reconsideration) is a separate decision.

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

Config A bit-identical to 3.0c production baseline. ✓ (validates loss.py kernel-builder refactor — restructured `_event_loss` path is numerically lossless)

**B−A: −$2,155,712 gross AAL (−23.5%)** — paradigm shift, not a small recalibration.
CHARACTERIZED BUT NOT ADJUDICATED: which AAL is closer to truth (logistic vs DS) is
exactly the Task 3 question, to be answered against HAZUS Table 5-46 average damage
states — not assumed in either direction.
The DS scheme has a qualitatively different curve shape (shoulder plateau + steeper rise)
vs. the smooth logistic, with RMSE 0.050–0.067 across classes.

**B−A per-construction decomposition (gross AAL, seed 42, 100k years):**

| Class | A gross AAL ($) | B gross AAL ($) | Delta ($) | Δ% class | Share of Δ |
|-------|-----------------|-----------------|-----------|----------|------------|
| Wood Frame | 3,314,116 | 2,148,461 | −1,165,655 | −35.2% | 54.1% |
| Masonry | 2,459,070 | 1,895,453 | −563,617 | −22.9% | 26.2% |
| Manufactured | 3,122,662 | 2,696,610 | −426,052 | −13.6% | 19.8% |
| RC | 255,373 | 254,984 | −389 | −0.2% | 0.02% |

PREDICTION CORRECTION: predicted Mfg-dominant, measured WF-dominant (54% of −$2.156M,
−35.2% class). WF's deficit window is wide: theta3=145 puts DS3 onset deep into Cat-3
winds, worst RMSE 0.067; the logistic–DS gap spans ~95–160 mph, exactly the most frequent
Cat-2/3 damage band, and WF carries 2.9× more TIV than Manufactured ($126.8M vs $43.7M).
Mfg 20% of delta (−13.6% class): theta3=110 closes the DS3 gap by ~125 mph, narrower
deficit window. Masonry 26% of delta — sub-proportional to its 50.7% TIV share (consistent
with RMSE 0.039, tightest calibration of any class). RC negligible as expected.

**C−B threshold decision — PORTFOLIO-CONTINGENT (deductible-absorbed-exactly, not statistically immaterial):**
- Probe: threshold=off kernel returns E[DR]=0.527% at gust=60 mph for Manufactured;
  threshold=on returns 0.0 — switch engaged correctly
- C ground-up AAL: $9,806,471 | B ground-up: $9,791,362 → **C−B ground-up = +$15,109/year**
  (measured positive; sub-threshold Manufactured damage exists)
- C gross AAL = B gross AAL = **$6,995,508.24 exactly → C−B gross = $0.00**
- Mechanism: gust 65–66.76 mph → Manufactured E[DR] ≈ 0.3–0.8% → ground-up
  ≈ $500–1,600 per location → fully absorbed by per-location deductible before gross.
  Resolution is deductible-absorbed-exactly, not statistically immaterial.
- PORTFOLIO-CONTINGENT: this resolution depends on the current deductible structure —
  re-evaluate when Phase 4 introduces real OED contracts. Default ds_gust_threshold
  decision deferred to Task 3 (when/if DS ships as production).

**Masonry-RC crossover (DS-mean, g < 71 mph):**
RC beta=0.20 fatter DS1 tail exceeds Masonry beta=0.15 for g ∈ [65, 69.8] mph (threshold=on
active range). Analytical g* = 71.0 mph; max violation 2.2e-5 DR at gust=65.71 mph.
Mechanism: per-class betas mean theta ordering does not imply E[DR] ordering — RC's beta=0.20
fat tail exceeds Masonry's beta=0.15 at low winds despite higher medians. Known, negligible
(< 2e-4 DR), accepted as a consequence of the literature-range beta calibration; revisit only
if Task 3 validation cares about the 65–70 mph gust band.
Well below 2e-4 bound. Test `test_cross_class_hierarchy` now asserts: strict hierarchy
g > 72 mph AND max sub-72 violation < 2e-4 (both with explicit assert, not silently weakened).

**141/141 tests pass (at Task 2b commit).** YAML-CSV weld test (`TestYamlCsvWeld`) enforces
bit-equality of YAML thetas/betas to `outputs/fragility_thetas.csv` — future hand-edit of
either file fails loudly.

## Task 3 outcomes — DONE 2026-06-12 (field validation vs HAZUS Elena 1985)
Read-only validation script: `analysis/validate_fragilities.py`. No config change, no
simulation, no production-switch decision. Analytically evaluates P(DS>=3|gust) for
Manufactured Housing at 8 Elena park gusts and compares to field-survey observations.

**Ex-ante acceptance criteria (frozen before model touched the data):**
- Primary: MdAE ≤ 16.455 pts  AND  MAE ≤ 19.560 pts
- Derivation: 1.5 × HAZUS certified model's own errors on the same 8 parks (HAZUS MdAE=10.97, MAE=13.04)

**PRIMARY TEST: FAIL**

| Park | Gust (mph) | z0 | Obs major% | HAZUS% | Ours% | Our err (pts) | HAZUS err (pts) |
|---|---|---|---|---|---|---|---|
| Trav Park Mobile Bay | 109 | 0.3 | 0.0 | 4.0 | 46.7 | **46.7** | 4.0 |
| Old Fort Village | 122 | 0.3 | 14.1 | 17.2 | 82.7 | **68.5** | 3.1 |
| Imperial Estates | 122 | 0.3 | 6.1 | 14.0 | 82.7 | **76.6** | 7.9 |
| Rolling Hills | 122 | 0.3 | 2.0 | 21.0 | 82.7 | **80.7** | 19.0 |
| Isle of Pines North | 124 | 0.3 | 23.0 | 29.7 | 86.2 | **63.2** | 6.7 |
| Isle of Pines South | 124 | 0.3 | 18.0 | 32.0 | 86.2 | **68.2** | 14.0 |
| Trade Winds Dauphin | 126 | 0.1 | 40.0 | 61.6 | 89.2 | **49.2** | 21.6 |
| Anchor Gautier | 126 | 0.3 | 4.0 | 32.0 | 89.2 | **85.2** | 28.0 |

Our MdAE = **68.36 pts** (criterion ≤16.5) | Our MAE = **67.27 pts** (criterion ≤19.6) — ~4× the limit.

**MECHANISM: Conceptual anchoring error — not a calibration miss.**
theta3 was anchored to the logistic midpoint (the wind speed at 50% MEAN damage ratio).
In the DS framework theta3 is the MEDIAN OF DS3 — the wind speed at which 50% PROBABILITY
of extensive/structural damage is reached. These are different physical quantities:
- "50% mean DR" integrates over all damage levels including DS1 and DS2 (light-moderate damage)
- "50% probability of DS3" means half the population experiences structural damage

The Task 2a/2b internal validation all passed (bit-identical, RMSE vs logistic 0.050–0.067,
cross-impl ≤1e-12) because the reference standard was the logistic, which shared the same
anchored midpoint. Internal consistency against a flawed reference cannot detect the error.
Only field validation against independently observed outcomes can.

**Terrain check: CONFIRMED (secondary signal)**
- Rough-terrain parks (z0=0.3): mean error 69.9 pts
- Exposed park (z0=0.1, Dauphin): error 49.2 pts
- Prediction (over-predict more at rough sites) confirmed; terrain bias is real but
  dwarfed by the anchoring bias (~4 vs ~50+ pts per-park).

**Discrimination verdict: DATA CAN DISCRIMINATE**
DS-mean over-predicts by ~4–6× the HAZUS benchmark error. The miss is far outside all
binomial confidence bands (Wilson 95% CIs on observed fractions, n=12–175). This is NOT
"both paradigms consistent with noise." The logistic E[DR] at Elena gusts (48–78%) also
over-predicts observed E[DR] (0.5–43%), but DS-mean P(DS>=3) is the more severely biased
metric for the major-damage comparison.

**Scope caveat:** Manufactured Housing only — single construction class, 8 parks, 1 storm.
WF/Masonry/RC: HAZUS Table 5-46 provides structural-level average damage states but no
equivalent park-level independent field validation for those classes. This finding is
specific to Manufactured Housing and to the major-damage metric.

**Production status:** `logistic_deterministic` remains default. The adjudication question
(which paradigm is closer to truth, and how to re-anchor theta3 if DS-mean is retained)
is a separate decision deferred to the user after reviewing this validation.

**Artifacts:** `outputs/fragility_validation.png` (3-panel: primary major-fraction per park
with binomial CI + HAZUS overlay; 5-class DS diagnostic for Trav Park and Dauphin; medians
vs Table 5-38 reference bands); `results/fragility_validation.csv` (not versioned).
**165/165 tests pass.** tests/test_fragility_validation.py: 24 tests (dataset integrity ×16,
major-fraction hand-checks ×3, determinism ×1, criterion constants ×2). Suite green at this commit.

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