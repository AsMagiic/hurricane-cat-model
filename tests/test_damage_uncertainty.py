"""
Tests for Step 3.1: secondary uncertainty with common-shock spatial correlation.

Five test classes:
  TestBitIdentical    -- uncertainty=off reproduces deterministic baseline exactly
  TestRngDiscipline   -- hazard rng stream uncontaminated (damage uses separate chain)
  TestCommonShock     -- shared quantile U under rho=1 (heterogeneous portfolio)
  TestBetaMoments     -- mean preserved, std ~ CV * mean (rho=0)
  TestEdgeCases       -- m=0 -> dr=0; [0,1] bounds; rho=1 fattens tail
"""

import numpy as np
import pytest
from scipy.special import betainc

import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import model.loss as _loss
from model.hazard import simulate_year

# ---------------------------------------------------------------------------
# 10-storm deterministic baseline (seed=42, uncertainty=off)
# ---------------------------------------------------------------------------
_BASELINE_10 = [
    (0.0, 0.0),
    (106205838.07609516, 84810152.78016408),
    (10786099.073262163, 6149425.553957316),
    (0.0, 0.0),
    (53118129.40082713, 39249444.656099275),
    (0.0, 0.0),
    (0.0, 0.0),
    (2203488.1104949424, 938664.7448912609),
    (37203827.86753815, 23624440.60258361),
    (7008304.487608565, 2202599.6645272905),
]


# ---------------------------------------------------------------------------
# Helper: collect first N portfolio losses under given damage_uncertainty switch
# ---------------------------------------------------------------------------
def _collect_n_event_losses(n, seed, uncertainty, rho=None, cv=None):
    """Run with given settings; return list of (ground_up, gross) sums for first n events."""
    from model.wind_field import wind_at_locations, StormParams

    old_du  = _loss._DAMAGE_UNCERTAINTY
    old_rho = _loss._DAMAGE_RHO
    old_cv  = _loss._DAMAGE_CV
    _loss._DAMAGE_UNCERTAINTY = uncertainty
    if rho is not None:
        _loss._DAMAGE_RHO = rho
    if cv is not None:
        _loss._DAMAGE_CV = cv

    rng        = np.random.default_rng(seed)
    damage_rng = np.random.default_rng([seed, 1])  # two-integer entropy, matching run_simulation

    results = []
    yr = 0
    while len(results) < n:
        yr += 1
        events = simulate_year(rng)
        for track, meta in events:
            winds = wind_at_locations(
                track,
                StormParams(rmax=meta["rmax"], b=meta["b"], dp_mb=meta["dp_mb"],
                            lat=meta["landfall_lat"], heading_deg=meta["heading_deg"],
                            vt_kmh=meta["translation_speed_kmh"]),
                _loss.lats, _loss.lons,
            )
            dmg_r = damage_rng if uncertainty == "on" else None
            gu, gr = _loss._event_loss(winds, dmg_r)
            results.append((float(gu.sum()), float(gr.sum())))
            if len(results) >= n:
                break

    _loss._DAMAGE_UNCERTAINTY = old_du
    _loss._DAMAGE_RHO = old_rho
    _loss._DAMAGE_CV  = old_cv
    return results


class TestBitIdentical:
    """uncertainty=off must be byte-for-byte identical to the deterministic baseline."""

    def test_10_event_ground_up_bit_identical(self, monkeypatch):
        monkeypatch.setattr(_loss, "_DAMAGE_UNCERTAINTY", "off")
        results = _collect_n_event_losses(10, seed=42, uncertainty="off")
        for i, ((exp_gu, exp_gr), (got_gu, got_gr)) in enumerate(zip(_BASELINE_10, results)):
            assert got_gu == exp_gu, f"event {i}: ground_up mismatch {got_gu} != {exp_gu}"
            assert got_gr == exp_gr, f"event {i}: gross mismatch {got_gr} != {exp_gr}"

    def test_smoke_run_aal_matches_3_0c_baseline(self, monkeypatch):
        """run_simulation with uncertainty=off must reproduce the 3.0c AAL baseline."""
        monkeypatch.setattr(_loss, "_DAMAGE_UNCERTAINTY", "off")
        # Use 5k years for speed; should be within 2% of the 100k baseline
        events_df, annual_df, *_ = _loss.run_simulation(n_years=5_000, seed=42)
        aal = float(annual_df["aggregate_gross"].mean())
        # 3.0c production AAL = 9,151,220; 5k-year estimate has noise, allow 5%
        assert abs(aal - 9_151_220) / 9_151_220 < 0.05, \
            f"smoke AAL {aal:,.0f} deviates > 5% from 3.0c baseline 9,151,220"


class TestRngDiscipline:
    """Hazard rng stream must be unaffected by whether damage uncertainty is on or off."""

    _N_YEARS = 20

    def test_hazard_stream_uncontaminated(self, monkeypatch):
        """
        Probe rng draws immediately after each simulate_year call.
        damage_rng uses two-integer entropy [seed, 1] — a completely separate
        SeedSequence tree from rng (seeded with single integer seed). rng's
        spawn counter is 0 before the year loop; storm slots are unchanged
        from pre-3.1. rng's PCG64 state after simulate_year must be identical
        with uncertainty=off vs uncertainty=on.
        """
        probe_draws = {}
        for switch in ("off", "on"):
            monkeypatch.setattr(_loss, "_DAMAGE_UNCERTAINTY", switch)
            rng = np.random.default_rng(99)
            yearly_probes = []
            for _ in range(self._N_YEARS):
                simulate_year(rng)
                # Probe the hazard rng's next 4 draws — must be switch-independent
                yearly_probes.append(rng.standard_normal(4).tolist())
            probe_draws[switch] = yearly_probes

        for yr, (off_vals, on_vals) in enumerate(
            zip(probe_draws["off"], probe_draws["on"])
        ):
            assert off_vals == on_vals, \
                f"Year {yr}: hazard rng probe differs between off and on: {off_vals} vs {on_vals}"


class TestCommonShock:
    """
    rho=1 -> all locations share the same quantile U (common shock).
    Tests with a heterogeneous portfolio to distinguish common-shock from homogeneity.
    Catches the #1 bug: drawing Z per-location instead of once per event.

    Portfolio means chosen to stay below the variance-cap boundary (m < 0.86 for cv=0.40)
    so Beta parameters are numerically well-conditioned for the round-trip test.
    """

    _DR_MEAN = np.array([0.10, 0.30, 0.50, 0.65, 0.80])
    _CV      = 0.40

    def test_shared_quantile_rho1(self, monkeypatch):
        """
        Under rho=1, recovered U_i = betainc(alpha_i, beta_i, realized_i) must be
        identical across all locations — they all received the same Z_event.
        Realized DR values must differ (different Beta distributions for different means).
        """
        monkeypatch.setattr(_loss, "_DAMAGE_RHO", 1.0)
        monkeypatch.setattr(_loss, "_DAMAGE_CV",  self._CV)
        monkeypatch.setattr(_loss, "n_loc",        len(self._DR_MEAN))

        rng      = np.random.default_rng(7)
        realized = _loss._damage_draw(self._DR_MEAN, rng)

        # Compute alpha, beta for each location
        alpha, beta_p = _loss._beta_params(self._DR_MEAN, self._CV)

        # Recover the quantile used at each location: U_i = betainc(alpha_i, beta_i, realized_i)
        recovered_u = betainc(alpha, beta_p, realized)

        # All recovered U must be equal (common shock -> single Z -> single Phi(Z))
        assert np.allclose(recovered_u, recovered_u[0], atol=1e-8), \
            f"rho=1 must yield identical U across all locations; got {recovered_u}"

        # Realized DRs must differ (each Beta evaluated at same percentile of its own dist)
        assert not np.allclose(realized, realized[0], atol=1e-4), \
            f"Heterogeneous portfolio must produce distinct realized DR; got {realized}"

    def test_comomovement_std_rho1_gt_rho0(self, monkeypatch):
        """
        Portfolio-level DR std at rho=1 must exceed rho=0 (common shock fattens tail).
        Portfolio mean must be preserved at both rho values (Beta is mean-preserving).
        """
        monkeypatch.setattr(_loss, "_DAMAGE_CV",  self._CV)
        monkeypatch.setattr(_loss, "n_loc",        len(self._DR_MEAN))
        det_total = float(self._DR_MEAN.sum())
        n_events  = 500

        port_totals = {}
        for rho_val, key in [(1.0, "rho1"), (0.0, "rho0")]:
            monkeypatch.setattr(_loss, "_DAMAGE_RHO", rho_val)
            rng = np.random.default_rng(13)
            port_totals[key] = np.array([
                _loss._damage_draw(self._DR_MEAN, rng).sum()
                for _ in range(n_events)
            ])

        assert port_totals["rho1"].std() > port_totals["rho0"].std(), \
            "rho=1 must produce higher portfolio DR std than rho=0"

        for key in ("rho1", "rho0"):
            mean_err = abs(port_totals[key].mean() - det_total) / det_total
            assert mean_err < 0.05, \
                f"{key}: portfolio mean {port_totals[key].mean():.4f} deviates " \
                f"{mean_err*100:.1f}% from deterministic {det_total:.4f}"


class TestBetaMoments:
    """
    With rho=0, independent noise washes out at portfolio level.
    Individual location moments: realized mean ≈ deterministic mean, std ≈ CV * mean.
    """

    _N_EVENTS = 800
    _DR_MEAN  = np.array([0.15, 0.30, 0.50, 0.70])
    _CV       = 0.40

    def test_mean_unbiased(self, monkeypatch):
        monkeypatch.setattr(_loss, "_DAMAGE_RHO", 0.0)
        monkeypatch.setattr(_loss, "_DAMAGE_CV",  self._CV)
        monkeypatch.setattr(_loss, "n_loc",        len(self._DR_MEAN))

        rng = np.random.default_rng(21)
        draws = np.array([_loss._damage_draw(self._DR_MEAN, rng) for _ in range(self._N_EVENTS)])
        # draws shape: (n_events, n_locs)
        realized_mean = draws.mean(axis=0)
        for i, (exp_m, got_m) in enumerate(zip(self._DR_MEAN, realized_mean)):
            rel_err = abs(got_m - exp_m) / exp_m
            assert rel_err < 0.05, \
                f"loc {i}: realized mean {got_m:.4f} deviates {rel_err*100:.1f}% from {exp_m:.4f}"

    def test_std_approx_cv_times_mean(self, monkeypatch):
        monkeypatch.setattr(_loss, "_DAMAGE_RHO", 0.0)
        monkeypatch.setattr(_loss, "_DAMAGE_CV",  self._CV)
        monkeypatch.setattr(_loss, "n_loc",        len(self._DR_MEAN))

        # Exclude extreme locations (m near 0 or 1) where Beta variance is capped
        dr_mid = np.array([0.25, 0.50])
        monkeypatch.setattr(_loss, "n_loc", len(dr_mid))
        rng = np.random.default_rng(22)
        draws = np.array([_loss._damage_draw(dr_mid, rng) for _ in range(self._N_EVENTS)])
        realized_std  = draws.std(axis=0)
        expected_std  = self._CV * dr_mid
        for i, (exp_s, got_s) in enumerate(zip(expected_std, realized_std)):
            rel_err = abs(got_s - exp_s) / exp_s
            assert rel_err < 0.15, \
                f"loc {i}: realized std {got_s:.4f} deviates {rel_err*100:.1f}% from {exp_s:.4f}"


class TestEdgeCases:
    """Edge case guards: m=0 -> dr=0; outputs in [0,1]; clamp on u works."""

    def test_zero_mean_stays_zero(self, monkeypatch):
        monkeypatch.setattr(_loss, "_DAMAGE_RHO", 0.5)
        monkeypatch.setattr(_loss, "_DAMAGE_CV",  0.40)
        n = 20
        monkeypatch.setattr(_loss, "n_loc", n)

        dr_mean = np.zeros(n)
        rng = np.random.default_rng(0)
        for _ in range(50):
            realized = _loss._damage_draw(dr_mean, rng)
            assert np.all(realized == 0.0), f"m=0 locations must always produce dr=0; got {realized}"

    def test_outputs_in_unit_interval(self, monkeypatch):
        monkeypatch.setattr(_loss, "_DAMAGE_RHO", 0.7)
        monkeypatch.setattr(_loss, "_DAMAGE_CV",  0.40)
        n = 50
        monkeypatch.setattr(_loss, "n_loc", n)
        rng = np.random.default_rng(33)

        for trial in range(100):
            dr_mean = rng.uniform(0.0, 0.95, n)
            dr_mean[:5] = 0.0   # include some zeros
            realized = _loss._damage_draw(dr_mean, rng)
            assert np.all(realized >= 0.0) and np.all(realized <= 1.0), \
                f"trial {trial}: realized DR outside [0,1]: min={realized.min()}, max={realized.max()}"

    def test_beta_params_positive_alpha_beta(self):
        """_beta_params must return strictly positive alpha and beta for all m in (0,1)."""
        m = np.linspace(0.01, 0.99, 200)
        alpha, beta_p = _loss._beta_params(m, cv=0.40)
        assert np.all(alpha  > 0), f"alpha has non-positive values: {alpha[alpha <= 0]}"
        assert np.all(beta_p > 0), f"beta has non-positive values: {beta_p[beta_p <= 0]}"
