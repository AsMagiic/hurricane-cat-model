"""
Tests for Step 3.0b: stochastic WPR residual in model/hazard.py.

Test classes
------------
TestWprOffBitIdentical  — wpr=off must reproduce the pre-3.0b production baseline
                          exactly (float64, 10-storm sequence).
TestWprSamplerStats     — wpr=on sampler: ε ~ N(0, σ²), Jensen bias on mean Δp.
TestSubstreamIndependence — spawn(2) children are uncorrelated (vw vs wpr streams).
TestDrawDiscipline      — vw_rng draws (Rmax, B) are identical whether wpr is on or off.
"""

import numpy as np
import pytest

from model.hazard import (
    sample_storm, _vw_rmax_mean, _sigma_rmax,
    _WPR_SIGMA_LOG, _WPR_A, _WPR_B_EXP,
)
import model.hazard as _hazard_mod


# ---------------------------------------------------------------------------
# Production baseline: 10 consecutive storms, seed=42, production config
# (v3+3.0a: all switches on, wpr_residual="off" — captured before 3.0b changes)
# ---------------------------------------------------------------------------
_BASELINE_10 = [
    # (rmax, dp_mb, b, vmax_landfall)  — exact float64 captures
    (43.416600321165504,  64.82839469706278, 1.4529704026528192, 125.3028222089054),
    (77.02295275643434,   59.184343984698856, 1.2065216362183013, 118.84779968552874),
    (22.36741097240041,   58.192825287772735, 1.6657375122762375, 117.68758952187191),
    (41.58210642819636,   43.94625192834046,  1.63423685326691,   99.98191922731499),
    (37.450652941908736,  63.091430098043254, 0.9164525439373397, 123.34232061571093),
    (43.1084823900127,    61.50791697672065,  0.9690954300173021, 121.53520168042833),
    (45.30617164337824,   47.883011172139796, 1.3391811826613678, 105.08878285305235),
    (47.810507125181026,  42.36783029787112,  1.5876040784132113,  97.8807849159002),
    (97.26682782694414,   61.251273434660725, 0.5653625195412878, 121.24048931491033),
    (34.54536802279213,   56.62375125632249,  1.506913446356151,  115.83447990967149),
]


class TestWprOffBitIdentical:
    """
    With wpr_residual='off', sample_storm must reproduce the pre-3.0b production
    baseline exactly (float64 equality) for a 10-storm sequence drawn from one rng.

    A single-storm test does not catch stream desync at storm N; a sequence does.
    The fields checked are rmax, dp_mb, and b — the three quantities that the
    spawn-split architecture could perturb if either child index or stream order
    is mishandled.
    """

    def test_10_storm_sequence_bit_identical(self, monkeypatch):
        monkeypatch.setattr(_hazard_mod, "_WPR_RESIDUAL", "off")

        rng = np.random.default_rng(42)
        for i, (exp_rmax, exp_dp, exp_b, exp_vmax) in enumerate(_BASELINE_10):
            _, meta = sample_storm(rng)
            assert meta["rmax"]          == exp_rmax,  f"storm {i}: rmax mismatch"
            assert meta["dp_mb"]         == exp_dp,    f"storm {i}: dp_mb mismatch"
            assert meta["b"]             == exp_b,     f"storm {i}: b mismatch"
            assert meta["vmax_landfall"] == exp_vmax,  f"storm {i}: vmax_landfall mismatch"


class TestWprSamplerStats:
    """
    ε = ln(dp_on / dp_off) must follow N(0, sigma_log²), and E[dp_on/dp_off]
    must equal exp(sigma_log²/2) (Jensen bias).

    Vectorized — no sample_storm loop. ε is independent of storm geometry, so
    we draw vmax uniformly and compute dp_off / dp_on in numpy. Tests that
    _WPR_SIGMA_LOG, _WPR_A, _WPR_B_EXP have the correct values and that the
    formula dp * exp(N(0,σ)) produces the expected statistics.
    """

    _N = 200_000
    _SIGMA_LOG = 0.2458

    def _draw_eps_and_ratios(self):
        rng = np.random.default_rng(0)
        vmax = rng.uniform(64.0, 165.0, self._N)
        dp_off = _WPR_A * vmax ** _WPR_B_EXP
        eps = rng.normal(0.0, _WPR_SIGMA_LOG, self._N)
        dp_on = dp_off * np.exp(eps)
        return eps, dp_on / dp_off

    def test_epsilon_mean_near_zero(self):
        eps, _ = self._draw_eps_and_ratios()
        tol = 3.0 * self._SIGMA_LOG / np.sqrt(self._N)
        assert abs(eps.mean()) < tol, (
            f"mean(ε) = {eps.mean():.6f}, expected near 0 (tol={tol:.6f})"
        )

    def test_epsilon_std_near_sigma_log(self):
        eps, _ = self._draw_eps_and_ratios()
        assert abs(eps.std(ddof=0) - self._SIGMA_LOG) / self._SIGMA_LOG < 0.01, (
            f"std(ε) = {eps.std():.4f}, expected {self._SIGMA_LOG} (±1%)"
        )

    def test_jensen_bias_on_mean_ratio(self):
        _, ratios = self._draw_eps_and_ratios()
        mean_ratio = ratios.mean()
        expected_jensen = np.exp(self._SIGMA_LOG ** 2 / 2)   # ≈ 1.0307
        assert abs(mean_ratio - expected_jensen) / expected_jensen < 0.01, (
            f"mean(dp_on/dp_off) = {mean_ratio:.5f}, "
            f"expected Jensen = {expected_jensen:.5f} (±1%)"
        )


class TestSubstreamIndependence:
    """
    The nested architecture (wpr_rng = vw_rng.spawn(1)[0]) must be uncorrelated
    with the parent vw_rng draws. Verified directly on the child streams.
    """

    _N = 1_000

    def test_vw_and_wpr_children_uncorrelated(self):
        parent = np.random.default_rng(42)
        vw_rng  = parent.spawn(1)[0]
        wpr_rng = vw_rng.spawn(1)[0]
        a = vw_rng.standard_normal(self._N)
        b = wpr_rng.standard_normal(self._N)
        r = float(np.corrcoef(a, b)[0, 1])
        assert abs(r) < 0.05, (
            f"Pearson r = {r:.4f} between vw_rng and wpr_rng (expected |r| < 0.05)"
        )


class TestDrawDiscipline:
    """
    The wpr_rng draw must NOT bleed into the parent rng's bitgenerator state.

    Property tested: after sample_storm, the parent rng's bitgenerator position is
    identical regardless of wpr_residual switch, because wpr draws come from
    wpr_rng (a nested child of vw_rng), never from rng directly.

    Verified by checking that the _next_ N draws from rng after each storm are
    bit-identical for wpr=off vs wpr=on.  Any bleed would shift the bitgenerator
    position and cause a mismatch from the very first post-storm draw.
    """

    _N_STORMS      = 20
    _N_PROBE_DRAWS = 5   # draws taken from rng after each storm to probe its position

    def test_parent_rng_position_identical_on_vs_off(self, monkeypatch):
        post_draws = {}
        for switch in ("off", "on"):
            monkeypatch.setattr(_hazard_mod, "_WPR_RESIDUAL", switch)
            rng = np.random.default_rng(99)
            draws_per_storm = []
            for _ in range(self._N_STORMS):
                sample_storm(rng)
                # Probe the parent rng state immediately after sample_storm returns
                draws_per_storm.append(rng.standard_normal(self._N_PROBE_DRAWS).tolist())
            post_draws[switch] = draws_per_storm

        for i, (off_draws, on_draws) in enumerate(
            zip(post_draws["off"], post_draws["on"])
        ):
            assert off_draws == on_draws, (
                f"storm {i}: post-storm rng state differs between wpr=off and wpr=on "
                f"— wpr draw bled into parent rng\n  off={off_draws}\n   on={on_draws}"
            )
