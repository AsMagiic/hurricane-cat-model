"""
Characterization and contract tests for the loss kernel refactor (v4 terrain seam).

Golden vectors were captured from model/loss._event_loss on HEAD (before the refactor)
using seed-99 wind draws and a seed-77 damage RNG, then stored in
tests/fixtures/golden_event_loss.npz.  These freeze the kernel math so any numerical
drift in compute_event_loss — not just a forwarding bug — fails immediately, without
waiting for the 100k-year run_all.py.
"""

import os
import numpy as np
import pytest

# Allow importing model package from project root
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import model.loss as _loss

_FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "golden_event_loss.npz")


@pytest.fixture(scope="module")
def golden():
    data = np.load(_FIXTURE)
    return data


@pytest.fixture(scope="module")
def test_winds(golden):
    return golden["test_winds"]


# ---------------------------------------------------------------------------
# TestGoldenDeterministic
# ---------------------------------------------------------------------------

class TestGoldenDeterministic:
    """
    compute_event_loss (deterministic path, dmg_rng=None) must reproduce the
    golden vectors captured from _event_loss on HEAD, element-wise.
    """

    def _call_pure_kernel(self, winds):
        gust_factors = np.full(len(_loss.tivs), _loss.GUST_FACTOR)
        gu, gr, dr = _loss.compute_event_loss(
            winds,
            tivs=_loss.tivs,
            deductibles=_loss.deductibles,
            pol_limits=_loss.pol_limits,
            gust_factors=gust_factors,
            vuln_kernel=_loss._vuln_kernel,
        )
        return gu, gr, dr

    def test_ground_up_matches_golden(self, golden, test_winds):
        gu, _, _ = self._call_pure_kernel(test_winds)
        np.testing.assert_array_equal(gu, golden["det_ground_up"])

    def test_gross_matches_golden(self, golden, test_winds):
        _, gr, _ = self._call_pure_kernel(test_winds)
        np.testing.assert_array_equal(gr, golden["det_gross"])


# ---------------------------------------------------------------------------
# TestGoldenStochastic
# ---------------------------------------------------------------------------

class TestGoldenStochastic:
    """
    compute_event_loss (stochastic path, dmg_rng=fixed-seed) must reproduce the
    golden vectors captured from _event_loss on HEAD with the same RNG seed.

    Note: the dmg_rng path calls _damage_draw, which closes over module-level
    n_loc / _DAMAGE_CV / _DAMAGE_RHO.  The golden is valid only for the current
    module configuration; changing those config values will (correctly) break it.
    """

    def _call_pure_kernel_stochastic(self, winds):
        gust_factors = np.full(len(_loss.tivs), _loss.GUST_FACTOR)
        dmg_rng = np.random.default_rng(77)   # same seed used when capturing the golden
        gu, gr, dr = _loss.compute_event_loss(
            winds,
            tivs=_loss.tivs,
            deductibles=_loss.deductibles,
            pol_limits=_loss.pol_limits,
            gust_factors=gust_factors,
            vuln_kernel=_loss._vuln_kernel,
            dmg_rng=dmg_rng,
        )
        return gu, gr, dr

    def test_ground_up_matches_golden(self, golden, test_winds):
        gu, _, _ = self._call_pure_kernel_stochastic(test_winds)
        np.testing.assert_array_equal(gu, golden["sto_ground_up"])

    def test_gross_matches_golden(self, golden, test_winds):
        _, gr, _ = self._call_pure_kernel_stochastic(test_winds)
        np.testing.assert_array_equal(gr, golden["sto_gross"])


# ---------------------------------------------------------------------------
# TestScalarVsUniformArrayGust
# ---------------------------------------------------------------------------

class TestScalarVsUniformArrayGust:
    """
    Scalar-broadcast gust and np.full(n, GUST_FACTOR) gust are element-wise equal.
    Documents the invariant the wrapper relies on.
    """

    def test_gust_elementwise_equal(self, test_winds):
        scalar_gust = test_winds * _loss.GUST_FACTOR
        array_gust  = test_winds * np.full(len(test_winds), _loss.GUST_FACTOR)
        np.testing.assert_array_equal(scalar_gust, array_gust)


# ---------------------------------------------------------------------------
# TestDrReturn
# ---------------------------------------------------------------------------

class TestDrReturn:
    """compute_event_loss returns dr as a valid (n_loc,) array."""

    def _pure_kernel(self, winds):
        gust_factors = np.full(len(_loss.tivs), _loss.GUST_FACTOR)
        return _loss.compute_event_loss(
            winds,
            tivs=_loss.tivs,
            deductibles=_loss.deductibles,
            pol_limits=_loss.pol_limits,
            gust_factors=gust_factors,
            vuln_kernel=_loss._vuln_kernel,
        )

    def test_dr_shape(self, test_winds):
        _, _, dr = self._pure_kernel(test_winds)
        assert dr.shape == (_loss.n_loc,)

    def test_dr_bounds(self, test_winds):
        _, _, dr = self._pure_kernel(test_winds)
        assert (dr >= 0.0).all(), "dr has values below 0"
        assert (dr <= 1.0).all(), "dr has values above 1"

    def test_dr_consistent_with_ground_up(self, test_winds):
        """ground_up == dr * tivs element-wise (verifies the kernel arithmetic)."""
        gu, _, dr = self._pure_kernel(test_winds)
        np.testing.assert_array_equal(gu, dr * _loss.tivs)
