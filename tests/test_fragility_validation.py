"""
Tests for analysis/validate_fragilities.py.

Focus:
  1. Dataset integrity: obs and hazus rows normalize within tolerance of 100.
  2. Major-fraction hand-checks on 2 parks (deterministic given known params).
  3. Determinism: compute_park_predictions produces identical output on repeat calls.
  4. Criterion constants: 1.5× HAZUS relationship is exact.
"""

import os
import sys

import numpy as np
import pytest
from scipy.special import ndtr

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from analysis.validate_fragilities import (
    _CRITERION_MAE,
    _CRITERION_MDAE,
    _ELENA_PARKS,
    _HAZUS_MAE,
    _HAZUS_MDAE,
    compute_park_predictions,
)
from calibration.fragility_thetas import exceedance_probs
from model_config import load_model_cfg


def _get_mfg_params():
    mcfg = load_model_cfg()
    ds = mcfg.vulnerability.damage_states
    mfg = ds["Manufactured"]
    thetas = np.array(mfg["thetas"])
    beta = float(mfg["beta"])
    cp = mcfg.vulnerability.construction_params
    logistic_params = {
        "cap":      float(cp["Manufactured"]["cap"]),
        "midpoint": float(cp["Manufactured"]["midpoint"]),
        "k":        float(cp["Manufactured"]["k"]),
    }
    return thetas, beta, logistic_params


# ---------------------------------------------------------------------------

class TestDataIntegrity:

    @pytest.mark.parametrize("park", _ELENA_PARKS)
    def test_obs_sums_near_100(self, park):
        """Observed DS percentages sum to 95-105 (source may round independently)."""
        s = sum(park["obs"])
        assert 95 <= s <= 105, (
            f"{park['name']}: obs sums to {s}, expected 95-105"
        )

    @pytest.mark.parametrize("park", _ELENA_PARKS)
    def test_hazus_sums_near_100(self, park):
        """HAZUS DS percentages sum to 90-105 (source rounding)."""
        s = sum(park["hazus"])
        assert 90 <= s <= 105, (
            f"{park['name']}: hazus sums to {s}, expected 90-105"
        )

    def test_eight_parks(self):
        assert len(_ELENA_PARKS) == 8

    def test_n_values_in_range(self):
        """Park unit counts are in the source-stated range 12-175."""
        for park in _ELENA_PARKS:
            assert 12 <= park["n"] <= 175, (
                f"{park['name']}: n={park['n']} outside [12,175]"
            )


# ---------------------------------------------------------------------------

class TestMajorFractionHandCheck:
    """Verify compute_park_predictions against direct ndtr calls."""

    def test_trav_park_major(self):
        """
        Trav Park Mobile Bay: gust=109 mph, Manufactured theta3=110.0, beta=0.11.
        P(DS>=3|109) = ndtr(ln(109/110.0)/0.11).
        """
        thetas, beta, logistic_params = _get_mfg_params()
        expected = float(ndtr(np.log(109.0 / thetas[2]) / beta))
        # Also verify via exceedance_probs directly
        computed = float(exceedance_probs(np.array([109.0]), thetas, beta)[2, 0])
        assert abs(computed - expected) < 1e-10

        # And via compute_park_predictions
        parks = [p for p in _ELENA_PARKS if p["name"] == "Trav Park Mobile Bay"]
        results = compute_park_predictions(parks, thetas, beta, logistic_params)
        assert abs(results[0]["our_major"] / 100 - expected) < 1e-10

    def test_dauphin_major(self):
        """
        Trade Winds Dauphin: gust=126 mph, Manufactured theta3=110.0, beta=0.11.
        P(DS>=3|126) = ndtr(ln(126/110.0)/0.11).
        """
        thetas, beta, logistic_params = _get_mfg_params()
        expected = float(ndtr(np.log(126.0 / thetas[2]) / beta))
        computed = float(exceedance_probs(np.array([126.0]), thetas, beta)[2, 0])
        assert abs(computed - expected) < 1e-10

        parks = [p for p in _ELENA_PARKS if p["name"] == "Trade Winds Dauphin"]
        results = compute_park_predictions(parks, thetas, beta, logistic_params)
        assert abs(results[0]["our_major"] / 100 - expected) < 1e-10

    def test_our_major_exceeds_hazus_threshold_at_high_gust(self):
        """At 122-126 mph our P(DS>=3) must exceed 50% given theta3=110 mph < gust."""
        thetas, beta, logistic_params = _get_mfg_params()
        for park in _ELENA_PARKS:
            if park["gust"] >= 122:
                p = float(exceedance_probs(
                    np.array([float(park["gust"])]), thetas, beta
                )[2, 0])
                assert p > 0.5, (
                    f"Expected P(DS>=3)>50% at g={park['gust']} mph "
                    f"since theta3={thetas[2]}<gust; got {p:.3f}"
                )


# ---------------------------------------------------------------------------

class TestDeterminism:

    def test_compute_park_predictions_identical_on_repeat(self):
        """Pure function: same inputs → identical outputs."""
        thetas, beta, logistic_params = _get_mfg_params()
        r1 = compute_park_predictions(_ELENA_PARKS, thetas, beta, logistic_params)
        r2 = compute_park_predictions(_ELENA_PARKS, thetas, beta, logistic_params)
        for a, b in zip(r1, r2):
            assert a["our_major"] == b["our_major"]
            assert a["obs_major"] == b["obs_major"]
            assert a["our_err"]   == b["our_err"]


# ---------------------------------------------------------------------------

class TestCriterionConstants:

    def test_criterion_is_1_5x_hazus(self):
        """Acceptance criteria are exactly 1.5× HAZUS own errors — frozen ex-ante."""
        assert abs(_CRITERION_MDAE - 1.5 * _HAZUS_MDAE) < 1e-9
        assert abs(_CRITERION_MAE  - 1.5 * _HAZUS_MAE)  < 1e-9

    def test_hazus_errors_match_spec(self):
        """HAZUS benchmark errors match the spec values used in criterion derivation."""
        assert abs(_HAZUS_MDAE - 10.97) < 0.01
        assert abs(_HAZUS_MAE  - 13.04) < 0.01
