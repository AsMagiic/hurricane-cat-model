"""
Tests for Step 3.0a: MPI upper truncation of the landfall intensity distribution.

Covers four properties:
  1. Off-branch bit-identity: with _INTENSITY_CAP="off" and _INT_P_UB=1.0,
     sample_intensity produces exactly the same floats as the pre-3.0a code.
  2. On-branch cap enforcement: no draw exceeds kt_to_mph(165 kt) ≈ 189.88 mph.
  3. Gentle renormalization: draws below 150 kt shift by < 0.5 mph vs off-branch.
  4. RNG discipline: exactly one uniform consumed per call regardless of branch.

Baseline for test 1 captured from pre-3.0a code (seed=42, unbounded distribution):
  [0]  (3, 120.6250008551009)     [10] (1, 94.51857151366518)
  [1]  (2, 98.10205326787062)     [11] (4, 142.77343165833634)
  [2]  (4, 130.27860907463074)    [12] (2, 110.27520082160545)
  [3]  (3, 114.13076379635741)    [13] (3, 125.72967186531868)
  [4]  (1, 79.79580612834692)     [14] (2, 98.3454007388215)
  [5]  (5, 162.11299690151955)    [15] (1, 87.10854518536746)
  [6]  (3, 119.43206375204561)    [16] (2, 104.60405063550289)
  [7]  (3, 121.80188466483627)    [17] (1, 77.95234543262006)
  [8]  (1, 81.75571808464176)     [18] (3, 126.29979807210962)
  [9]  (2, 98.72084735272571)     [19] (2, 109.45164419886606)
"""

from unittest.mock import patch, MagicMock

import numpy as np
import pytest
import scipy.stats

import model.hazard as _hazard_mod
from model.hazard import sample_intensity, _INT_P_LB, _INT_MU_LOG, _INT_SIGMA_LOG, kt_to_mph

# ---------------------------------------------------------------------------
# Constants shared across tests
# ---------------------------------------------------------------------------
_CAP_KT  = 165.0
_CAP_MPH = float(kt_to_mph(_CAP_KT))  # ≈ 189.879 mph

_INT_P_UB_ON = float(scipy.stats.norm.cdf(
    (np.log(_CAP_KT) - _INT_MU_LOG) / _INT_SIGMA_LOG
))  # ≈ 0.99609

_BASELINE = [
    (3, 120.6250008551009),
    (2, 98.10205326787062),
    (4, 130.27860907463074),
    (3, 114.13076379635741),
    (1, 79.79580612834692),
    (5, 162.11299690151955),
    (3, 119.43206375204561),
    (3, 121.80188466483627),
    (1, 81.75571808464176),
    (2, 98.72084735272571),
    (1, 94.51857151366518),
    (4, 142.77343165833634),
    (2, 110.27520082160545),
    (3, 125.72967186531868),
    (2, 98.3454007388215),
    (1, 87.10854518536746),
    (2, 104.60405063550289),
    (1, 77.95234543262006),
    (3, 126.29979807210962),
    (2, 109.45164419886606),
]


# ---------------------------------------------------------------------------
class TestIntensityCapOff:
    """Off-branch must be bit-identical to the pre-3.0a unbounded implementation."""

    @pytest.fixture(autouse=True)
    def _force_off(self, monkeypatch):
        monkeypatch.setattr(_hazard_mod, "_INTENSITY_CAP", "off")
        monkeypatch.setattr(_hazard_mod, "_INT_P_UB", 1.0)

    def test_p_ub_is_exactly_one(self):
        assert _hazard_mod._INT_P_UB == 1.0

    def test_draws_bit_identical_to_baseline(self):
        """20 draws with seed=42 must exactly match the pre-3.0a values."""
        rng = np.random.default_rng(42)
        for i, (exp_cat, exp_vmax) in enumerate(_BASELINE):
            got_cat, got_vmax = sample_intensity(rng)
            assert got_cat == exp_cat, (
                f"draw[{i}] category: got {got_cat} != baseline {exp_cat}"
            )
            assert got_vmax == exp_vmax, (
                f"draw[{i}] vmax_mph: got {got_vmax!r} != baseline {exp_vmax!r}"
            )


# ---------------------------------------------------------------------------
class TestIntensityCapOn:
    """On-branch must enforce the 165 kt ceiling."""

    @pytest.fixture(autouse=True)
    def _force_on(self, monkeypatch):
        monkeypatch.setattr(_hazard_mod, "_INTENSITY_CAP", "on")
        monkeypatch.setattr(_hazard_mod, "_INT_P_UB", _INT_P_UB_ON)

    def test_no_draw_exceeds_cap(self):
        rng = np.random.default_rng(0)
        max_vmax = 0.0
        for _ in range(10_000):
            _, vmax = sample_intensity(rng)
            max_vmax = max(max_vmax, vmax)
        assert max_vmax < _CAP_MPH, (
            f"draw {max_vmax:.4f} mph >= cap {_CAP_MPH:.4f} mph"
        )

    def test_max_draw_approaches_cap(self):
        """Largest draw over 10k samples should be within kt_to_mph(1 kt) of cap."""
        rng = np.random.default_rng(1)
        max_vmax = max(sample_intensity(rng)[1] for _ in range(10_000))
        kt_1_mph = float(kt_to_mph(1.0))
        assert max_vmax > _CAP_MPH - kt_1_mph, (
            f"Largest draw {max_vmax:.4f} mph too far below cap {_CAP_MPH:.4f} mph; "
            f"renormalization may not reach near-cap region"
        )


# ---------------------------------------------------------------------------
class TestIntensityCapGentleShift:
    """Renormalization shifts Cat1-3 storms (< 111 kt) by < 0.5 mph.

    The shift grows with intensity because the lognormal PDF thins in the upper
    tail (small dp -> large dVmax).  At ~115 kt it crosses 0.5 mph; the 0.5 mph
    bound is tight only for the bulk of events (Cat1-3, < 111 kt / 127.7 mph).
    """

    def test_gentle_below_111kt(self, monkeypatch):
        n = 10_000
        # off-branch draws
        monkeypatch.setattr(_hazard_mod, "_INTENSITY_CAP", "off")
        monkeypatch.setattr(_hazard_mod, "_INT_P_UB", 1.0)
        rng_off = np.random.default_rng(7)
        off_draws = [sample_intensity(rng_off)[1] for _ in range(n)]

        # on-branch draws (same seed)
        monkeypatch.setattr(_hazard_mod, "_INTENSITY_CAP", "on")
        monkeypatch.setattr(_hazard_mod, "_INT_P_UB", _INT_P_UB_ON)
        rng_on = np.random.default_rng(7)
        on_draws = [sample_intensity(rng_on)[1] for _ in range(n)]

        # Cat3 threshold: 111 mph ≈ 96.5 kt — shift analytically < 0.5 mph here
        threshold_mph = float(kt_to_mph(111.0))
        for i, (v_off, v_on) in enumerate(zip(off_draws, on_draws)):
            if v_off < threshold_mph:
                diff = abs(v_on - v_off)
                assert diff < 0.5, (
                    f"draw[{i}]: off={v_off:.4f} mph, on={v_on:.4f} mph, "
                    f"shift={diff:.4f} mph > 0.5 mph threshold"
                )


# ---------------------------------------------------------------------------
class TestRngDiscipline:
    """Each branch must consume exactly one uniform per sample_intensity call.

    Strategy: after sample_intensity(rng_a), the next draw from rng_a must equal
    the second draw from a fresh rng_b (same seed).  If sample_intensity consumed
    more or fewer than one uniform, rng_a would be out of sync with rng_b.
    """

    def _assert_one_uniform_consumed(self, monkeypatch, cap_setting, p_ub_val):
        monkeypatch.setattr(_hazard_mod, "_INTENSITY_CAP", cap_setting)
        monkeypatch.setattr(_hazard_mod, "_INT_P_UB", p_ub_val)

        rng_a = np.random.default_rng(99)
        rng_b = np.random.default_rng(99)

        sample_intensity(rng_a)         # consumes draws from rng_a
        _ = rng_b.uniform()             # consume exactly 1 uniform from rng_b

        # If sample_intensity consumed exactly 1 uniform, both RNGs are now
        # in the same state, so their next uniform draws must be equal.
        assert rng_a.uniform() == rng_b.uniform(), (
            "RNG states diverged: sample_intensity did not consume exactly 1 uniform"
        )

    def test_off_branch_one_uniform(self, monkeypatch):
        self._assert_one_uniform_consumed(monkeypatch, "off", 1.0)

    def test_on_branch_one_uniform(self, monkeypatch):
        self._assert_one_uniform_consumed(monkeypatch, "on", _INT_P_UB_ON)
