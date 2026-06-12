"""
Tests for calibration/fragility_thetas.py.

Focus:
  1. Bit-stable determinism: re-running calibrate_class() reproduces the
     committed outputs/fragility_thetas.csv exactly (same algorithm, same fixed
     multistarts -> same Nelder-Mead trajectory -> same result).
  2. Sanity checks from DoD: thetas strictly increasing, SEP respected,
     cross-class hierarchy, betas inside effective grid bounds, E[DR] monotone.

Pure-function tests (edr, logistic_dr) do not read config or files.
"""

import os

import numpy as np
import pandas as pd
import pytest

from calibration.fragility_thetas import (
    _BETA_GRID,
    _BETA_MAX,
    _BETA_MIN,
    _CLASS_ORDER,
    _FIT_GRID,
    _THETA1_REF_MIDPOINT,
    _THETA1_REF_MPH,
    _feasible_betas,
    calibrate_class,
    check_cross_class_hierarchy,
    edr,
    exceedance_probs,
    logistic_dr,
)
from model_config import load_model_cfg

_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CSV     = os.path.join(_ROOT, "outputs", "fragility_thetas.csv")
_FIXTURE = None   # loaded once per session by _get_committed_csv()


def _get_committed_csv() -> pd.DataFrame:
    global _FIXTURE
    if _FIXTURE is None:
        if not os.path.exists(_CSV):
            pytest.skip(f"Committed CSV not found: {_CSV} — run fragility_thetas.py first")
        _FIXTURE = pd.read_csv(_CSV)
    return _FIXTURE


def _cparams() -> dict:
    return load_model_cfg().vulnerability.construction_params


# ---------------------------------------------------------------------------
# Pure function tests
# ---------------------------------------------------------------------------

class TestEdr:

    def test_zero_at_very_low_gust(self):
        """E[DR] ≈ 0 when gust << theta1 for any class."""
        thetas = np.array([88.0, 100.0, 145.0, 165.0])
        result = edr(np.array([10.0]), thetas, beta=0.13)
        assert result[0] < 1e-6

    def test_approaches_one_at_extreme_gust(self):
        """E[DR] approaches sum(delta_lr) = 1.0 at very high gust."""
        thetas = np.array([88.0, 100.0, 145.0, 165.0])
        result = edr(np.array([1000.0]), thetas, beta=0.13)
        assert abs(result[0] - 1.0) < 1e-6

    def test_monotone_by_construction(self):
        """edr() is monotone non-decreasing in gust — summation-by-parts guarantee."""
        thetas = np.array([88.0, 100.0, 145.0, 165.0])
        g = np.linspace(50.0, 300.0, 200)
        vals = edr(g, thetas, beta=0.13)
        assert np.all(np.diff(vals) >= -1e-12)

    def test_upper_bound_is_sum_delta_lr(self):
        """E[DR] <= 1.0 everywhere (sum of delta_lr = 1.0)."""
        thetas = np.array([88.0, 100.0, 145.0, 165.0])
        g = np.linspace(65.0, 300.0, 500)
        vals = edr(g, thetas, beta=0.13)
        assert np.all(vals <= 1.0 + 1e-9)


class TestLogisticDr:

    def test_zero_below_threshold(self):
        """logistic_dr = 0 for g < 65 mph."""
        dr = logistic_dr(np.array([40.0, 60.0, 64.9]), cap=1.0, midpoint=145.0, k=0.05)
        assert np.all(dr == 0.0)

    def test_at_midpoint(self):
        """logistic_dr(midpoint) = cap / 2 (exactly)."""
        cap = 0.9
        dr  = logistic_dr(np.array([165.0]), cap=cap, midpoint=165.0, k=0.05)
        assert abs(dr[0] - cap / 2.0) < 1e-10

    def test_monotone_above_threshold(self):
        g  = np.linspace(65.0, 220.0, 200)
        dr = logistic_dr(g, cap=1.0, midpoint=145.0, k=0.05)
        assert np.all(np.diff(dr) >= 0)


class TestExceedanceProbs:

    def test_shape(self):
        """exceedance_probs returns (4, n) array."""
        thetas = np.array([88.0, 100.0, 145.0, 165.0])
        g = np.linspace(65.0, 200.0, 50)
        p = exceedance_probs(g, thetas, beta=0.13)
        assert p.shape == (4, 50)

    def test_decreasing_in_k(self):
        """P(DS>=k) >= P(DS>=k+1) for all k (larger damage harder to reach)."""
        thetas = np.array([88.0, 100.0, 145.0, 165.0])
        g = np.linspace(65.0, 220.0, 100)
        p = exceedance_probs(g, thetas, beta=0.13)
        for i in range(3):
            assert np.all(p[i] >= p[i + 1] - 1e-12)


# ---------------------------------------------------------------------------
# Feasibility check
# ---------------------------------------------------------------------------

class TestFeasibility:

    def test_beta_025_infeasible_for_all_classes(self):
        """
        theta3/theta1 = 145/88 = 1.6477 < exp(2*0.25) = 1.6487 for all classes.
        beta=0.25 must be excluded from the feasible grid.
        """
        th1 = _THETA1_REF_MPH
        th3 = _THETA1_REF_MIDPOINT
        feas = _feasible_betas(th1, th3)
        assert 0.25 not in [round(b, 10) for b in feas], (
            "beta=0.25 should be infeasible (exp(0.5) > 145/88)"
        )

    def test_effective_max_is_024(self):
        th1 = _THETA1_REF_MPH
        th3 = _THETA1_REF_MIDPOINT
        feas = _feasible_betas(th1, th3)
        assert round(max(feas), 2) == 0.24

    def test_beta_010_feasible(self):
        th1 = _THETA1_REF_MPH
        th3 = _THETA1_REF_MIDPOINT
        feas = _feasible_betas(th1, th3)
        assert any(abs(b - 0.10) < 1e-9 for b in feas)


# ---------------------------------------------------------------------------
# Determinism: re-run reproduces committed CSV exactly
# ---------------------------------------------------------------------------

class TestDeterminism:

    def test_all_classes_reproduce_csv(self):
        """
        Re-running calibrate_class() for each class reproduces the committed
        CSV to the precision written (4 decimal places for thetas/beta, 6 for RMSE).
        """
        committed = _get_committed_csv()
        cparams   = _cparams()

        for _, row in committed.iterrows():
            cls     = row["class"]
            result  = calibrate_class(cls, cparams[cls])

            for key in ("beta",):
                assert round(result[key], 4) == round(row[key], 4), (
                    f"{cls}: {key} mismatch — got {result[key]:.4f}, "
                    f"committed {row[key]:.4f}"
                )
            for key in ("theta1", "theta2", "theta3", "theta4"):
                assert round(result[key], 4) == round(row[key], 4), (
                    f"{cls}: {key} mismatch — got {result[key]:.4f}, "
                    f"committed {row[key]:.4f}"
                )
            assert round(result["rmse"], 6) == round(row["rmse"], 6), (
                f"{cls}: rmse mismatch — got {result['rmse']:.6f}, "
                f"committed {row['rmse']:.6f}"
            )


# ---------------------------------------------------------------------------
# Sanity checks (DoD)
# ---------------------------------------------------------------------------

class TestSanityChecks:

    @pytest.fixture(scope="class")
    def all_results(self):
        cparams = _cparams()
        return [calibrate_class(cls, cparams[cls]) for cls in _CLASS_ORDER]

    def test_thetas_strictly_increasing(self, all_results):
        for r in all_results:
            thetas = [r["theta1"], r["theta2"], r["theta3"], r["theta4"]]
            diffs  = [thetas[i+1] - thetas[i] for i in range(3)]
            assert all(d > 0 for d in diffs), (
                f"{r['cls']}: thetas not strictly increasing: {thetas}"
            )

    def test_sep_respected(self, all_results):
        for r in all_results:
            thetas = [r["theta1"], r["theta2"], r["theta3"], r["theta4"]]
            sep    = np.exp(r["beta"])
            for i in range(3):
                ratio = thetas[i+1] / thetas[i]
                assert ratio >= sep - 1e-6, (
                    f"{r['cls']}: SEP violated: theta{i+2}/theta{i+1}={ratio:.4f} < SEP={sep:.4f}"
                )

    def test_cross_class_hierarchy(self, all_results):
        check_cross_class_hierarchy(all_results)  # raises on violation

    def test_betas_inside_effective_bounds(self, all_results):
        """Beta must be inside [0.10, 0.24], not pinned beyond effective bounds."""
        for r in all_results:
            assert 0.10 <= r["beta"] <= 0.24, (
                f"{r['cls']}: beta={r['beta']:.2f} outside effective grid [0.10, 0.24]"
            )

    def test_edr_monotone_per_class(self, all_results):
        g = np.linspace(65.0, 230.0, 300)
        for r in all_results:
            thetas = np.array([r["theta1"], r["theta2"], r["theta3"], r["theta4"]])
            vals   = edr(g, thetas, r["beta"])
            diffs  = np.diff(vals)
            assert np.all(diffs >= -1e-12), (
                f"{r['cls']}: E[DR] not monotone. Min diff={diffs.min():.2e}"
            )

    def test_rmse_reasonable(self, all_results):
        """RMSE < 0.10 for all classes (rough sanity — large RMSE signals bad fit)."""
        for r in all_results:
            assert r["rmse"] < 0.10, (
                f"{r['cls']}: RMSE={r['rmse']:.4f} exceeds 0.10 threshold"
            )
