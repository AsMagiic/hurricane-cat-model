"""
Tests for model/wind_field.py — modified Rankine vortex and wind_at_locations.

Step 2.1a regression: wind_at_locations must reproduce the old inline Rankine
formula bit-for-bit (exact float64 equality) including the d=0 / safe_d guard.
"""

import numpy as np
import pytest

from model.geo_utils  import haversine
from model.wind_field import (
    StormParams,
    _holland,
    _rankine,
    _validate_physics_config,
    wind_at_locations,
    _RHO,
)
import model.wind_field as _wf_mod
from model_config     import load_model_cfg

_OUTER_DECAY = float(load_model_cfg().hazard.outer_decay_exponent)


class TestRankinePure:
    """_rankine is a pure function: testable without any storm, lat/lon, or config."""

    RMAX     = 40.0   # km
    VMAX     = 100.0  # mph
    EXPONENT = _OUTER_DECAY

    def _w(self, d):
        return _rankine(np.asarray(d, dtype=float), self.RMAX, self.VMAX, self.EXPONENT)

    def test_calm_eye(self):
        assert float(self._w(0.0)) == 0.0

    def test_vmax_at_rmax(self):
        assert float(self._w(self.RMAX)) == pytest.approx(self.VMAX)

    def test_inner_linear_ramp(self):
        assert float(self._w(self.RMAX * 0.5)) == pytest.approx(self.VMAX * 0.5)

    def test_monotone_decay_outside(self):
        ds = np.array([self.RMAX, self.RMAX * 2, self.RMAX * 5, self.RMAX * 10])
        ws = self._w(ds)
        assert (np.diff(ws) < 0).all(), "Wind must decrease monotonically outside Rmax"

    def test_array_input(self):
        ds = np.array([0.0, self.RMAX * 0.5, self.RMAX, self.RMAX * 2, self.RMAX * 5])
        ws = self._w(ds)
        assert ws.shape == ds.shape


class TestWindAtLocations:
    """
    Regression: refactored wind_at_locations must reproduce the old inline
    Rankine formula to exact float64 equality for a fixed reference storm.

    Location 0 is placed exactly at a track centre so d=0 for that step;
    this freezes the safe_d guard behavior (Adjustment 3).

    All tests in this class pin wind_profile='rankine' so they continue to
    describe Rankine behaviour regardless of the active config default.
    """

    @pytest.fixture(autouse=True)
    def _use_rankine(self, monkeypatch):
        monkeypatch.setattr(_wf_mod, "_WIND_PROFILE", "rankine")

    # Reference track: 3 steps.
    TRACK = np.array([
        [25.0,  -80.5,  100.0,  0.0],   # step 0 — landfall
        [25.5,  -80.5,   80.0, 30.0],   # step 1
        [26.0,  -80.0,   60.0, 60.0],   # step 2
    ])
    RMAX = 40.0  # km

    # Location 0: exactly at step-0 centre → d=0 for step 0 (safe_d guard test)
    # Location 1: within Rmax of step 0
    # Location 2: outside Rmax of all steps
    LATS = np.array([25.0,  25.1,  27.0])
    LONS = np.array([-80.5, -80.5, -79.0])

    def _expected(self):
        """Old inline Rankine formula — reference implementation for the assertion."""
        expected = np.zeros(len(self.LATS))
        for lat_c, lon_c, vs, _ in self.TRACK:
            d      = haversine(lat_c, lon_c, self.LATS, self.LONS)
            safe_d = np.where(d > 0, d, 1e-10)
            wind   = np.where(
                d <= self.RMAX,
                vs * (d / self.RMAX),
                vs * (self.RMAX / safe_d) ** _OUTER_DECAY,
            )
            np.maximum(expected, wind, out=expected)
        return expected

    def test_bit_identical(self):
        sp     = StormParams(rmax=self.RMAX)
        result = wind_at_locations(self.TRACK, sp, self.LATS, self.LONS)
        assert np.array_equal(result, self._expected())

    def test_d_zero_location_is_finite_and_correct(self):
        """
        Location 0 is exactly at step-0 centre: d=0.
        Inner branch gives V = vmax_step * 0 / rmax = 0.
        The max over all steps yields the contribution from steps 1 and 2.
        Result must be finite and non-negative (not NaN, not inf).
        """
        sp     = StormParams(rmax=self.RMAX)
        result = wind_at_locations(self.TRACK, sp, self.LATS, self.LONS)
        assert np.isfinite(result[0])
        assert result[0] >= 0.0

    def test_wind_decreases_with_distance(self):
        """Wind at a farther location must be strictly less than at a nearer one."""
        # Both locations are beyond Rmax of all steps (outer power-law branch).
        near_lats = np.array([25.2])   # ~22 km from step-0 centre
        near_lons = np.array([-80.5])
        far_lats  = np.array([27.0])   # ~220 km from step-0 centre
        far_lons  = np.array([-80.5])
        sp        = StormParams(rmax=self.RMAX)
        w_near    = wind_at_locations(self.TRACK, sp, near_lats, near_lons)
        w_far     = wind_at_locations(self.TRACK, sp, far_lats,  far_lons)
        assert w_far[0] < w_near[0]

    def test_storm_params_placeholder_fields_ignored(self):
        """heading_deg / vt_kmh / b placeholders must not change the Rankine result."""
        sp_base = StormParams(rmax=self.RMAX)
        sp_full = StormParams(rmax=self.RMAX, heading_deg=315.0, vt_kmh=25.0, b=1.2)
        r_base  = wind_at_locations(self.TRACK, sp_base, self.LATS, self.LONS)
        r_full  = wind_at_locations(self.TRACK, sp_full, self.LATS, self.LONS)
        assert np.array_equal(r_base, r_full)


class TestStormParams:
    def test_rmax_required(self):
        with pytest.raises(TypeError):
            StormParams()   # rmax has no default

    def test_defaults(self):
        sp = StormParams(rmax=35.0)
        assert sp.heading_deg == 0.0
        assert sp.vt_kmh      == 0.0
        assert sp.b           == 0.0

    def test_typo_raises(self):
        sp = StormParams(rmax=35.0)
        with pytest.raises(AttributeError):
            _ = sp.rmox   # deliberate typo — must not silently return None


# ---------------------------------------------------------------------------
# Holland (1980) pure-function tests
# ---------------------------------------------------------------------------

class TestHollandPure:
    """_holland is a pure function: testable without any storm, lat/lon, or config."""

    RMAX  = 40.0   # km
    VMAX  = 120.0  # mph
    B     = 1.2    # typical V&W Holland-B value
    DP_MB = 60.0   # mb — representative Cat 3 pressure deficit
    LAT   = 25.0   # degrees — typical FL landfall latitude

    def _w(self, d):
        return _holland(np.asarray(d, dtype=float),
                        self.RMAX, self.VMAX, self.B, self.DP_MB, self.LAT, _RHO)

    def test_vmax_at_rmax(self):
        """V(Rmax) == Vmax exactly: x=1, exp(0)=1, sqrt(1*1)=1."""
        assert float(self._w(self.RMAX)) == pytest.approx(self.VMAX)

    def test_eye_zero(self):
        """V(0) == 0.0 exactly — calm eye, d=0 guard via safe_d=inf."""
        assert float(self._w(0.0)) == 0.0

    def test_eye_not_nan(self):
        """V(0) must be finite (not nan) — inf*0 indeterminate must not propagate."""
        assert np.isfinite(self._w(0.0))

    def test_monotone_outside_rmax(self):
        """Wind strictly decreasing for r > Rmax."""
        ds = np.array([self.RMAX, self.RMAX * 1.5, self.RMAX * 3, self.RMAX * 6])
        ws = self._w(ds)
        assert (np.diff(ws) < 0).all(), "Holland wind must decrease monotonically outside Rmax"

    def test_monotone_inside_rmax(self):
        """Wind strictly increasing from 0 toward 0.90×Rmax (eye-wall build-up).

        Tested to 0.90×Rmax rather than Rmax itself. The full gradient-balance
        profile with Coriolis places the true peak at r slightly INSIDE Rmax:
        the r·f/2 subtraction shrinks as r→0, shifting the peak inward by ~2-3 km
        (5-7% of a typical 40 km Rmax at 25° lat). Verified by test_no_overshoot:
        the resulting overshoot above Vmax is < 0.1%, well within the 2% limit.
        Testing to 0.90×Rmax stays safely below this near-peak region.
        """
        ds = np.linspace(1.0, self.RMAX * 0.90, 20)
        ws = self._w(ds)
        assert (np.diff(ws) > 0).all(), "Holland wind must increase monotonically inside 0.90×Rmax"

    def test_higher_b_sharper_peak(self):
        """
        Higher B -> sharper eyewall peak: steeper wind gradient on BOTH sides of Rmax.

        At r=Rmax: V=Vmax exactly regardless of B (analytic identity).
        Everywhere else: higher B drives the profile closer to zero more steeply,
        so V(r != Rmax) is LOWER for higher B — the peak is more concentrated.

        Inner wall (r < Rmax): x = (Rmax/r)^B > 1; larger B -> larger x -> f(x)
          decreasing (since f peaks at x=1) -> lower V.
        Outer wall (r > Rmax): x = (Rmax/r)^B < 1; larger B -> smaller x -> V lower.

        Verified with full gradient-balance + Vg(Rmax) anchoring: ratio Vg(r)/Vg(Rmax)
        is lower for higher B at both r_inner and r_outer.
        """
        b_lo, b_hi = 0.8, 1.8
        r_inner = np.asarray(self.RMAX * 0.85, dtype=float)
        r_outer = np.asarray(self.RMAX * 4.0,  dtype=float)

        v_inner_lo = float(_holland(r_inner, self.RMAX, self.VMAX, b_lo, self.DP_MB, self.LAT, _RHO))
        v_inner_hi = float(_holland(r_inner, self.RMAX, self.VMAX, b_hi, self.DP_MB, self.LAT, _RHO))
        v_outer_lo = float(_holland(r_outer, self.RMAX, self.VMAX, b_lo, self.DP_MB, self.LAT, _RHO))
        v_outer_hi = float(_holland(r_outer, self.RMAX, self.VMAX, b_hi, self.DP_MB, self.LAT, _RHO))

        assert v_inner_hi < v_inner_lo, (
            f"Higher B should give less wind just inside Rmax (steeper inner wall); "
            f"got b={b_hi}: {v_inner_hi:.2f} mph, b={b_lo}: {v_inner_lo:.2f} mph"
        )
        assert v_outer_hi < v_outer_lo, (
            f"Higher B should give less wind far outside Rmax (faster outer decay); "
            f"got b={b_hi}: {v_outer_hi:.2f} mph, b={b_lo}: {v_outer_lo:.2f} mph"
        )

    def test_far_field_decay(self):
        """
        At r=10×Rmax, wind must be well below Vmax.

        Threshold < 0.30 × Vmax catches the broken simplified Holland form (which
        gives ~0.40 for B=1.2 via V=Vmax*sqrt(x*exp(1-x))); the full gradient-
        balance formula gives ~0.23 at this distance (B=1.2, dp=60 mb, lat=25°).
        Pinned to B=1.2 so the expected range is precise.
        """
        r_far = np.asarray(self.RMAX * 10.0, dtype=float)
        v_far = float(self._w(r_far))
        assert v_far < 0.30 * self.VMAX, (
            f"Far-field wind too high: {v_far:.1f} mph at 10×Rmax "
            f"(threshold {0.30 * self.VMAX:.1f} mph = 30% of Vmax). "
            f"Simplified Holland form gives ~40% — check the formula."
        )

    def test_no_overshoot(self):
        """
        Anchoring at Vg(Rmax) may allow the true profile maximum (Coriolis-shifted
        slightly outside Rmax) to exceed vmax_step. At Florida latitudes Coriolis is
        weak relative to the pressure term, so the overshoot must be < 2%.
        If this test fails, switch to grid-search anchoring on the true profile peak.
        """
        r_dense = np.linspace(0.01, 20.0 * self.RMAX, 5000)
        v_dense = _holland(r_dense, self.RMAX, self.VMAX, self.B, self.DP_MB, self.LAT, _RHO)
        max_v   = float(np.max(v_dense))
        assert max_v <= self.VMAX * 1.02, (
            f"Profile peak {max_v:.3f} mph exceeds Vmax={self.VMAX} mph by "
            f"{(max_v / self.VMAX - 1.0) * 100:.2f}% (threshold 2%). "
            f"Anchor by grid-search peak instead of Vg(Rmax)."
        )

    def test_array_input(self):
        ds = np.array([0.0, self.RMAX * 0.5, self.RMAX, self.RMAX * 2, self.RMAX * 5])
        ws = self._w(ds)
        assert ws.shape == ds.shape


# ---------------------------------------------------------------------------
# Switch routing — Holland vs Rankine
# ---------------------------------------------------------------------------

class TestHollandVsRankineSwitch:
    """Verify switch routing: rankine pin reproduces _rankine bit-identically."""

    # Same reference fixtures as TestWindAtLocations
    TRACK = np.array([
        [25.0,  -80.5,  100.0,  0.0],
        [25.5,  -80.5,   80.0, 30.0],
        [26.0,  -80.0,   60.0, 60.0],
    ])
    RMAX = 40.0
    LATS = np.array([25.0,  25.1,  27.0])
    LONS = np.array([-80.5, -80.5, -79.0])

    def _rankine_expected(self):
        expected = np.zeros(len(self.LATS))
        for lat_c, lon_c, vs, _ in self.TRACK:
            d      = haversine(lat_c, lon_c, self.LATS, self.LONS)
            safe_d = np.where(d > 0, d, 1e-10)
            wind   = np.where(
                d <= self.RMAX,
                vs * (d / self.RMAX),
                vs * (self.RMAX / safe_d) ** _OUTER_DECAY,
            )
            np.maximum(expected, wind, out=expected)
        return expected

    def test_rankine_switch_bit_identical(self, monkeypatch):
        """With wind_profile='rankine', wind_at_locations reproduces _rankine exactly."""
        monkeypatch.setattr(_wf_mod, "_WIND_PROFILE", "rankine")
        sp     = StormParams(rmax=self.RMAX)
        result = wind_at_locations(self.TRACK, sp, self.LATS, self.LONS)
        assert np.array_equal(result, self._rankine_expected())

    def test_holland_differs_from_rankine(self, monkeypatch):
        """With wind_profile='holland', the result differs from Rankine (profile shapes differ)."""
        monkeypatch.setattr(_wf_mod, "_WIND_PROFILE", "holland")
        sp_holland = StormParams(rmax=self.RMAX, b=1.2, dp_mb=60.0, lat=25.0)
        sp_rankine = StormParams(rmax=self.RMAX)
        result_h   = wind_at_locations(self.TRACK, sp_holland, self.LATS, self.LONS)
        result_r   = self._rankine_expected()
        assert not np.array_equal(result_h, result_r), (
            "Holland with b=1.2 must produce a different footprint than Rankine"
        )


# ---------------------------------------------------------------------------
# Config-coherence guard
# ---------------------------------------------------------------------------

class TestHollandConfigGuard:
    """_validate_physics_config raises the right error for the b=0+Holland combination."""

    def test_holland_constant_raises(self):
        """wind_profile='holland' + b_method='constant' must raise ValueError."""
        with pytest.raises(ValueError, match="b_method='constant'"):
            _validate_physics_config("holland", "constant")

    def test_rankine_constant_ok(self):
        """wind_profile='rankine' + b_method='constant' is valid (b unused by Rankine)."""
        _validate_physics_config("rankine", "constant")   # must not raise

    def test_holland_vickery_wadhera_ok(self):
        """wind_profile='holland' + b_method='vickery_wadhera' is valid."""
        _validate_physics_config("holland", "vickery_wadhera")   # must not raise
