"""
Tests for model/wind_field.py — modified Rankine vortex and wind_at_locations.

Step 2.1a regression: wind_at_locations must reproduce the old inline Rankine
formula bit-for-bit (exact float64 equality) including the d=0 / safe_d guard.
"""

import numpy as np
import pytest

from model.geo_utils  import haversine
from model.wind_field import StormParams, _rankine, wind_at_locations
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
    """

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
