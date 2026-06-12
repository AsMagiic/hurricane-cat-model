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
    _apply_asymmetry,
    _validate_physics_config,
    wind_at_locations,
    _RHO,
    _ASYM_FRAC,
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
        monkeypatch.setattr(_wf_mod, "_ASYMMETRY_ON", False)

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
        monkeypatch.setattr(_wf_mod, "_ASYMMETRY_ON", False)
        sp     = StormParams(rmax=self.RMAX)
        result = wind_at_locations(self.TRACK, sp, self.LATS, self.LONS)
        assert np.array_equal(result, self._rankine_expected())

    def test_holland_differs_from_rankine(self, monkeypatch):
        """With wind_profile='holland', the result differs from Rankine (profile shapes differ)."""
        monkeypatch.setattr(_wf_mod, "_WIND_PROFILE", "holland")
        monkeypatch.setattr(_wf_mod, "_ASYMMETRY_ON", False)
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


# ---------------------------------------------------------------------------
# Phase 2.2 — Translation asymmetry
# ---------------------------------------------------------------------------

class TestAsymmetry:
    """
    Tests for _apply_asymmetry() and its integration into wind_at_locations.

    Storm geometry used throughout:
        Track moving DUE NORTH along lon = -80.5 (heading_deg = 0.0).
        RIGHT side = EAST (bearing ~90°) — should get +a·Vt enhancement.
        LEFT  side = WEST (bearing ~270°) — should get -a·Vt reduction.
    """

    RMAX  = 40.0   # km
    VMAX  = 120.0  # mph
    B     = 1.2
    DP_MB = 60.0   # mb
    LAT   = 25.0   # degrees

    # North-moving track along lon=-80.5; step spacing ~55.6 km (0.5° lat).
    TRACK = np.array([
        [25.0, -80.5, 120.0,   0.0],
        [25.5, -80.5,  96.0,  55.6],
        [26.0, -80.5,  72.0, 111.2],
    ])

    def test_mirror_not_flipped(self):
        """
        THE mandatory mirror test: storm moving north (heading=0°).
        East location (bearing=90°) must have HIGHER corrected wind than
        west location (bearing=270°) when both have the same symmetric wind.

        sin(90°-0°)=+1 -> east enhanced; sin(270°-0°)=-1 -> west reduced.
        A sign error on (bearing-heading) or a wrong definition of 'right'
        would mirror the asymmetry, putting the strong side on the west.
        """
        v_sym   = np.array([80.0, 80.0])
        brg     = np.array([90.0, 270.0])   # east, west
        vt_mph  = 20.0
        result  = _apply_asymmetry(v_sym, brg, 0.0, vt_mph, _ASYM_FRAC)
        assert result[0] > result[1], (
            f"MIRROR BUG: east (right) wind {result[0]:.2f} mph must exceed "
            f"west (left) wind {result[1]:.2f} mph. "
            f"sin(90°-0°)=+1 should enhance east; sin(270°-0°)=-1 should reduce west."
        )

    def test_asymmetry_off_bitidentical(self, monkeypatch):
        """
        With _ASYMMETRY_ON=False, wind_at_locations is bit-identical to the
        pure symmetric Holland profile (bearing() is never called; the correction
        block is skipped entirely — no floating-point perturbation).
        """
        monkeypatch.setattr(_wf_mod, "_ASYMMETRY_ON", False)
        monkeypatch.setattr(_wf_mod, "_WIND_PROFILE", "holland")

        sp   = StormParams(rmax=self.RMAX, b=self.B, dp_mb=self.DP_MB, lat=self.LAT,
                           heading_deg=0.0, vt_kmh=30.0)
        lats = np.array([25.0, 25.5, 26.5])
        lons = np.array([-80.0, -79.5, -82.0])

        result_off = wind_at_locations(self.TRACK, sp, lats, lons)

        # Reference: manually accumulate symmetric Holland over track steps
        lats_a = np.asarray(lats, float)
        lons_a = np.asarray(lons, float)
        expected = np.zeros(len(lats_a))
        for lat_c, lon_c, vmax_step, _ in self.TRACK:
            d = haversine(lat_c, lon_c, lats_a, lons_a)
            wind = _holland(d, self.RMAX, vmax_step, self.B, self.DP_MB, self.LAT, _RHO)
            np.maximum(expected, wind, out=expected)

        assert np.array_equal(result_off, expected), (
            "asymmetry=off must produce bit-identical output to the symmetric "
            "Holland profile (bearing/correction block must not execute)"
        )

    def test_right_greater_than_left(self, monkeypatch):
        """
        Integration: east location (right of north-moving storm) must have
        higher maximum wind than west location (left) at equal distance.
        """
        monkeypatch.setattr(_wf_mod, "_ASYMMETRY_ON", True)
        monkeypatch.setattr(_wf_mod, "_WIND_PROFILE", "holland")

        sp = StormParams(rmax=self.RMAX, b=self.B, dp_mb=self.DP_MB, lat=self.LAT,
                         heading_deg=0.0, vt_kmh=30.0)

        # ~100 km east and west of the track centreline (lon=-80.5)
        w_east = wind_at_locations(
            self.TRACK, sp, np.array([25.5]), np.array([-79.5]))
        w_west = wind_at_locations(
            self.TRACK, sp, np.array([25.5]), np.array([-81.5]))

        assert w_east[0] > w_west[0], (
            f"MIRROR BUG: east wind {w_east[0]:.1f} mph must exceed "
            f"west wind {w_west[0]:.1f} mph for north-moving storm (heading=0°)"
        )

    def test_asymmetry_magnitude(self):
        """
        Right-left wind spread is bounded by 2·a·Vt_mph.

        If raw km/h were passed instead of mph the spread would be ~1.6× too large;
        this test catches that unit error.
        """
        vt_kmh = 30.0
        vt_mph = vt_kmh * 0.621371
        v_sym  = np.array([80.0, 80.0])
        brg    = np.array([90.0, 270.0])   # right, left

        result = _apply_asymmetry(v_sym, brg, 0.0, vt_mph, _ASYM_FRAC)
        spread = float(result[0] - result[1])
        bound  = 2.0 * _ASYM_FRAC * vt_mph

        assert spread <= bound * 1.001, (
            f"Right-left spread {spread:.2f} mph > 2·a·Vt = {bound:.2f} mph. "
            f"Unit error? Raw km/h ({vt_kmh:.0f}) would give "
            f"{2 * _ASYM_FRAC * vt_kmh:.1f} mph spread."
        )

    def test_no_negative_wind(self):
        """max(0,...) clip ensures no location ever receives negative wind."""
        v_sym  = np.array([5.0, 2.0, 0.5, 0.0])   # near-zero left-flank periphery
        brg    = np.array([270.0, 270.0, 270.0, 270.0])   # all on the weak side
        vt_mph = 25.0   # large enough to drive subtraction below zero without the clip
        result = _apply_asymmetry(v_sym, brg, 0.0, vt_mph, _ASYM_FRAC)
        assert (result >= 0.0).all(), (
            f"Negative wind after asymmetry correction: {result}"
        )

    def test_clip_only_below_damage_threshold(self):
        """
        Physical refinement test: the max(0,...) clip must fire ONLY in the
        sub-damage-threshold periphery (V_sym < ~50 mph sustained).

        If the clip ever activates where V_sym >= 50 mph, the additive+clip form
        would zero out wind at a loss-relevant location — a physical error in the
        loss calculation, not just a peripheral artefact.

        Worst-case setup: left side (sin = -1), correction = -a·Vt_mph.
        Clip fires when V_sym < a·Vt_mph. We use a conservative high-end FL
        translation speed (50 km/h, ~31 mph) so a·Vt_mph ≈ 15.5 mph — the
        maximum plausible clip activation threshold.
        """
        DAMAGE_THRESHOLD_MPH = 50.0
        VT_KMH = 50.0               # well above FL mean (23.6 km/h), maximises pressure
        vt_mph = VT_KMH * 0.621371
        max_correction = _ASYM_FRAC * vt_mph   # ~15.5 mph for a=0.5

        r_dense = np.linspace(0.01, 20.0 * self.RMAX, 5000)
        v_sym   = _holland(r_dense, self.RMAX, self.VMAX, self.B, self.DP_MB, self.LAT, _RHO)

        # Locations where max(0,...) would fire: left-side corrected wind < 0
        clips_here = (v_sym - max_correction) < 0.0
        v_at_clip  = v_sym[clips_here]

        if len(v_at_clip) > 0:
            max_clip = float(v_at_clip.max())
            assert max_clip < DAMAGE_THRESHOLD_MPH, (
                f"max(0,...) clip activates at V_sym = {max_clip:.1f} mph, which is "
                f">= the damage threshold ({DAMAGE_THRESHOLD_MPH:.0f} mph sustained). "
                f"The additive+clip asymmetry is physically inappropriate here — it "
                f"would force zero wind at a loss-relevant location. "
                f"Reduce asymmetry_fraction or switch to a non-additive formulation."
            )
