"""
Tests for Kaplan-DeMaria (1995) inland decay in model/hazard.py.

Formula: V(t) = Vb + (V0 - Vb) * exp(-alpha * t)
  V0    = landfall Vmax [mph], R=1 (landfall-anchor, Jing-Lin 2019 convention)
  Vb    = kt_to_mph(26.7 kt)  [exact: 26.7 × 1852/1609.344] ≈ 30.726 mph background wind
  alpha = 0.095 h^-1
  t     = cum_dist_km / vt_kmh  [h]   (km / (km/h) = h, alpha in h^-1 -> consistent)
"""

import numpy as np
import pytest

import model.hazard as _haz
from model.hazard import build_track, _KD_VB_MPH, _KD_ALPHA, _EFOLD_KM, _STEP_KM


# Reference storm: Cat-3 strength, moderate translation speed.
_V0       = 120.0    # mph
_VT       = 20.0     # km/h
_LAT      = 25.5
_LON      = -80.5
_HDG      = 330.0    # deg (NNW)
_RMAX     = 40.0     # km


class TestKaplanDeMaria:

    @pytest.fixture(autouse=True)
    def _use_kd(self, monkeypatch):
        """Force kaplan_demaria decay for every test in this class."""
        monkeypatch.setattr(_haz, "_DECAY_METHOD", "kaplan_demaria")

    def _track(self, vt=_VT):
        return build_track(_LAT, _LON, _V0, _HDG, _RMAX, vt)

    # ------------------------------------------------------------------
    def test_row0_equals_vmax(self):
        """Row 0 (landfall, t=0): V = Vb + (V0-Vb)*exp(0) = V0 exactly."""
        track = self._track()
        assert track[0, 2] == pytest.approx(_V0, abs=1e-10)

    def test_monotone_decay(self):
        """Vmax decreases (or stays flat) at every inland step."""
        track = self._track()
        diffs = np.diff(track[:, 2])
        assert np.all(diffs <= 0), f"Non-monotone step found: {diffs}"

    def test_approaches_vb(self):
        """Very slow storm (vt=vt_min_kmh): last row is within 1 mph of Vb."""
        track = self._track(vt=_haz._VT_MIN_KMH)
        last_v = track[-1, 2]
        assert abs(last_v - _KD_VB_MPH) < 1.0, (
            f"Last step {last_v:.2f} mph not near Vb={_KD_VB_MPH:.2f} mph"
        )

    def test_hand_computed_value(self):
        """
        V0=120 mph, vt=20 km/h, cum_dist=60 km -> t=3 h
        V = 30.726 + (120 - 30.726) * exp(-0.095*3)
          = 30.726 + 89.274 * exp(-0.285)
          = 30.726 + 89.274 * 0.75199...
          ≈ 97.86 mph
        """
        track = self._track(vt=20.0)
        # row 2 = cum_dist = 2 * step_km
        step_km = _STEP_KM
        row_idx = round(60.0 / step_km)   # 60 / 30 = 2
        t = 60.0 / 20.0                   # 3 h
        expected = _KD_VB_MPH + (_V0 - _KD_VB_MPH) * np.exp(-_KD_ALPHA * t)
        assert track[row_idx, 2] == pytest.approx(expected, abs=0.1)

    def test_unit_chain(self):
        """
        Unit consistency: t [h] = cum_dist [km] / vt [km/h]; alpha [h^-1].
        vt=60 km/h, cum_dist=120 km -> t=2 h exactly.
        """
        step_km = _STEP_KM
        row_idx = round(120.0 / step_km)  # 120 / 30 = 4
        track = self._track(vt=60.0)
        t = 120.0 / 60.0                  # 2 h
        expected = _KD_VB_MPH + (_V0 - _KD_VB_MPH) * np.exp(-_KD_ALPHA * t)
        assert track[row_idx, 2] == pytest.approx(expected, abs=1e-6)

    def test_vt_guard_no_divide_by_zero(self):
        """vt=0 must not raise; inland steps must be < V0 (decay occurred)."""
        track = self._track(vt=0.0)
        assert track[1, 2] < _V0

    def test_efold_bit_identical(self, monkeypatch):
        """
        With decay_method=efold, build_track reproduces vmax*exp(-cum_dist/efold)
        for all rows — array_equal (exact float, no tolerance).
        """
        monkeypatch.setattr(_haz, "_DECAY_METHOD", "efold")
        track = build_track(_LAT, _LON, _V0, _HDG, _RMAX, _VT)
        for i, row in enumerate(track):
            cum_dist = i * _STEP_KM
            expected = _V0 * np.exp(-cum_dist / _EFOLD_KM)
            assert row[2] == expected, (
                f"Row {i}: got {row[2]}, expected {expected} (efold path not bit-identical)"
            )
