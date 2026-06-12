"""
Tests for model/geo_utils.py — haversine and bearing.

bearing() is tested with exact cardinal cases at the equator, where the
forward-azimuth formula has simple closed-form results. Tolerance 0.01° is
tighter than any geophysical noise that could matter to the asymmetry model.
"""

import numpy as np
import pytest

from model.geo_utils import bearing, haversine


class TestBearingCardinal:
    """Cardinal-direction cases — the 'known answer' tests for bearing()."""

    TOL = 0.01   # degrees; ~1 km at equator — more than adequate

    # All tests are at or near the equator to avoid polar-distortion effects.

    def test_due_north(self):
        """Bearing from (0,0) to (1,0) must be 0° (due north)."""
        b = float(bearing(0.0, 0.0, 1.0, 0.0))
        assert abs(b - 0.0) < self.TOL or abs(b - 360.0) < self.TOL, (
            f"Due-north bearing should be 0°, got {b:.4f}°"
        )

    def test_due_east(self):
        """Bearing from (0,0) to (0,1) must be 90° (due east)."""
        b = float(bearing(0.0, 0.0, 0.0, 1.0))
        assert abs(b - 90.0) < self.TOL, (
            f"Due-east bearing should be 90°, got {b:.4f}°"
        )

    def test_due_south(self):
        """Bearing from (0,0) to (-1,0) must be 180° (due south)."""
        b = float(bearing(0.0, 0.0, -1.0, 0.0))
        assert abs(b - 180.0) < self.TOL, (
            f"Due-south bearing should be 180°, got {b:.4f}°"
        )

    def test_due_west(self):
        """Bearing from (0,0) to (0,-1) must be 270° (due west)."""
        b = float(bearing(0.0, 0.0, 0.0, -1.0))
        assert abs(b - 270.0) < self.TOL, (
            f"Due-west bearing should be 270°, got {b:.4f}°"
        )

    def test_northeast(self):
        """Bearing from (0,0) to (1,1) must be in the NE quadrant (0°-90°)."""
        b = float(bearing(0.0, 0.0, 1.0, 1.0))
        assert 0.0 < b < 90.0, f"NE bearing should be 0°-90°, got {b:.4f}°"

    def test_range_zero_to_360(self):
        """bearing() output must always be in [0, 360)."""
        lats2 = np.array([ 1.0, -1.0,  0.0,  0.0,  2.0, -2.0])
        lons2 = np.array([ 0.0,  0.0,  1.0, -1.0,  3.0, -3.0])
        bs = bearing(0.0, 0.0, lats2, lons2)
        assert (bs >= 0.0).all() and (bs < 360.0).all(), (
            f"All bearings must be in [0, 360): got {bs}"
        )

    def test_array_output_shape(self):
        """bearing() with a scalar centre and array targets returns same shape as targets."""
        lats2 = np.array([1.0, -1.0, 0.0, 0.0])
        lons2 = np.array([0.0,  0.0, 1.0, -1.0])
        bs = bearing(0.0, 0.0, lats2, lons2)
        assert bs.shape == lats2.shape


class TestBearingVsHaversine:
    """bearing() and haversine() must accept the same input patterns without error."""

    def test_scalar_inputs(self):
        d = haversine(25.0, -80.5, 26.0, -80.5)
        b = bearing(25.0, -80.5, 26.0, -80.5)
        assert d > 0
        assert 0.0 <= float(b) < 360.0

    def test_broadcast_scalar_centre_array_locations(self):
        lats = np.array([24.0, 25.0, 26.0, 27.0])
        lons = np.full(4, -80.5)
        d = haversine(25.0, -80.5, lats, lons)
        b = bearing(25.0, -80.5, lats, lons)
        assert d.shape == lats.shape
        assert b.shape == lats.shape
