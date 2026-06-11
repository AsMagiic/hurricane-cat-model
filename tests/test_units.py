"""
Round-trip and known-value tests for model/units.py.
"""

import os
import sys

import numpy as np
import pytest

# model/ has no __init__.py; add it to sys.path for direct module import.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model"))
from units import kt_to_mph, kt_to_ms, ms_to_kt, mph_to_kt

_ROUNDTRIP_TOL = 1e-9
_KNOWN_MPH_TOL = 0.1    # per spec: 64 kt ≈ 73.6 mph, tolerance 0.1 mph


class TestKnownValue:
    def test_64kt_to_mph(self):
        """64 kt ≈ 73.6 mph (NHC 1.15078 factor; tolerance 0.1 mph per spec)."""
        assert abs(float(kt_to_mph(64.0)) - 73.6) <= _KNOWN_MPH_TOL


class TestRoundTrips:
    _VALUES = [0.0, 1.0, 34.0, 64.0, 96.0, 137.0, 180.0]

    def test_kt_mph_kt(self):
        for v in self._VALUES:
            recovered = float(mph_to_kt(kt_to_mph(v)))
            assert abs(recovered - v) <= _ROUNDTRIP_TOL, \
                f"kt -> mph -> kt failed for {v} kt: recovered {recovered}"

    def test_kt_ms_kt(self):
        for v in self._VALUES:
            recovered = float(ms_to_kt(kt_to_ms(v)))
            assert abs(recovered - v) <= _ROUNDTRIP_TOL, \
                f"kt -> m/s -> kt failed for {v} kt: recovered {recovered}"


class TestArrayInput:
    def test_shape_preserved(self):
        arr = np.array([0.0, 34.0, 64.0, 96.0])
        assert kt_to_mph(arr).shape == arr.shape
        assert kt_to_ms(arr).shape == arr.shape
        assert mph_to_kt(arr).shape == arr.shape
        assert ms_to_kt(arr).shape == arr.shape

    def test_array_consistent_with_scalar(self):
        arr = np.array([34.0, 64.0, 96.0])
        mph_arr = kt_to_mph(arr)
        for i, v in enumerate(arr):
            assert abs(float(mph_arr[i]) - float(kt_to_mph(v))) <= _ROUNDTRIP_TOL


class TestEdgeCases:
    def test_zero(self):
        assert float(kt_to_mph(0.0)) == 0.0
        assert float(kt_to_ms(0.0)) == 0.0
        assert float(mph_to_kt(0.0)) == 0.0
        assert float(ms_to_kt(0.0)) == 0.0

    def test_negative(self):
        """Negative values convert symmetrically (useful for wind anomalies)."""
        assert float(kt_to_mph(-10.0)) == pytest.approx(-float(kt_to_mph(10.0)))
        assert float(kt_to_ms(-10.0)) == pytest.approx(-float(kt_to_ms(10.0)))
