"""
Unit tests for V&W (2008) Rmax and Holland-B pure functions in model/hazard.py,
plus a bit-identical RNG regression test for the legacy switch path.

Test classes
------------
TestSigmaRmax     — heteroscedastic σ branch selection
TestVwRmaxMean    — deterministic Rmax formula
TestVwBMean       — deterministic B formula
TestVwBCensoring  — B clamped to [b_min, b_max]
TestRngRegression — sample_storm bit-identical under legacy switches
"""

import numpy as np
import pytest

from model.hazard import (
    _sigma_rmax,
    _vw_rmax_mean,
    _vw_b_mean,
    _vw_rmax_sample,
    _vw_b_sample,
    sample_storm,
    _VW_RMAX_SIG_LOW,
    _VW_RMAX_SIG_MID_A,
    _VW_RMAX_SIG_MID_B,
    _VW_RMAX_SIG_HIGH,
    _VW_RMAX_DP_BREAK_LO,
    _VW_RMAX_DP_BREAK_HI,
    _VW_RMAX_INTERCEPT,
    _VW_RMAX_DP2_COEFF,
    _VW_RMAX_LAT_COEFF,
    _VW_B_INTERCEPT,
    _VW_B_RMAX_COEFF,
    _VW_B_LAT_COEFF,
    _VW_B_MIN,
    _VW_B_MAX,
)
import model.hazard as _hazard_mod


class TestSigmaRmax:
    """σ branch selection for the heteroscedastic ln(Rmax) error."""

    def test_branch_low(self):
        """Δp ≤ 87 → σ = sigma_low_dp (0.448)."""
        assert _sigma_rmax(80.0) == pytest.approx(_VW_RMAX_SIG_LOW)

    def test_branch_mid(self):
        """87 < Δp ≤ 120 → σ = a − b·Δp."""
        dp = 100.0
        expected = _VW_RMAX_SIG_MID_A - _VW_RMAX_SIG_MID_B * dp
        assert _sigma_rmax(dp) == pytest.approx(expected)

    def test_branch_high(self):
        """Δp > 120 → σ = sigma_high_dp (0.186)."""
        assert _sigma_rmax(130.0) == pytest.approx(_VW_RMAX_SIG_HIGH)

    def test_boundary_lo(self):
        """Δp exactly at lower breakpoint uses the low-dp branch (≤ 87)."""
        assert _sigma_rmax(_VW_RMAX_DP_BREAK_LO) == pytest.approx(_VW_RMAX_SIG_LOW)

    def test_boundary_hi(self):
        """Δp exactly at upper breakpoint uses the mid-dp branch (87 < Δp ≤ 120)."""
        dp = _VW_RMAX_DP_BREAK_HI
        expected = _VW_RMAX_SIG_MID_A - _VW_RMAX_SIG_MID_B * dp
        assert _sigma_rmax(dp) == pytest.approx(expected)


class TestVwRmaxMean:
    """Deterministic component of ln(Rmax_km)."""

    def test_exact_formula(self):
        """_vw_rmax_mean(50, 25) reproduces hand-computed value to float precision."""
        dp, lat = 50.0, 25.0
        expected = (
            _VW_RMAX_INTERCEPT
            + _VW_RMAX_DP2_COEFF * dp ** 2
            + _VW_RMAX_LAT_COEFF * lat
        )
        assert abs(_vw_rmax_mean(dp, lat) - expected) < 1e-12

    def test_decreasing_in_dp(self):
        """
        ln(Rmax) is strictly decreasing in Δp (dp2_coeff < 0).
        Physically: stronger storms (larger Δp) → smaller Rmax.
        """
        lat = 26.0
        dps = [30.0, 50.0, 80.0, 100.0, 120.0]
        means = [_vw_rmax_mean(dp, lat) for dp in dps]
        assert all(means[i] > means[i + 1] for i in range(len(means) - 1))

    def test_increasing_in_lat(self):
        """ln(Rmax) is strictly increasing in latitude (lat_coeff > 0)."""
        dp = 60.0
        lats = [22.0, 24.0, 26.0, 28.0, 30.0]
        means = [_vw_rmax_mean(dp, lat) for lat in lats]
        assert all(means[i] < means[i + 1] for i in range(len(means) - 1))

    def test_sample_positive(self):
        """_vw_rmax_sample returns positive km across a range of storm intensities."""
        sub_rng = np.random.default_rng(0)
        for dp, lat in [(30.0, 22.0), (60.0, 26.0), (100.0, 28.0), (140.0, 30.0)]:
            rmax = _vw_rmax_sample(dp, lat, sub_rng)
            assert rmax > 0.0, f"Negative Rmax {rmax} at dp={dp}, lat={lat}"


class TestVwBMean:
    """Deterministic component of Holland B."""

    def test_exact_formula(self):
        """_vw_b_mean(20, 25) reproduces hand-computed value to float precision."""
        rmax_km, lat = 20.0, 25.0
        expected = (
            _VW_B_INTERCEPT
            + _VW_B_RMAX_COEFF * rmax_km
            + _VW_B_LAT_COEFF * lat
        )
        assert abs(_vw_b_mean(rmax_km, lat) - expected) < 1e-12

    def test_decreasing_with_rmax(self):
        """
        B is strictly decreasing with Rmax (rmax_coeff < 0).
        Physically: compact storms (small Rmax) have higher B (sharper peak).
        """
        lat = 26.0
        rmax_vals = [10.0, 20.0, 40.0, 60.0, 100.0]
        b_vals = [_vw_b_mean(r, lat) for r in rmax_vals]
        assert all(b_vals[i] > b_vals[i + 1] for i in range(len(b_vals) - 1))


class TestVwBCensoring:
    """B is always clamped to [b_min, b_max]."""

    def test_censoring_bounds_over_many_samples(self):
        """200 random (rmax, lat) draws all produce B in [0.5, 2.5]."""
        rng = np.random.default_rng(1)
        for _ in range(200):
            rmax_km = float(rng.uniform(5.0, 150.0))
            lat     = float(rng.uniform(20.0, 35.0))
            sub_rng = rng.spawn(1)[0]
            b = _vw_b_sample(rmax_km, lat, sub_rng)
            assert _VW_B_MIN <= b <= _VW_B_MAX, (
                f"B={b:.4f} outside [{_VW_B_MIN}, {_VW_B_MAX}] "
                f"at rmax_km={rmax_km:.1f}, lat={lat:.1f}"
            )

    def test_lower_clamp(self):
        """Very large Rmax drives B mean well below b_min → censored to 0.5."""
        # B mean at rmax_km=9999, lat=26: 1.881 + (-0.00557)*9999 + (-0.01295)*26 ≈ -53.9
        # Even without error, this is deep in the lower tail → always clamps.
        class _ZeroNormal:
            """Stub sub_rng that returns 0.0 for normal()."""
            def normal(self, loc, scale):
                return 0.0

        b = _vw_b_sample(9999.0, 26.0, _ZeroNormal())
        assert b == pytest.approx(_VW_B_MIN)

    def test_upper_clamp(self):
        """Very small Rmax drives B mean well above b_max → censored to 2.5."""
        # B mean at rmax_km=0, lat=0: 1.881 (above b_min but below b_max).
        # Need a large positive error to exceed 2.5, OR use negative rmax coefficient.
        # Check via a large positive noise draw.
        class _HighNormal:
            """Stub sub_rng that returns 100.0 for normal()."""
            def normal(self, loc, scale):
                return 100.0

        b = _vw_b_sample(10.0, 25.0, _HighNormal())
        assert b == pytest.approx(_VW_B_MAX)


class TestRngRegression:
    """
    With rmax_method='uniform' and b_method='constant', sample_storm must
    reproduce the v2 baseline bit-identically: exact float64 equality on the
    complete track array and every metadata field.

    Baseline captured from the pre-V&W implementation:
        rng = np.random.default_rng(42)
        track, meta = sample_storm(rng)   [before any changes to hazard.py]

    Key canaries:
        translation_speed_kmh — gamma draw (rng draw 4, BEFORE spawn)
        rmax                  — uniform draw (rng draw 6, AFTER spawn)
    Both must match exactly to prove the legacy stream is unperturbed.
    """

    # Exact float64 literals from pre-V&W capture (seed=42, full precision)
    _BASELINE_TRACK = np.array([
        [24.712854668727026,  -82.22062328926987, 125.72967186531868,  0.0],
        [24.874334465715954,  -82.45919977021335,  97.91836690402097, 30.0],
        [25.035814262704882,  -82.69808705164886,  76.25890082192467, 60.0],
        [25.19729405969381,   -82.93728784688808,  59.390491676279524, 90.0],
        [25.35877385668274,   -83.17680488493,     46.25336142448225, 120.0],
        [25.520253653671666,  -83.41664091064575,  36.022154097071486, 150.0],
        [25.681733450660595,  -83.65679868496571,  28.05408181871809, 180.0],
        [25.843213247649523,  -83.89728098506856,  21.848540888766912, 210.0],
        [26.00469304463845,   -84.13809060457272,  17.01566075313928, 240.0],
        [26.16617284162738,   -84.37923035372994,  13.25180991902224, 270.0],
        [26.327652638616307,  -84.62070305962104,  10.320519942047932, 300.0],
    ])

    _BASELINE_META = {
        "category":              3,
        "vmax_landfall":         125.72967186531868,
        "rmax":                  41.08535497068328,
        "landfall_lat":          24.712854668727026,
        "landfall_lon":          -82.22062328926987,
        "regime":                "atlantic",
        "heading_deg":           306.68928886003926,
        "translation_speed_kmh": 25.838849077209947,
    }

    def test_legacy_stream_unchanged(self, monkeypatch):
        """
        track (np.array_equal) and all pre-existing metadata fields must match
        exactly (==, not approx). dp_mb and b are new fields not in the baseline
        capture — only b is verified here (must be 0.0 under 'constant' switch).
        """
        monkeypatch.setattr(_hazard_mod, "_RMAX_METHOD", "uniform")
        monkeypatch.setattr(_hazard_mod, "_B_METHOD", "constant")
        monkeypatch.setattr(_hazard_mod, "_DECAY_METHOD", "efold")

        rng = np.random.default_rng(42)
        track, meta = sample_storm(rng)

        # Track: exact float64 array equality
        assert np.array_equal(track, self._BASELINE_TRACK), (
            f"Track mismatch — RNG stream was perturbed.\n"
            f"Got:\n{track}\nExpected:\n{self._BASELINE_TRACK}"
        )

        # Legacy metadata fields — exact equality (not approx)
        for key, expected in self._BASELINE_META.items():
            actual = meta[key]
            if isinstance(expected, float):
                assert actual == expected, (
                    f"meta[{key!r}] = {actual!r} != baseline {expected!r}"
                )
            else:
                assert actual == expected, (
                    f"meta[{key!r}] = {actual!r} != baseline {expected!r}"
                )

        # New field: b must be 0.0 under 'constant' switch
        assert meta["b"] == 0.0, f"Expected b=0.0 under 'constant' switch; got {meta['b']}"
