"""
Tests for model/scenario.py — Commit A scope (track builder).

Golden fixtures: tests/fixtures/hurdat2_fixture.txt — extracted from the real
HURDAT2 file (data/raw/hurdat2-1851-2025-02272026.txt), trimmed to essential
fixes only.  Tests do NOT depend on the real HURDAT2 file being present.
"""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import model.scenario as sc
from model.units import kt_to_mph
from model import hazard as _hazard

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "hurdat2_fixture.txt")


# ---------------------------------------------------------------------------
# TestLoadStorm
# ---------------------------------------------------------------------------

class TestLoadStorm:
    def test_andrew_unique_storm_id(self):
        df = sc.load_storm("ANDREW", 1992, hurdat2_path=FIXTURE)
        assert df["storm_id"].nunique() == 1

    def test_ian_unique_storm_id(self):
        df = sc.load_storm("IAN", 2022, hurdat2_path=FIXTURE)
        assert df["storm_id"].nunique() == 1

    def test_case_insensitive(self):
        upper = sc.load_storm("ANDREW", 1992, hurdat2_path=FIXTURE)
        lower = sc.load_storm("andrew", 1992, hurdat2_path=FIXTURE)
        assert len(upper) == len(lower)

    def test_unknown_storm_raises(self):
        with pytest.raises(ValueError):
            sc.load_storm("NOTREAL", 2022, hurdat2_path=FIXTURE)

    def test_wrong_year_raises(self):
        with pytest.raises(ValueError):
            sc.load_storm("ANDREW", 1991, hurdat2_path=FIXTURE)


# ---------------------------------------------------------------------------
# TestLandfallPick
# ---------------------------------------------------------------------------

class TestLandfallPick:
    def test_andrew_fl_not_la(self):
        """Primary: picks FL landfall (~80.3W), not LA landfall (~91.5W)."""
        df = sc.load_storm("ANDREW", 1992, hurdat2_path=FIXTURE)
        lf = sc._pick_landfall(df, sc._fl_bbox)
        # FL bbox: lon_min=-87.6, lon_max=-79.5
        assert lf["lon"] >= sc._fl_bbox.lon_min, "landfall too far west (outside FL bbox)"
        assert lf["lon"] <= sc._fl_bbox.lon_max, "landfall too far east (outside FL bbox)"
        assert lf["lat"] >= sc._fl_bbox.lat_min
        assert lf["lat"] <= sc._fl_bbox.lat_max

    def test_ian_cayo_costa_highest_vmax(self):
        """Among two FL 'L' fixes (110 kt and 130 kt), picks the 130-kt Cayo Costa fix."""
        df = sc.load_storm("IAN", 2022, hurdat2_path=FIXTURE)
        lf = sc._pick_landfall(df, sc._fl_bbox)
        assert lf["vmax_kt"] == pytest.approx(130.0), (
            f"expected 130 kt (Cayo Costa FL landfall), got {lf['vmax_kt']} kt"
        )

    def test_andrew_la_fix_outside_bbox(self):
        """Confirm the LA landfall fix is outside the FL bbox (sanity check on fixture)."""
        df = sc.load_storm("ANDREW", 1992, hurdat2_path=FIXTURE)
        la = df[df["record_id"] == "L"].sort_values("vmax_kt").iloc[0]
        assert la["lon"] < sc._fl_bbox.lon_min or la["lat"] > sc._fl_bbox.lat_max, (
            "Expected the lowest-vmax 'L' fix to be outside FL bbox"
        )


# ---------------------------------------------------------------------------
# TestBuildTrackArray
# ---------------------------------------------------------------------------

class TestBuildTrackArray:
    @pytest.fixture(scope="class")
    def andrew_track(self):
        df = sc.load_storm("ANDREW", 1992, hurdat2_path=FIXTURE)
        interp_df = sc._interpolate_track(df)
        return sc._build_track_array(interp_df), df, interp_df

    def test_shape(self, andrew_track):
        track, orig_df, _ = andrew_track
        assert track.ndim == 2
        assert track.shape[1] == 4
        assert track.shape[0] > len(orig_df), (
            "Interpolated track must have more rows than the original 4 fixes"
        )

    def test_cum_dist_starts_at_zero(self, andrew_track):
        track, _, _ = andrew_track
        assert track[0, 3] == pytest.approx(0.0)

    def test_cum_dist_monotone(self, andrew_track):
        track, _, _ = andrew_track
        diffs = np.diff(track[:, 3])
        assert np.all(diffs >= -1e-9), "cum_dist_km must be non-decreasing"

    def test_vmax_mph_at_original_times(self):
        """At each original fix time, interpolated vmax_step matches kt_to_mph(observed)."""
        orig_df  = sc.load_storm("ANDREW", 1992, hurdat2_path=FIXTURE)
        interp_df = sc._interpolate_track(orig_df)
        track     = sc._build_track_array(interp_df)

        for _, row in orig_df.iterrows():
            if math.isnan(row["vmax_kt"]):
                continue
            mask = interp_df["datetime"] == row["datetime"]
            assert mask.any(), (
                f"Original fix time {row['datetime']} not found in interpolated track"
            )
            idx = interp_df.index[mask][0]
            expected_mph = float(kt_to_mph(row["vmax_kt"]))
            assert track[idx, 2] == pytest.approx(expected_mph, abs=1e-9)


# ---------------------------------------------------------------------------
# TestStormParams
# ---------------------------------------------------------------------------

class TestStormParams:
    def _params(self, name, year):
        df = sc.load_storm(name, year, hurdat2_path=FIXTURE)
        lf = sc._pick_landfall(df, sc._fl_bbox)
        pf = sc._prev_fix(df, lf)
        return sc._build_storm_params(lf, pf)

    def test_heading_in_range(self):
        sp = self._params("ANDREW", 1992)
        assert 0.0 <= sp.heading_deg < 360.0

    def test_vt_positive(self):
        sp = self._params("ANDREW", 1992)
        assert sp.vt_kmh > 0.0

    def test_rmax_above_floor(self):
        for name, year in [("ANDREW", 1992), ("IAN", 2022)]:
            sp = self._params(name, year)
            assert sp.rmax >= float(_hazard._RMAX_FLOOR_KM), (
                f"{name} {year}: rmax={sp.rmax:.2f} km < floor={_hazard._RMAX_FLOOR_KM} km"
            )

    def test_b_in_vw_range(self):
        for name, year in [("ANDREW", 1992), ("IAN", 2022)]:
            sp = self._params(name, year)
            assert float(_hazard._VW_B_MIN) <= sp.b <= float(_hazard._VW_B_MAX), (
                f"{name} {year}: b={sp.b:.3f} not in "
                f"[{_hazard._VW_B_MIN}, {_hazard._VW_B_MAX}]"
            )

    def test_dp_mb_positive(self):
        for name, year in [("ANDREW", 1992), ("IAN", 2022)]:
            sp = self._params(name, year)
            assert sp.dp_mb > 0.0, f"{name} {year}: dp_mb={sp.dp_mb} not positive"

    def test_determinism_track_builder(self):
        """Two calls with the same inputs return bit-identical results."""
        df = sc.load_storm("ANDREW", 1992, hurdat2_path=FIXTURE)
        lf = sc._pick_landfall(df, sc._fl_bbox)
        pf = sc._prev_fix(df, lf)

        sp1 = sc._build_storm_params(lf, pf)
        sp2 = sc._build_storm_params(lf, pf)
        assert sp1.rmax        == sp2.rmax
        assert sp1.heading_deg == sp2.heading_deg
        assert sp1.vt_kmh      == sp2.vt_kmh
        assert sp1.b           == sp2.b
        assert sp1.dp_mb       == sp2.dp_mb

        interp1 = sc._interpolate_track(df)
        interp2 = sc._interpolate_track(df)
        np.testing.assert_array_equal(
            sc._build_track_array(interp1),
            sc._build_track_array(interp2),
        )


# ---------------------------------------------------------------------------
# TestRunScenario — Commit B scope
# ---------------------------------------------------------------------------

class TestRunScenario:
    """End-to-end run_scenario tests using the vendored HURDAT2 fixture."""

    @pytest.fixture(scope="class")
    def ian_results(self):
        footprint, ground_up, gross, dr = sc.run_scenario(
            "IAN", 2022, hurdat2_path=FIXTURE
        )
        return footprint, ground_up, gross, dr

    def test_determinism(self):
        """Two run_scenario calls with identical inputs return bit-identical arrays."""
        r1 = sc.run_scenario("ANDREW", 1992, hurdat2_path=FIXTURE)
        r2 = sc.run_scenario("ANDREW", 1992, hurdat2_path=FIXTURE)
        for a, b in zip(r1, r2):
            np.testing.assert_array_equal(a, b)

    def test_dr_shape_and_bounds(self, ian_results):
        _, _, _, dr = ian_results
        from model.exposure_io import OED_LOC_PATH, OED_ACC_PATH, load_oed_exposure
        n_loc = len(load_oed_exposure(OED_LOC_PATH, OED_ACC_PATH))
        assert dr.shape == (n_loc,)
        assert (dr >= 0.0).all(), "dr has negative values"
        assert (dr <= 1.0).all(), "dr has values above 1"

    def test_ground_up_equals_dr_tiv(self, ian_results):
        """ground_up == dr * tivs element-wise (verifies kernel arithmetic)."""
        _, ground_up, _, dr = ian_results
        from model.exposure_io import OED_LOC_PATH, OED_ACC_PATH, load_oed_exposure
        tivs = load_oed_exposure(OED_LOC_PATH, OED_ACC_PATH)["tiv"].to_numpy(dtype=float)
        np.testing.assert_allclose(ground_up, dr * tivs, atol=1e-9)

    def test_ian_hurricane_force(self, ian_results):
        """At least one portfolio location sees hurricane-force wind (>74 mph)."""
        footprint, _, _, _ = ian_results
        assert footprint.max() > 74.0, (
            f"No location sees hurricane-force wind. max footprint = {footprint.max():.1f} mph"
        )

    def test_ian_vmax_not_exceeded(self, ian_results):
        """Max footprint does not exceed 1.25x the observed landfall vmax (unit sanity)."""
        footprint, _, _, _ = ian_results
        df = sc.load_storm("IAN", 2022, hurdat2_path=FIXTURE)
        lf = sc._pick_landfall(df, sc._fl_bbox)
        ian_vmax_mph = float(kt_to_mph(lf["vmax_kt"]))
        assert footprint.max() <= ian_vmax_mph * 1.25, (
            f"max footprint {footprint.max():.1f} mph exceeds 1.25x landfall vmax "
            f"({ian_vmax_mph:.1f} mph) — likely a unit error or wrong storm pick"
        )
