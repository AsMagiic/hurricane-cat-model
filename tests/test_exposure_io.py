"""
Tests for model/exposure_io.py  (Step 4.1).

TestRoundTrip
    load_oed_exposure() must return exact values matching the frozen legacy
    exposure_reference.csv fixture on all compared columns.
    - String columns (location_id, state, county, construction, occupancy):
      element-wise equality (list ==).
    - Numeric columns (lat, lon, tiv, deductible, limit): np.array_equal
      (exact, zero tolerance — no rounding or arithmetic in the read path).

TestOedValidity
    Required OED fields present and non-null; code values in valid ranges;
    DedType and LimitType == 0 (Amount); LocPerilsCovered == "WTC";
    single account row with correct fields.

TestAccFileMissing
    load_oed_exposure raises ValueError when required account columns are absent.

TestLocFileMissing
    load_oed_exposure raises ValueError when required location columns are absent.
"""

import os
import numpy as np
import pandas as pd
import pytest

_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LOC     = os.path.join(_ROOT, "data", "oed", "location.csv")
_ACC     = os.path.join(_ROOT, "data", "oed", "account.csv")
_REF     = os.path.join(_ROOT, "tests", "fixtures", "exposure_reference.csv")

from model.exposure_io import load_oed_exposure


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_ref() -> pd.DataFrame:
    return pd.read_csv(_REF)


def _load_oed() -> pd.DataFrame:
    return load_oed_exposure(_LOC, _ACC)


# ---------------------------------------------------------------------------
# TestRoundTrip
# ---------------------------------------------------------------------------

class TestRoundTrip:
    """load_oed_exposure() returns the same data as the frozen reference CSV."""

    @pytest.fixture(scope="class")
    def ref(self):
        return _load_ref()

    @pytest.fixture(scope="class")
    def oed(self):
        return _load_oed()

    def test_row_count(self, ref, oed):
        assert len(oed) == len(ref), (
            f"Row count mismatch: OED {len(oed)} vs reference {len(ref)}"
        )

    def test_location_id_exact(self, ref, oed):
        assert list(oed["location_id"]) == list(ref["location_id"]), \
            "location_id mismatch"

    def test_state_exact(self, ref, oed):
        assert list(oed["state"]) == list(ref["state"]), \
            "state mismatch"

    def test_county_exact(self, ref, oed):
        assert list(oed["county"]) == list(ref["county"]), \
            "county mismatch"

    def test_construction_exact(self, ref, oed):
        assert list(oed["construction"]) == list(ref["construction"]), \
            "construction mismatch (OrgConstructionCode round-trip failed)"

    def test_occupancy_exact(self, ref, oed):
        assert list(oed["occupancy"]) == list(ref["occupancy"]), \
            "occupancy mismatch (OrgOccupancyCode round-trip failed)"

    def test_lat_exact(self, ref, oed):
        assert np.array_equal(oed["lat"].to_numpy(), ref["lat"].to_numpy()), \
            "lat not bit-identical"

    def test_lon_exact(self, ref, oed):
        assert np.array_equal(oed["lon"].to_numpy(), ref["lon"].to_numpy()), \
            "lon not bit-identical"

    def test_tiv_exact(self, ref, oed):
        assert np.array_equal(
            oed["tiv"].to_numpy(dtype=np.int64),
            ref["tiv"].to_numpy(dtype=np.int64),
        ), "tiv not bit-identical"

    def test_deductible_exact(self, ref, oed):
        assert np.array_equal(
            oed["deductible"].to_numpy(dtype=np.int64),
            ref["deductible"].to_numpy(dtype=np.int64),
        ), "deductible not bit-identical (DedType=Amount stores pre-rounded integer)"

    def test_limit_exact(self, ref, oed):
        assert np.array_equal(
            oed["limit"].to_numpy(dtype=np.int64),
            ref["limit"].to_numpy(dtype=np.int64),
        ), "limit not bit-identical"

    def test_column_order(self, oed):
        expected = [
            "location_id", "state", "county", "lat", "lon",
            "tiv", "construction", "occupancy", "deductible", "limit",
        ]
        assert list(oed.columns) == expected, \
            f"Column order mismatch: {list(oed.columns)}"


# ---------------------------------------------------------------------------
# TestOedValidity
# ---------------------------------------------------------------------------

class TestOedValidity:
    """OED files satisfy structural and domain constraints."""

    @pytest.fixture(scope="class")
    def loc(self):
        return pd.read_csv(_LOC)

    @pytest.fixture(scope="class")
    def acc(self):
        return pd.read_csv(_ACC)

    # ---- required Loc fields ------------------------------------------------
    def test_loc_required_fields_present(self, loc):
        for col in ("LocNumber", "CountryCode", "LocPerilsCovered", "LocCurrency"):
            assert col in loc.columns, f"Required Loc column missing: {col}"

    def test_loc_required_fields_non_null(self, loc):
        for col in ("LocNumber", "CountryCode", "LocPerilsCovered", "LocCurrency"):
            assert not loc[col].isnull().any(), f"Nulls in required Loc column: {col}"

    # ---- required Acc fields ------------------------------------------------
    def test_acc_required_fields_present(self, acc):
        for col in ("AccNumber", "AccCurrency", "PolNumber", "PolPerilsCovered"):
            assert col in acc.columns, f"Required Acc column missing: {col}"

    def test_acc_single_row(self, acc):
        assert len(acc) == 1, f"Expected 1 account row, got {len(acc)}"

    # ---- code value ranges --------------------------------------------------
    def test_construction_codes_in_range(self, loc):
        # OED ConstructionCode for residential wood/masonry/concrete/manufactured:
        # 5050, 5100, 5150, 5350 — all within 5000-5399 residential range.
        valid = {5050, 5100, 5150, 5350}
        actual = set(loc["ConstructionCode"].unique())
        assert actual.issubset(valid), f"Unexpected ConstructionCode values: {actual - valid}"

    def test_occupancy_codes_in_range(self, loc):
        # OED OccupancyCode for SF/Condo/MH: 1051, 1055 — within 1050-1099 residential.
        valid = {1051, 1055}
        actual = set(loc["OccupancyCode"].unique())
        assert actual.issubset(valid), f"Unexpected OccupancyCode values: {actual - valid}"

    # ---- financial columns: exact spec names present, unprefixed variants absent ----
    def test_financial_spec_columns_present(self, loc):
        for col in ("LocDedCode1Building", "LocDedType1Building",
                    "LocDed1Building", "LocLimitType1Building", "LocLimit1Building"):
            assert col in loc.columns, f"Required OED financial column missing: {col}"

    def test_unprefixed_financial_columns_absent(self, loc):
        for col in ("DedCode1Building", "DedType1Building", "LimitType1Building"):
            assert col not in loc.columns, (
                f"Non-spec column {col!r} present — use Loc-prefixed name"
            )

    def test_ded_type_is_amount(self, loc):
        assert (loc["LocDedType1Building"] == 0).all(), \
            "LocDedType1Building must be 0 (Amount) for bit-identity"

    def test_limit_type_is_amount(self, loc):
        assert (loc["LocLimitType1Building"] == 0).all(), \
            "LocLimitType1Building must be 0 (Amount)"

    # ---- peril code ---------------------------------------------------------
    def test_loc_peril_wtc(self, loc):
        assert (loc["LocPerilsCovered"] == "WTC").all(), \
            "All LocPerilsCovered must be 'WTC'"

    def test_acc_peril_wtc(self, acc):
        assert acc["PolPerilsCovered"].iloc[0] == "WTC", \
            "PolPerilsCovered must be 'WTC'"

    # ---- currency -----------------------------------------------------------
    def test_loc_currency_usd(self, loc):
        assert (loc["LocCurrency"] == "USD").all(), "LocCurrency must be USD"

    def test_acc_currency_usd(self, acc):
        assert acc["AccCurrency"].iloc[0] == "USD", "AccCurrency must be USD"

    # ---- OccupancyCode is NOT uniquely invertible ---------------------------
    def test_occupancy_code_not_invertible(self, loc):
        # Single Family (1051) and Mobile Home (1051) share the same OccupancyCode.
        # This verifies that OccupancyCode alone cannot distinguish them.
        sf_mask  = loc["OrgOccupancyCode"] == "Single Family"
        mh_mask  = loc["OrgOccupancyCode"] == "Mobile Home"
        assert loc.loc[sf_mask, "OccupancyCode"].iloc[0] == \
               loc.loc[mh_mask, "OccupancyCode"].iloc[0], \
            "Expected Single Family and Mobile Home to share OccupancyCode=1051"

    # ---- provenance fields present ------------------------------------------
    def test_org_construction_scheme_present(self, loc):
        assert "OrgConstructionScheme" in loc.columns
        assert (loc["OrgConstructionScheme"] == "MODEL").all()

    def test_org_occupancy_scheme_present(self, loc):
        assert "OrgOccupancyScheme" in loc.columns
        assert (loc["OrgOccupancyScheme"] == "MODEL").all()


# ---------------------------------------------------------------------------
# TestMissingRequiredFields
# ---------------------------------------------------------------------------

class TestMissingRequiredFields:
    """load_oed_exposure raises ValueError when required fields are absent."""

    def test_missing_loc_field_raises(self, tmp_path):
        # Create a location file missing LocNumber
        bad_loc = pd.DataFrame({
            "CountryCode": ["US"],
            "LocPerilsCovered": ["WTC"],
            "LocCurrency": ["USD"],
        })
        acc = pd.read_csv(_ACC)
        loc_path = str(tmp_path / "loc_bad.csv")
        acc_path = str(tmp_path / "acc.csv")
        bad_loc.to_csv(loc_path, index=False)
        acc.to_csv(acc_path, index=False)
        with pytest.raises(ValueError, match="LocNumber"):
            load_oed_exposure(loc_path, acc_path)

    def test_missing_acc_field_raises(self, tmp_path):
        # Create an account file missing PolNumber
        bad_acc = pd.DataFrame({"AccNumber": ["A0001"], "AccCurrency": ["USD"]})
        loc = pd.read_csv(_LOC)
        loc_path = str(tmp_path / "loc.csv")
        acc_path = str(tmp_path / "acc_bad.csv")
        loc.to_csv(loc_path, index=False)
        bad_acc.to_csv(acc_path, index=False)
        with pytest.raises(ValueError, match="PolNumber"):
            load_oed_exposure(loc_path, acc_path)
