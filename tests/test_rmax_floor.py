"""
Tests for Step 3.0c: physical Rmax floor in model/hazard.py.

Test classes
------------
TestFloorOffBitIdentical  — floor=off reproduces the post-3.0b baseline exactly
                            (float64, 10-storm sequence from seed=42).
TestFloorOnClips          — floor=on: no storm falls below 8 km; floor fires for
                            storms below it; storms above floor are unchanged.
TestBCouplesToFlooredRmax — B is computed at the floored Rmax, not the raw value.
TestFloorRngDiscipline    — floor consumes no RNG draws; parent state unchanged.
"""

import numpy as np
import pytest

from model.hazard import sample_storm, _vw_b_mean, _VW_B_RMAX_COEFF, _VW_B_MIN, _VW_B_MAX
import model.hazard as _hazard_mod

# ---------------------------------------------------------------------------
# Production baseline: 10 consecutive storms, seed=42, v3+3.0b, wpr=off, floor=off
# (bit-identical to v3+3.0a; captured before 3.0c changes)
# ---------------------------------------------------------------------------
_BASELINE_10 = [
    # (rmax_km, dp_mb, b, vmax_mph)
    (43.416600321165504,  64.82839469706278,  1.4529704026528192, 125.3028222089054),
    (77.02295275643434,   59.184343984698856, 1.2065216362183013, 118.84779968552874),
    (22.36741097240041,   58.192825287772735, 1.6657375122762375, 117.68758952187191),
    (41.58210642819636,   43.94625192834046,  1.63423685326691,   99.98191922731499),
    (37.450652941908736,  63.091430098043254, 0.9164525439373397, 123.34232061571093),
    (43.1084823900127,    61.50791697672065,  0.9690954300173021, 121.53520168042833),
    (45.30617164337824,   47.883011172139796, 1.3391811826613678, 105.08878285305235),
    (47.810507125181026,  42.36783029787112,  1.5876040784132113,  97.8807849159002),
    (97.26682782694414,   61.251273434660725, 0.5653625195412878, 121.24048931491033),
    (34.54536802279213,   56.62375125632249,  1.506913446356151,  115.83447990967149),
]

# Storms from _BASELINE_10 with rmax < _FLOOR_KM_TEST = 40.0 km.
# Indices derived by inspection above; rmax values confirmed float64.
# Storms with rmax >= 40 are NOT in this set and must be unchanged by the floor.
#
# For each floored storm, b_on = b_off + _VW_B_RMAX_COEFF * (40.0 - rmax_off).
# Verified analytically that b_on_expected is inside [b_min, b_max] for all three:
#   storm 2: b_on ≈ 1.666 - 0.098 = 1.568  ∈ [0.5, 2.5]
#   storm 4: b_on ≈ 0.916 - 0.014 = 0.902  ∈ [0.5, 2.5]
#   storm 9: b_on ≈ 1.507 - 0.030 = 1.477  ∈ [0.5, 2.5]
_FLOOR_KM_TEST = 40.0   # raised so 3/10 baseline storms trigger the floor reliably
_FLOORED_IDX   = {2, 4, 9}   # storm indices from _BASELINE_10 with rmax < 40 km


class TestFloorOffBitIdentical:
    """
    floor=off must reproduce the post-3.0b production baseline exactly (float64).

    A 10-storm sequence catches stream desync at storm N that a single-storm
    test misses. Fields checked: rmax, dp_mb, b, vmax_landfall.
    """

    def test_10_storm_sequence_bit_identical(self, monkeypatch):
        monkeypatch.setattr(_hazard_mod, "_RMAX_FLOOR", "off")

        rng = np.random.default_rng(42)
        for i, (exp_rmax, exp_dp, exp_b, exp_vmax) in enumerate(_BASELINE_10):
            _, meta = sample_storm(rng)
            assert meta["rmax"]          == exp_rmax, f"storm {i}: rmax mismatch"
            assert meta["dp_mb"]         == exp_dp,   f"storm {i}: dp_mb mismatch"
            assert meta["b"]             == exp_b,    f"storm {i}: b mismatch"
            assert meta["vmax_landfall"] == exp_vmax, f"storm {i}: vmax_landfall mismatch"


class TestFloorOnClips:
    """floor=on clips Rmax to the floor value; storms above are unchanged."""

    def test_no_storm_below_physical_floor(self, monkeypatch):
        """500 storms at physical floor (8 km) — none should fall below it."""
        monkeypatch.setattr(_hazard_mod, "_RMAX_FLOOR", "on")
        monkeypatch.setattr(_hazard_mod, "_RMAX_FLOOR_KM", 8.0)
        rng = np.random.default_rng(42)
        for i in range(500):
            _, meta = sample_storm(rng)
            assert meta["rmax"] >= 8.0, (
                f"storm {i}: Rmax {meta['rmax']:.3f} km below physical floor 8.0 km"
            )

    def test_floor_fires_at_raised_km(self, monkeypatch):
        """With floor raised to 40 km, 3 of the first 10 baseline storms are clipped."""
        monkeypatch.setattr(_hazard_mod, "_RMAX_FLOOR", "on")
        monkeypatch.setattr(_hazard_mod, "_RMAX_FLOOR_KM", _FLOOR_KM_TEST)
        rng = np.random.default_rng(42)
        floored_count = 0
        for _ in range(10):
            _, meta = sample_storm(rng)
            if meta["rmax"] == _FLOOR_KM_TEST:
                floored_count += 1
        assert floored_count == len(_FLOORED_IDX), (
            f"Expected {len(_FLOORED_IDX)} storms floored at {_FLOOR_KM_TEST} km, "
            f"got {floored_count}"
        )

    def test_storms_above_floor_bit_identical(self, monkeypatch):
        """Storms with raw rmax >= floor must be bit-identical floor=off vs floor=on."""
        results = {}
        for switch in ("off", "on"):
            monkeypatch.setattr(_hazard_mod, "_RMAX_FLOOR", switch)
            monkeypatch.setattr(_hazard_mod, "_RMAX_FLOOR_KM", _FLOOR_KM_TEST)
            rng = np.random.default_rng(42)
            results[switch] = [sample_storm(rng)[1] for _ in range(10)]

        for i, (m_off, m_on) in enumerate(zip(results["off"], results["on"])):
            if i not in _FLOORED_IDX:   # above floor — must be unchanged
                assert m_on["rmax"] == m_off["rmax"], (
                    f"storm {i}: above-floor rmax changed ({m_off['rmax']} → {m_on['rmax']})"
                )
                assert m_on["b"]    == m_off["b"], (
                    f"storm {i}: above-floor b changed ({m_off['b']} → {m_on['b']})"
                )


class TestBCouplesToFlooredRmax:
    """
    B is computed at the floored Rmax, not at the raw sub-physical value.

    Uses floor=40 km and the seed=42 10-storm sequence. Storms 2, 4, 9 have
    rmax_off < 40 km (from _BASELINE_10). For all three the coupling identity

        b_on = b_off + _VW_B_RMAX_COEFF * (floor_km - rmax_off)

    holds exactly because:
      (a) both b_off and b_on share the same vw_rng noise draw (same seed, same
          draw position — floor adds no new RNG calls);
      (b) neither B is clipped: b_on_expected ∈ {1.568, 0.902, 1.477} ⊂ [0.5, 2.5].

    A guard assertion inside the loop verifies (b) so that if test parameters
    ever change and clipping becomes possible, the test fails explicitly rather
    than silently passing with a wrong assertion.
    """

    def test_b_couples_to_floored_rmax(self, monkeypatch):
        floor_km = _FLOOR_KM_TEST
        results  = {}
        for switch in ("off", "on"):
            monkeypatch.setattr(_hazard_mod, "_RMAX_FLOOR", switch)
            monkeypatch.setattr(_hazard_mod, "_RMAX_FLOOR_KM", floor_km)
            rng = np.random.default_rng(42)
            results[switch] = [sample_storm(rng)[1] for _ in range(10)]

        coupling_exercised = 0
        for i in _FLOORED_IDX:
            m_off = results["off"][i]
            m_on  = results["on"][i]

            assert m_on["rmax"] == floor_km, (
                f"storm {i}: floor=on rmax should be {floor_km}, got {m_on['rmax']}"
            )

            # b_off shares the same vw_rng noise draw as b_on (floor adds no draws).
            # Shifting the mean by _VW_B_RMAX_COEFF*(floor - rmax_off) gives b_on.
            rmax_off = m_off["rmax"]
            b_off    = m_off["b"]
            b_shift  = _VW_B_RMAX_COEFF * (floor_km - rmax_off)
            b_on_expected = b_off + b_shift

            # Guard: b_on_expected must be inside [b_min, b_max] — no clipping.
            # If this fires, update _FLOOR_KM_TEST or the test seed.
            assert _VW_B_MIN < b_on_expected < _VW_B_MAX, (
                f"storm {i}: b_on_expected={b_on_expected:.4f} outside "
                f"({_VW_B_MIN}, {_VW_B_MAX}) — clipping would invalidate the "
                "coupling identity; update test parameters"
            )

            assert abs(m_on["b"] - b_on_expected) < 1e-10, (
                f"storm {i}: b_on={m_on['b']:.10f} != expected {b_on_expected:.10f} "
                f"(b_off={b_off:.6f}, shift={b_shift:.6f}); "
                "B did not couple to the floored Rmax"
            )
            coupling_exercised += 1

        assert coupling_exercised == len(_FLOORED_IDX), (
            f"Only {coupling_exercised}/{len(_FLOORED_IDX)} coupling assertions ran — "
            "check _FLOORED_IDX matches the current baseline"
        )


class TestFloorRngDiscipline:
    """
    The floor is a pure max() — it consumes no RNG draws. The parent rng's
    bitgenerator position must be identical after each storm regardless of the
    floor switch.

    Verified by probing rng with N standard-normal draws immediately after each
    storm and comparing floor=off vs floor=on draws element-by-element.
    """

    _N_STORMS      = 20
    _N_PROBE_DRAWS = 5

    def test_parent_rng_position_identical_on_vs_off(self, monkeypatch):
        post_draws = {}
        for switch in ("off", "on"):
            monkeypatch.setattr(_hazard_mod, "_RMAX_FLOOR", switch)
            monkeypatch.setattr(_hazard_mod, "_RMAX_FLOOR_KM", 8.0)
            rng = np.random.default_rng(99)
            draws_per_storm = []
            for _ in range(self._N_STORMS):
                sample_storm(rng)
                draws_per_storm.append(rng.standard_normal(self._N_PROBE_DRAWS).tolist())
            post_draws[switch] = draws_per_storm

        for i, (off_d, on_d) in enumerate(
            zip(post_draws["off"], post_draws["on"])
        ):
            assert off_d == on_d, (
                f"storm {i}: post-storm rng state differs floor=off vs floor=on "
                "— floor consumed an unexpected RNG draw\n"
                f"  off={off_d}\n   on={on_d}"
            )
