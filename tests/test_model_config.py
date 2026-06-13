"""
Tests for the env-var physics-switch override layer in model_config.py.

Covered behaviour
-----------------
  1. No env vars set        -> load_model_cfg() returns YAML defaults exactly (no-op proof).
  2. Single override        -> the targeted leaf is updated; YAML default is overwritten.
  3. Invalid value          -> ValueError raised, message names the bad env var.
  4. Missing leaf guard     -> if the target attr doesn't exist in the tree, raises ValueError
                               rather than silently creating an orphan attribute.
  5. Non-physics untouched  -> an active override does not perturb simulation.seed,
                               hazard.efold_km, or reinsurance.layers.
  6. All five overrides     -> all five switches updated in a single call.

All tests use monkeypatch.setenv so no env state leaks between tests.
"""

import pytest
from model_config import load_model_cfg, _PHYSICS_OVERRIDES, _apply_physics_overrides, _NS


# YAML production defaults (must match config/model_v3.yaml)
_YAML_DEFAULTS = {
    "wind_profile":          "holland",
    "rmax_method":           "vickery_wadhera",
    "b_method":              "vickery_wadhera",
    "translation_asymmetry": "on",
    "decay_method":          "kaplan_demaria",
}

# Alternate values (each different from the YAML default)
_ALTERNATES = {
    "wind_profile":          "rankine",
    "rmax_method":           "uniform",
    "b_method":              "constant",
    "translation_asymmetry": "off",
    "decay_method":          "efold",
}

# Env-var name for each attribute (derived from _PHYSICS_OVERRIDES)
_ATTR_TO_ENV = {attr: env for env, (attr, _) in _PHYSICS_OVERRIDES.items()}


class TestPhysicsOverrides:

    def test_no_overrides_returns_yaml_defaults(self):
        """With no CATMODEL_* env vars, all five switches equal the YAML defaults."""
        tree = load_model_cfg()
        phys = tree.hazard.physics
        for attr, expected in _YAML_DEFAULTS.items():
            actual = getattr(phys, attr)
            assert actual == expected, (
                f"hazard.physics.{attr}: expected YAML default {expected!r}, got {actual!r}"
            )

    def test_single_override_applied(self, monkeypatch):
        """CATMODEL_WIND_PROFILE=rankine overrides the YAML value of 'holland'."""
        monkeypatch.setenv("CATMODEL_WIND_PROFILE", "rankine")
        tree = load_model_cfg()
        assert tree.hazard.physics.wind_profile == "rankine"

    def test_invalid_value_raises_valueerror(self, monkeypatch):
        """An unrecognised value raises ValueError naming the offending env var."""
        monkeypatch.setenv("CATMODEL_WIND_PROFILE", "banana")
        with pytest.raises(ValueError, match="CATMODEL_WIND_PROFILE"):
            load_model_cfg()

    def test_missing_leaf_raises_not_orphan(self, monkeypatch):
        """
        If the target attribute is absent from the tree, _apply_physics_overrides
        must raise ValueError — not silently create an orphan attribute.

        Simulate by passing a physics _NS that lacks 'wind_profile'.
        """
        monkeypatch.setenv("CATMODEL_WIND_PROFILE", "rankine")

        # Build a minimal tree stub that looks like tree.hazard.physics
        # but is missing the wind_profile attribute.
        phys_stub = _NS({
            "rmax_method":           "vickery_wadhera",
            "b_method":              "vickery_wadhera",
            "translation_asymmetry": "on",
            "decay_method":          "kaplan_demaria",
            # wind_profile intentionally absent
        })
        hazard_stub = _NS({"physics": phys_stub})
        tree_stub   = _NS({"hazard": hazard_stub})

        with pytest.raises(ValueError, match="wind_profile"):
            _apply_physics_overrides(tree_stub)

        # Confirm the orphan was NOT created
        assert not hasattr(phys_stub, "wind_profile"), (
            "setattr created an orphan attribute despite the missing-leaf guard"
        )

    def test_non_physics_config_untouched(self, monkeypatch):
        """An active override does not perturb non-physics leaves."""
        monkeypatch.setenv("CATMODEL_WIND_PROFILE", "rankine")
        tree = load_model_cfg()
        assert tree.simulation.seed == 42
        assert tree.hazard.efold_km == 120
        assert len(tree.summary.return_periods) == 8  # reinsurance migrated to config/reinsurance.yaml

    def test_all_five_overrides_work(self, monkeypatch):
        """All five CATMODEL_* vars can be set simultaneously to alternate values."""
        for attr, alt_val in _ALTERNATES.items():
            monkeypatch.setenv(_ATTR_TO_ENV[attr], alt_val)
        tree = load_model_cfg()
        phys = tree.hazard.physics
        for attr, expected in _ALTERNATES.items():
            actual = getattr(phys, attr)
            assert actual == expected, (
                f"hazard.physics.{attr}: expected override {expected!r}, got {actual!r}"
            )
