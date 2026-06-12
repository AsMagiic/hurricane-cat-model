"""
Config loader for the Florida hurricane cat model.

Reads config/exposure.yaml     -> load_exposure_cfg()
Reads config/model_v3.yaml    -> load_model_cfg()
Reads config/calibration.yaml -> load_calibration_cfg()

Every leaf entry {value, units, source} is validated at load time; the raw
value is returned so callers see plain Python scalars/lists, not wrappers.

Attribute access
----------------
    ecfg = load_exposure_cfg()
    ecfg.n_locations          # int 1000
    ecfg.county_centroids     # dict {"Miami-Dade": [25.61, -80.40], ...}

    mcfg = load_model_cfg()
    mcfg.simulation.seed      # int 42
    mcfg.frequency.lambda_rate            # float 0.7
    mcfg.hazard.coast_polyline            # list of [lat, lon] lists
    mcfg.vulnerability.construction_params  # plain dict-of-dicts
    mcfg.reinsurance.layers               # list of plain dicts

Special-cased nodes (reconstructed as plain Python containers)
--------------------------------------------------------------
  construction_params  ->  {type_name: {param: value}}
      Preserves space-separated key names ("Wood Frame", "Reinforced Concrete")
      so callers can index by CONSTRUCTION_PARAMS[construction].
  reinsurance.layers   ->  [{name, attachment, limit}, ...]
      Matches the list-of-dicts format the existing pipeline expects.

Path conventions
----------------
Each script adds ROOT to sys.path (one level above its own directory),
then imports: from model_config import load_exposure_cfg, load_model_cfg
"""

import os

try:
    import yaml
except ImportError as exc:
    raise ImportError("pyyaml required: pip install pyyaml") from exc

_ROOT    = os.path.dirname(os.path.abspath(__file__))
_CFG_DIR = os.path.join(_ROOT, "config")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class _NS:
    """Lightweight attribute-access namespace for config sub-trees."""
    def __init__(self, mapping):
        for k, v in mapping.items():
            setattr(self, k, v)
    def __repr__(self):
        return f"_NS({list(vars(self).keys())})"


def _is_leaf(node):
    """A leaf is a dict that carries a 'value' key."""
    return isinstance(node, dict) and "value" in node


def _validate_leaf(node, path):
    for field in ("units", "source"):
        if field not in node:
            raise ValueError(
                f"Config error at '{path}': leaf has 'value' but missing '{field}'"
            )


def _materialize_construction_params(node, path):
    """
    Reconstruct construction_params as a plain dict-of-dicts.

    Input YAML shape:
        TypeName:
          param: {value: X, units: Y, source: Z}
          ...

    Output:
        {"TypeName": {"param": X, ...}, ...}

    Key names with spaces (e.g. "Wood Frame") are preserved exactly.
    """
    result = {}
    for type_name, params in node.items():
        row = {}
        for param, leaf in params.items():
            p = f"{path}.{type_name}.{param}"
            if not _is_leaf(leaf):
                raise ValueError(
                    f"Config error at '{p}': expected leaf {{value, units, source}}, "
                    f"got {type(leaf)}"
                )
            _validate_leaf(leaf, p)
            row[param] = leaf["value"]
        result[type_name] = row
    return result


def _materialize_layers(lst, path):
    """
    Reconstruct reinsurance layers as a list of plain dicts.

    Input YAML shape (list):
        - name:       {value: ..., units: ..., source: ...}
          attachment: {value: ..., units: ..., source: ...}
          limit:      {value: ..., units: ..., source: ...}

    Output:
        [{"name": "Layer 1", "attachment": 60000000, "limit": 40000000}, ...]
    """
    result = []
    for i, layer in enumerate(lst):
        row = {}
        for k, v in layer.items():
            p = f"{path}[{i}].{k}"
            if _is_leaf(v):
                _validate_leaf(v, p)
                row[k] = v["value"]
            else:
                row[k] = v
        result.append(row)
    return result


def _materialize(node, path=""):
    """Recursively materialize a config node."""
    if not isinstance(node, dict):
        return node
    if _is_leaf(node):
        _validate_leaf(node, path)
        return node["value"]
    mapping = {}
    for k, v in node.items():
        child = f"{path}.{k}" if path else k
        if k == "construction_params" and isinstance(v, dict):
            mapping[k] = _materialize_construction_params(v, child)
        elif k == "layers" and isinstance(v, list):
            mapping[k] = _materialize_layers(v, child)
        else:
            mapping[k] = _materialize(v, child)
    return _NS(mapping)


def _load(path):
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return _materialize(raw)


# ---------------------------------------------------------------------------
# Physics-switch env-var overrides (Paso 2.4)
# ---------------------------------------------------------------------------

_PHYSICS_OVERRIDES = {
    "CATMODEL_WIND_PROFILE":          ("wind_profile",          frozenset({"rankine", "holland"})),
    "CATMODEL_RMAX_METHOD":           ("rmax_method",           frozenset({"uniform", "vickery_wadhera"})),
    "CATMODEL_B_METHOD":              ("b_method",              frozenset({"constant", "vickery_wadhera"})),
    "CATMODEL_TRANSLATION_ASYMMETRY": ("translation_asymmetry", frozenset({"on", "off"})),
    "CATMODEL_DECAY_METHOD":          ("decay_method",          frozenset({"efold", "kaplan_demaria"})),
    "CATMODEL_INTENSITY_CAP":         ("intensity_cap",         frozenset({"on", "off"})),
}


def _apply_physics_overrides(tree):
    """Override hazard.physics switches from CATMODEL_* env vars. No-op if none are set."""
    phys = tree.hazard.physics
    for env_key, (attr, allowed) in _PHYSICS_OVERRIDES.items():
        val = os.environ.get(env_key)
        if val is None:
            continue
        if val not in allowed:
            raise ValueError(
                f"Environment variable {env_key}={val!r} is not a recognised value. "
                f"Allowed: {sorted(allowed)}"
            )
        if not hasattr(phys, attr):
            raise ValueError(
                f"Environment variable {env_key} targets hazard.physics.{attr}, "
                f"but that leaf does not exist in the materialized config tree. "
                f"Check that config/model_v3.yaml contains hazard.physics.{attr}."
            )
        setattr(phys, attr, val)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_exposure_cfg():
    """Load and validate config/exposure.yaml.  Returns _NS."""
    return _load(os.path.join(_CFG_DIR, "exposure.yaml"))


def load_model_cfg():
    """Load and validate config/model_v3.yaml.  Returns _NS."""
    tree = _load(os.path.join(_CFG_DIR, "model_v3.yaml"))
    _apply_physics_overrides(tree)
    return tree


def load_calibration_cfg():
    """Load and validate config/calibration.yaml.  Returns _NS."""
    return _load(os.path.join(_CFG_DIR, "calibration.yaml"))
