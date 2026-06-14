"""
Level-2 loss-domain zone damage check.

Runs two deterministic historical storm scenarios (Andrew 1992, Ian 2022) over the
synthetic FL portfolio and aggregates per-location ground-up losses to county-level
damage ratios.

Coarse sanity check only — NOT a dollar reconciliation, NOT a vulnerability validation.
Zero RNG draws; fully deterministic. Results written to outputs/zone_damage_check.md.
"""

import os
import sys

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from model.scenario import run_scenario
from model.exposure_io import load_oed_exposure, OED_LOC_PATH, OED_ACC_PATH

_OUTPUT_PATH = os.path.join(_ROOT, "outputs", "zone_damage_check.md")

STORM_CONFIGS = {
    "ANDREW_1992": {"name": "ANDREW", "year": 1992},
    "IAN_2022":    {"name": "IAN",    "year": 2022},
}

_KNOWN_ABSENT = ["Charlotte"]


def compute_county_dr(ground_up: np.ndarray, exp_df: pd.DataFrame) -> pd.Series:
    """
    Compute county DR = sum(ground_up) / sum(tiv) per county.

    Parameters
    ----------
    ground_up : (n_loc,) float64 -- per-location ground-up loss (USD)
    exp_df    : pd.DataFrame     -- exposure with 'county' and 'tiv' columns;
                                   positionally aligned with ground_up

    Returns
    -------
    pd.Series -- index=county, values=DR in [0, 1], sorted descending by DR
    """
    assert len(ground_up) == len(exp_df), (
        f"Positional misalignment: ground_up[{len(ground_up)}] != exp_df[{len(exp_df)}]"
    )
    df = pd.DataFrame({
        "county":    exp_df["county"].to_numpy(),
        "tiv":       exp_df["tiv"].to_numpy(dtype=float),
        "ground_up": ground_up,
    })
    grp = df.groupby("county")[["ground_up", "tiv"]].sum()
    return (grp["ground_up"] / grp["tiv"]).sort_values(ascending=False)


def run_zone_damage(
    name: str,
    year: int,
    loc_path=None,
    acc_path=None,
    hurdat2_path=None,
) -> dict:
    """
    Run one storm scenario and return county damage ratios.

    Parameters
    ----------
    name         : str      -- storm name (e.g. 'ANDREW')
    year         : int      -- calendar year (e.g. 1992)
    loc_path     : str|None -- OED location CSV; None -> canonical OED_LOC_PATH
    acc_path     : str|None -- OED account CSV;  None -> canonical OED_ACC_PATH
    hurdat2_path : str|None -- HURDAT2 file;     None -> configured default

    Returns
    -------
    dict:
      'county_dr'       : pd.Series -- county -> DR, sorted descending
      'absent_counties' : list[str] -- _KNOWN_ABSENT entries not in county_dr
    """
    lp = loc_path if loc_path is not None else OED_LOC_PATH
    ap = acc_path if acc_path is not None else OED_ACC_PATH

    _, ground_up, _, _ = run_scenario(name, year, hurdat2_path, lp, ap)
    exp_df = load_oed_exposure(lp, ap)
    county_dr = compute_county_dr(ground_up, exp_df)

    return {
        "county_dr":       county_dr,
        "absent_counties": [c for c in _KNOWN_ABSENT if c not in county_dr.index],
    }


def write_report(andrew: dict, ian: dict, output_path: str = _OUTPUT_PATH) -> None:
    """Write the zone damage check report to output_path."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    adr = andrew["county_dr"]
    idr = ian["county_dr"]

    def _g(dr, county):
        return dr.get(county, float("nan"))

    def _p1(v):
        return f"{v * 100:.1f}"

    def _p0(v):
        return f"{v * 100:.0f}"

    andrew_notes = {
        "Miami-Dade":   "Impact zone — direct landfall, 145 kt",
        "Broward":      "Adjacent, on-track",
        "Collier":      "Near exit path — see over-spread note",
        "Palm Beach":   "Off-track (north of Andrew's center) — over-spread signal",
        "Monroe":       "On Andrew's west-to-east exit track",
        "Lee":          "SW Gulf coast — over-spread",
        "Pinellas":     "Truly peripheral — west coast, far north of track",
        "Hillsborough": "Truly peripheral",
    }
    ian_notes = {
        "Lee":          "Impact zone — direct landfall, 130 kt",
        "Collier":      "Immediately south of Lee, on-track",
        "Pinellas":     "Peripheral — Gulf coast, north of track",
        "Hillsborough": "Peripheral — Gulf coast, north of track",
        "Palm Beach":   "East coast — on Ian's cross-peninsula path",
        "Broward":      "East coast, south of Ian's path",
        "Monroe":       "Keys — south of Ian's path",
        "Miami-Dade":   "East coast, far south of Ian's path",
    }

    def _table(dr, notes):
        rows = ["| County | DR (%) | Notes |", "|---|---|---|"]
        for county, val in dr.items():
            rows.append(f"| {county} | {_p1(val)} | {notes.get(county, '')} |")
        return "\n".join(rows)

    am  = _g(adr, "Miami-Dade")
    abr = _g(adr, "Broward")
    aco = _g(adr, "Collier")
    apb = _g(adr, "Palm Beach")
    api = _g(adr, "Pinellas")
    il  = _g(idr, "Lee")
    ico = _g(idr, "Collier")
    ipi = _g(idr, "Pinellas")
    ihi = _g(idr, "Hillsborough")

    # --- build the report in sections ---
    parts = []

    parts.append("\n".join([
        "# Zone Damage Check — Level-2 Loss-Domain Backtest",
        "",
        "Model: FL hurricane cat model v3 | Generated by validation/zone_damage.py",
        "",
        "---",
        "",
        "**Scope and confounds.** Coarse sanity check on the loss footprint — NOT a dollar",
        "reconciliation, NOT a vulnerability validation. Three material confounds inflate all",
        "impact-county DRs and broaden footprints: (1) ~2× Rmax over-spread (Level-1a: modeled",
        "R64 88–179 nm vs NHC radii); (2) open-terrain-uniform vulnerability (no site-exposure",
        "layer); (3) synthetic coastal-only portfolio. The robust findings are the internal",
        "concentration gradient and the cross-storm footprint shape — model-internal comparisons",
        "that need no external dollar figure to validate.",
        "",
        "## Andrew 1992 — county damage ratios",
        "",
        _table(adr, andrew_notes),
        "",
        f"**Impact zone.** Miami-Dade: {_p1(am)}%. Broward (adjacent): {_p1(abr)}%.",
        "",
        f"**Concentration gradient.** Miami-Dade {_p0(am)}% → Palm Beach {_p0(apb)}%"
        f" → Pinellas {_p1(api)}% — a",
        "steep drop confirms the loss footprint is concentrated at the impact county and",
        "attenuates sharply with distance from the core.",
        "",
        "**Over-spread propagation (named finding).**",
        f"- *Clean off-track example — Palm Beach {_p1(apb)}%*: Palm Beach lies north of Andrew's",
        "  compact core, away from its landfall-to-exit path. The real Andrew spared most of",
        "  Palm Beach; the model still assigns this DR — a clear signature of the Level-1a",
        "  R64 over-spread (~80 nm modeled vs ~50–60 nm observed in the NW/SW quadrants).",
        f"- *Exit-path inflation — Collier {_p1(aco)}%*: Andrew's track heading at landfall is ~277°",
        "  (nearly due west), placing Collier on the storm's exit path along the SW Gulf coast.",
        "  The model's extended R64 (~80 nm) reaches Collier at this heading; real Andrew's",
        "  smaller SW-quadrant radii confined major damage to Miami-Dade. Collier's elevated DR",
        "  combines legitimate exit-path exposure with over-spread amplification — not a",
        "  clean off-track example.",
        "",
        "**Qualitative external note.** Andrew caused widespread destruction in south Miami-Dade",
        "(tens of thousands of homes destroyed, catastrophic damage to Homestead/Florida City area).",
        f"Model's {_p1(am)}% impact-county DR is order-of-magnitude consistent with that description.",
        "No dollar reconciliation attempted.",
        "",
        "## Ian 2022 — county damage ratios",
        "",
        _table(idr, ian_notes),
        "",
        f"**Impact zone.** Lee: {_p1(il)}%. Collier (immediately south, on-track): {_p1(ico)}%.",
    ]))

    if "Charlotte" in ian["absent_counties"]:
        parts.append("\n".join([
            "",
            "**Charlotte County (known absent).** Charlotte was a real Ian-impact county",
            "(direct landfall corridor). Zero locations in the synthetic coastal portfolio →",
            "absent from all DR results; this is a documented portfolio limitation, not a model failure.",
        ]))

    parts.append("\n".join([
        "",
        f"**Peripheral gradient.** Pinellas {_p1(ipi)}%, Hillsborough {_p1(ihi)}% — Gulf-coast counties",
        "north of Ian's track. Peripherals are non-trivial (~7–8%), reflecting Ian's broader",
        "track crossing the peninsula.",
        "",
        "**Qualitative external note.** Ian caused catastrophic damage in Lee County",
        "(Sanibel, Fort Myers Beach) and significant structural damage in Collier. Model's",
        f"Lee DR ({_p1(il)}%) is order-of-magnitude consistent with documented widespread",
        "destruction. No dollar reconciliation attempted.",
        "",
        "## Cross-storm footprint: concentration vs spread",
        "",
        "| Storm | Impact county | DR | Off-track/peripheral | DR |",
        "|---|---|---|---|---|",
        f"| Andrew 1992 | Miami-Dade | {_p1(am)}% | Palm Beach (off-track) | {_p1(apb)}% |",
        f"| Ian 2022    | Lee        | {_p1(il)}% | Pinellas (peripheral)  | {_p1(ipi)}% |",
        "",
        "**Finding.** Both impact counties saturate near the vulnerability ceiling (~80–83% DR);",
        "the model correctly identifies both events as locally catastrophic. The",
        f"{abs(am - il) * 100:.1f} ppt margin between them is not informative at saturation — it is retained",
        "as a deterministic regression guard, not a physical conclusion.",
        "",
        "The discriminating signal is **footprint shape, not peak level**:",
        f"- *Andrew*: sharp concentration — Miami-Dade {_p0(am)}% drops to Palm Beach {_p0(apb)}%"
        f" (off-track) and Pinellas {_p1(api)}%. Compact and peaked.",
        f"- *Ian*: broad distribution — Lee {_p0(il)}%, but peripherals at ~{_p0(ihi)}–{_p0(ipi)}%"
        f" (vs Andrew's {_p1(api)}%). Wider damage spread across the portfolio.",
        "",
        "This contrast — concentrated vs broad damage footprint — is the **loss-domain echo of",
        "the Level-1a compact-vs-broad radii finding**: Andrew was a compact, fast-moving Cat-5",
        "that caused extreme but localized damage; Ian was a slower, broader storm with a wider",
        "damaging-wind swath. The loss domain reproduces the same structural distinction.",
        "",
        "## Summary",
        "",
        "| Check | Finding | Confounds |",
        "|---|---|---|",
        f"| Andrew impact gradient | Miami-Dade {_p0(am)}% >> Palm Beach {_p0(apb)}%"
        f" >> Pinellas {_p1(api)}% | 2× over-spread, open-terrain vuln |",
        f"| Ian gradient | Lee {_p0(il)}%, peripherals ~{_p0(ihi)}–{_p0(ipi)}%"
        " (less sharp than Andrew) | broader track, peninsula crossing |",
        f"| Over-spread propagation | Palm Beach {_p0(apb)}% (clean off-track);"
        f" Collier {_p0(aco)}% (exit-path + over-spread) | Level-1a Rmax 2× |",
        "| Charlotte absent | 0 portfolio locs in Charlotte County | synthetic portfolio limitation |",
        "| Cross-storm footprint | Both saturate at peak; signal is concentration vs spread"
        " | saturation masks peak ordering |",
        "| Qualitative consistency | Both impact counties consistent with catastrophic"
        " damage descriptions | not a numerical match |",
    ]))

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts) + "\n")


def main():
    print("Running Andrew 1992 scenario...")
    andrew = run_zone_damage("ANDREW", 1992)
    print("Running Ian 2022 scenario...")
    ian = run_zone_damage("IAN", 2022)

    write_report(andrew, ian)

    print("\nAndrew 1992 county DRs:")
    for county, dr in andrew["county_dr"].items():
        print(f"  {county:20s}  {dr * 100:6.2f}%")

    print("\nIan 2022 county DRs:")
    for county, dr in ian["county_dr"].items():
        print(f"  {county:20s}  {dr * 100:6.2f}%")

    if andrew["absent_counties"] or ian["absent_counties"]:
        print(
            f"\nAbsent: Andrew={andrew['absent_counties']}, Ian={ian['absent_counties']}"
        )

    print(f"\nReport written to {_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
