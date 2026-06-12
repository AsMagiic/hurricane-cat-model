#!/usr/bin/env python
"""
Attribution waterfall: quantify each physics component's contribution to
the v2 -> v3 AAL / OEP-100 / OEP-250 change.

Runs run_all.py as fresh subprocesses (one per switch combination) so that
module-level import caching is never an issue.  CATMODEL_* env-var overrides
(added to model_config.py in Paso 2.4 part 1) set each config without
mutating config/model_v3.yaml, which stays the versioned source of truth.

Seed=42 is fixed in config/model_v3.yaml and is not exposed in _PHYSICS_OVERRIDES.
All 8 configs therefore share the same stochastic catalog; switches only change how
each event is evaluated (track geometry, wind profile, decay).  The deltas are
purely physics, not Monte Carlo noise.

Usage
-----
    python analysis/waterfall.py           # full 100k-year runs (~7 min)
    python analysis/waterfall.py --quick   # 1k-year smoke test (~60s)
"""

import argparse
import csv
import os
import subprocess
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(ROOT, "results", "summary_metrics.csv")
OUT_DIR  = os.path.join(ROOT, "outputs")
RES_DIR  = os.path.join(ROOT, "results")
RUN_ALL  = os.path.join(ROOT, "run_all.py")

# ---------------------------------------------------------------------------
# Switch configurations
# ---------------------------------------------------------------------------
_WP = "CATMODEL_WIND_PROFILE"
_RM = "CATMODEL_RMAX_METHOD"
_BM = "CATMODEL_B_METHOD"
_AS = "CATMODEL_TRANSLATION_ASYMMETRY"
_DC = "CATMODEL_DECAY_METHOD"

MAIN_CONFIGS = [
    ("Config 0 -- v2 baseline",
     {_WP: "rankine",  _RM: "uniform",         _BM: "constant",        _AS: "off", _DC: "efold"}),
    ("Config 1 -- +Rmax V&W",
     {_WP: "rankine",  _RM: "vickery_wadhera", _BM: "constant",        _AS: "off", _DC: "efold"}),
    ("Config 2 -- +Holland & B",
     {_WP: "holland",  _RM: "vickery_wadhera", _BM: "vickery_wadhera", _AS: "off", _DC: "efold"}),
    ("Config 3 -- +Asymmetry",
     {_WP: "holland",  _RM: "vickery_wadhera", _BM: "vickery_wadhera", _AS: "on",  _DC: "efold"}),
    ("Config 4 -- v3 full",
     {_WP: "holland",  _RM: "vickery_wadhera", _BM: "vickery_wadhera", _AS: "on",  _DC: "kaplan_demaria"}),
]

# One-at-a-time sensitivity: each component isolated from the v2 baseline.
# S1 (v2 + Rmax only) == Config 1 -- reuse that result, no extra subprocess.
# S2 keeps rmax_method=uniform to isolate Holland+B from the Rmax effect;
#    Holland requires b_method=vickery_wadhera, so this run is "+Holland&B".
SENSITIVITY_CONFIGS = [
    ("S2 -- +Holland&B only",
     {_WP: "holland",  _RM: "uniform",         _BM: "vickery_wadhera", _AS: "off", _DC: "efold"}),
    ("S3 -- +Asymmetry only",
     {_WP: "rankine",  _RM: "uniform",         _BM: "constant",        _AS: "on",  _DC: "efold"}),
    ("S4 -- +Decay only",
     {_WP: "rankine",  _RM: "uniform",         _BM: "constant",        _AS: "off", _DC: "kaplan_demaria"}),
]

# ---------------------------------------------------------------------------
# Bit-identical anchor self-check (full mode only)
# ---------------------------------------------------------------------------
# Values confirmed from Paso 2.3 validation (full 100k-year runs, seed=42).
_V2_ANCHORS = {"aal": 3.584,  "oep100":  58.28, "oep250":  84.85}
_V3_ANCHORS = {"aal": 9.171,  "oep100": 113.44, "oep250": 147.15}
_ANCHOR_TOL = {"aal": 0.005,  "oep100":   0.05, "oep250":   0.05}

# ---------------------------------------------------------------------------
# Metric names
# ---------------------------------------------------------------------------
METRICS = ["aal", "oep100", "oep250"]
_METRIC_TITLES = {
    "aal":    "AAL (gross, AEP)",
    "oep100": "OEP-100 (gross)",
    "oep250": "OEP-250 (gross)",
}

# ---------------------------------------------------------------------------
# CSV reader
# ---------------------------------------------------------------------------
def _read_metrics(path):
    """
    Extract AAL, OEP-100, OEP-250 gross from summary_metrics.csv.

    Schema (confirmed against model/summary.py _save_csv):
        rows:    AAL_M | PML_1in100_M | PML_1in250_M
        columns: aep_gross_M | aep_net_M | oep_gross_M | oep_net_M

    AAL uses aep_gross_M (AEP-basis = mean annual aggregate loss; matches the
    3.584M / 9.171M anchors reported throughout Phase 2 validation).
    OEP PMLs use oep_gross_M (per-occurrence return-period loss).
    """
    with open(path, newline="") as f:
        rows = {r["metric"]: r for r in csv.DictReader(f)}
    try:
        return {
            "aal":    float(rows["AAL_M"]["aep_gross_M"]),
            "oep100": float(rows["PML_1in100_M"]["oep_gross_M"]),
            "oep250": float(rows["PML_1in250_M"]["oep_gross_M"]),
        }
    except KeyError as exc:
        available_metrics = list(rows.keys())
        available_cols = list(next(iter(rows.values())).keys()) if rows else []
        raise KeyError(
            f"Expected key {exc} missing in {path}. "
            f"Metrics found: {available_metrics}. "
            f"Columns found: {available_cols}."
        ) from exc


# ---------------------------------------------------------------------------
# Subprocess runner
# ---------------------------------------------------------------------------
def _run_config(label, config_env, run_all_args):
    """Run run_all.py with config_env overrides; return parsed metrics dict."""
    print(f"  {label}...", end=" ", flush=True)
    t_before = time.time()
    env = {**os.environ, **config_env}
    try:
        subprocess.run(
            [sys.executable, RUN_ALL] + run_all_args,
            env=env, cwd=ROOT, check=True,
            capture_output=True, text=True,
        )
    except subprocess.CalledProcessError as exc:
        tail = (exc.stderr or exc.stdout or "")[-3000:]
        sys.exit(f"\n[FAIL] '{label}' exited {exc.returncode}:\n{tail}")
    if os.path.getmtime(CSV_PATH) < t_before:
        raise RuntimeError(
            f"summary_metrics.csv mtime did not advance after '{label}'. "
            "CSV may be stale from a crashed or no-op run."
        )
    elapsed = time.time() - t_before
    result = _read_metrics(CSV_PATH)
    print(
        f"done ({elapsed:.1f}s)"
        f"  AAL={result['aal']:.3f}M"
        f"  OEP-100={result['oep100']:.2f}M"
        f"  OEP-250={result['oep250']:.2f}M"
    )
    return result


# ---------------------------------------------------------------------------
# Attribution table
# ---------------------------------------------------------------------------
def _fmt_d(d):
    return "     --    " if d is None else f"{d:+.3f}"


def _print_and_return_table(cumulative, iso_labels, iso_results, c0, c4):
    """Print the attribution table; return (iso_deltas, interaction) for CSV."""
    sep = "=" * 100
    dashes = "-" * 100
    print()
    print(sep)
    print("  ATTRIBUTION WATERFALL -- Florida Homeowners Cat Model  v2 -> v3")
    print("  AAL: AEP-basis gross ($M)  |  OEP-100, OEP-250: per-occurrence gross ($M)")
    print(sep)
    hdr = (f"  {'Step':<28}"
           f"{'AAL ($M)':>10}{'dAAL':>10}"
           f"  {'OEP-100':>10}{'dOEP-100':>11}"
           f"  {'OEP-250':>10}{'dOEP-250':>11}")
    print(hdr)
    print(dashes)

    prev = None
    for label, res in cumulative:
        da   = None if prev is None else res["aal"]    - prev["aal"]
        do1  = None if prev is None else res["oep100"] - prev["oep100"]
        do2  = None if prev is None else res["oep250"] - prev["oep250"]
        print(
            f"  {label:<28}"
            f"{res['aal']:>10.3f}{_fmt_d(da):>10}"
            f"  {res['oep100']:>10.2f}{_fmt_d(do1):>11}"
            f"  {res['oep250']:>10.2f}{_fmt_d(do2):>11}"
        )
        prev = res

    total = {m: c4[m] - c0[m] for m in METRICS}
    print(dashes)
    print(
        f"  {'Total v2 -> v3':<28}"
        f"{'':>10}{_fmt_d(total['aal']):>10}"
        f"  {'':>10}{_fmt_d(total['oep100']):>11}"
        f"  {'':>10}{_fmt_d(total['oep250']):>11}"
    )

    print()
    print("  ONE-AT-A-TIME SENSITIVITY (from v2 baseline; component interactions excluded)")
    print("  S1 (+Rmax only) = Config 1 result reused -- no extra subprocess.")
    print("  S2: Holland+B isolated with rmax_method=uniform; Holland requires b_method=vickery_wadhera.")
    print(dashes)

    iso_deltas = {}
    for lbl, res in zip(iso_labels, iso_results):
        d = {m: res[m] - c0[m] for m in METRICS}
        iso_deltas[lbl] = d
        print(
            f"  {lbl:<28}"
            f"{res['aal']:>10.3f}{_fmt_d(d['aal']):>10}"
            f"  {res['oep100']:>10.2f}{_fmt_d(d['oep100']):>11}"
            f"  {res['oep250']:>10.2f}{_fmt_d(d['oep250']):>11}"
        )

    iso_sum = {m: sum(iso_deltas[l][m] for l in iso_labels) for m in METRICS}
    interaction = {m: total[m] - iso_sum[m] for m in METRICS}

    print(dashes)
    print(
        f"  {'Sum of isolated':<28}"
        f"{'':>10}{_fmt_d(iso_sum['aal']):>10}"
        f"  {'':>10}{_fmt_d(iso_sum['oep100']):>11}"
        f"  {'':>10}{_fmt_d(iso_sum['oep250']):>11}"
    )
    print(
        f"  {'Interaction term':<28}"
        f"{'':>10}{_fmt_d(interaction['aal']):>10}"
        f"  {'':>10}{_fmt_d(interaction['oep100']):>11}"
        f"  {'':>10}{_fmt_d(interaction['oep250']):>11}"
    )
    print(sep)
    print()

    return iso_deltas, interaction


# ---------------------------------------------------------------------------
# CSV save
# ---------------------------------------------------------------------------
def _save_csvs(cumulative, iso_labels, iso_results, iso_deltas, interaction, c0, c4):
    """Save waterfall_attribution.csv to outputs/ (versioned) and results/."""
    fieldnames = (
        ["step"]
        + [f"{m}_M" for m in METRICS]
        + [f"delta_{m}_M" for m in METRICS]
    )
    rows = []

    prev = None
    for label, res in cumulative:
        row = {"step": label}
        for m in METRICS:
            row[f"{m}_M"] = round(res[m], 3)
            row[f"delta_{m}_M"] = "" if prev is None else round(res[m] - prev[m], 3)
        rows.append(row)
        prev = res

    rows.append({"step": "---", **{f: "" for f in fieldnames if f != "step"}})

    for lbl, res in zip(iso_labels, iso_results):
        row = {"step": lbl}
        for m in METRICS:
            row[f"{m}_M"] = round(res[m], 3)
            row[f"delta_{m}_M"] = round(iso_deltas[lbl][m], 3)
        rows.append(row)

    iso_sum = {m: sum(iso_deltas[l][m] for l in iso_labels) for m in METRICS}
    rows.append({
        "step": "Sum of isolated",
        **{f"{m}_M": "" for m in METRICS},
        **{f"delta_{m}_M": round(iso_sum[m], 3) for m in METRICS},
    })
    rows.append({
        "step": "Interaction",
        **{f"{m}_M": "" for m in METRICS},
        **{f"delta_{m}_M": round(interaction[m], 3) for m in METRICS},
    })

    for out_dir in (OUT_DIR, RES_DIR):
        path = os.path.join(out_dir, "waterfall_attribution.csv")
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Waterfall figure
# ---------------------------------------------------------------------------
_STEP_COLORS = [
    "#888888",  # v2 baseline
    "#4C72B0",  # +Rmax
    "#DD8452",  # +Holland & B
    "#55A868",  # +Asymmetry
    "#C44E52",  # +Decay (K-D)
]
_STEP_XLABELS = ["v2\n(base)", "+Rmax\nV&W", "+Holland\n& B", "+Asym", "+Decay\n(v3)"]
_LEGEND_LABELS = ["v2 baseline", "+Rmax V&W", "+Holland & B", "+Asymmetry", "+Decay (K-D)"]


def _plot_waterfall(cumulative, out_path):
    """3-panel waterfall cascade (one per metric), saved to out_path."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    fig.suptitle(
        "Physics Attribution: v2 -> v3  |  Florida Homeowners",
        fontsize=13, fontweight="bold",
    )

    x_pos = list(range(len(cumulative)))

    for ax, metric in zip(axes, METRICS):
        values = [r[metric] for _, r in cumulative]

        # Bar geometry: bar 0 is the absolute baseline; bars 1-4 are floating deltas.
        bottoms = [0.0]
        heights = [values[0]]
        for i in range(1, len(values)):
            delta = values[i] - values[i - 1]
            bottoms.append(min(values[i - 1], values[i]))
            heights.append(abs(delta))

        ax.bar(x_pos, heights, bottom=bottoms, color=_STEP_COLORS,
               edgecolor="white", linewidth=0.8, width=0.6)

        # Connector dashes between steps
        for i in range(len(values) - 1):
            ax.plot([i + 0.31, i + 0.69], [values[i], values[i]],
                    color="#555555", linewidth=0.8, linestyle="--")

        # Annotations: absolute value for bar 0; signed delta for bars 1-4
        ax.text(0, bottoms[0] + heights[0] / 2,
                f"{values[0]:.2f}", ha="center", va="center",
                fontsize=8, color="white", fontweight="bold")
        for i in range(1, len(values)):
            delta = values[i] - values[i - 1]
            mid_y = values[i - 1] + delta / 2
            sign = "+" if delta >= 0 else ""
            ax.text(i, mid_y, f"{sign}{delta:.2f}",
                    ha="center", va="center", fontsize=8,
                    color="white", fontweight="bold")

        ax.set_xticks(x_pos)
        ax.set_xticklabels(_STEP_XLABELS, fontsize=8)
        ax.set_ylabel("$M", fontsize=9)
        ax.set_title(_METRIC_TITLES[metric], fontsize=10)
        ax.set_xlim(-0.5, len(values) - 0.5)
        ax.set_ylim(0, max(values) * 1.20)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    legend_handles = [
        mpatches.Patch(color=c, label=l)
        for c, l in zip(_STEP_COLORS, _LEGEND_LABELS)
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.04), fontsize=9)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Attribution waterfall: v2 -> v3 physics decomposition."
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Pass --quick to run_all.py (1k-year runs, fast iteration).",
    )
    args = parser.parse_args()
    run_all_args = ["--quick"] if args.quick else []

    print()
    print("=" * 60)
    print("  Attribution Waterfall  --  8 subprocess configs")
    mode = "--quick (1k years, MC noise expected)" if args.quick else "full 100k years"
    print(f"  Mode: {mode}")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # 5 main waterfall configs
    # -----------------------------------------------------------------------
    print("\n[1/2] Main waterfall configs:")
    main_results = []
    for label, config_env in MAIN_CONFIGS:
        main_results.append(_run_config(label, config_env, run_all_args))
    c0, c1, c2, c3, c4 = main_results  # noqa: F841 (c2, c3 unused directly but kept for clarity)

    # -----------------------------------------------------------------------
    # 3 sensitivity configs  (S1 = c1, no extra subprocess)
    # -----------------------------------------------------------------------
    print("\n[2/2] Sensitivity configs:")
    s_results = []
    for label, config_env in SENSITIVITY_CONFIGS:
        s_results.append(_run_config(label, config_env, run_all_args))
    s1_res, s2_res, s3_res, s4_res = c1, s_results[0], s_results[1], s_results[2]

    # -----------------------------------------------------------------------
    # Bit-identical self-check (full mode only)
    # -----------------------------------------------------------------------
    if not args.quick:
        print("\nChecking bit-identical anchors (full mode)...")
        failed = False
        for conf_label, result, anchor in [
            ("Config 0 (v2)", c0, _V2_ANCHORS),
            ("Config 4 (v3)", c4, _V3_ANCHORS),
        ]:
            for key in METRICS:
                diff = abs(result[key] - anchor[key])
                ok = diff <= _ANCHOR_TOL[key]
                status = "[OK]  " if ok else "[FAIL]"
                print(
                    f"  {status} {conf_label} {key}: "
                    f"got {result[key]:.4f}  expected {anchor[key]:.4f}  "
                    f"diff {diff:.4f}  tol +/-{_ANCHOR_TOL[key]}"
                )
                if not ok:
                    failed = True
        if failed:
            sys.exit(
                "\n[ABORT] Anchor mismatch -- env-var override or subprocess wiring is broken."
            )
    else:
        print("\n[NOTE] Anchor self-check skipped in --quick mode (MC noise at 1k years).")

    # -----------------------------------------------------------------------
    # Build attribution and print table
    # -----------------------------------------------------------------------
    cumulative = list(zip(
        ["v2 baseline", "+Rmax V&W", "+Holland & B", "+Asymmetry", "+Decay (v3)"],
        main_results,
    ))
    iso_labels  = ["+Rmax only (=C1)", "+Holland&B only(*)", "+Asymmetry only", "+Decay only"]
    iso_results = [s1_res, s2_res, s3_res, s4_res]

    iso_deltas, interaction = _print_and_return_table(
        cumulative, iso_labels, iso_results, c0, c4
    )

    # -----------------------------------------------------------------------
    # Save CSV
    # -----------------------------------------------------------------------
    _save_csvs(cumulative, iso_labels, iso_results, iso_deltas, interaction, c0, c4)

    # -----------------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------------
    _plot_waterfall(cumulative, os.path.join(OUT_DIR, "waterfall.png"))

    print("\nDone.")


if __name__ == "__main__":
    main()
