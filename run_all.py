"""
Pipeline orchestrator -- Florida hurricane cat model.

Runs every step in order, aborting with a clear message on first failure.
Each step is timed independently; a summary table is printed at the end.

Steps
-----
  1  data/generate_exposure.py   synthetic FL portfolio (1k locations, 500M TIV)
  2  model/hazard.py              stochastic moving-track hazard (demo + asserts)
  3  model/vulnerability.py       HAZUS-anchored damage curves (demo + asserts)
  4  model/loss.py                100k-year loss catalog (ground-up + gross + EventId)
  5  model/reinsurance.py         per-occurrence XoL programme + gross vs net OEP
  5.5 model/outputs.py            standard YLT + SELT (sim -> ylt.csv + elt.csv)
  6  model/summary.py             metrics table (reads ylt.csv) + outputs/ep_master.png

Usage
-----
    python run_all.py            full pipeline (~60s on a modern laptop)
    python run_all.py --quick    smoke run: loss.py uses 1,000 years (~15s total)
"""

import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))


def _run_step(name, cmd):
    """Execute one pipeline step.  Returns elapsed seconds; aborts on failure."""
    print()
    print("=" * 66)
    print(f"  {name}")
    print("=" * 66)
    t0      = time.perf_counter()
    result  = subprocess.run(cmd, cwd=ROOT)
    elapsed = time.perf_counter() - t0
    if result.returncode != 0:
        print()
        print(f"[FAIL]  '{name}' exited with code {result.returncode}.")
        print("        Pipeline aborted.")
        sys.exit(result.returncode)
    print(f"\n[OK]  {name} -- {elapsed:.1f}s")
    return elapsed


def main():
    parser = argparse.ArgumentParser(
        description="Run the full Florida hurricane cat model pipeline."
    )
    parser.add_argument(
        "--quick", action="store_true",
        help=(
            "Smoke run: pass --quick to loss.py so it uses 1,000 simulated years "
            "instead of 100,000.  Tests the entire pipeline end-to-end in ~15s."
        ),
    )
    parser.add_argument(
        "--results-dir", default="results",
        help=(
            "Directory for summary_metrics.csv output (default: results/). "
            "Used by analysis/waterfall.py to isolate subprocess runs from the "
            "production results/summary_metrics.csv."
        ),
    )
    args = parser.parse_args()

    py = sys.executable

    loss_cmd = [py, "-m", "model.loss"]
    if args.quick:
        loss_cmd.append("--quick")

    summary_cmd = [py, "-m", "model.summary"]
    if args.results_dir != "results":
        summary_cmd += ["--results-dir", args.results_dir]

    steps = [
        ("Step 1 | Exposure generation   (data/generate_exposure.py)",
         [py, "-m", "data.generate_exposure"]),

        ("Step 2 | Hazard module          (model/hazard.py -- demo + asserts)",
         [py, "-m", "model.hazard"]),

        ("Step 3 | Vulnerability module   (model/vulnerability.py -- demo + asserts)",
         [py, "-m", "model.vulnerability"]),

        ("Step 4 | Loss simulation        (model/loss.py)",
         loss_cmd),

        ("Step 5 | Reinsurance            (model/reinsurance.py)",
         [py, "-m", "model.reinsurance"]),

        ("Step 5.5 | YLT + SELT build     (model/outputs.py)",
         [py, "-m", "model.outputs"]),

        ("Step 6 | Summary + master plot  (model/summary.py)",
         summary_cmd),
    ]

    t_start = time.perf_counter()
    timings = []

    for name, cmd in steps:
        elapsed = _run_step(name, cmd)
        timings.append((name, elapsed))

    total = time.perf_counter() - t_start

    print()
    print("=" * 66)
    print("PIPELINE OK" + ("  (--quick mode: 1,000 simulated years)" if args.quick else ""))
    print("=" * 66)
    for name, t in timings:
        label = name.split("|")[-1].strip()
        print(f"  {label:<48} {t:>7.1f}s")
    print(f"  {'TOTAL':<48} {total:>7.1f}s")
    print()
    print("Outputs:")
    print("  outputs/vulnerability_curves.png")
    print("  outputs/ep_gross_vs_net.png")
    print("  outputs/ep_master.png")
    print("  results/events.csv          (one row per event, EventId first column)")
    print("  results/annual_losses.csv")
    print("  results/events_net.csv      (EventId carried through from events.csv)")
    print("  results/ylt.csv             (Year Loss Table — single source for EP metrics)")
    print("  results/elt.csv             (Sampled Event Loss Table — one row per event)")
    print("  results/summary_metrics.csv")
    print()


if __name__ == "__main__":
    main()
