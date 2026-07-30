"""Plot extinction fraction vs. nutrient/antibiotic timing offset (famine protocol).

Consumes the timing-sensitivity sweep produced by
scripts/simulate/sim_famine_timing.sbatch (-> simulation_test_controlenv_wnutr.py):
for each rep batch it writes
    {RESULTS_DIR}/a<ab:.2f>_n<nutrient>_value_check/<PROTOCOL>/extinction_summary_rep<rep>.pkl
each holding, per timing offset (in 12-min decision steps),
    final_num_cells[offset] = [final population per rep]   (0.0 => extinction)
plus offsets_steps / offsets_min / delta_t / num_of_reps.

This script pools the per-rep final populations across ALL rep batches, computes the
extinction fraction per offset (fraction with final count == 0, matching
plot_control.py: extinction = 1 if final cell count == 0), bootstraps a mean +/- std
over reps (n_bootstraps, like plot_control.py), and plots extinction fraction vs offset.
Offset sign convention (from the sim): + = nutrient reduced AFTER antibiotic applied.

It ALSO writes the pooled numbers to a CSV alongside the figures. Every output filename
carries the protocol/antibiotic/nutrient tag AND a run timestamp, so re-running never
overwrites earlier figures or CSVs.

Run with the RL_bact env python (the pkls were written with its numpy):
    /storage/home/hcoda1/0/jkratz3/r-jkratz3-0/envs/RL_bact/bin/python \
        plot_famine_timing.py \
        --results_dir /storage/project/r-sbanerjee347-0/jkratz3/pnas_rl/famine_timing_sensitivity \
        --protocol Famine --antibiotic 3.72 --nutrient_range 1_3

    (all four flags default to the values below, so it can be run with no args.)

Output (written under plotting/figures_jpg and plotting/figures_pdf):
    famine_timing_<PROTOCOL>_a<ab:.2f>_n<nutrient>_<YYYYmmdd_HHMMSS>.jpg / .pdf
    famine_timing_<PROTOCOL>_a<ab:.2f>_n<nutrient>_<YYYYmmdd_HHMMSS>.csv   (pooled stats)
"""

import argparse
import glob
import os
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

# purple = famine palette used for the manual Famine protocol in plot_control.py
BAR_COLOR = "purple"
N_BOOTSTRAPS = 1000

FIG_DIR = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results_dir", type=Path,
                   default=Path("/storage/project/r-sbanerjee347-0/jkratz3/pnas_rl/famine_timing_sensitivity"),
                   help="sweep output root (the RESULTS_DIR of sim_famine_timing.sbatch)")
    p.add_argument("--protocol", default="Famine", help="Famine or Feast")
    p.add_argument("--antibiotic", type=float, default=3.72, help="antibiotic 'on' level")
    p.add_argument("--nutrient_range", default="1_3", help="underscore-joined nutrient actions")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for the bootstrap")
    return p.parse_args()


def load_pooled_finals(cell_dir):
    """Pool per-rep final populations across all extinction_summary_rep*.pkl in cell_dir.

    Returns (finals_by_offset, delta_t, n_batches):
        finals_by_offset : dict offset(int) -> np.array of final populations (all reps)
    """
    summary_files = sorted(glob.glob(str(cell_dir / "extinction_summary_rep*.pkl")))
    if not summary_files:
        raise FileNotFoundError(
            f"no extinction_summary_rep*.pkl found in {cell_dir}. "
            "Has the sweep (sim_famine_timing.sbatch) run and finished?"
        )

    finals_by_offset = {}
    delta_t = None
    for sf in summary_files:
        with open(sf, "rb") as f:
            s = pickle.load(f)
        delta_t = s["delta_t"]
        for offset, finals in s["final_num_cells"].items():
            finals_by_offset.setdefault(int(offset), []).extend(finals)

    finals_by_offset = {k: np.asarray(v, dtype=float)
                        for k, v in sorted(finals_by_offset.items())}
    return finals_by_offset, delta_t, len(summary_files)


def bootstrap_extinction(extinction, n_boot, rng):
    """Bootstrap mean +/- std of the extinction indicator vector (matches plot_control.py)."""
    boot_means = [np.mean(rng.choice(extinction, size=len(extinction), replace=True))
                  for _ in range(n_boot)]
    return float(np.mean(boot_means)), float(np.std(boot_means))


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    ab_tag = f"a{args.antibiotic:.2f}"
    cell_dir = (args.results_dir /
                f"{ab_tag}_n{args.nutrient_range}_value_check" / args.protocol)

    finals_by_offset, delta_t, n_batches = load_pooled_finals(cell_dir)

    offsets = np.array(sorted(finals_by_offset))          # in decision steps
    offsets_min = offsets * delta_t * 60.0                 # in minutes

    ext_frac, boot_mean, boot_std, n_reps = [], [], [], []
    for off in offsets:
        finals = finals_by_offset[off]
        extinction = (finals == 0).astype(int)
        ext_frac.append(float(np.mean(extinction)))
        bm, bs = bootstrap_extinction(extinction, N_BOOTSTRAPS, rng)
        boot_mean.append(bm)
        boot_std.append(bs)
        n_reps.append(len(extinction))
    ext_frac = np.array(ext_frac)
    boot_mean = np.array(boot_mean)
    boot_std = np.array(boot_std)

    print(f"Pooled {n_batches} rep-batch summaries from {cell_dir}")
    for off, om, ef, bm, bs, n in zip(offsets, offsets_min, ext_frac,
                                      boot_mean, boot_std, n_reps):
        print(f"  offset {off:+d} steps ({om:+.0f} min): "
              f"extinction {ef:.3f} (boot {bm:.3f}+/-{bs:.3f}), n={n}")

    # ----- plot: extinction fraction vs offset -----
    fig, ax = plt.subplots(figsize=(6, 4.5))
    x = np.arange(len(offsets))
    ax.bar(x, boot_mean, yerr=boot_std, capsize=5,
           color=BAR_COLOR, edgecolor=BAR_COLOR, alpha=0.85)
    # highlight the simultaneous (offset 0) baseline
    if 0 in set(offsets):
        zi = int(np.where(offsets == 0)[0][0])
        ax.bar(x[zi], boot_mean[zi], yerr=boot_std[zi], capsize=5,
               color="black", edgecolor="black", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{om:+.0f}\n({off:+d})" for off, om in zip(offsets, offsets_min)])
    ax.set_xlabel("Nutrient-reduction offset [min]  (decision steps)\n"
                  "(+ = nutrient reduced after antibiotic)")
    ax.set_ylabel("Extinction fraction")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"{args.protocol} timing sensitivity  "
                 f"({ab_tag}, n={args.nutrient_range}, {n_reps[0]} reps/offset)")
    fig.tight_layout()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"famine_timing_{args.protocol}_{ab_tag}_n{args.nutrient_range}_{stamp}"
    (FIG_DIR / "figures_jpg").mkdir(exist_ok=True)
    (FIG_DIR / "figures_pdf").mkdir(exist_ok=True)
    jpg = FIG_DIR / "figures_jpg" / f"{base}.jpg"
    pdf = FIG_DIR / "figures_pdf" / f"{base}.pdf"
    csv = FIG_DIR / "figures_jpg" / f"{base}.csv"
    fig.savefig(jpg, dpi=600, bbox_inches="tight")
    fig.savefig(pdf, dpi=600, bbox_inches="tight")

    with open(csv, "w") as f:
        f.write("offset_steps,offset_min,extinction_frac,bootstrap_mean,bootstrap_std,n_reps\n")
        for off, om, ef, bm, bs, n in zip(offsets, offsets_min, ext_frac,
                                          boot_mean, boot_std, n_reps):
            f.write(f"{off},{om:.1f},{ef:.4f},{bm:.4f},{bs:.4f},{n}\n")

    print(f"Saved: {jpg}")
    print(f"Saved: {pdf}")
    print(f"Saved: {csv}")


if __name__ == "__main__":
    main()
