#!/usr/bin/env python
"""Draw the three figures of the README from the CIF files and result tables.

    python make_figures.py

Writes figures/fig_glass.png, figures/fig_host.png, figures/fig_aflow_speed.png. Existing files are overwritten.
Colours are the Okabe-Ito set, distinguishable under the common forms of colour-vision deficiency.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ase.io import read

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import check_structures as cs  # noqa: E402

BLUE, ORANGE, GREEN, VERM, SKY, GREY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#56B4E9", "#555555"
plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.7, "axes.titlesize": 9,
                     "axes.titleweight": "normal", "axes.titlelocation": "left", "xtick.major.width": 0.7, "ytick.major.width": 0.7, "legend.frameon": False})
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)


def glass_cells():
    cells = {}
    for p in sorted((ROOT / "structures" / "glass").glob("glass_*.cif")):
        ctz, d3, seed = p.stem.replace("glass_", "").split("_")
        cells[(int(ctz[3:]), d3 == "D3", int(seed[4:]))] = read(str(p))
    return cells


def fig_glass() -> None:
    tbl = pd.read_csv(ROOT / "results" / "analysis" / "glass_metrics.csv")
    cells = glass_cells()
    fig, ax = plt.subplots(1, 3, figsize=(10.2, 3.1), constrained_layout=True)

    a = ax[0]
    expt = {30: 3.05, 40: 3.24, 50: 3.34}
    for c, v in expt.items():
        a.plot([c - 2.2, c + 2.2], [v, v], color="black", lw=1.6, solid_capstyle="butt")
    a.text(50.6, 3.34, "measured\nglass-ceramic", va="center", fontsize=7.5, color=GREY)
    d = tbl[tbl.d3 == 1]
    a.scatter(d.ctz + np.where(d.seed == 0, -0.7, 0.7), d.density_final_g_cm3, s=26, color=VERM, label="MACE-MPA-0 + D3", zorder=3)
    n = tbl[tbl.d3 == 0]
    a.scatter(n.ctz + np.where(n.seed == 0, -0.7, 0.7), n.density_final_g_cm3, s=26, color=BLUE, label="MACE-MPA-0, no D3", zorder=3)
    a.set_xlim(26, 62)
    a.set_xticks([30, 40, 50])
    a.set_xlabel("CTZ content (wt % label of Wan et al.)")
    a.set_ylabel("density (g/cm$^3$)")
    a.set_title("A  density after the quench")
    a.legend(loc="lower right", fontsize=7.5, handletextpad=0.2)

    b = ax[1]
    nets = (("Si, B, Al", cs.FORMERS, BLUE), ("+ Ti", cs.FORMERS + ("Ti",), ORANGE), ("+ Ti, Zr", cs.FORMERS + ("Ti", "Zr"), GREEN))
    for label, net, col in nets:
        means = []
        for c in (30, 40, 50):
            v = [cs.bridging_fraction(cells[(c, True, s)], net) for s in (0, 1)]
            b.scatter([c - 0.5, c + 0.5], v, s=14, color=col, alpha=0.55, zorder=2)
            means.append(np.mean(v))
        b.plot([30, 40, 50], means, color=col, lw=1.6, zorder=3)
        b.text(51.2, means[-1], label, color=GREY, va="center", fontsize=7.5)
    b.set_xlim(27, 58)
    b.set_xticks([30, 40, 50])
    b.set_ylim(0, 0.9)
    b.set_xlabel("CTZ content (wt %)")
    b.set_ylabel("fraction of O bonded to two or more\nnetwork atoms")
    b.set_title("B  bridging oxygen depends on the network")

    c = ax[2]
    for ctz in (30, 40, 50):
        for s in (0, 1):
            at = cells[(ctz, True, s)]
            c.scatter(ctz + (-0.7 if s == 0 else 0.7), cs.nd_scan(at, 3.1), s=34, color=VERM, zorder=3)
    c.axhline(2, color=GREY, lw=0.6, ls=":")
    c.set_xlim(26, 56)
    c.set_xticks([30, 40, 50])
    c.set_yticks([1, 2, 3, 4])
    c.set_ylim(0.5, 4.6)
    c.set_xlabel("CTZ content (wt %)")
    c.set_ylabel("largest group of Nd sharing oxygen\n(6 Nd per cell)")
    c.set_title("C  Nd clustering, D3 runs")
    fig.savefig(OUT / "fig_glass.png", dpi=190)
    plt.close(fig)


def fig_host() -> None:
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.1), constrained_layout=True)
    a = ax[0]
    for recipe, col, lab in (("he6", VERM, "Si/Al/Ti/Zr/Sn/Ge + Cs + Sr"), ("csal", BLUE, "Si/Al + 3 Cs")):
        df = pd.read_csv(ROOT / "results" / "host" / recipe / "energies.csv")
        e = np.sort(df["E_rel_eV"].to_numpy())
        a.step(e, np.arange(1, len(e) + 1) / len(e), where="post", color=col, lw=1.6, label=lab)
    a.set_xlabel("E - E$_{min}$ (eV), 100 random arrangements")
    a.set_ylabel("cumulative fraction")
    a.set_title("A  energy spread in chabazite")
    a.legend(loc="lower right", fontsize=7.5)
    b = ax[1]
    df = pd.read_csv(ROOT / "results" / "host" / "csal" / "energies.csv")
    rng = np.random.default_rng(0)
    b.scatter(df["Cs_CN"] + rng.uniform(-0.03, 0.03, len(df)), df["E_rel_eV"], s=16, color=BLUE, alpha=0.7)
    b.set_xlabel("mean Cs-O coordination (Cs-O < 3.9 Å)")
    b.set_ylabel("E - E$_{min}$ (eV)")
    b.set_title("B  Si/Al + 3 Cs: Spearman -0.32")
    fig.savefig(OUT / "fig_host.png", dpi=190)
    plt.close(fig)


def fig_aflow_speed() -> None:
    fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.2), constrained_layout=True)
    a = ax[0]
    tab = pd.read_csv(ROOT / "results" / "aflow" / "aflow_vs_mace.csv").set_index("name").drop("chabazite_framework_SiO2")
    x = np.arange(len(tab))
    a.bar(x - 0.2, tab["dV_V_mace_d30_vs_aflow_pct"], 0.38, color=BLUE, label="MACE-MPA-0")
    a.bar(x + 0.2, tab["dV_V_mace_d31_vs_aflow_pct"], 0.38, color=VERM, label="MACE-MPA-0 + D3")
    a.axhline(0, color="black", lw=0.7)
    a.set_xticks(x)
    a.set_xticklabels([n.split("_")[1] for n in tab.index], rotation=55, ha="right", fontsize=7.5)
    a.set_ylabel("cell volume vs AFLOW cell (%)")
    a.set_title("A  seven waste phases")
    a.legend(loc="lower left", fontsize=7.5)
    b = ax[1]
    rows = pd.DataFrame(json.loads((ROOT / "results" / "cxx" / "bench_results.json").read_text())["rows"])
    for col, lab, c, m in (("cxx_all_partial_rdfs_s", "C++ cell list, all pairs", VERM, "o"), ("ase_neighbor_list_s", "ASE neighbour list", GREEN, "s"), ("crisp_one_pair_rdf_s", "CRISP, one pair", BLUE, "^")):
        r = rows.dropna(subset=[col])
        b.plot(r["n_atoms"], r[col], marker=m, ms=4.5, lw=1.6, color=c, label=lab)
    b.set_xscale("log")
    b.set_yscale("log")
    b.set_xlabel("atoms")
    b.set_ylabel("time for g(r), one thread (s)")
    b.set_title("B  g(r) cost")
    b.legend(loc="lower right", fontsize=7.5)
    fig.savefig(OUT / "fig_aflow_speed.png", dpi=190)
    plt.close(fig)


if __name__ == "__main__":
    fig_glass()
    fig_host()
    fig_aflow_speed()
    print("wrote", ", ".join(sorted(p.name for p in OUT.glob("*.png"))))
