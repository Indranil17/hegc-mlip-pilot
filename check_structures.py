#!/usr/bin/env python
"""Recompute the claims of the README from the CIF files in structures/ and the result tables in results/.

Uses only ase, numpy, scipy, pandas and spglib. No simulation code, no force field, no C++ module is needed, and nothing
is read from a file that this script does not name. Every line prints PASS or FAIL with the number found.

    python check_structures.py

Sections: 1 glass cells, 2 chabazite host, 3 AFLOW phases, 4 ring counts, 5 stored benchmarks.
"""
from __future__ import annotations

import itertools
import json
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import spglib
from ase import units
from ase.io import read
from ase.neighborlist import neighbor_list
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent
STRUCT, RES, DATA = ROOT / "structures", ROOT / "results", ROOT / "data"

# metal-oxygen bond cutoffs in angstrom (first shell of the known oxides, short of the second shell)
CUT = {"Si": 2.0, "B": 1.9, "Al": 2.2, "Ti": 2.5, "Zr": 2.8, "Mo": 2.3, "Nd": 3.1, "Na": 3.2, "Ca": 3.1}
FORMERS = ("Si", "B", "Al")
CHARGE = {"Si": 4, "B": 3, "Al": 3, "Na": 1, "Ca": 2, "Ti": 4, "Zr": 4, "Nd": 3, "Mo": 6, "Sn": 4, "Ge": 4, "Cs": 1, "Sr": 2, "O": -2}
HOST_CUT = {"Cs": 3.9, "Sr": 3.2, "Al": 2.0}
SEED = 0
N_BOOT, N_PERM = 2000, 20000

bad: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    if not ok:
        bad.append(name)


def note(text: str) -> None:
    print(f"[NOTE] {text}")


# ---------------------------------------------------------------------------------------------------------------- glass
def bonds(atoms, cut: dict):
    """Centre-O pairs inside each centre's cutoff: (centre index, oxygen index, distance, symbols)."""
    sym = np.array(atoms.get_chemical_symbols())
    i, j, d = neighbor_list("ijd", atoms, max(cut.values()) + 0.01)
    m = np.array([sym[a] in cut and sym[b] == "O" and r <= cut[sym[a]] for a, b, r in zip(i, j, d)])
    return i[m], j[m], d[m], sym


def shared_oxygen_sizes(c, o, n, sym, el):
    """Size of the cluster each atom of element el belongs to; atoms are joined when they share an oxygen."""
    idx = np.where(sym == el)[0]
    sel = np.isin(c, idx)
    g = coo_matrix((np.ones(sel.sum()), (c[sel], o[sel])), shape=(n, n))
    _, lab = connected_components(g + g.T, directed=False)
    size = {k: int((lab[idx] == lab[k]).sum()) for k in idx}
    return np.array([size[k] for k in idx])


def glass_metrics(atoms, cut=CUT, formers=FORMERS) -> dict:
    n = len(atoms)
    c, o, d, sym = bonds(atoms, cut)
    out = {}
    for el in CUT:
        idx = np.where(sym == el)[0]
        cn = np.array([(c == k).sum() for k in idx])
        out[f"CN_{el}"] = float(cn.mean())
        if el == "B":
            out["B3_fraction"], out["B4_fraction"] = float((cn == 3).mean()), float((cn == 4).mean())
        if el in ("Si", "Mo"):
            out[f"{el}_CN4_fraction"] = float((cn == 4).mean())
    nfor = np.bincount(o[np.isin(sym[c], formers)], minlength=n)
    ox = np.where(sym == "O")[0]
    out["O_bridging"], out["O_nonbridging"], out["O_free"] = (float((nfor[ox] >= 2).mean()), float((nfor[ox] == 1).mean()), float((nfor[ox] == 0).mean()))
    si = np.where(sym == "Si")[0]
    bo = np.zeros(n, dtype=int)
    for a, b in zip(c, o):
        if sym[a] == "Si":
            bo[a] += int(nfor[b] >= 2)
    for q in range(5):
        out[f"Q{q}_Si"] = float((bo[si] == q).mean())
    for el in ("Mo", "Nd"):
        sizes = shared_oxygen_sizes(c, o, n, sym, el)
        out[f"{el}_largest_cluster"] = int(sizes.max())
        out[f"{el}_fraction_in_shared_O_clusters"] = float((sizes >= 2).mean())
    return out


def bridging_fraction(atoms, networks, shift: float = 0.0) -> float:
    """Fraction of oxygen bonded to two or more atoms of the listed elements (cutoffs from CUT, all moved by shift)."""
    c, o, d, sym = bonds(atoms, {el: CUT[el] + shift for el in networks})
    cnt = np.bincount(o, minlength=len(atoms))
    ox = np.where(sym == "O")[0]
    return float((cnt[ox] >= 2).mean())


def mo_scan(atoms, cut):
    """Mean Mo coordination, share of Mo with exactly four O, and whether any O is shared between two Mo."""
    c, o, d, sym = bonds(atoms, {"Mo": cut})
    mo = np.where(sym == "Mo")[0]
    cn = np.array([(c == k).sum() for k in mo])
    sizes = shared_oxygen_sizes(c, o, len(atoms), sym, "Mo")
    return cn, bool((sizes == 1).all())


def nd_scan(atoms, cut):
    c, o, d, sym = bonds(atoms, {"Nd": cut})
    return int(shared_oxygen_sizes(c, o, len(atoms), sym, "Nd").max())


def section_glass() -> None:
    print("\n== 1. melt-quenched glass cells (structures/glass)")
    tbl = pd.read_csv(RES / "analysis" / "glass_metrics.csv").set_index("tag")
    cells = {}
    for p in sorted((STRUCT / "glass").glob("glass_*.cif")):
        ctz, d3, seed = p.stem.replace("glass_", "").split("_")
        cells[f"{ctz}_d3{int(d3 == 'D3')}_{seed}"] = read(str(p))
    check("8 glass cells found", len(cells) == 8, str(sorted(cells)))

    worst, n_cmp = 0.0, 0
    for tag, at in cells.items():
        for k, v in glass_metrics(at).items():
            worst = max(worst, abs(float(tbl.loc[tag, k]) - v))
            n_cmp += 1
        q = sum(CHARGE[s] for s in at.get_chemical_symbols())
        rho = at.get_masses().sum() * 1.66053906660 / at.get_volume()
        check(f"{tag}: formal charge zero", q == 0, f"charge {q}, {len(at)} atoms, {at.get_chemical_formula()}")
        check(f"{tag}: density of the CIF cell within 1 % of the tabulated hold mean", abs(rho / tbl.loc[tag, "density_final_g_cm3"] - 1) < 0.01,
              f"CIF {rho:.3f}, table {tbl.loc[tag, 'density_final_g_cm3']:.3f} g/cm3")
    check("coordination, bridging O, Q^n, Mo and Nd cluster numbers equal the tabulated ones", worst < 1e-9, f"{n_cmp} numbers, largest difference {worst:.1e}")

    # what the nine-oxide mixtures contain: from Table 1 of the paper (glass/composition.py) and from the atoms of the cells
    import importlib.util

    spec = importlib.util.spec_from_file_location("composition", ROOT / "glass" / "composition.py")
    comp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(comp)
    from ase.data import atomic_masses, atomic_numbers

    def mass_of(formula):
        return sum(atomic_masses[atomic_numbers[el]] * n for el, n in formula.items())

    ox_mass = {ox: mass_of(f) for ox, f in comp.FORMULA.items()}
    nominal, cellwt = {}, {}
    for ctz in (30, 40, 50):
        w = {ox: comp.TABLE1_MOL[ox][ctz] * ox_mass[ox] for ox in ox_mass}
        nominal[ctz] = 100 * (w["CaO"] + w["TiO2"] + w["ZrO2"]) / sum(w.values())
        at = cells[f"ctz{ctz}_d31_seed0"]
        n = {el: at.get_chemical_symbols().count(el) for el in ("Ca", "Ti", "Zr")}
        cellwt[ctz] = 100 * (n["Ca"] * ox_mass["CaO"] + n["Ti"] * ox_mass["TiO2"] + n["Zr"] * ox_mass["ZrO2"]) / at.get_masses().sum()
    check("CaO + TiO2 + ZrO2 are 27.4, 35.2 and 43.1 wt % of the Table 1 mixtures and 27.5, 35.3 and 42.7 wt % of the cells (the label CTZ-x is the paper's name, not this number)",
          [round(nominal[c], 1) for c in (30, 40, 50)] == [27.4, 35.2, 43.1] and [round(cellwt[c], 1) for c in (30, 40, 50)] == [27.5, 35.3, 42.7],
          f"Table 1 {', '.join(f'{nominal[c]:.1f}' for c in (30, 40, 50))}; cells {', '.join(f'{cellwt[c]:.1f}' for c in (30, 40, 50))}")
    check("the Table 1 columns for CTZ-30, 40 and 50 sum to 100 mol % within 0.05", all(abs(sum(comp.TABLE1_MOL[ox][c] for ox in comp.TABLE1_MOL) - 100) < 0.05 for c in (30, 40, 50)),
          ", ".join(f"{sum(comp.TABLE1_MOL[ox][c] for ox in comp.TABLE1_MOL):.2f}" for c in (30, 40, 50)))

    # the melt destroyed the starting packing
    ratio = tbl["msd_cation_end_melt_A2"] / tbl["msd_ref_ell2_A2"]
    check("cation mean squared displacement at the end of the 5000 K melt is 24 to 32 times (V/N)^(2/3), 131 to 177 A^2", round(ratio.min()) == 24 and round(ratio.max()) == 32 and
          round(tbl["msd_cation_end_melt_A2"].min()) == 131 and round(tbl["msd_cation_end_melt_A2"].max()) == 177,
          f"{ratio.min():.1f} to {ratio.max():.1f} times, {tbl['msd_cation_end_melt_A2'].min():.0f} to {tbl['msd_cation_end_melt_A2'].max():.0f} A^2")

    # density
    rho = tbl["density_final_g_cm3"]
    expt = tbl["density_expt_glass_ceramic_g_cm3"]
    d3 = tbl[tbl.d3 == 1]
    err = 100 * (rho / expt - 1)
    check("with D3 the density is 6.0 to 11.7 % below the glass-ceramic value", round(err[tbl.d3 == 1].max(), 1) == -6.0 and round(err[tbl.d3 == 1].min(), 1) == -11.7,
          f"{err[tbl.d3 == 1].min():.1f} to {err[tbl.d3 == 1].max():.1f} %")
    check("without D3 it is 23.5 to 23.9 % below", round(err[tbl.d3 == 0].max(), 1) == -23.5 and round(err[tbl.d3 == 0].min(), 1) == -23.9,
          f"{err[tbl.d3 == 0].min():.1f} to {err[tbl.d3 == 0].max():.1f} %")
    with_, without = rho[(tbl.ctz == 40) & (tbl.d3 == 1)].to_numpy(), rho[(tbl.ctz == 40) & (tbl.d3 == 0)].to_numpy()
    up = sorted(100 * (a / b - 1) for a in with_ for b in without)
    check("D3 raises the CTZ-40 density by 15.4 to 22.6 % (four run pairs)", round(up[0], 1) == 15.4 and round(up[-1], 1) == 22.6, f"{up[0]:.1f} to {up[-1]:.1f} %")
    means = d3.groupby("ctz")["density_final_g_cm3"].mean()
    check("D3 mean densities 2.86, 2.94, 3.13 g/cm3", [round(v, 2) for v in means] == [2.86, 2.94, 3.13], str([round(v, 3) for v in means]))
    # exact permutation test, two runs per group: the smallest attainable p-value
    vals = {c: d3[d3.ctz == c]["density_final_g_cm3"].to_numpy() for c in (30, 40, 50)}
    pvals = []
    for a, b in ((30, 40), (40, 50), (30, 50)):
        pooled = np.concatenate([vals[a], vals[b]])
        obs = vals[b].mean() - vals[a].mean()
        diffs = [pooled[list(s)].mean() - np.delete(pooled, list(s)).mean() for s in itertools.combinations(range(4), 2)]
        p1 = float(np.mean(np.array(diffs) >= obs - 1e-12))
        pvals.append(p1)
        note(f"CTZ-{a} against CTZ-{b}: mean step {obs:+.3f} g/cm3, exact permutation p (one-sided) {p1:.3f}, 6 labelings")
    check("no density step is resolved: the smallest one-sided p of the three steps is 1/6 = 0.17 (the smallest value 2 runs per group can give)", round(min(pvals), 2) == 0.17, f"p = {', '.join(f'{x:.2f}' for x in pvals)}")
    check("pressure of the final state -0.39 to -0.27 GPa in all D3 and no-D3 runs", round(tbl["pressure_final_GPa"].min(), 2) == -0.39 and round(tbl["pressure_final_GPa"].max(), 2) == -0.27,
          f"{tbl['pressure_final_GPa'].min():.2f} to {tbl['pressure_final_GPa'].max():.2f} GPa")

    # quench-to-quench scatter of the energy per atom at fixed composition (two D3 runs each)
    scat = [1000 * abs(np.diff(d3[d3.ctz == c]["epot_final_eV_atom"].to_numpy())[0]) for c in (30, 40, 50)]
    check("energy per atom of the two quenches of one composition differs by 11.2, 18.3 and 9.6 meV/atom", [round(x, 1) for x in scat] == [11.2, 18.3, 9.6], ", ".join(f"{x:.1f}" for x in scat))

    # cost of the runs
    summ = {t: json.loads((RES / "glass" / f"summary_{t}.json").read_text()) for t in cells}
    pph = {t: 3.6 / (v["ms_per_step"] / 1000) for t, v in summ.items()}
    wall = {t: v["wall_s"] / 3600 for t, v in summ.items()}
    dd = [t for t in cells if "_d31_" in t]
    nn = [t for t in cells if "_d30_" in t]
    check("with D3 8.4 to 13.8 ps per GPU-hour and 5.6 to 9.2 h per 77 ps; without D3 19 to 41 ps per GPU-hour",
          (round(min(pph[t] for t in dd), 1), round(max(pph[t] for t in dd), 1), round(min(wall[t] for t in dd), 1), round(max(wall[t] for t in dd), 1)) == (8.4, 13.8, 5.6, 9.2)
          and round(min(pph[t] for t in nn)) == 19 and round(max(pph[t] for t in nn)) == 41,
          f"D3 {min(pph[t] for t in dd):.1f} to {max(pph[t] for t in dd):.1f} ps/h, {min(wall[t] for t in dd):.1f} to {max(wall[t] for t in dd):.1f} h; no D3 {min(pph[t] for t in nn):.0f} to {max(pph[t] for t in nn):.0f} ps/h")
    ms = [summ[t]["ms_per_step"] for t in dd]
    check("same job, different GPU node: step time differs by a factor of 1.65 (D3 runs)", round(max(ms) / min(ms), 2) == 1.65, f"{min(ms):.0f} to {max(ms):.0f} ms per step")
    check("ring counts of the Si,B,Al network are 0.03 to 0.18 per network atom", round(tbl["rings_per_T"].min(), 2) == 0.03 and round(tbl["rings_per_T"].max(), 2) == 0.18, f"{tbl['rings_per_T'].min():.3f} to {tbl['rings_per_T'].max():.3f} (stored, from the ring counter)")

    # local coordination
    si4 = [glass_metrics(a)["Si_CN4_fraction"] for a in cells.values()]
    check("Si is four-coordinate: CN = 4 for 97 to 100 % of the Si in every cell", round(100 * min(si4)) == 97 and round(100 * max(si4)) == 100, f"{100 * min(si4):.1f} to {100 * max(si4):.1f} %")
    b3 = [glass_metrics(a)["B3_fraction"] for a in cells.values()]
    check("three-coordinate boron: 0.75 to 1.0 of the B atoms", round(min(b3), 2) >= 0.75 and max(b3) <= 1.0, f"{min(b3):.2f} to {max(b3):.2f}")
    bl = {}
    for el in ("Si", "B", "Mo", "Ti", "Zr", "Nd", "Na", "Ca"):
        v = []
        for at in cells.values():
            c, o, d, sym = bonds(at, {el: CUT[el]})
            v.append(d.mean())
        bl[el] = (min(v), max(v))
    note("mean M-O distance inside the cutoff, range over cells (A): " + ", ".join(f"{el} {lo:.2f}-{hi:.2f}" for el, (lo, hi) in bl.items()))

    # mean coordination and Q^n of Si, D3 runs, mean of the two seeds
    print("   coordination (cutoffs in A: " + ", ".join(f"{el} {CUT[el]}" for el in CUT) + "), D3 runs, mean of two seeds")
    gm = {(ctz, sd): glass_metrics(cells[f"ctz{ctz}_d31_seed{sd}"]) for ctz in (30, 40, 50) for sd in (0, 1)}
    mean = {ctz: {k: np.mean([gm[(ctz, sd)][k] for sd in (0, 1)]) for k in gm[(ctz, 0)]} for ctz in (30, 40, 50)}
    for k in ("CN_Si", "CN_B", "B3_fraction", "CN_Al", "CN_Ti", "CN_Zr", "CN_Mo", "CN_Nd", "CN_Na", "CN_Ca"):
        print(f"     {k:12s} " + "  ".join(f"CTZ-{ctz} {mean[ctz][k]:.2f}" for ctz in (30, 40, 50)))
    for label, nets in (("Si,B,Al", FORMERS), ("Si,B,Al,Ti,Zr", FORMERS + ("Ti", "Zr"))):
        print(f"   Q^n of Si over the {label} network, D3 runs, mean of two seeds (Q0 .. Q4)")
        for ctz in (30, 40, 50):
            qq = np.mean([[glass_metrics(cells[f"ctz{ctz}_d31_seed{sd}"], formers=nets)[f"Q{n}_Si"] for n in range(5)] for sd in (0, 1)], axis=0)
            print(f"     CTZ-{ctz}  " + "  ".join(f"{x:.2f}" for x in qq))

    # network definition and bridging oxygen
    print("   bridging oxygen by network definition (D3 runs, mean of two seeds)")
    rows = {}
    for label, nets in (("Si,B,Al", FORMERS), ("Si,B,Al,Ti", FORMERS + ("Ti",)), ("Si,B,Al,Ti,Zr", FORMERS + ("Ti", "Zr"))):
        rows[label] = {ctz: np.mean([bridging_fraction(at, nets) for t, at in cells.items() if t.startswith(f"ctz{ctz}_d31")]) for ctz in (30, 40, 50)}
        print(f"     {label:15s} CTZ-30 {rows[label][30]:.3f}  CTZ-40 {rows[label][40]:.3f}  CTZ-50 {rows[label][50]:.3f}")
    check("bridging O over Si,B,Al falls with CTZ: 0.37, 0.30, 0.22", [round(rows["Si,B,Al"][c], 2) for c in (30, 40, 50)] == [0.37, 0.30, 0.22])
    check("with Ti added it does not fall: 0.57, 0.57, 0.54", [round(rows["Si,B,Al,Ti"][c], 2) for c in (30, 40, 50)] == [0.57, 0.57, 0.54])
    check("with Ti and Zr added it rises: 0.71, 0.74, 0.76", [round(rows["Si,B,Al,Ti,Zr"][c], 2) for c in (30, 40, 50)] == [0.71, 0.74, 0.76])
    for shift in (-0.1, 0.1):
        v = {ctz: [bridging_fraction(cells[f"ctz{ctz}_d31_seed{sd}"], FORMERS, shift) for sd in (0, 1)] for ctz in (30, 40, 50)}
        apart = max(v[40]) < min(v[30]) and max(v[50]) < min(v[40])
        check(f"all Si,B,Al cutoffs moved by {shift:+.1f} A: bridging O still falls with CTZ and the two runs of each composition do not overlap",
              apart and [round(np.mean(v[c]), 2) for c in (30, 40, 50)] == [0.37, 0.30, 0.22], ", ".join(f"{np.mean(v[c]):.3f}" for c in (30, 40, 50)))
    q = {ctz: np.mean([glass_metrics(at)["Q4_Si"] for t, at in cells.items() if t.startswith(f"ctz{ctz}_d31")]) for ctz in (30, 40, 50)}
    note(f"Q4 fraction of Si over the Si,B,Al network (D3 runs): {q[30]:.2f}, {q[40]:.2f}, {q[50]:.2f}; it changes with the network definition as well")

    # Mo
    print("   Mo against the Mo-O cutoff (8 cells, 4 Mo each)")
    iso = {}
    for cut in (1.9, 2.0, 2.3, 2.6):
        cn_all, isolated = [], 0
        for at in cells.values():
            cn, ok = mo_scan(at, cut)
            cn_all += list(cn)
            isolated += ok
        iso[cut] = isolated
        cn_all = np.array(cn_all)
        print(f"     cutoff {cut:.1f}: mean CN {cn_all.mean():.2f}, CN = 4 for {100 * np.mean(cn_all == 4):.0f} % of Mo, CN = 5 for {100 * np.mean(cn_all == 5):.0f} %, no O shared between Mo in {isolated} of 8 cells")
    check("no O shared between two Mo in 7 of 8 cells at 2.0 and 2.3 A, 6 of 8 at 2.6 A", (iso[2.0], iso[2.3], iso[2.6]) == (7, 7, 6), str(iso))
    dmo = []
    for at in cells.values():
        sym = np.array(at.get_chemical_symbols())
        i, j, d = neighbor_list("ijd", at, 12.0)
        m = (sym[i] == "Mo") & (sym[j] == "Mo")
        dmo.append(d[m].min())
    check("shortest Mo-Mo distance in any cell is 3.33 A", round(min(dmo), 2) == 3.33, f"{min(dmo):.3f} A")

    # Nd
    print("   Nd: largest group of Nd that share oxygen, against the Nd-O cutoff")
    big = {}
    for cut in (2.9, 3.1, 3.3):
        big[cut] = {t: nd_scan(at, cut) for t, at in cells.items() if "_d31_" in t}
    for t in sorted(big[3.1]):
        print(f"     {t:18s} " + "  ".join(f"{cut}: {big[cut][t]}" for cut in big))
    ok50 = all(min(big[c][t] for t in big[c] if t.startswith("ctz50")) >= 3 for c in big)
    ok_other = all(max(big[c][t] for t in big[c] if not t.startswith("ctz50")) <= 2 for c in big)
    check("Nd groups of 3 or 4 occur only at CTZ-50; at most 2 elsewhere (D3 runs, all three cutoffs)", ok50 and ok_other)
    dnd = {}
    for t, at in cells.items():
        if "d31" not in t:
            continue
        sym = np.array(at.get_chemical_symbols())
        i, j, d = neighbor_list("ijd", at, 12.0)
        dnd[t] = d[(sym[i] == "Nd") & (sym[j] == "Nd")].min()
    low = min(v for t, v in dnd.items() if t.startswith("ctz50")), max(v for t, v in dnd.items() if t.startswith("ctz50"))
    oth = min(v for t, v in dnd.items() if not t.startswith("ctz50")), max(v for t, v in dnd.items() if not t.startswith("ctz50"))
    check("shortest Nd-Nd distance 3.68 to 3.70 A at CTZ-50 and 4.11 to 5.59 A at CTZ-30 and 40", [round(x, 2) for x in low + oth] == [3.68, 3.70, 4.11, 5.59],
          f"{low[0]:.2f}-{low[1]:.2f} and {oth[0]:.2f}-{oth[1]:.2f} A")
    note("each cell holds 4 Mo and 6 Nd and each composition has 2 cells: the Mo and Nd findings are observations, not statistics")

    # the motif files cut out of the cells
    m = {p.stem: read(str(p)) for p in sorted((STRUCT / "motifs").glob("*.cif"))}
    nd4, nd2 = m["nd_cluster_ctz50_seed0"], m["nd_largest_ctz40_seed0"]
    check("motif nd_cluster_ctz50_seed0: 4 Nd joined through shared O", nd4.get_chemical_symbols().count("Nd") == 4 and nd_scan(nd4, CUT["Nd"]) == 4, nd4.get_chemical_formula())
    check("motif nd_largest_ctz40_seed0: 2 Nd joined through shared O", nd2.get_chemical_symbols().count("Nd") == 2 and nd_scan(nd2, CUT["Nd"]) == 2, nd2.get_chemical_formula())
    for name, want in (("mo_cn4_ctz30_d31_seed0", 4), ("mo_cn5_ctz30_d31_seed0", 5)):
        cn, iso = mo_scan(m[name], CUT["Mo"])
        check(f"motif {name}: one Mo with {want} O, no O shared with another Mo", list(cn) == [want] and iso, m[name].get_chemical_formula())
    pair = m["mo_shared_O_pair_ctz50_d31_seed0"]
    cn, iso = mo_scan(pair, CUT["Mo"])
    sym = np.array(pair.get_chemical_symbols())
    dmm = pair.get_distance(*np.where(sym == "Mo")[0], mic=True)
    check("motif mo_shared_O_pair_ctz50_d31_seed0: two Mo that share O, Mo-Mo 3.33 A", (not iso) and round(dmm, 2) == 3.33, f"{pair.get_chemical_formula()}, Mo-Mo {dmm:.3f} A, CN {[int(x) for x in cn]}")


# ---------------------------------------------------------------------------------------------------------------- host
def boltzmann_neff(e_rel: np.ndarray, T: float) -> float:
    w = np.exp(-e_rel / (units.kB * T))
    p = w / w.sum()
    return float(np.exp(-(p * np.log(p + 1e-300)).sum()))


def tsite_group(fw):
    """Permutations of the T sites induced by the space group of the framework, and the T-T adjacency (shared bridging O)."""
    sym = np.array(fw.get_chemical_symbols())
    t = np.where(sym == "Si")[0]
    pos = {k: n for n, k in enumerate(t)}
    i, j, d = neighbor_list("ijd", fw, 1.9)
    adj = {n: set() for n in range(len(t))}
    for o in np.where(sym == "O")[0]:
        nb = sorted({int(b) for a, b in zip(i, j) if a == o and sym[b] == "Si"})
        if len(nb) == 2:
            adj[pos[nb[0]]].add(pos[nb[1]])
            adj[pos[nb[1]]].add(pos[nb[0]])
    # the framework is slightly distorted: spglib finds R3 (9 operations) at 1e-3 and R-3m (36 operations) at 1e-1
    ds = spglib.get_symmetry((fw.cell[:], fw.get_scaled_positions(), fw.numbers), symprec=1e-1)
    frac = fw.get_scaled_positions()[t]
    perms = []
    for R, tr in zip(ds["rotations"], ds["translations"]):
        img = (frac @ R.T + tr) % 1.0
        perm = []
        for x in img:
            dd = frac - x
            dd -= np.round(dd)
            perm.append(int(np.argmin(np.linalg.norm(dd @ fw.cell[:], axis=1))))
        assert len(set(perm)) == len(perm), "a symmetry operation does not map the T sites onto themselves"
        perms.append(tuple(perm))
    return t, adj, sorted(set(perms))


def canonical(subset, perms):
    return min(tuple(sorted(p[s] for s in subset)) for p in perms)


def section_host() -> None:
    print("\n== 2. chabazite host with several T cations (structures/host, results/host)")
    fw = read(str(DATA / "CHA_framework.cif"))
    t_idx, adj, perms = tsite_group(fw)
    # arrangement space
    n_t = len(t_idx)
    comp = {k: [c for c in itertools.combinations(range(n_t), k) if all(b not in adj[a] for a, b in itertools.combinations(c, 2))] for k in (2, 3)}
    cls = {k: {canonical(c, perms) for c in comp[k]} for k in comp}
    check("framework: 36 T sites, 36 space-group operations (R-3m at tolerance 0.1 A), every T has 4 T neighbours", (n_t, len(perms)) == (36, 36) and all(len(v) == 4 for v in adj.values()),
          f"{n_t} T sites, {len(perms)} operations")
    check("Loewenstein-compliant placements: 558 for 2 Al (21 classes), 4908 for 3 Al (139 classes)", (len(comp[2]), len(cls[2]), len(comp[3]), len(cls[3])) == (558, 21, 4908, 139),
          f"{len(comp[2])} ({len(cls[2])}), {len(comp[3])} ({len(cls[3])})")
    he6 = comp[3].__len__() * 33 * 32 * 31 * 30
    check("six-species recipe: 4.82e9 labelled T patterns, at least 1.34e8 symmetry-distinct", abs(he6 / 1e9 - 4.82) < 0.01 and he6 / 36 > 1.33e8, f"{he6:.3e} labelled, {he6 / 36:.3e} distinct at least")

    stats = {}
    for recipe in ("he6", "csal"):
        d = RES / "host" / recipe
        df = pd.read_csv(d / "energies.csv")
        structs = read(str(d / "structures_relaxed.extxyz"), index=":")
        check(f"{recipe}: 100 structures", len(structs) == 100 == len(df))
        q = [sum(CHARGE[s] for s in at.get_chemical_symbols()) for at in structs]
        check(f"{recipe}: all 100 arrangements are charge balanced", set(q) == {0}, f"formal charges {sorted(set(q))}")
        viol = 0
        for at in structs:
            c, o, dd, sym = bonds(at, {"Al": HOST_CUT["Al"]})
            viol += int((np.bincount(o, minlength=len(at))[sym == "O"] >= 2).sum())
        check(f"{recipe}: no oxygen bonded to two Al (Loewenstein) in any structure", viol == 0, f"{viol} violations")
        # Cs and Sr coordination from the CIF-quality geometry
        cn_cs, cn_sr = [], []
        for at in structs:
            c, o, dd, sym = bonds(at, HOST_CUT)
            for el, store in (("Cs", cn_cs), ("Sr", cn_sr)):
                for k in np.where(sym == el)[0]:
                    store.append(int((c == k).sum()))
        if recipe == "he6":
            check("he6: Cs coordination equals the tabulated one in all structures", np.allclose(cn_cs, df["Cs_CN"]), f"mean {np.mean(cn_cs):.2f} +/- {np.std(cn_cs, ddof=1):.2f}")
            check("he6: Sr coordination equals the tabulated one", np.allclose(cn_sr, df["Sr_CN"]), f"mean {np.mean(cn_sr):.2f} +/- {np.std(cn_sr, ddof=1):.2f}")
        else:
            cs_mean = np.mean(cn_cs)
            note(f"csal: Cs coordination per cation over 300 Cs: {cs_mean:.2f} +/- {np.std(cn_cs, ddof=1):.2f}")
            per_struct = np.array(cn_cs).reshape(100, 3).mean(axis=1)
            check("csal: Cs coordination per structure equals the tabulated one", np.allclose(per_struct, df["Cs_CN"]), f"mean {per_struct.mean():.2f} +/- {per_struct.std(ddof=1):.2f}")
        # first cation-O peak of the large cations (bin width 0.1 A, normalised by the shell volume)
        for el, want in (("Cs", 3.25), ("Sr", 2.55)):
            if el == "Sr" and recipe != "he6":
                continue
            edges = np.linspace(0, 6.5, 66)
            h, n_el = np.zeros(65), 0
            for at in structs:
                sym = np.array(at.get_chemical_symbols())
                i, j, dd = neighbor_list("ijd", at, 6.5)
                h += np.histogram(dd[(sym[i] == el) & (sym[j] == "O")], edges)[0]
                n_el += int((sym == el).sum())
            r = 0.5 * (edges[1:] + edges[:-1])
            g = h / (n_el * 4 * np.pi * r**2 * 0.1 * (72 / structs[0].get_volume()))
            check(f"{recipe}: first {el}-O peak at {want} A", abs(r[np.argmax(g[:40])] - want) < 1e-9, f"{r[np.argmax(g[:40])]:.2f} A")
        # energies
        e = df["E_rel_eV"].to_numpy()
        conv = df["max_force_eV_A"].to_numpy() <= 0.1
        rng = np.random.default_rng(SEED)
        boot = [boltzmann_neff(rng.choice(e, size=len(e)), 1300.0) for _ in range(N_BOOT)]
        stats[recipe] = dict(rng=e.max(), std=e.std(ddof=1), std_conv=e[conv].std(ddof=1), iqr=np.subtract(*np.percentile(e, [75, 25])),
                             n_eff=[boltzmann_neff(e, T) for T in (300.0, 800.0, 1300.0)], ci=np.percentile(boot, [2.5, 97.5]), n_unconv=int((~conv).sum()))
        s = stats[recipe]
        print(f"   {recipe}: range {s['rng']:.2f} eV, std {s['std']:.2f}, IQR {s['iqr']:.2f}, n_eff at 300/800/1300 K {s['n_eff'][0]:.1f}/{s['n_eff'][1]:.1f}/{s['n_eff'][2]:.1f}, "
              f"bootstrap 95 % interval at 1300 K {s['ci'][0]:.1f} to {s['ci'][1]:.1f}, relaxations above 0.1 eV/A: {s['n_unconv']}")
        if recipe == "csal":
            rho, p_asym = spearmanr(df["E_rel_eV"], df["Cs_CN"])
            rng = np.random.default_rng(SEED)
            perm = [spearmanr(df["E_rel_eV"], rng.permutation(df["Cs_CN"].to_numpy()))[0] for _ in range(N_PERM)]
            p_perm = (1 + np.sum(np.abs(perm) >= abs(rho))) / (N_PERM + 1)
            check("csal: energy falls with Cs coordination, Spearman -0.32, p about 1e-3", round(rho, 2) == -0.32 and 5e-4 <= p_perm <= 2e-3, f"rho {rho:.3f}, asymptotic p {p_asym:.4f}, permutation p {p_perm:.4f} (two-sided, {N_PERM} draws)")
        else:
            rho, p = spearmanr(df["E_rel_eV"], df["Cs_CN"])
            check("he6: no relation between energy and Cs coordination", p > 0.05, f"rho {rho:.3f}, p {p:.2f}")
    a, b = stats["he6"], stats["csal"]
    check("energy range 8.06 against 1.43 eV, std 1.03 against 0.28 eV", (round(a["rng"], 2), round(a["std"], 2), round(b["rng"], 2), round(b["std"], 2)) == (8.06, 1.03, 1.43, 0.28))
    check("std ratio 3.7 on the converged runs, interquartile range ratio 2.0", round(a["std_conv"] / b["std_conv"], 1) == 3.7 and round(a["iqr"] / b["iqr"], 1) == 2.0,
          f"{a['std_conv'] / b['std_conv']:.2f}, {a['iqr'] / b['iqr']:.2f}")
    check("effective number of structures at 300, 800, 1300 K: 1.0, 1.1, 1.9 (six species) and 1.0, 3.2, 18.3 (Cs only)",
          [round(x, 1) for x in a["n_eff"]] == [1.0, 1.1, 1.9] and [round(x, 1) for x in b["n_eff"]] == [1.0, 3.2, 18.3])
    check("the 95 % bootstrap intervals of n_eff(1300 K) do not overlap", a["ci"][1] < b["ci"][0], f"{a['ci'][0]:.1f}-{a['ci'][1]:.1f} against {b['ci'][0]:.1f}-{b['ci'][1]:.1f}")
    check("3 of 100 relaxations per recipe stop above 0.1 eV/A", a["n_unconv"] == 3 and b["n_unconv"] == 3)

    # sampling coverage of the Al patterns in the Cs-only recipe
    structs = read(str(RES / "host" / "csal" / "structures_relaxed.extxyz"), index=":")
    frac_fw = fw.get_scaled_positions()[t_idx]
    sampled, worst = set(), 0.0
    for at in structs:
        sym = np.array(at.get_chemical_symbols())
        sites = []
        for x in at.get_scaled_positions()[sym == "Al"]:
            dd = frac_fw - x
            dd -= np.round(dd)
            dist = np.linalg.norm(dd @ at.cell[:], axis=1)
            sites.append(int(np.argmin(dist)))
            worst = max(worst, dist.min())
        sampled.add(canonical(sites, perms))
    check("the 100 Cs-only structures hold 100 different Al patterns, 100 of the 139 classes", len(sampled) == 100, f"{len(sampled)} classes; farthest relaxed Al from its framework T site {worst:.2f} A")
    note("the Cs-only sample covers most of its Al patterns, the six-species sample about 1 in 10^6 of its T patterns: the two effective ensemble sizes are not measured on equal footing")


# ---------------------------------------------------------------------------------------------------------------- AFLOW
def section_aflow() -> None:
    print("\n== 3. AFLOW phases (structures/aflow)")
    tab = pd.read_csv(RES / "aflow" / "aflow_vs_mace.csv").set_index("name")
    n_ok = 0
    for p in sorted((STRUCT / "aflow").glob("*.cif")):
        at = read(str(p))
        sg = spglib.get_symmetry_dataset((at.cell[:], at.get_scaled_positions(), at.numbers), symprec=1e-3).number
        name = p.stem
        if name in tab.index and not np.isnan(tab.loc[name, "aflux_sg"]):
            ok = sg == int(tab.loc[name, "aflux_sg"]) == int(tab.loc[name, "aflow_sg"])
            n_ok += ok
            check(f"{name}: space group {sg} equals the AFLOW binary and the AFLOW database", ok, f"{at.get_chemical_formula()}, {len(at)} atoms, {tab.loc[name, 'aflow_label']}")
    check("7 of 7 phases agree", n_ok == 7)
    cha = read(str(STRUCT / "host" / "cha_silica_framework.cif"))
    s3 = spglib.get_symmetry_dataset((cha.cell[:], cha.get_scaled_positions(), cha.numbers), symprec=1e-3).number
    s1 = spglib.get_symmetry_dataset((cha.cell[:], cha.get_scaled_positions(), cha.numbers), symprec=1e-1).number
    check("chabazite framework: spglib 146 at tolerance 1e-3, 166 at 1e-1 (AFLOW binary: 166)", (s3, s1) == (146, 166) and int(tab.loc["chabazite_framework_SiO2", "aflow_sg"]) == 166, f"{s3}, {s1}")
    d0 = tab["dV_V_mace_d30_vs_aflow_pct"].drop("chabazite_framework_SiO2")
    d1 = tab["dV_V_mace_d31_vs_aflow_pct"].drop("chabazite_framework_SiO2")
    shift = -100 * (tab["V_mace_d31_A3_atom"] / tab["V_mace_d30_A3_atom"] - 1).drop("chabazite_framework_SiO2")
    check("MACE-MPA-0 volumes without D3: -2.8 to +1.5 % of the AFLOW cell", round(d0.min(), 1) == -2.8 and round(d0.max(), 1) == 1.5, f"{d0.min():.2f} to {d0.max():.2f} %")
    check("with D3: -4.7 to -1.4 % of the AFLOW cell", round(d1.min(), 1) == -4.7 and round(d1.max(), 1) == -1.4, f"{d1.min():.2f} to {d1.max():.2f} %")
    check("D3 lowers each volume by 1.6 to 2.9 %", round(shift.min(), 1) == 1.6 and round(shift.max(), 1) == 2.9, f"{shift.min():.2f} to {shift.max():.2f} %")
    v = [read(str(STRUCT / "aflow" / f"{n}.cif")).get_volume() / len(read(str(STRUCT / "aflow" / f"{n}.cif"))) for n in d0.index]
    check("volume per atom of each CIF equals the AFLOW value in the table", np.allclose(v, tab.loc[d0.index, "V_aflow_pbe_A3_atom"], rtol=1e-6))
    note("AFLOW's entries use DFT+U for some phases (ldau_type 0 and 2), so these differences mix model error and calculation settings; the CIF served per entry is its relaxed cell")


# ---------------------------------------------------------------------------------------------------------------- rings
def neighbours(adj, node):
    t, s = node
    for nt, ds in adj[t]:
        yield (nt, (s[0] + ds[0], s[1] + ds[1], s[2] + ds[2]))


def periodic_net(atoms, cutoff=1.9):
    """T-T adjacency of a framework: T atoms (Si) joined through a bridging O, with the lattice offset of each link."""
    sym = np.array(atoms.get_chemical_symbols())
    i, j, S = neighbor_list("ijS", atoms, cutoff)
    adj = {int(k): set() for k in np.where(sym == "Si")[0]}
    for o in np.where(sym == "O")[0]:
        nb = [(int(b), tuple(int(x) for x in s)) for a, b, s in zip(i, j, S) if a == o and sym[b] == "Si"]
        if len(nb) == 2:
            (a, sa), (b, sb) = nb
            adj[a].add((b, tuple(y - x for x, y in zip(sa, sb))))
            adj[b].add((a, tuple(x - y for x, y in zip(sa, sb))))
    return {k: tuple(v) for k, v in adj.items()}


def ball(adj, src, depth, cache):
    hit = cache.get(src)
    if hit is not None and hit[0] >= depth:
        return hit[1]
    dist = {src: 0}
    q = deque([src])
    while q:
        u = q.popleft()
        if dist[u] >= depth:
            continue
        for w in neighbours(adj, u):
            if w not in dist:
                dist[w] = dist[u] + 1
                q.append(w)
    cache[src] = (depth, dist)
    return dist


def shortest_path_ring(adj, ring, cache) -> bool:
    """Franzblau: every pair of ring members is as close along the ring as in the whole net."""
    s = len(ring)
    for a in range(s):
        du = ball(adj, ring[a], s // 2, cache)
        for b in range(a + 1, s):
            if du.get(ring[b]) != min(b - a, s - (b - a)):
                return False
    return True


def ring_counts_bruteforce(adj, root_atom, max_size, cache):
    """Number of shortest-path rings of each size through one T atom, by listing every simple cycle through it."""
    root = (root_atom, (0, 0, 0))
    counts, path, on = {}, [root], {root}
    nb_cache = {}

    def nb(node):
        r = nb_cache.get(node)
        if r is None:
            r = nb_cache[node] = set(neighbours(adj, node))
        return r

    def dfs():
        u = path[-1]
        for w in nb(u):
            if w == root and len(path) >= 3:
                if path[1] < path[-1] and shortest_path_ring(adj, tuple(path), cache):
                    counts[len(path)] = counts.get(len(path), 0) + 1
                continue
            if w in on or len(path) >= max_size:
                continue
            path.append(w)
            on.add(w)
            dfs()
            path.pop()
            on.discard(w)

    dfs()
    return counts


def section_rings() -> None:
    print("\n== 4. shortest-path ring counts, independent depth-first enumeration (structures/nets)")
    dia = {0: tuple((1, s) for s in ((0, 0, 0), (-1, 0, 0), (0, -1, 0), (0, 0, -1))), 1: tuple((0, s) for s in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)))}
    cub = {0: tuple((0, s) for s in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)))}
    c = ring_counts_bruteforce(dia, 0, 12, {})
    check("diamond: 12 six-rings through every atom and no other ring up to size 12", c == {6: 12}, str(c))
    c = ring_counts_bruteforce(cub, 0, 12, {})
    check("simple cubic: 12 four-rings and 24 six-rings through every atom", c == {4: 12, 6: 24}, str(c))
    stored = {r["case"]: r for r in json.loads((RES / "cxx" / "bench_rings_16.json").read_text())["rows"]}
    for code, key in (("LTA", "LTA (IZA)"), ("CHA", "CHA (IZA)"), ("FER", "FER (IZA)")):
        at = read(str(STRUCT / "nets" / f"{code}.cif"))
        adj = periodic_net(at)
        cache = {}
        per_t = {t: ring_counts_bruteforce(adj, t, 12, cache) for t in adj}
        total = sum(sum(v.values()) for v in per_t.values())
        sizes = sorted({n for v in per_t.values() for n in v})
        want = stored[key]["rings_total"]
        check(f"{code}: {len(adj)} T atoms, {total} rings through T atoms up to size 12, equal to the stored count {want} (C++ and ZIPPER)", total == want, f"ring sizes present {sizes}")


# ---------------------------------------------------------------------------------------------------------------- stored runs
def section_stored() -> None:
    print("\n== 5. stored benchmark and verification files (results/cxx, results/analysis)")
    b = json.loads((RES / "cxx" / "bench_results.json").read_text())["rows"]
    sp = {r["n_atoms"]: r["crisp_one_pair_rdf_s"] / r["cxx_all_partial_rdfs_s"] for r in b if r["n_atoms"] in (300, 1000, 3000)}
    check("g(r), one thread: 18, 277, 903 times faster than CRISP at 300, 1000, 3000 atoms (C++ all pairs, CRISP one pair)", [round(sp[n]) for n in (300, 1000, 3000)] == [18, 277, 903],
          str({n: round(v) for n, v in sp.items()}))
    n = np.array([r["n_atoms"] for r in b if r["n_atoms"] >= 1000], dtype=float)
    t = np.array([r["cxx_all_partial_rdfs_s"] for r in b if r["n_atoms"] >= 1000])
    slope = np.polyfit(np.log(n), np.log(t), 1)[0]
    cr = [(r["n_atoms"], r["crisp_one_pair_rdf_s"]) for r in b if r.get("crisp_one_pair_rdf_s") is not None]
    slope_cr = np.polyfit(np.log([x[0] for x in cr]), np.log([x[1] for x in cr]), 1)[0]
    check("cell-list cost grows linearly with atoms (log-log slope 1.0, 1,000 to 100,000 atoms); CRISP's grows with slope 2.2 (300 to 3,000 atoms)", round(slope, 1) == 1.0 and round(slope_cr, 1) == 2.2,
          f"slopes {slope:.2f} and {slope_cr:.2f}")
    big = [r for r in b if r["n_atoms"] == 100000]
    check("g(r) for 100,000 atoms takes 0.59 s", bool(big) and round(big[0]["cxx_all_partial_rdfs_s"], 2) == 0.59, f"{big[0]['cxx_all_partial_rdfs_s']:.2f} s" if big else "")
    rows = json.loads((RES / "cxx" / "bench_rings_16.json").read_text())["rows"]
    check("ring counts identical to the Python reference on 16 of 16 structures", len(rows) == 16 and all(r["identical_counts"] for r in rows))
    s1 = [r["speedup_1thread"] for r in rows]
    check("ring enumerator 5.9 to 12.0 times faster on one thread", round(min(s1), 1) == 5.9 and round(max(s1), 1) == 12.0, f"{min(s1):.1f} to {max(s1):.1f}")
    v = json.loads((RES / "analysis" / "verification_cluster.json").read_text())
    check("5,280 g(r) comparisons with CRISP, largest difference 0", v["A"]["n_comparisons"] == 5280 and v["A"]["max_abs_diff"] == 0.0, f"{v['A']['n_comparisons']} comparisons, largest {v['A']['max_abs_diff']}")
    r = json.loads((RES / "analysis" / "rings_bruteforce.json").read_text())
    check("independent ring enumeration agrees with both implementations on all roots tested", r["all_pass"] and all(c["agree"] == c["roots_tested"] for c in r["cases"]))


if __name__ == "__main__":
    section_glass()
    section_host()
    section_aflow()
    section_rings()
    section_stored()
    print(f"\n{'ALL PASS' if not bad else 'FAILED: ' + '; '.join(bad)}")
    raise SystemExit(1 if bad else 0)
