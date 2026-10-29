#!/usr/bin/env python
"""Write the CIF files in structures/ from the raw simulation output.

glass/    final cell of each melt-quench run (results/glass/final_*.xyz)
motifs/   local environments cut out of the glass cells: Nd clusters that share oxygen, MoO4 and MoO5 groups, a Mo-O-Mo pair
host/     chabazite framework and the lowest and highest energy relaxed arrangement of each recipe (results/host/*)
aflow/    the relaxed AFLOW cells of the seven waste phases and the chabazite framework (data/aflow)
nets/     IZA frameworks LTA, CHA and FER used for the ring-count check (data/iza)

Run from the repository root:  python analysis/export_structures.py [--force]
Existing files are kept unless --force is given.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from ase import Atoms
from ase.io import read, write
from ase.neighborlist import neighbor_list
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "structures"
FORCE = "--force" in sys.argv
ORDER = ["Si", "Al", "B", "Ti", "Zr", "Sn", "Ge", "Mo", "Nd", "Na", "Ca", "Cs", "Sr", "La", "O"]  # O last, so the CIF formula reads cleanly
CUT = {"Si": 2.0, "B": 1.9, "Al": 2.2, "Ti": 2.5, "Zr": 2.8, "Mo": 2.3, "Nd": 3.1, "Na": 3.2, "Ca": 3.1}


def put(atoms: Atoms, path: Path, wrap: bool = True) -> None:
    """Write a P1 CIF with unique atom labels (Si1, Si2, ...), positions wrapped into the cell."""
    if path.exists() and not FORCE:
        print(f"  exists, kept: {path.relative_to(ROOT)}")
        return
    a = atoms.copy()
    a.calc = None
    a.arrays.pop("momenta", None)
    a.arrays.pop("forces", None)
    a.info = {}
    rank = {el: k for k, el in enumerate(ORDER)}
    a = a[sorted(range(len(a)), key=lambda k: (rank.get(a.get_chemical_symbols()[k], len(ORDER) - 1), k))]
    if wrap:
        a.wrap()
    count: dict = {}
    labels = []
    for s in a.get_chemical_symbols():
        count[s] = count.get(s, 0) + 1
        labels.append(f"{s}{count[s]}")
    write(str(path), a, format="cif", labels=[labels], wrap=False)
    print(f"  wrote {path.relative_to(ROOT)}  ({len(a)} atoms)")


def bonds(atoms: Atoms, cut: dict):
    sym = np.array(atoms.get_chemical_symbols())
    i, j, d = neighbor_list("ijd", atoms, max(cut.values()) + 0.01)
    keep = [(a, b) for a, b, r in zip(i, j, d) if sym[a] in cut and sym[b] == "O" and r <= cut[sym[a]]]
    return np.array([k[0] for k in keep], dtype=int), np.array([k[1] for k in keep], dtype=int), sym


def clusters(atoms: Atoms, el: str, cut: float):
    """Atoms of element el joined through shared oxygen: list of (member indices, oxygen indices)."""
    cs, os_, sym = bonds(atoms, {el: cut})
    n = len(atoms)
    g = coo_matrix((np.ones(len(cs)), (cs, os_)), shape=(n, n))
    _, lab = connected_components(g + g.T, directed=False)
    out = []
    for c in sorted(set(lab[np.where(sym == el)[0]])):
        members = [int(k) for k in np.where((sym == el) & (lab == c))[0]]
        oxy = sorted({int(o) for m, o in zip(cs, os_) if m in members})
        out.append((members, oxy))
    return out


def motif(atoms: Atoms, members, oxy, pad: float = 6.0) -> Atoms:
    """Cut atoms out of a periodic cell, unwrapped around the first member (minimum image), in a padded box."""
    idx = list(members) + list(oxy)
    ref = atoms.positions[members[0]]
    pos = []
    for k in idx:
        d = atoms.positions[k] - ref
        d = atoms.cell.cartesian_positions(atoms.cell.scaled_positions(d) - np.round(atoms.cell.scaled_positions(d)))
        pos.append(ref + d)
    pos = np.array(pos)
    lo, hi = pos.min(axis=0) - pad, pos.max(axis=0) + pad
    return Atoms([atoms.get_chemical_symbols()[k] for k in idx], positions=pos - lo, cell=hi - lo, pbc=True)


def export_glass() -> None:
    print("glass cells")
    cells = {}
    for p in sorted((ROOT / "results" / "glass").glob("final_*.xyz")):
        tag = p.stem.replace("final_", "")  # ctz30_d31_seed0
        ctz, d3, seed = tag.split("_")
        name = f"glass_{ctz}_{'D3' if d3 == 'd31' else 'noD3'}_{seed}.cif"
        at = read(str(p))
        cells[tag] = at
        put(at, OUT / "glass" / name)
    print("motifs")
    # Nd: largest shared-oxygen cluster in CTZ-50 (3 or 4 Nd) and in CTZ-40 (at most 2), Nd-O cutoff 3.1 A
    for tag, name in (("ctz50_d31_seed0", "nd_cluster_ctz50_seed0"), ("ctz40_d31_seed0", "nd_largest_ctz40_seed0")):
        cl = max(clusters(cells[tag], "Nd", CUT["Nd"]), key=lambda c: len(c[0]))
        put(motif(cells[tag], *cl), OUT / "motifs" / f"{name}.cif", wrap=False)
    # Mo at 2.3 A: one Mo with exactly four O, one with five, and a Mo-O-Mo pair if any cell has one
    found = {}
    for tag, at in cells.items():
        cs, os_, sym = bonds(at, {"Mo": CUT["Mo"]})
        for c in clusters(at, "Mo", CUT["Mo"]):
            mem, oxy = c
            if len(mem) == 1:
                cn = int((cs == mem[0]).sum())
                found.setdefault(f"mo_cn{cn}", (tag, c))
            elif len(mem) >= 2:
                found.setdefault("mo_shared_O_pair", (tag, c))
    for key, (tag, (mem, oxy)) in sorted(found.items()):
        if key in ("mo_cn4", "mo_cn5", "mo_shared_O_pair"):
            put(motif(cells[tag], mem, oxy), OUT / "motifs" / f"{key}_{tag}.cif", wrap=False)


def export_host() -> None:
    print("host")
    put(read(str(ROOT / "data" / "CHA_framework.cif")), OUT / "host" / "cha_silica_framework.cif")
    for recipe in ("he6", "csal"):
        d = ROOT / "results" / "host" / recipe
        df = pd.read_csv(d / "energies.csv")
        structs = read(str(d / "structures_relaxed.extxyz"), index=":")
        ok = df[df["max_force_eV_A"] <= 0.1]  # relaxed to 0.1 eV/A or better
        for label, row in (("lowest", ok.loc[ok["E_eV"].idxmin()]), ("highest", ok.loc[ok["E_eV"].idxmax()])):
            k = int(row["idx"])
            put(structs[k], OUT / "host" / f"{recipe}_{label}_E.cif")
            print(f"    {recipe} {label}: structure {k}, E - Emin = {row['E_rel_eV']:.3f} eV, max force {row['max_force_eV_A']:.3f} eV/A")


def export_aflow() -> None:
    print("aflow")
    for d in sorted((ROOT / "data" / "aflow").iterdir()):
        if (d / "CONTCAR.relax.vasp").exists():
            put(read(str(d / "CONTCAR.relax.vasp")), OUT / "aflow" / f"{d.name}.cif")


def export_nets() -> None:
    print("nets")
    for code in ("LTA", "CHA", "FER"):
        put(read(str(ROOT / "data" / "iza" / f"{code}.extxyz")), OUT / "nets" / f"{code}.cif")


if __name__ == "__main__":
    export_glass()
    export_host()
    export_aflow()
    export_nets()
