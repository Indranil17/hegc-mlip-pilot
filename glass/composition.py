"""Glass compositions of the zirconolite-powellite borosilicate glass-ceramics of Wan et al., Materials 14, 5747 (2021).

Table 1 of that paper (nominal compositions, mol %). The CTZ-20 column sums to 97.4 mol %, which is a
typesetting or transcription problem in the source, so CTZ-20 is not used here.
Bulk densities are those the paper reports for the crystallised glass-ceramics (g cm^-3), given in the
order CTZ-20, 30, 40, 50; they are not glass densities.
"""
from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.data import atomic_masses, atomic_numbers

TABLE1_MOL = {
    "SiO2": {30: 43.85, 40: 38.63, 50: 33.11},
    "B2O3": {30: 9.64, 40: 8.50, 50: 7.28},
    "Na2O": {30: 13.16, 40: 11.60, 50: 9.94},
    "Al2O3": {30: 1.08, 40: 0.95, 50: 0.82},
    "CaO": {30: 9.04, 40: 10.53, 50: 12.11},
    "TiO2": {30: 12.07, 40: 16.53, 50: 21.26},
    "ZrO2": {30: 5.25, 40: 7.19, 50: 9.25},
    "Nd2O3": {30: 2.75, 40: 2.82, 50: 2.90},
    "MoO3": {30: 3.15, 40: 3.24, 50: 3.33},
}
EXPT_DENSITY = {30: 3.05, 40: 3.24, 50: 3.34}  # g/cm3, glass-ceramic bulk density from the paper

FORMULA = {  # oxide -> {element: count}
    "SiO2": {"Si": 1, "O": 2},
    "B2O3": {"B": 2, "O": 3},
    "Na2O": {"Na": 2, "O": 1},
    "Al2O3": {"Al": 2, "O": 3},
    "CaO": {"Ca": 1, "O": 1},
    "TiO2": {"Ti": 1, "O": 2},
    "ZrO2": {"Zr": 1, "O": 2},
    "Nd2O3": {"Nd": 2, "O": 3},
    "MoO3": {"Mo": 1, "O": 3},
}


def atoms_per_unit(ox: str) -> int:
    return sum(FORMULA[ox].values())


def choose_counts(ctz: int, target_atoms: int = 350, window: int = 30):
    """Integer numbers of oxide units that reproduce the nominal mol % best, with about target_atoms atoms."""
    mol = {ox: TABLE1_MOL[ox][ctz] for ox in TABLE1_MOL}
    tot = sum(mol.values())
    mol = {ox: 100.0 * v / tot for ox, v in mol.items()}
    best = None
    for s in np.arange(0.3, 4.0, 0.002):
        counts = {ox: int(round(s * v)) for ox, v in mol.items()}
        if counts["Nd2O3"] < 1 or counts["MoO3"] < 1 or counts["Al2O3"] < 1:
            continue
        n_atoms = sum(counts[ox] * atoms_per_unit(ox) for ox in counts)
        if abs(n_atoms - target_atoms) > window:
            continue
        units = sum(counts.values())
        err = sum(abs(100.0 * counts[ox] / units - mol[ox]) for ox in counts)
        if best is None or err < best[0] - 1e-9:
            best = (err, counts, n_atoms)
    if best is None:
        raise RuntimeError("no composition found")
    err, counts, n_atoms = best
    units = sum(counts.values())
    actual = {ox: 100.0 * counts[ox] / units for ox in counts}
    return counts, n_atoms, mol, actual, err


def element_counts(counts: dict) -> dict:
    out: dict = {}
    for ox, n in counts.items():
        for el, k in FORMULA[ox].items():
            out[el] = out.get(el, 0) + n * k
    return out


def box_length_for_density(el_counts: dict, density_g_cm3: float) -> float:
    mass_amu = sum(atomic_masses[atomic_numbers[el]] * n for el, n in el_counts.items())
    volume = mass_amu * 1.66053906660 / density_g_cm3  # angstrom^3
    return volume ** (1.0 / 3.0)


def random_pack(el_counts: dict, L: float, seed: int, tries: int = 200) -> Atoms:
    """Random sequential placement in a cubic periodic box; cations first, then oxygen.

    Minimum distances (angstrom): cation-cation 2.2, cation-O 1.5, O-O 1.9. They are relaxed by 4 % after
    each failed attempt, so a very dense target still ends with a valid, if loose, starting structure.
    """
    rng = np.random.default_rng(seed)
    cations = [el for el in el_counts if el != "O"]
    order = [el for el in cations for _ in range(el_counts[el])]
    rng.shuffle(order)
    order += ["O"] * el_counts["O"]
    scale = 1.0
    for attempt in range(tries):
        pos, syms = [], []
        ok = True
        for el in order:
            placed = False
            for _ in range(3000):
                p = rng.random(3) * L
                good = True
                if pos:
                    arr = np.asarray(pos)
                    d = arr - p
                    d -= L * np.round(d / L)
                    r = np.sqrt((d**2).sum(axis=1))
                    for q, rq in zip(syms, r):
                        dmin = 1.9 if (el == "O" and q == "O") else (2.2 if (el != "O" and q != "O") else 1.5)
                        if rq < dmin * scale:
                            good = False
                            break
                if good:
                    pos.append(p)
                    syms.append(el)
                    placed = True
                    break
            if not placed:
                ok = False
                break
        if ok:
            atoms = Atoms(syms, positions=pos, cell=[L, L, L], pbc=True)
            atoms.info["packing_scale"] = scale
            return atoms
        scale *= 0.96
    raise RuntimeError("random packing failed")
