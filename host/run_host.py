#!/usr/bin/env python
"""Pilot: multi-cation site disorder in a crystalline host, enumerated by ZIPPER and ranked with MACE.

Host: chabazite (CHA), 36 T-sites in the hexagonal cell. Two recipes, both charge balanced by construction:
  he6  : T-sites Si + 3 Al + Ti + Zr + Sn + Ge (six T species), counter cations Cs + Sr
  csal : T-sites Si + 3 Al, counter cations 3 Cs (the single-substituent reference)
ZIPPER places the substituents on symmetry-distinct T-site patterns (Loewenstein's rule on), places the counter
cations, and relaxes positions with MACE-MPA-0 plus D3 at fixed cell. This script then records the energies,
the Boltzmann-weighted ensemble at three temperatures, and the first-shell environment of the large cations.

The sample is random, not exhaustive: the number of symmetry-distinct arrangements is far larger than the
number sampled, so energies are a pilot of the method and not a converged ensemble.

Example: python host/run_host.py --recipe he6 --n 100 --model models/mace-mpa-0-medium.model
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from ase import units
from ase.io import write

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cxx" / "python"))
sys.path.insert(0, str(ROOT / "cxx" / "build"))

from zeozipper import MixedCationGenerator  # noqa: E402
from zeozipper.mace import get_mace_calculator, relax_with_mace  # noqa: E402
import hegc  # noqa: E402

RECIPES = {
    "he6": {"t": {"Al": 3, "Ti": 1, "Zr": 1, "Sn": 1, "Ge": 1}, "counter": {"Cs": 1, "Sr": 1}},
    "csal": {"t": {"Al": 3}, "counter": {"Cs": 3}},
}
CATION_O_CUTOFF = {"Cs": 3.9, "Sr": 3.2, "Al": 2.0, "Ti": 2.4, "Zr": 2.6, "Sn": 2.5, "Ge": 2.1, "Si": 2.0}


def environment(atoms) -> dict:
    """First-shell oxygen counts of the extra-framework cations and the shortest cation-Al distance."""
    cen = {el: c for el, c in CATION_O_CUTOFF.items() if el in set(atoms.get_chemical_symbols())}
    res = hegc.connectivity(atoms, cen, ligand="O")
    sym = np.array(atoms.get_chemical_symbols())
    out = {}
    for el in ("Cs", "Sr"):
        idx = np.where(sym == el)[0]
        if len(idx):
            out[f"{el}_CN"] = float(np.mean(res["cn"][idx]))
    i, j, d = hegc.neighbor_pairs(atoms, 8.0)
    for el in ("Cs", "Sr"):
        m = ((sym[i] == el) & (sym[j] == "Al")) | ((sym[j] == el) & (sym[i] == "Al"))
        out[f"{el}_Al_min_A"] = float(d[m].min()) if m.any() else float("nan")
    return out


def boltzmann(e_rel_eV: np.ndarray, T: float) -> tuple[float, float]:
    """Effective number of structures exp(-sum p ln p) and the mean relative energy (eV) at temperature T."""
    w = np.exp(-e_rel_eV / (units.kB * T))
    p = w / w.sum()
    return float(np.exp(-(p * np.log(p + 1e-300)).sum())), float((p * e_rel_eV).sum())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recipe", choices=sorted(RECIPES), required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--d3", type=int, default=1)
    ap.add_argument("--fmax", type=float, default=0.05)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--cif", default=str(ROOT / "data" / "CHA_framework.cif"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    out = Path(a.out) / a.recipe
    out.mkdir(parents=True, exist_ok=True)
    r = RECIPES[a.recipe]
    calc = get_mace_calculator(model=a.model, device=a.device, dispersion=bool(a.d3), dtype=a.dtype)
    prov = getattr(calc, "provenance", None) or getattr(calc, "_provenance", None)
    gen = MixedCationGenerator(a.cif, t_recipe=r["t"], counter_recipe=r["counter"], seed=a.seed, output_dir=str(out / "zipper_raw"))
    t0 = time.perf_counter()
    structs = gen.generate(a.n, max_attempts=50 * a.n, mace_calc=calc, mace_relax=True, relax_fmax=a.fmax, relax_steps=a.steps)
    t_gen = time.perf_counter() - t0
    print(f"[{a.recipe}] {len(structs)} structures in {t_gen:.0f} s ({t_gen / max(len(structs), 1):.1f} s each)", flush=True)

    rows, keep = [], []
    for k, s in enumerate(structs):
        s = s.copy()
        s.calc = calc
        e = float(s.get_potential_energy())
        fmax = float(np.abs(s.get_forces()).max())
        row = {"idx": k, "E_eV": e, "E_eV_atom": e / len(s), "max_force_eV_A": fmax, "n_atoms": len(s), "formula": s.get_chemical_formula()}
        row.update(environment(s))
        sites = s.info.get("species_sites", {})
        row["t_sites"] = json.dumps({el: sorted(int(i) for i in v) for el, v in sites.items()})
        rows.append(row)
        s.info["E_eV"] = e
        s.info["idx"] = k
        s.calc = None
        keep.append(s)
    df = pd.DataFrame(rows)
    df["E_rel_eV"] = df["E_eV"] - df["E_eV"].min()
    df.to_csv(out / "energies.csv", index=False)
    write(str(out / "structures_relaxed.extxyz"), keep, format="extxyz")

    summ = {"recipe": a.recipe, "t_recipe": r["t"], "counter_recipe": r["counter"], "n_structures": len(df), "seed": a.seed,
            "model": Path(a.model).name, "d3": a.d3, "dtype": a.dtype, "fmax_target": a.fmax, "relax_steps_max": a.steps,
            "generate_and_relax_wall_s": t_gen, "n_atoms": int(df["n_atoms"].iloc[0]) if len(df) else 0,
            "E_rel_range_eV": float(df["E_rel_eV"].max()), "E_rel_std_eV": float(df["E_rel_eV"].std(ddof=1)) if len(df) > 1 else float("nan"),
            "n_not_converged": int((df["max_force_eV_A"] > 2 * a.fmax).sum()), "provenance": str(prov)}
    for T in (300.0, 800.0, 1300.0):
        n_eff, e_mean = boltzmann(df["E_rel_eV"].to_numpy(), T)
        summ[f"n_eff_{int(T)}K"] = n_eff
        summ[f"mean_E_rel_eV_{int(T)}K"] = e_mean
    (out / "summary.json").write_text(json.dumps(summ, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: v for k, v in summ.items() if k not in ("provenance",)}, default=float), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
