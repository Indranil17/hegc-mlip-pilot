#!/usr/bin/env python
"""Cross-check three things on the AFLOW reference phases pulled by aflux_pull.py.

1. Symmetry: the AFLOW 4.1 binary (--aflowSG, --prototype, --pearson_symbol) against spglib and against the space group stored in the
   AFLOW database (AFLUX keyword spacegroup_relax).
2. Volume per atom: the ICSD experimental cell, the AFLOW (DFT-PBE, mostly with +U) relaxed cell, and MACE-MPA-0 relaxed with and without
   D3 dispersion. The AFLOW cell is the starting point of the MACE relaxation (positions and cell, float64).
3. AFLOW-style record: one JSON line per structure with AFLOW keyword names plus the MLIP results, the format in which a larger glass-ceramic
   dataset could be stored.

Run:  python aflow/run_aflow_mace.py --aflow-bin <path to aflow> --model models/mace-mpa-0-medium.model --out results/aflow
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import spglib
from ase.filters import FrechetCellFilter
from ase.io import read
from ase.optimize import BFGS

ROOT = Path(__file__).resolve().parents[1]


def sym_spglib(atoms, symprec=1e-3) -> int:
    cell = (atoms.cell[:], atoms.get_scaled_positions(), atoms.numbers)
    ds = spglib.get_symmetry_dataset(cell, symprec=symprec)
    return int(ds.number) if ds is not None else -1


def aflow_cmd(binary: str, opt: str, poscar_text: str, home: str) -> str:
    env = dict(os.environ, HOME=home)
    r = subprocess.run([binary, opt], input=poscar_text, capture_output=True, text=True, timeout=300, env=env)
    lines = [ln for ln in r.stdout.splitlines() if ln.strip() and not ln.startswith("*")]
    return "\n".join(lines)


def aflow_info(binary: str, atoms_path: Path, home: str) -> dict:
    from ase.io import write
    tmp = Path(home) / "tmp_poscar"
    write(str(tmp), read(str(atoms_path)), format="vasp", direct=True)
    text = tmp.read_text()
    sg_txt = aflow_cmd(binary, "--aflowSG", text, home)
    proto = aflow_cmd(binary, "--prototype", text, home)
    pearson = aflow_cmd(binary, "--pearson_symbol", text, home)
    m = re.search(r"#\s*(\d+)", sg_txt)
    label = re.search(r"AFLOW label\s*:\s*(\S+)", proto)
    return {"aflow_sg_text": sg_txt.strip().splitlines()[0] if sg_txt.strip() else "", "aflow_sg": int(m.group(1)) if m else -1,
            "aflow_label": label.group(1) if label else "", "aflow_pearson": pearson.strip().splitlines()[0] if pearson.strip() else ""}


def relax(atoms, calc, fmax=0.02, steps=400):
    a = atoms.copy()
    a.calc = calc
    opt = BFGS(FrechetCellFilter(a), logfile=None)
    opt.run(fmax=fmax, steps=steps)
    return a, opt.nsteps, float(np.abs(a.get_forces()).max())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aflow-bin", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default=str(ROOT / "data" / "aflow"))
    ap.add_argument("--extra-cif", default=str(ROOT / "data" / "CHA_framework.cif"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--home", default=str(ROOT / "aflow_home"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from mace.calculators import mace_mp

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    Path(a.home).mkdir(parents=True, exist_ok=True)
    sel = pd.read_csv(Path(a.data) / "aflux_selection.csv")
    items = []
    for _, r in sel.iterrows():
        d = Path(a.data) / r["name"]
        if bool(r["found"]) and (d / "CONTCAR.relax.vasp").exists():
            items.append((r["name"], d / "CONTCAR.relax.vasp", d / "icsd.cif", r))
    if Path(a.extra_cif).exists():
        items.append(("chabazite_framework_SiO2", Path(a.extra_cif), None, None))

    calcs = {d3: mace_mp(model=a.model, device=a.device, default_dtype="float64", dispersion=bool(d3)) for d3 in (0, 1)}
    rows, records = [], []
    for name, poscar, icsd, row in items:
        t0 = time.perf_counter()
        at = read(str(poscar))
        info = aflow_info(a.aflow_bin, poscar, a.home)
        rec = {"name": name, "natoms": len(at), "formula": at.get_chemical_formula(), "spglib_sg_1e-3": sym_spglib(at, 1e-3),
               "spglib_sg_1e-1": sym_spglib(at, 1e-1), **info}
        if row is not None:
            rec["aflux_sg"] = int(row["spacegroup_relax"])
            rec["aflux_pearson"] = row["Pearson"]
            rec["aflux_volume_atom_A3"] = float(row["volume_atom_A3"])
            rec["aflow_ldau_type"] = row["ldau_type"]
            rec["aflux_enthalpy_formation_atom_eV"] = None if pd.isna(row["enthalpy_formation_atom_eV"]) else float(row["enthalpy_formation_atom_eV"])
        rec["V_aflow_pbe_A3_atom"] = at.get_volume() / len(at)
        if icsd is not None and icsd.exists():
            try:
                rec["V_icsd_A3_atom"] = read(str(icsd)).get_volume() / len(read(str(icsd)))
            except Exception as exc:  # partial occupancies or an unreadable cif
                rec["V_icsd_A3_atom"] = float("nan")
                rec["icsd_note"] = repr(exc)[:80]
        for d3, calc in calcs.items():
            rel, nsteps, fmax = relax(at, calc)
            rec[f"V_mace_d3{d3}_A3_atom"] = rel.get_volume() / len(rel)
            rec[f"mace_d3{d3}_steps"] = nsteps
            rec[f"mace_d3{d3}_fmax"] = fmax
            rec[f"mace_d3{d3}_spglib_sg_1e-2"] = sym_spglib(rel, 1e-2)
        rec["wall_s"] = time.perf_counter() - t0
        for k in ("V_mace_d30", "V_mace_d31"):
            col = f"{k}_A3_atom"
            rec[f"dV_{k}_vs_aflow_pct"] = 100.0 * (rec[col] / rec["V_aflow_pbe_A3_atom"] - 1.0)
            if "V_icsd_A3_atom" in rec and np.isfinite(rec["V_icsd_A3_atom"]):
                rec[f"dV_{k}_vs_icsd_pct"] = 100.0 * (rec[col] / rec["V_icsd_A3_atom"] - 1.0)
        if "V_icsd_A3_atom" in rec and np.isfinite(rec["V_icsd_A3_atom"]):
            rec["dV_aflow_vs_icsd_pct"] = 100.0 * (rec["V_aflow_pbe_A3_atom"] / rec["V_icsd_A3_atom"] - 1.0)
        rows.append(rec)
        # AFLOW-style keywords next to the MLIP results
        records.append({"compound": rec["formula"], "species": sorted(set(at.get_chemical_symbols())), "natoms": len(at),
                        "spacegroup_relax": rec.get("aflow_sg"), "Pearson_symbol_relax": rec.get("aflow_pearson"), "prototype": rec.get("aflow_label"),
                        "volume_atom": rec["V_mace_d31_A3_atom"], "dft_type": "MACE-MPA-0+D3(BJ)", "source_aurl": None if row is None else row["aurl"],
                        "volume_atom_aflow_pbe": rec["V_aflow_pbe_A3_atom"]})
        print(f"{name:28s} sg aflow {rec['aflow_sg']:4d} spglib {rec['spglib_sg_1e-3']:4d}  dV(MACE noD3 vs AFLOW) {rec['dV_V_mace_d30_vs_aflow_pct']:+.2f}%  "
              f"dV(MACE D3 vs AFLOW) {rec['dV_V_mace_d31_vs_aflow_pct']:+.2f}%  {rec['wall_s']:.0f} s", flush=True)
    pd.DataFrame(rows).to_csv(out / "aflow_vs_mace.csv", index=False)
    with open(out / "aflowlib_like.jsonl", "w") as fh:
        for r in records:
            fh.write(json.dumps(r, default=float) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
