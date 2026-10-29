#!/usr/bin/env python
"""Pull waste-form reference phases from the public AFLOW database.

For each target phase it queries AFLUX (https://aflow.org/API/aflux/) with the species set, picks one ICSD entry of
the right reduced formula and space group (preferring one that has a formation enthalpy), downloads that entry's
PBE-relaxed structure (CONTCAR.relax.vasp) and its CIF through the AFLOW REST interface, and writes a selection table
and a log of the exact queries. Zirconolite (Ca-Zr-Ti-O) is queried too; the public catalogue returned no entry for it.
Pollucite is targeted as the cubic Ia-3d phase (space group 230); AFLUX holds only a tetragonal I4_1/acd entry of CsAlSi2O6, so it counts as not found.

Run locally (needs internet):  python aflow/aflux_pull.py --out data/aflow
"""
from __future__ import annotations

import argparse
import json
import math
import re
from functools import reduce
from pathlib import Path

import pandas as pd
import requests

AFLUX = "https://aflow.org/API/aflux/?"
REST = "https://aflowlib.duke.edu/AFLOWDATA/"
KEYS = "compound,auid,aurl,spacegroup_relax,Pearson_symbol_relax,prototype,volume_atom,density,enthalpy_formation_atom,Egap,ldau_type,species,nspecies,natoms"

TARGETS = [  # name, species (alphabetical), reduced formula, space group, role
    ("powellite_CaMoO4", ["Ca", "Mo", "O"], {"Ca": 1, "Mo": 1, "O": 4}, 88, "Mo host in glass-ceramic [Wan 2021]"),
    ("perovskite_CaTiO3", ["Ca", "O", "Ti"], {"Ca": 1, "O": 3, "Ti": 1}, 62, "competing phase at CTZ-50"),
    ("perovskite_NdAlO3", ["Al", "Nd", "O"], {"Al": 1, "Nd": 1, "O": 3}, 167, "competing phase at CTZ-20"),
    ("pyrochlore_La2Zr2O7", ["La", "O", "Zr"], {"La": 2, "O": 7, "Zr": 2}, 227, "pyrochlore host"),
    ("pyrochlore_Nd2Zr2O7", ["Nd", "O", "Zr"], {"Nd": 2, "O": 7, "Zr": 2}, 227, "pyrochlore host, Nd as actinide surrogate"),
    ("pollucite_CsAlSi2O6", ["Al", "Cs", "O", "Si"], {"Al": 1, "Cs": 1, "O": 6, "Si": 2}, 230, "Cs host"),
    ("baddeleyite_ZrO2", ["O", "Zr"], {"O": 2, "Zr": 1}, 14, "ZrO2 in the glass"),
    ("rutile_TiO2", ["O", "Ti"], {"O": 2, "Ti": 1}, 136, "TiO2 in the glass"),
    ("zirconolite_CaZrTi2O7", ["Ca", "O", "Ti", "Zr"], {"Ca": 1, "O": 7, "Ti": 2, "Zr": 1}, None, "main actinide host (not in the public catalogue)"),
]


def parse_formula_from_aurl(aurl: str) -> dict:
    base = aurl.rstrip("/").split("/")[-1].split("_ICSD_")[0]
    counts = {el: int(n) for el, n in re.findall(r"([A-Z][a-z]?)(\d+)", base)}
    g = reduce(math.gcd, counts.values()) if counts else 1
    return {el: n // g for el, n in counts.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/aflow")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log, rows = [], []
    for name, species, formula, sg, role in TARGETS:
        q = f"species({','.join(species)}),nspecies({len(species)}),{KEYS},paging(0)"
        r = requests.get(AFLUX + q, timeout=180)
        data = r.json() if r.ok else []
        entries = list(data.values()) if isinstance(data, dict) else data  # paging(0) returns {"1 of N": entry, ...}
        cand = [e for e in entries if "_ICSD_" in e.get("aurl", "") and parse_formula_from_aurl(e["aurl"]) == formula
                and (sg is None or e.get("spacegroup_relax") == sg)]
        cand.sort(key=lambda e: (e.get("enthalpy_formation_atom") is None, int(e["aurl"].split("_ICSD_")[-1])))
        log.append({"target": name, "query": AFLUX + q, "http": r.status_code, "n_entries_species_set": len(entries),
                    "n_matching_formula_and_sg": len(cand)})
        print(f"{name:26s} entries {len(entries):4d}  matching {len(cand):3d}", flush=True)
        if not cand:
            rows.append({"name": name, "role": role, "found": False})
            continue
        e = cand[0]
        path = e["aurl"].split(":", 1)[1]
        d = out / name
        d.mkdir(exist_ok=True)
        base = REST + path.split("AFLOWDATA/", 1)[1] + "/"
        icsd = path.split("/")[-1]
        for fname, dest in ((f"CONTCAR.relax.vasp", "CONTCAR.relax.vasp"), (f"{icsd}.cif", "icsd.cif")):
            rr = requests.get(base + fname, timeout=120)
            if rr.ok:
                (d / dest).write_text(rr.text, encoding="utf-8")
        rows.append({"name": name, "role": role, "found": True, "aurl": e["aurl"], "auid": e.get("auid"), "compound": e.get("compound"),
                     "spacegroup_relax": e.get("spacegroup_relax"), "Pearson": e.get("Pearson_symbol_relax"), "volume_atom_A3": e.get("volume_atom"),
                     "density_g_cm3": e.get("density"), "enthalpy_formation_atom_eV": e.get("enthalpy_formation_atom"), "Egap_eV": e.get("Egap"),
                     "ldau_type": e.get("ldau_type"), "n_entries_for_formula": len(cand)})
    pd.DataFrame(rows).to_csv(out / "aflux_selection.csv", index=False)
    (out / "aflux_query_log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    print("wrote", out / "aflux_selection.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
