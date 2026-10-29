#!/usr/bin/env python
"""Melt-quench one multi-cation borosilicate glass of Wan et al. (CTZ-30, 40 or 50) with a MACE foundation model.

Protocol (1 fs steps): random packing at the paper's density, FIRE pre-relaxation, NVT Langevin at T_melt for
melt_ps, then NPT (Berendsen thermostat and barostat) with a linear ramp to T_end at rate K/ps, then NPT at
T_end for hold_ps. The volume is free from the start of the ramp, so the final density is a prediction.
D3(BJ) dispersion is added on request. Every run records the mean squared displacement at the end of the melt,
wall time per step, and the density averaged over the last half of the hold.

Example (GPU):
  python glass/run_glass.py --ctz 40 --seed 0 --d3 1 --model models/mace-mpa-0-medium.model --out results/glass
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from ase import units
from ase.io import write
from ase.md.langevin import Langevin
from ase.md.nptberendsen import NPTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.optimize import FIRE

sys.path.insert(0, str(Path(__file__).resolve().parent))
import composition as comp  # noqa: E402


def make_calc(model: str, device: str, dtype: str, d3: bool):
    from mace.calculators import mace_mp

    return mace_mp(model=model, device=device, default_dtype=dtype, dispersion=d3)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ctz", type=int, required=True, choices=[30, 40, 50])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--d3", type=int, default=1)
    p.add_argument("--model", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float32")
    p.add_argument("--natoms", type=int, default=350)
    p.add_argument("--t-melt", type=float, default=5000.0)
    p.add_argument("--melt-ps", type=float, default=20.0)
    p.add_argument("--rate-K-per-ps", type=float, default=100.0)
    p.add_argument("--t-end", type=float, default=300.0)
    p.add_argument("--hold-ps", type=float, default=10.0)
    p.add_argument("--dt-fs", type=float, default=1.0)
    p.add_argument("--friction-per-fs", type=float, default=0.01)
    p.add_argument("--save-every", type=int, default=200)
    p.add_argument("--quick", action="store_true", help="smoke test: --quick-steps steps in each of the three stages")
    p.add_argument("--quick-steps", type=int, default=30)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"ctz{a.ctz}_d3{a.d3}_seed{a.seed}"

    counts, n_atoms, nominal, actual, err = comp.choose_counts(a.ctz, a.natoms)
    el_counts = comp.element_counts(counts)
    rho0 = comp.EXPT_DENSITY[a.ctz]
    L0 = comp.box_length_for_density(el_counts, rho0)
    atoms = comp.random_pack(el_counts, L0, a.seed)
    print(f"[{tag}] {n_atoms} atoms, box {L0:.2f} A at {rho0} g/cm3; units {counts}; composition error {err:.2f} mol% (L1)", flush=True)

    t_wall0 = time.perf_counter()
    calc = make_calc(a.model, a.device, a.dtype, bool(a.d3))
    atoms.calc = calc
    t_load = time.perf_counter() - t_wall0

    # remove the worst overlaps of the random packing before dynamics
    opt = FIRE(atoms, maxstep=0.1, logfile=None)
    opt.run(fmax=1.0, steps=5 if a.quick else 400)
    f_after = float(np.abs(atoms.get_forces()).max())
    print(f"[{tag}] pre-relax: {opt.nsteps} FIRE steps, max |F| {f_after:.2f} eV/A, E/atom {atoms.get_potential_energy() / len(atoms):.3f} eV", flush=True)

    ramp_ps = (a.t_melt - a.t_end) / a.rate_K_per_ps
    n_melt, n_ramp, n_hold = (int(round(x * 1000.0 / a.dt_fs)) for x in (a.melt_ps, ramp_ps, a.hold_ps))
    if a.quick:
        n_melt = n_ramp = n_hold = a.quick_steps
    total = n_melt + n_ramp + n_hold
    print(f"[{tag}] {a.t_melt:.0f} K for {n_melt} steps (NVT), ramp {n_ramp} steps, hold {n_hold} steps (NPT); {total} steps", flush=True)

    rng = np.random.default_rng(a.seed)
    MaxwellBoltzmannDistribution(atoms, temperature_K=a.t_melt, rng=rng)
    Stationary(atoms)
    p0 = atoms.get_positions().copy()
    sym = np.asarray(atoms.get_chemical_symbols())
    series, frames = [], []
    state = {"phase": "melt"}

    dyn = Langevin(atoms, a.dt_fs * units.fs, temperature_K=a.t_melt, friction=a.friction_per_fs / units.fs)

    def target_T(step: int) -> float:
        if step < n_melt:
            return a.t_melt
        if step < n_melt + n_ramp:
            return a.t_melt - a.rate_K_per_ps * (step - n_melt) * a.dt_fs / 1000.0
        return a.t_end

    msd_end_melt: dict = {}
    t_start = time.perf_counter()

    def observe():
        step = state["step0"] + dyn.nsteps
        T_set = target_T(step)
        dyn.set_temperature(temperature_K=T_set)
        if step == n_melt and not msd_end_melt:
            d2 = ((atoms.get_positions() - p0) ** 2).sum(axis=1)
            ell2 = (atoms.get_volume() / len(atoms)) ** (2.0 / 3.0)
            msd_end_melt.update({"msd_cation_A2": float(d2[sym != "O"].mean()), "msd_O_A2": float(d2[sym == "O"].mean()), "ell2_A2": float(ell2)})
        if step % a.save_every == 0 or step == total:
            mass = atoms.get_masses().sum() * 1.66053906660 / atoms.get_volume()
            s = atoms.get_stress(voigt=False)
            series.append({"step": step, "t_ps": step * a.dt_fs / 1000.0, "T_set_K": T_set,
                           "T_K": atoms.get_kinetic_energy() / len(atoms) / (1.5 * units.kB),
                           "epot_eV_atom": atoms.get_potential_energy() / len(atoms), "volume_A3": atoms.get_volume(),
                           "density_g_cm3": mass, "pressure_GPa": -float(np.trace(s)) / 3.0 / units.GPa, "phase": state["phase"]})
            if step >= n_melt + n_ramp:
                frames.append(atoms.copy())

    crashed = None
    state["step0"] = 0
    dyn.attach(observe, interval=1)
    try:
        dyn.run(n_melt)
        # switch to NPT for ramp and hold; velocities carry over
        state["phase"] = "ramp"
        state["step0"] = n_melt
        npt = NPTBerendsen(atoms, a.dt_fs * units.fs, temperature_K=a.t_melt, pressure_au=0.0, taut=100 * units.fs,
                           taup=1000 * units.fs, compressibility_au=3e-6 / units.bar, fixcm=True)

        def observe_npt():
            step = state["step0"] + npt.nsteps
            if step >= n_melt + n_ramp and state["phase"] == "ramp":
                state["phase"] = "hold"
            T_set = target_T(step)
            npt.set_temperature(temperature_K=T_set)
            if step % a.save_every == 0 or step == total:
                mass = atoms.get_masses().sum() * 1.66053906660 / atoms.get_volume()
                s = atoms.get_stress(voigt=False)
                series.append({"step": step, "t_ps": step * a.dt_fs / 1000.0, "T_set_K": T_set,
                               "T_K": atoms.get_kinetic_energy() / len(atoms) / (1.5 * units.kB),
                               "epot_eV_atom": atoms.get_potential_energy() / len(atoms), "volume_A3": atoms.get_volume(),
                               "density_g_cm3": mass, "pressure_GPa": -float(np.trace(s)) / 3.0 / units.GPa, "phase": state["phase"]})
                if step >= n_melt + n_ramp:
                    frames.append(atoms.copy())

        npt.attach(observe_npt, interval=1)
        npt.run(n_ramp + n_hold)
    except Exception as exc:  # keep whatever was written
        crashed = repr(exc)
    wall = time.perf_counter() - t_start

    df = pd.DataFrame(series)
    df.to_csv(out / f"series_{tag}.csv", index=False)
    done = int(df["step"].max()) if len(df) else 0
    hold = df[df["phase"] == "hold"]
    half = hold[hold["t_ps"] >= hold["t_ps"].min() + 0.5 * (hold["t_ps"].max() - hold["t_ps"].min())] if len(hold) > 2 else hold
    rho_final = float(half["density_g_cm3"].mean()) if len(half) else float("nan")
    d2 = ((atoms.get_positions() - p0) ** 2).sum(axis=1)
    summary = {
        "tag": tag, "ctz": a.ctz, "seed": a.seed, "d3": a.d3, "n_atoms": len(atoms), "finished": crashed is None, "error": crashed or "",
        "steps_done": done, "steps_planned": total, "wall_s": wall, "ms_per_step": 1000.0 * wall / max(done, 1), "model_load_s": t_load,
        "density_start_g_cm3": rho0, "density_final_g_cm3": rho_final, "density_expt_glass_ceramic_g_cm3": rho0,
        "pressure_final_GPa": float(half["pressure_GPa"].mean()) if len(half) else float("nan"),
        "epot_final_eV_atom": float(half["epot_eV_atom"].mean()) if len(half) else float("nan"),
        "counts_oxide_units": counts, "element_counts": el_counts, "nominal_mol_pct": nominal, "actual_mol_pct": actual, "composition_error_L1_mol_pct": err,
        "prerelax_max_force_eV_A": f_after, "msd_end_melt": msd_end_melt, "t_melt_K": a.t_melt, "melt_ps": a.melt_ps,
        "rate_K_per_ps": a.rate_K_per_ps, "hold_ps": a.hold_ps, "dt_fs": a.dt_fs, "dtype": a.dtype, "device": a.device,
        "model": Path(a.model).name, "quick": a.quick,
    }
    (out / f"summary_{tag}.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    write(str(out / f"final_{tag}.xyz"), atoms, format="extxyz")
    if frames:
        write(str(out / f"hold_frames_{tag}.xyz"), frames[-min(len(frames), 25):], format="extxyz")
    print(f"[{tag}] done: {done}/{total} steps, {summary['ms_per_step']:.1f} ms/step, density {rho_final:.3f} g/cm3, crashed={crashed}", flush=True)
    return 0 if crashed is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
