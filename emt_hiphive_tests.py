from contextlib import redirect_stdout
from io import StringIO
from typing import List, Optional, Tuple

from ase import Atoms
import ase.build
from ase.io.trajectory import Trajectory
from ase.md import MDLogger
from ase.md.langevin import Langevin
from ase.units import fs

from asap3 import EMT
from calorine.tools import get_force_constants, relax_structure
from hiphive import ClusterSpace, StructureContainer, ForceConstantPotential
from hiphive.utilities import prepare_structures
import numpy as np
from tqdm import tqdm
from trainstation import Optimizer

from phonopy import Phonopy

# from phonopy.structure.atoms import PhonopyAtoms


TEMPERATURE_SERIES = [10, 300, 1000]
PHONON_SUPERCELL = [[-4, 4, 4], [4, -4, 4], [4, 4, -4]]
UNITCELL_SUPERCELL = [4, 4, 4]

# Hiphive cluster model: 2nd-order only, 6Å
CLUSTER_CUTOFFS = [
    6.0,
]


def get_opt_structures() -> Tuple[Atoms, Atoms]:
    prim = ase.build.bulk("Al")
    prim.calc = EMT()

    relax_structure(prim, fmax=1e-6)
    prim.write("prim.extxyz")

    unitcell = ase.build.make_supercell(
        prim, [[-1, 1, 1], [1, -1, 1], [1, 1, -1]]
    )
    unitcell.write("unitcell.extxyz")

    return prim, unitcell


def get_direct_phonons(
    prim: Atoms, filename: str = "phonopy_harmonic_fc.yaml"
) -> Phonopy:
    """Use phonopy to get low-T force constants by finite-displacements

    Write to yaml file for further analysis
    """
    phonon = get_force_constants(prim, EMT(), PHONON_SUPERCELL)
    phonon.save(filename=filename, settings={"force_constants": True})
    return phonon


def run_md(
    atoms: Atoms,
    temperature: float = 300,
    steps: int = 5000,
    log_interval: int = 10,
    traj_interval: int = 100,
    timestep_fs: float = 2.0,
    log_file: str = "-",
    trajectory: Optional[str] = None,
) -> List[Atoms]:
    """Run Langevin dynamics"""
    dyn = Langevin(
        atoms,
        timestep=timestep_fs * fs,
        temperature_K=temperature,
        friction=(0.01 / fs),
    )

    logger = MDLogger(
        dyn, atoms, log_file, header=True, stress=False, peratom=True, mode="w"
    )
    dyn.attach(logger, interval=log_interval)

    if trajectory:
        traj_writer = Trajectory(trajectory, "w", atoms)
        dyn.attach(traj_writer.write, interval=traj_interval)

    for _ in tqdm(dyn.irun(steps), total=steps):
        pass

    # dyn.run(steps)

    if trajectory:
        return ase.io.read(trajectory, index=":")
    else:
        return [atoms]


class PrintCapture:
    def __init__(self, callback, callback_interval=2):
        self._io = StringIO()
        self._callback = callback
        self._callback_interval = callback_interval
        self._lines_caught = 0

    def write(self, *args):
        self._io.write(*args)
        self._lines_caught += 1
        if self._lines_caught == self._callback_interval:
            self._lines_caught = 0
            self._callback()

    def get_last_line(self):
        return self._io.getvalue().split("\n")[-2]


def run_sc(
    atoms: Atoms,
    cs: ClusterSpace,
    temperature: float = 300,
    n_structures: int = 50,
    n_iterations: int = 50,
    alpha=0.2,
) -> ForceConstantPotential:

    from hiphive.self_consistent_phonons import self_consistent_harmonic_model

    progress = tqdm(total=n_iterations)
    callback = progress.update

    with redirect_stdout(PrintCapture(callback)) as scp_output:
        parameters_traj = self_consistent_harmonic_model(
            atoms, EMT(), cs, temperature, alpha, n_iterations, n_structures
        )
        last_line = scp_output.get_last_line()

    progress.close()

    print("    " + last_line)
    print("    writing force constant potential...")

    fcp = ForceConstantPotential(cs, parameters_traj[-1])
    fcp.write(f"scp_{temperature}K.fcp")

    return fcp


def main() -> None:
    print("Optimising structure...")
    prim, unitcell = get_opt_structures()
    supercell = unitcell.copy() * UNITCELL_SUPERCELL

    print("Calculating harmonic phonons by direct method...")
    phonon = get_direct_phonons(prim)

    plt = phonon.auto_band_structure(
        plot=True, write_yaml=True, filename="phonopy_harmonic_bands.yaml"
    )
    plt.savefig("phonopy_harmonic.pdf")

    # When we get to comparing plots, auto_band_structure won't be flexible
    # enough (unless we write a parser for the band.yaml output: looks
    # tedious!)  Then preferred approach will be to phonon.run_band_structure()
    # followed by phonon.get_band_structure_dict(), with qpoints from
    # phonopy.phonon.band_structure.get_band_qpoints_by_seekpath

    print(f"Setting up cluster space (cutoffs: {CLUSTER_CUTOFFS})...")
    cs = ClusterSpace(supercell, CLUSTER_CUTOFFS)

    for temperature in TEMPERATURE_SERIES:
        print(f"\n\nMolecular dynamics at {temperature}K: ")
        atoms = supercell.copy()
        atoms.calc = EMT()

        print("Equilibrating...")
        atoms = run_md(
            atoms,
            temperature=temperature,
            log_file=f"md_eq_{temperature}K.log",
        )[-1]

        print("Production run...")
        training_structures = run_md(
            atoms,
            temperature=temperature,
            trajectory=f"md_{temperature}K.traj",
            log_file=f"md_{temperature}K.log",
        )

        print("Getting displacement statistics...")
        training_structures = prepare_structures(
            training_structures, supercell
        )

        abs_displacements = np.abs(
            [s.arrays["displacements"] for s in training_structures]
        )
        print(
            f"    avg: {np.mean(abs_displacements):6.4f}, "
            f"max: {np.max(abs_displacements)}"
        )

        print("Fitting effective harmonic model...")
        sc = StructureContainer(cs)
        for s in training_structures:
            sc.add_structure(s)
        opt = Optimizer(sc.get_fit_data(), train_size=0.9)
        opt.train()
        stats = opt.summary
        print(f"    RMSE (training): {stats['rmse_train']:6.3e}")
        print(f"    RMSE (test):     {stats['rmse_test']:6.3e}")
        fcp = ForceConstantPotential(cs, opt.parameters)
        fcp.write(f"ehm_{temperature}K.fcp")

        print("Running self-consistent-phonons...")
        run_sc(supercell.copy(), cs, temperature=temperature)


if __name__ == "__main__":
    main()
