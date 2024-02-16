from contextlib import redirect_stdout
from io import StringIO
from typing import List, NamedTuple, Optional, Tuple

from ase import Atoms
import ase.build
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io.trajectory import Trajectory
from ase.md import MDLogger
from ase.md.langevin import Langevin
from ase.units import fs

from asap3 import EMT
from calorine.tools import get_force_constants, relax_structure
from hiphive import ClusterSpace, StructureContainer, ForceConstantPotential
from hiphive.self_consistent_phonons import self_consistent_harmonic_model
from hiphive.utilities import prepare_structures
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from trainstation import Optimizer

from phonopy import Phonopy
from phonopy.phonon.band_structure import (
    BandPlot,
    get_band_qpoints_by_seekpath,
)
from phonopy.structure.atoms import PhonopyAtoms


TEMPERATURE_SERIES = [10, 300, 1000]
PHONON_SUPERCELL = [[-4, 4, 4], [4, -4, 4], [4, 4, -4]]
UNITCELL_SUPERCELL = [4, 4, 4]

# Hiphive cluster model: 2nd-order only, 6Å
CLUSTER_CUTOFFS = [
    6.0,
]

# Number of structures sampled in each iteration of self-consistent phonons
N_PHONON_RATTLED_STRUCTURES = 50

# Choose classical or Bose statistics for phonon rattling;
# Bose is more realistic, classical is more comparable to MD
QM_STATISTICS = False


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
    def __init__(self, callback, callback_interval=4, first_block_bonus=2):
        self._io = StringIO()
        self._callback = callback
        self._callback_interval = callback_interval
        self._lines_caught = 0 - first_block_bonus

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
    n_structures: int = N_PHONON_RATTLED_STRUCTURES,
    n_iterations: int = 20,
    alpha=0.2,
) -> ForceConstantPotential:

    progress = tqdm(total=n_iterations)
    callback = progress.update

    with redirect_stdout(PrintCapture(callback)) as scp_output:
        parameters_traj = self_consistent_harmonic_model(
            atoms,
            EMT(),
            cs,
            temperature,
            alpha,
            n_iterations,
            n_structures,
            QM_statistics=QM_STATISTICS,
        )
        last_line = scp_output.get_last_line()

    progress.close()

    print("    " + last_line)
    print("    writing force constant potential...")

    fcp = ForceConstantPotential(cs, parameters_traj[-1])
    fcp.write(f"scp_{temperature}K.fcp")

    return fcp


def phonopy_from_fcp(
    fcp: ForceConstantPotential, supercell_matrix=PHONON_SUPERCELL
) -> Phonopy:
    prim = fcp.primitive_structure
    atoms_phonopy = PhonopyAtoms(
        symbols=prim.get_chemical_symbols(),
        scaled_positions=prim.get_scaled_positions(),
        cell=prim.cell,
    )

    phonopy = Phonopy(atoms_phonopy, supercell_matrix=supercell_matrix)
    supercell = phonopy.get_supercell()
    supercell = Atoms(
        cell=supercell.cell,
        numbers=supercell.numbers,
        pbc=True,
        scaled_positions=supercell.get_scaled_positions(),
    )

    fcs = fcp.get_force_constants(supercell)
    phonopy.set_force_constants(fcs.get_fc_array(order=2))

    return phonopy


class PhonopyBandPath(NamedTuple):
    bands: List[np.ndarray]
    labels: List[Tuple[str, str]]
    connections: List[bool]


def setup_phonopy_bands_axes(
    band_spec: PhonopyBandPath,
) -> Tuple[Figure, np.ndarray]:
    ncols = band_spec.connections.count(False)
    fig, axes = plt.subplots(ncols=ncols)
    return fig, axes


def fit_to_structures(
    structures: List[Atoms],
    ref_cell: Atoms,
    cs: ClusterSpace,
    train_size: float = 0.9,
    save_filename: str = None,
) -> ForceConstantPotential:
    print("Getting displacement statistics...")
    training_structures = prepare_structures(structures, ref_cell)

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
    if train_size < 1:
        print(f"    RMSE (test):     {stats['rmse_test']:6.3e}")

    fcp = ForceConstantPotential(cs, opt.parameters)

    if save_filename:
        fcp.write(save_filename)
    return fcp


def main() -> None:
    band_plots = {"md": {}, "sc": {}, "rattled": {}}

    print("Optimising structure...")
    prim, unitcell = get_opt_structures()
    supercell = unitcell.copy() * UNITCELL_SUPERCELL

    print("Calculating harmonic phonons by direct method...")
    phonon = get_direct_phonons(prim)
    phonopy_atoms = PhonopyAtoms(
        symbols=prim.get_chemical_symbols(),
        scaled_positions=prim.get_scaled_positions(),
        cell=prim.cell,
    )

    band_spec = PhonopyBandPath(
        *get_band_qpoints_by_seekpath(
            phonopy_atoms, 500, is_const_interval=True
        )
    )

    phonon.run_band_structure(
        band_spec.bands,
        path_connections=band_spec.connections,
        labels=band_spec.labels,
    )
    band_plots["direct"] = phonon.get_band_structure_dict()

    # When we get to comparing plots, auto_band_structure won't be flexible
    # enough (unless we write a parser for the band.yaml output: looks
    # tedious!)  Then preferred approach will be to phonon.run_band_structure()
    # followed by phonon.get_band_structure_dict(), with qpoints from
    # phonopy.phonon.band_structure.get_band_qpoints_by_seekpath

    def band_dict_from_fcp(fcp: ForceConstantPotential) -> dict:
        phonon = phonopy_from_fcp(fcp)
        phonon.run_band_structure(
            band_spec.bands,
            path_connections=band_spec.connections,
            labels=band_spec.labels,
        )
        return phonon.get_band_structure_dict()

    print(f"Setting up cluster space (cutoffs: {CLUSTER_CUTOFFS})...")
    cs = ClusterSpace(supercell, CLUSTER_CUTOFFS)

    for temperature in TEMPERATURE_SERIES:
        print("\n\n{:#^79}".format(f" {temperature}K "))

        print("{:-^79}".format(" NVT Molecular dynamics "))
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

        fcp = fit_to_structures(
            training_structures,
            supercell,
            cs,
            save_filename=f"ehm_{temperature}K.fcp",
        )

        print("Calculating band structure...")
        band_plots["md"][temperature] = band_dict_from_fcp(fcp)

        print("\n{:-^79}".format(" Phonon rattling with FC from MD "))
        print("Getting rattled structures from effective harmonic model...")
        from hiphive.structure_generation import (
            generate_phonon_rattled_structures,
        )

        fc2 = fcp.get_force_constants(supercell).get_fc_array(
            order=2, format="ase"
        )

        rattled_structures = generate_phonon_rattled_structures(
            supercell.copy(),
            fc2,
            N_PHONON_RATTLED_STRUCTURES,
            temperature,
            QM_statistics=QM_STATISTICS,
        )

        # Setup one Atoms for calculating forces of all rattled structures;
        # this saves on re-instantiating EMT calculator every time.
        calc_atoms = rattled_structures[0].copy()
        calc_atoms.calc = EMT()

        for structure in rattled_structures:
            calc_atoms.positions = structure.positions
            structure.calc = SinglePointCalculator(structure,
                                                   forces=calc_atoms.get_forces())

        fcp = fit_to_structures(
            rattled_structures,
            supercell,
            cs,
            save_filename=f"rattled_{temperature}K.fcp",
        )

        print("Calculating band structure...")
        band_plots["rattled"][temperature] = band_dict_from_fcp(fcp)

        print("\n{:-^79}".format(" Self-consistent phonon rattling "))
        fcp = run_sc(supercell.copy(), cs, temperature=temperature)
        print("Calculating band structure...")
        band_plots["sc"][temperature] = band_dict_from_fcp(fcp)

    print("Plotting band structures...")

    for temperature in TEMPERATURE_SERIES:
        fig, axes = setup_phonopy_bands_axes(band_spec)
        plotter = BandPlot(axes)
        for band_data, label, fmt in [
            (band_plots["direct"], "Direct", "C0-"),
            (band_plots["md"][temperature], f"MD {temperature}K", "C1-"),
            (band_plots["sc"][temperature], f"self-consistent {temperature}K", "C2-"),
            (band_plots["rattled"][temperature], f"phonon-rattled from {temperature}K MD", "C3-"),
        ]:

            plotter.plot(
                band_data["distances"],
                band_data["frequencies"],
                band_spec.connections,
                label=label,
                fmt=fmt,
            )

        plotter.decorate(
            band_spec.labels,
            band_spec.connections,
            band_data["frequencies"],
            band_data["distances"],
        )
        fig.savefig(f"compare_bands_{temperature}.pdf")


if __name__ == "__main__":
    main()
