from typing import List, Optional, Tuple

from ase import Atoms
import ase.build
from ase.io.trajectory import Trajectory
from ase.md import MDLogger
from ase.md.langevin import Langevin
from ase.units import fs

from asap3 import EMT
from calorine.tools import get_force_constants, relax_structure
from phonopy import Phonopy
# from phonopy.structure.atoms import PhonopyAtoms


TEMPERATURE_SERIES = [10, 300, 1000]
PHONON_SUPERCELL = [[-4, 4, 4], [4, -4, 4], [4, 4, -4]]
UNITCELL_SUPERCELL = [4, 4, 4]


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

    dyn.run(steps)

    if trajectory:
        return ase.io.read(trajectory, index=":")
    else:
        return [atoms]


def main() -> None:
    prim, unitcell = get_opt_structures()
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

    for temperature in TEMPERATURE_SERIES:
        print(f"Molecular dynamics at {temperature}K: ")
        atoms = unitcell.copy() * UNITCELL_SUPERCELL
        atoms.calc = EMT()

        print("Equilibrating...")
        atoms = run_md(atoms, temperature=temperature, log_file=f"md_eq_{temperature}K.log")[-1]

        print("Production run...")
        atoms = run_md(
            atoms, temperature=temperature, trajectory=f"md_{temperature}K.traj", log_file=f"md_{temperature}K.log"
        )


if __name__ == "__main__":
    main()
