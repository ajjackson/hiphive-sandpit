from typing import Tuple

from asap3 import EMT
from ase import Atoms
import ase.build
from calorine.tools import get_force_constants, relax_structure
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms


TEMPERATURE_SERIES = [10, 300, 1000]
PHONON_SUPERCELL = [[-4, 4, 4], [4, -4, 4], [4, 4, -4]]


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


def get_direct_phonons(prim, filename="phonopy_harmonic_fc.yaml") -> Phonopy:
    """Use phonopy to get low-T force constants by finite-displacements

    Write to"""
    phonon = get_force_constants(prim, EMT(), PHONON_SUPERCELL)
    phonon.save(filename=filename, settings={"force_constants": True})
    return phonon


def main() -> None:
    prim, _ = get_opt_structures()
    phonon = get_direct_phonons(prim)

    plt = phonon.auto_band_structure(plot=True,
                                     write_yaml=True,
                                     filename="phonopy_harmonic_bands.yaml")
    plt.savefig("phonopy_harmonic.pdf")

    # When we get to comparing plots, auto_band_structure won't be flexible
    # enough (unless we write a parser for the band.yaml output: looks
    # tedious!)  Then preferred approach will be to phonon.run_band_structure()
    # followed by phonon.get_band_structure_dict(), with qpoints from
    # phonopy.phonon.band_structure.get_band_qpoints_by_seekpath


if __name__ == "__main__":
    main()
