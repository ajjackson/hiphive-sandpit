#! /usr/bin/env python3

from argparse import ArgumentParser
from pathlib import Path

from ase import Atoms
from hiphive import ForceConstantPotential
import numpy as np
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms


def get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("fcp_file", type=Path)
    parser.add_argument(
        "output_file",
        type=Path,
        nargs="?",
        default="phonopy_params_from_fcp.yaml",
    )
    parser.add_argument(
        "--dim", type=int, nargs="+", default=[-4, 4, 4, 4, -4, 4, 4, 4, -4]
    )
    return parser


def main() -> None:
    args = get_parser().parse_args()

    if len(args.dim) in (1, 3):
        supercell_matrix = np.eye(3) * args.dim
    elif len(args.dim) == 9:
        supercell_matrix = np.array(args.dim).reshape((3, 3))
    else:
        raise ValueError(
            f"Cannot interpret supercell matrix from {len(args.dim)}-vector"
        )

    fcp = ForceConstantPotential.read(str(args.fcp_file))
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

    phonopy.save(
        settings={"force_constants": True}, filename=str(args.output_file)
    )


if __name__ == "__main__":
    main()
