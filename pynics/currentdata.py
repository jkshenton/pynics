"""Unified current density data container for CASTEP and VASP calculations.

Provides :class:`CurrentData`, a single object that holds the parsed current
density field together with structural information (lattice, atom positions)
regardless of the originating code.

CASTEP stores the full 3×3 current–response tensor J[ix,iy,iz,α,β] where α
is the spatial component of the current and β is the B-field direction.

VASP stores only the three spatial components (jx, jy, jz) of the current
in response to one specific B-field direction per file (NMRCURBX → Bx, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class CurrentData:
    """Parsed current density data from CASTEP or VASP.

    Attributes:
        format (str): Source code, either ``'castep'`` or ``'vasp'``.
        grid (list[int]): FFT grid dimensions ``[nx, ny, nz]``.
        lattice (np.ndarray): Lattice matrix, shape ``(3, 3)``.  Rows are
            the lattice vectors **a**, **b**, **c** in Ångströms.
        positions (np.ndarray | None): Atomic positions, shape
            ``(n_atoms, 3)``, Cartesian Ångströms.  ``None`` if unavailable.
        species (list[str] | None): Element symbols.  ``None`` if unavailable.
        chi (np.ndarray | None): Magnetic susceptibility tensor, shape
            ``(9,)`` (flat row-major), in internal CASTEP units.
            ``None`` for VASP data.
        current (np.ndarray): Current density.

            * **CASTEP**: shape ``(nx, ny, nz, 3, 3)`` where the last two
              axes are ``[spatial_component, B_field_direction]``.
              Compatible with :class:`pynics.nics.NicsCompute`.
            * **VASP**: shape ``(3, nx, ny, nz)`` where the first axis is
              the spatial component (jx, jy, jz).
    """

    format: str
    grid: List[int]
    lattice: np.ndarray
    current: np.ndarray
    positions: Optional[np.ndarray] = None
    species: Optional[List[str]] = None
    chi: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def get_current_density(self, b_index: int = 0) -> np.ndarray:
        """Return the current density as a ``(3, nx, ny, nz)`` array.

        For CASTEP data the *b_index* selects which applied B-field direction
        (0=Bx, 1=By, 2=Bz) to extract from the full 3×3 response tensor.
        For VASP data *b_index* is ignored (the file already contains only one
        direction).

        Returns:
            np.ndarray: shape ``(3, nx, ny, nz)`` — jx, jy, jz components.
        """
        if self.format == "castep":
            # current shape: (nx, ny, nz, 3, 3)
            # axis -1 is B-field direction, axis -2 is spatial component
            j_xyz = self.current[..., :, b_index]  # (nx, ny, nz, 3)
            return np.moveaxis(j_xyz, -1, 0)  # → (3, nx, ny, nz)
        else:
            # current shape: (3, nx, ny, nz)
            return self.current

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_castep(
        cls,
        cell_path: str,
        current_path: str = "current.dat",
    ) -> "CurrentData":
        """Load current density from a CASTEP calculation.

        Arguments:
            cell_path (str): Path to the CASTEP ``.cell`` file.  Used to
                read the lattice and atom positions via ASE.
            current_path (str): Path to the ``current.dat`` binary file.
                Defaults to ``'current.dat'``.

        Returns:
            CurrentData: Parsed data with ``format='castep'``.
        """
        from ase.io import read as ase_read

        from pynics.binparse import CurrentFile

        cfile = CurrentFile(current_path)
        atoms = ase_read(cell_path)

        nx = cfile.grid[0]
        ny = cfile.grid[1]
        nz = cfile.grid[2]

        # Convert nested list [nx][ny][nz][3x3] → numpy (nx, ny, nz, 3, 3)
        current_array = np.array(cfile.current)  # (nx, ny, nz, 3, 3)

        return cls(
            format="castep",
            grid=list(cfile.grid),
            lattice=np.array(atoms.get_cell()),
            current=current_array,
            positions=atoms.get_positions(),
            species=list(atoms.get_chemical_symbols()),
            chi=np.array(cfile.chi),
        )

    @classmethod
    def from_vasp(
        cls,
        nmrcurb_path: str,
        poscar_path: Optional[str] = None,
    ) -> "CurrentData":
        """Load current density from a VASP NMRCURB{X,Y,Z} file.

        Arguments:
            nmrcurb_path (str): Path to the NMRCURBX, NMRCURBY, or NMRCURBZ
                file.
            poscar_path (str | None): Optional path to a POSCAR/CONTCAR file.
                If given, atom positions are read from there (more reliable);
                otherwise positions embedded in the NMRCURB file are used.

        Returns:
            CurrentData: Parsed data with ``format='vasp'``.
        """
        from pynics.binparse.vasp_current import VaspCurrentFile

        vcf = VaspCurrentFile(nmrcurb_path)

        if poscar_path is not None:
            from ase.io.vasp import read_vasp

            atoms = read_vasp(poscar_path)
            positions = atoms.get_positions()
            species = list(atoms.get_chemical_symbols())
        else:
            positions = vcf.positions
            # Expand species list from (species_list, counts) pair
            species = []
            for sym, count in zip(vcf.species, vcf.counts):
                species.extend([sym] * count)

        return cls(
            format="vasp",
            grid=list(vcf.grid),
            lattice=vcf.lattice,
            current=vcf.current,  # (3, nx, ny, nz)
            positions=positions,
            species=species,
            chi=None,
        )
