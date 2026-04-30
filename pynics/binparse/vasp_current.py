"""Parser for VASP NMR current density files (NMRCURBX, NMRCURBY, NMRCURBZ).

These are plain-text files in POSCAR/CHGCAR format containing the induced
current density response to an applied magnetic field in the x, y, or z
direction respectively.
"""

import numpy as np


class VaspCurrentError(Exception):
    pass


class VaspCurrentFile:
    """Read and parse a VASP NMRCURB{X,Y,Z} current density file.

    The file format is a POSCAR header followed by 3 (or 4) blocks of
    volumetric scalar data representing the three Cartesian components of the
    induced current density (jx, jy, jz) at each real-space grid point.

    Attributes:
        system (str): System name from the first line of the file.
        lattice (np.ndarray): Lattice matrix, shape (3, 3).  Rows are the
            three lattice vectors **a**, **b**, **c** in Ångströms.
        species (list[str]): Element symbols in the order they appear in the
            POSCAR header.
        counts (list[int]): Number of atoms per species.
        positions (np.ndarray): Atomic positions, shape (n_atoms, 3).
            Cartesian coordinates in Ångströms.
        grid (list[int]): FFT grid dimensions [nx, ny, nz].
        current (np.ndarray): Current density array, shape (3, nx, ny, nz).
            ``current[0]`` = jx, ``current[1]`` = jy, ``current[2]`` = jz.
    """

    def __init__(self, fname):
        """Parse *fname* and populate attributes.

        Arguments:
            fname (str | path-like): Path to the NMRCURB{X,Y,Z} file.
        """
        with open(fname, "r") as f:
            self._parse(f)

    def _parse(self, f):
        # --- POSCAR header -------------------------------------------------
        self.system = f.readline().strip()
        scale = float(f.readline())

        lattice_rows = []
        for _ in range(3):
            lattice_rows.append([float(x) for x in f.readline().split()])
        self.lattice = scale * np.array(lattice_rows)  # (3, 3)

        self.species = f.readline().split()
        self.counts = [int(x) for x in f.readline().split()]

        coord_line = f.readline()
        is_fractional = "direct" in coord_line.lower()

        n_atoms = sum(self.counts)
        raw_pos = []
        for _ in range(n_atoms):
            raw_pos.append([float(x) for x in f.readline().split()[:3]])
        raw_pos = np.array(raw_pos)  # (n_atoms, 3)

        if is_fractional:
            # fractional → Cartesian: pos_cart = frac @ lattice
            self.positions = raw_pos @ self.lattice
        else:
            self.positions = scale * raw_pos

        # blank line between positions and grid
        f.readline()

        # --- Grid and volumetric data --------------------------------------
        self.grid = [int(x) for x in f.readline().split()]
        nx, ny, nz = self.grid

        n_pts = nx * ny * nz
        n_lines = int(np.ceil(n_pts / 5))

        tmp = []
        # The file contains 3 component blocks (jx, jy, jz) each followed by
        # a blank line.  A 4th incomplete block may exist; we read it
        # gracefully and discard any extra values.
        for _block in range(4):
            for _line in range(n_lines):
                line = f.readline()
                if not line:
                    break
                for val in line.split():
                    tmp.append(float(val))
            f.readline()  # trailing blank line after each block
            if len(tmp) >= 3 * n_pts:
                break

        if len(tmp) < 3 * n_pts:
            raise VaspCurrentError(
                f"Expected at least {3 * n_pts} values in {n_pts}-point grid "
                f"(3 components), got {len(tmp)}."
            )

        # Reshape: VASP stores data with z as the fastest index →  (nz, ny, nx)
        # after reshape.  Transpose to (nx, ny, nz).
        jden = np.array(tmp[: 3 * n_pts])
        j = jden.reshape(3, nz, ny, nx)
        j = np.transpose(j, (0, 3, 2, 1))  # → (3, nx, ny, nz)

        self.current = j  # (3, nx, ny, nz)
