import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Union

import numpy as np

from ..constants import ANGSTROM_PER_BOHR


def upf_file2atomic_densities(
        filename: Path, 
        atom_type: str
    ) -> Dict[str, List[Union[str, float]]]:
    """
    Parses an UltraSoft Pseudopotential (UPF) file suitable for PAW calculations
    to extract atomic densities, including valence, core, and total densities.
    The function returns a dictionary containing these densities along with
    the atom type and distance grid values. All densities are in (absolute) 
    elemental charges per cubic angstrom.
    Parameters
    ----------
    filename : Path
        The path of the .UPF file containing the UltraSoft Pseudopotential information.
    atom_type : str
        The atomic species (element symbol) matching the file.

    Returns
    -------
    Dict[str, List[Union[str, float]]]
        A dictionary containing the following key-value pairs:
            - 'qubox_density_atomic_atom_type':
              List of repeated atom types (str) corresponding to the densities.
            - 'qubox_density_atomic_rgrid':
              List of radial grid points (float) for atomic densities in angstroms.
            - 'qubox_density_atomic_valence':
              List of valence electron density (float) values for the atom type.
            - 'qubox_density_atomic_core':
              List of core electron density (float) values for the atom type.
            - 'qubox_density_atomic_total':
              List of total electron density (float) values for the atom type.
    """
    tree = ET.parse(filename)

    root = tree.getroot()

    distance_mesh = None
    rho_atom = None
    ae_nlcc = None
    for child in root:
        if child.tag.upper() == 'PP_MESH':
            for grandchild in child:
                if grandchild.tag.upper() == 'PP_R':
                    distance_mesh = list(map(float, grandchild.text.split()))
        elif child.tag.upper() == 'PP_RHOATOM':
            rho_atom = list(map(float, child.text.split()))
        if child.tag.upper() == 'PP_PAW':
            for grandchild in child:
                if grandchild.tag.upper() == 'PP_AE_NLCC':
                    ae_nlcc = list(map(float, grandchild.text.split()))
    assert all(item is not None for item in (distance_mesh, rho_atom, ae_nlcc)), 'Could not find all entries, not a PAW .UPF file?'

    distance_grid = np.array(distance_mesh) * ANGSTROM_PER_BOHR
    valence_density = np.array(rho_atom) / 4 / np.pi / distance_grid**2 / ANGSTROM_PER_BOHR
    core_density = np.array(ae_nlcc) / ANGSTROM_PER_BOHR**3
    total_density = valence_density + core_density
    atom_types = [atom_type] * distance_grid.shape[0]
    
    return {
        '_qubox_density_atomic_atom_type': atom_types,
        '_qubox_density_atomic_rgrid': list(distance_grid),
        '_qubox_density_atomic_valence': list(valence_density),
        '_qubox_density_atomic_core': list(core_density),
        '_qubox_density_atomic_total': list(total_density)
    }