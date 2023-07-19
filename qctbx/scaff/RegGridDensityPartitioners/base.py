from ..DensityPartitionerBase import DensityPartitioner
from ...conversions import cell_dict2atom_sites_dict

from scipy.integrate import simps
import numpy as np
from typing import Dict, Any

def calc_f0j_core(
    cell_dict: Dict[str, Any],
    refln_dict: Dict[str, Any],
    qubox_density_atomic_dicts: Dict[str, Any]
):
    """
    Calculates the core density in Fourier space.

    Args:
        cell_dict (Dict[str, Any]): A dictionary representing the unit cell.
        refln_dict (Dict[str, Any]): A dictionary representing the reflection data.
        qubox_density_atomic_dict (Dict[str, Dict[str, Any]]): A dictionary of 
            dictionaries with elements as key and the entries being dictionaries
            containing the atomic density on a real-space grid. Used entries within
            the individual elements are _qubox_density_atomic_rgrid and 
            _qubox_density_atomic_core

    Returns:
        Dict[str, Any]: A dictionary of core density values in Fourier space, keyed by element.
    """
    cell_mat_m = cell_dict2atom_sites_dict(cell_dict)['_atom_sites_Cartn_tran_matrix']
    cell_mat_f = np.linalg.inv(cell_mat_m).T
    hkl = np.stack(tuple((np.array(refln_dict[f'_refln_index_{idx}']) for idx in ('h', 'k', 'l'))), axis=1)
    g_ks = np.linalg.norm(np.einsum('xy, zy -> zx', cell_mat_f, hkl), axis=-1)

    f0j_core_dict = {}
    n_elec_core = {}
    for element, atomic_entries in qubox_density_atomic_dicts.items():
        r = np.array(atomic_entries['_qubox_density_atomic_rgrid'])
        core_density = np.array(atomic_entries['_qubox_density_atomic_core'])
        gr = r[None,:] * g_ks[:,None]
        j0 = np.zeros_like(gr)
        j0[gr != 0] = np.sin(2 * np.pi * gr[gr != 0]) / (2 * np.pi * gr[gr != 0])
        j0[gr == 0] = 1
        y00_factor = 0.5 * np.pi**(-0.5)
        f0j_core_dict[element] = simps(4 * np.pi * r**2  * core_density * j0, x=r) * y00_factor
        n_elec_core[element] = simps(4 * np.pi * r**2  * core_density, x=r)
    return f0j_core_dict, n_elec_core

class RegGridDensityPartitioner(DensityPartitioner):
    pass