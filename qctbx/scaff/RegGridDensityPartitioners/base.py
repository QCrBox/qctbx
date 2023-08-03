from typing import Any, Dict, Union
import warnings

import numpy as np
from scipy.integrate import simps

from ..base_classes import DensityPartitioner
from ..util import dict_merge
from ...io.cif import read_settings_cif, settings_cif2kwargs
from ...conversions import cell_dict2atom_sites_dict

def calc_f0j_core(
    cell_dict: Dict[str, Any],
    refln_dict: Dict[str, Any],
    qctbx_density_atomic_dict: Dict[str, Any]
):
    """
    Calculates the core density in Fourier space.

    Args:
        cell_dict (Dict[str, Any]): A dictionary representing the unit cell.
        refln_dict (Dict[str, Any]): A dictionary representing the reflection data.
        qubox_density_atomic_dict (Dict[str, Dict[str, Any]]): A dictionary of
            dictionaries with elements as key and the entries being dictionaries
            containing the atomic density on a real-space grid. Used entries within
            the individual elements are _qctbx_density_atomic_rgrid and
            _qctbx_density_atomic_core

    Returns:
        Dict[str, Any]: A dictionary of core density values in Fourier space, keyed by element.
    """
    cell_mat_m = cell_dict2atom_sites_dict(cell_dict)['_atom_sites_Cartn_tran_matrix']
    cell_mat_f = np.linalg.inv(cell_mat_m).T
    hkl = np.stack(tuple((np.array(refln_dict[f'_refln_index_{idx}']) for idx in ('h', 'k', 'l'))), axis=1)
    g_ks = np.linalg.norm(np.einsum('xy, zy -> zx', cell_mat_f, hkl), axis=-1)

    f0j_core_dict = {}
    n_elec_core = {}
    for element, atomic_entries in separate_atoms_in_dict(qctbx_density_atomic_dict).items():
        r = np.array(atomic_entries['_qctbx_density_atomic_rgrid'])
        core_density = np.array(atomic_entries['_qctbx_density_atomic_core'])
        gr = r[None,:] * g_ks[:,None]
        j0 = np.zeros_like(gr)
        j0[gr != 0] = np.sin(2 * np.pi * gr[gr != 0]) / (2 * np.pi * gr[gr != 0])
        j0[gr == 0] = 1
        f0j_core_dict[element] = simps(4 * np.pi * r**2  * core_density * j0, x=r)
        n_elec_core[element] = simps(4 * np.pi * r**2  * core_density, x=r)
    return f0j_core_dict, n_elec_core


def separate_atoms_in_dict(qctbx_density_atomic_dict):
    atom_type_list = qctbx_density_atomic_dict['_qctbx_density_atomic_atom_type']
    atom_types = set(atom_type_list)
    separated_dict = {}
    for atom_type in atom_types:
        separated_dict[atom_type] = {key: [
            value for value, inner_atom_type in zip(values, atom_type_list) if inner_atom_type == atom_type
        ] for key, values in qctbx_density_atomic_dict.items()}
    return separated_dict


class RegGridDensityPartitioner(DensityPartitioner):
    _density_type = None
    _cif_entry_start = '_qctbx_reggridpartition_'
    available_args = ('software', 'method', 'density_type', 'specific_options', 'calc_options', 'qctbx_density_atomic_dict')
    def __init__(
        self,
        method:str=None,
        density_type:str=None,
        qctbx_density_atomic_dict: Dict[str, Union[str, float]]=None,
        specific_options: Dict[str, Any]=None,
        calc_options: Dict[str, Any]=None
    ):
        self.method = method
        self.density_type = density_type
        self.qctbx_density_atomic_dict = qctbx_density_atomic_dict
        if specific_options is None:
            self.specific_options = {}
        else:
            self.specific_options = specific_options
        if calc_options is None:
            self.calc_options = {}
        else:
            self.calc_options = calc_options

    @classmethod
    def from_settings_cif(cls, scif_path, block_name):
        #TODO There should possibly be a central settings_cif wrapper
        settings_cif = read_settings_cif(scif_path, block_name)

        dict_entries = ('specific_options', 'calc_options')
        type_funcs = {
            'method': str,
            'density_type': str,
        }

        kwargs = settings_cif2kwargs(
            settings_cif,
            cls._cif_entry_start,
            dict_entries,
            type_funcs,
            cls.available_args
        )

        if '_qctbx_density_atomic_atom_type' in settings_cif:
            kwargs['qctbx_density_atomic_dict'] = {
                '_qctbx_density_atomic_atom_type': [str(val) for val in settings_cif['_qctbx_density_atomic_atom_type']],
                '_qctbx_density_atomic_rgrid': [float(val) for val in settings_cif['_qctbx_density_atomic_rgrid']],
                '_qctbx_density_atomic_valence': [float(val) for val in settings_cif['_qctbx_density_atomic_valence']],
                '_qctbx_density_atomic_core': [float(val) for val in settings_cif['_qctbx_density_atomic_core']],
                '_qctbx_density_atomic_total': [float(val) for val in settings_cif['_qctbx_density_atomic_total']]
            }

        new_obj = cls(**kwargs)

        return new_obj

    @property
    def density_type(self):
        return self._density_type

    @density_type.setter
    def density_type(self, value):
        if value not in ('total', 'valence'):
            raise NotImplementedError('density_type can be either valence or total')
        self._density_type = value

    def update_from_dict(self, update_dict, update_if_present=True):

        for key in update_dict.keys():
            if key not in self.available_args:
                warnings.warn(f'Could not find matching property for key: {key}')

        condition = (self.method is None) or update_if_present
        if condition and 'method' in update_dict:
            self.method = update_dict['method']

        condition = (self.density_type is None) or update_if_present
        if condition and 'density_type' in update_dict:
            self.density_type = update_dict['density_type']

        condition = (self.qctbx_density_atomic_dict is None) or update_if_present
        if condition and 'qctbx_density_atomic_dict' in update_dict:
            self.qctbx_density_atomic_dict = update_dict['qctbx_density_atomic_dict']

        #dictionaries are merged instead of replaced
        updates = update_dict.get('specific_options', {})
        if update_if_present:
            self.specific_options = dict_merge(self.specific_options, updates)
        else:
            self.specific_options = dict_merge(updates, self.specific_options)

        updates = update_dict.get('calc_options', {})
        if update_if_present:
            self.calc_options = dict_merge(self.calc_options, updates)
        else:
            self.calc_options = dict_merge(updates, self.calc_options)

