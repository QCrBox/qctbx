from typing import Dict, List, Any, Tuple
from pathlib import Path
import numpy as np
import re
from copy import deepcopy
from collections import OrderedDict
from .io.tsc import TSCFile, TSCBFile
from .conversions import expand_atom_site_table_symm, split_error, cell_dict2atom_sites_dict, create_hkl_dmin
from abc import abstractmethod
from .custom_typing import Path

class F0jSource:

    @abstractmethod          
    def calc_f0j(
        self,
        atom_site_dict: Dict[str, List[Any]],
        cell_dict: Dict[str, Any],
        space_group_dict: Dict[str, Any],
        refln_dict: Dict[str, Any]
    ):
        pass

    def write_tsc(
        self,
        tsc_filename: Path,
        atom_site_dict: Dict[str, List[Any]],
        cell_dict: Dict[str, Any],
        space_group_dict: Dict[str, Any],
        refln_dict: Dict[str, Any],
        tsc_title='qctbx-export'
    ):
        #TODO: Implement culling of inversion equivalent reflections
        f0j = self.calc_f0j(atom_site_dict, cell_dict, space_group_dict, refln_dict)
        if tsc_filename.endswith('.tscb'):
            new_tsc = TSCBFile()
        else:
            new_tsc = TSCFile()
        new_tsc.scatterers = list(atom_site_dict['_atom_site_label'])
        new_data = {
            (h, k, l): form_factors for h, k, l, form_factors in zip(refln_dict['_refln_index_h'], refln_dict['_refln_index_k'], refln_dict['_refln_index_l'], f0j.T)
        }
        new_tsc.header['TITLE'] = tsc_title
        new_tsc.data = new_data
        new_tsc.to_file(tsc_filename)

    def cctbx2tsc(
        self, 
        structure,
        miller_array,
        tsc_filename: Path,
        tsc_title='qctbx tsc file'
    ):
        # to not be limited by accuracy of cif block strings
        labels = [sc.label for sc in structure.scatterers()]
        type_symbols = [sc.element_symbol() for sc in structure.scatterers()]

        fract_x = np.empty(len(structure.scatterers()))
        fract_y = np.empty(len(structure.scatterers()))
        fract_z = np.empty(len(structure.scatterers()))
        occupancies = np.empty(len(structure.scatterers()))

        for index, sc in enumerate(structure.scatterers()):
            fract_x[index], fract_y[index], fract_z[index] = sc.site
            occupancies[index] = sc.occupancy

        atom_site_dict = {
            '_atom_site_label': labels,
            '_atom_site_type_symbol': type_symbols,
            '_atom_site_fract_x': fract_x,
            '_atom_site_fract_y': fract_y,
            '_atom_site_fract_z': fract_z,
            '_atom_site_occupancy': occupancies
        }

        cif_dict = structure.as_cif_block()

        cell_dict = {
            key.replace('.', '_'): float(value) for key, value in cif_dict.items() if key.startswith('_cell')
        }

        space_group_dict = {
            key.replace('.', '_'): list(value) for key, value in cif_dict.items() if key.startswith('_space_group_')
        }

        h, k, l = np.array(miller_array.expand_to_p1().indices()).T

        refln_dict = {
            '_refln_index_h': h,
            '_refln_index_k': k,
            '_refln_index_l': l,
        }

        self.write_tsc(
            tsc_filename,
            atom_site_dict,
            cell_dict,
            space_group_dict,
            refln_dict,
            tsc_title
        )

    def cif2tsc(
        self,
        cif_filename: Path,
        cif_dataset: str,
        tsc_filename: Path,
        tsc_title='qctbx-export'
    ):
        from iotbx import cif

        cif_model = cif.reader(cif_filename).model()
        block = cif_model[cif_dataset]

        cell_dict = {
            key: split_error(block[key])[0] for key in ('_cell_length_a', '_cell_length_b', '_cell_length_c', '_cell_angle_alpha', '_cell_angle_beta', '_cell_angle_gamma')
        }

        cell_mat_m = cell_dict2atom_sites_dict(cell_dict)['_atom_sites_Cartn_tran_matrix']
        cell_mat_f = np.linalg.inv(cell_mat_m).T

        if '_reflns_d_resolution_high' in block:
            reslim = float(block['_reflns_d_resolution_high']) - 0.01
        elif '_diffrn_reflns_theta_max' in block and '_diffrn_radiation_wavelength' in block:
            theta = np.deg2rad(float(block['_diffrn_reflns_theta_max']))
            reslim = float(block['_diffrn_radiation_wavelength']) / (2 * np.sin(theta)) - 0.001
        elif '_refln_index_h' in block:
            hkl = np.stack([np.array(block[f'_refln_index_{mil}'], dtype=np.float64) for mil in ('h', 'k', 'l')], axis=1)
            r_star = np.linalg.norm(np.einsum('xy, zy -> zx', cell_mat_f, hkl), axis=1)
            reslim = 1 / r_star.max() - 0.001
        elif '_diffrn_refln_index_h' in block:
            hkl = np.stack([np.array(block[f'_diffrn_refln_index_{mil}'], dtype=np.float64) for mil in ('h', 'k', 'l')], axis=1)
            r_star = np.linalg.norm(np.einsum('xy, zy -> zx', cell_mat_f, hkl), axis=1)
            reslim = 1 / r_star.max() - 0.001
        else:
            raise NotImplementedError('Could not determine the resolution from the given cif entries. Give either reflns_d_resolution_high, diffrn_reflns_theta_max and diffrn_radiation_wavelength or the (diffrn_)refln_index entries.')

        refln_dict = create_hkl_dmin(cell_dict, reslim)

        space_group_dict = {
            '_space_group_symop_id': block.get('_space_group_symop_id', np.arange(1, len(block['_space_group_symop_operation_xyz']) + 1)),
            '_space_group_symop_operation_xyz': [val.upper() for val in block['_space_group_symop_operation_xyz']]
        }

        atom_site_keys = ( 
            '_atom_site_label',
            '_atom_site_type_symbol'
        )

        atom_site_dict = {
            key: block[key] for key in atom_site_keys
        }

        split_keys = (    
            '_atom_site_fract_x',
            '_atom_site_fract_y',
            '_atom_site_fract_z'
        )
        for key in split_keys:
            vals = block[key]
            numerical = np.array([split_error(val)[0] for val in vals])
            atom_site_dict[key] = numerical

        add_keys = ('_atom_site_disorder_assembly', '_atom_site_disorder_group')

        for key in add_keys:
            atom_site_dict[key] = block.get(key, ['.'] * len(atom_site_dict['_atom_site_label']))

        self.write_tsc(
            tsc_filename,
            atom_site_dict,
            cell_dict,
            space_group_dict,
            refln_dict,
            tsc_title
        )





