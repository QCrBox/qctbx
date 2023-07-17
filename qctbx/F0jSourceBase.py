from typing import Dict, List, Any, Tuple
import numpy as np
import re
from copy import deepcopy
from collections import OrderedDict
from .io.tsc import TSCFile
from .conversions import expand_atom_site_table_symm
from abc import abstractmethod

class F0jSource:
    def cctbx2tsc(
        self, 
        structure,
        miller_array,
        tsc_filename,
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
        tsc_filename: str,
        atom_site_dict: Dict[str, List[Any]],
        cell_dict: Dict[str, Any],
        space_group_dict: Dict[str, Any],
        refln_dict: Dict[str, Any],
        tsc_title='qctbx-export'
    ):
        #TODO: Implement culling of inversion equivalent reflections

        f0j = self.calc_f0j(atom_site_dict, cell_dict, space_group_dict, refln_dict)

        new_tsc = TSCFile()

        new_tsc.scatterers = list(atom_site_dict['_atom_site_label'])

        new_data = {
            (h, k, l): form_factors for h, k, l, form_factors in zip(refln_dict['_refln_index_h'], refln_dict['_refln_index_k'], refln_dict['_refln_index_l'], f0j.T)
        }
        new_tsc.header['TITLE'] = tsc_title

        new_tsc.data = new_data

        new_tsc.to_file(tsc_filename)





