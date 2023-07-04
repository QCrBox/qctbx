from typing import Dict, List, Any, Tuple
import numpy as np
import re
from copy import deepcopy
from collections import OrderedDict
from ..io.tsc import TSCFile
from .conversions import expand_atom_site_table_symm

class LCAOF0jEvaluation:
    charge_dict = {}

    def __init__(self, density_calculator, partitioner, expand_positions={}, use_charges=False):
        self.density_calculator = density_calculator
        self.partitioner = partitioner
        self.expand_positions = expand_positions
        self.use_charges = use_charges
        if use_charges:
            raise NotImplementedError('cluster charge calculations are not implemented yet')
        
    def expand_symm(self, atom_site_dict):
        return expand_atom_site_table_symm(atom_site_dict, self.expand_positions, check_special=False)
        
    def calc_f0j(
        self,
        atom_site_dict: Dict[str, List[Any]],
        cell_dict: Dict[str, Any],
        space_group_dict: Dict[str, Any],
        refln_dict: Dict[str, Any]
    ):
        if len(self.expand_positions) > 0:
            atom_site_dict_exp = self.expand_symm(atom_site_dict)
        else:
            atom_site_dict_exp = atom_site_dict

        wavefunction_path = self.density_calculator.calculate_density(atom_site_dict_exp, cell_dict)

        f0j, charges = self.partitioner.calc_f0j(
            list(atom_site_dict['_atom_site_label']),
            atom_site_dict_exp,
            cell_dict,
            space_group_dict,
            refln_dict,
            wavefunction_path
        )

        return f0j

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





