from typing import Dict, List, Any, Tuple
import numpy as np
import re
from copy import deepcopy
from collections import OrderedDict
from .RegGridDensityCalculators.RegGridDensityCalculatorBase import RegGridDensityCalculator
from ..io.tsc import TSCFile
from ..conversions import expand_atom_site_table_symm
from ..F0jSourceBase import F0jSource

class ScaffF0jSource(F0jSource):
    charge_dict = {}

    def __init__(self, density_calculator, partitioner, expand_positions={}, use_charges=False):
        self.density_calculator = density_calculator
        self.partitioner = partitioner
        self.expand_positions = expand_positions
        self.use_charges = use_charges
        if use_charges:
            raise NotImplementedError('cluster charge calculations are not implemented yet')
              
    def calc_f0j(
        self,
        atom_site_dict: Dict[str, List[Any]],
        cell_dict: Dict[str, Any],
        space_group_dict: Dict[str, Any],
        refln_dict: Dict[str, Any]
    ):
        if isinstance(self.density_calculator, RegGridDensityCalculator):
            expand_positions = {op: 'all' for op in space_group_dict['_space_group_symop_operation_xyz']}
            atom_site_dict_exp = expand_atom_site_table_symm(atom_site_dict, expand_positions, cell_dict, check_special=True)
        elif len(self.expand_positions) > 0:
            atom_site_dict_exp = expand_atom_site_table_symm(atom_site_dict, self.expand_positions, check_special=False)
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




