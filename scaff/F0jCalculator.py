from typing import Dict, List, Any, Tuple
import numpy as np
import re
from copy import deepcopy
from collections import OrderedDict
from ..io.tsc import TSCFile

def symm_to_matrix_vector(instruction: str) -> Tuple[np.ndarray, np.ndarray]:
    """Converts a symmetry instruction into a symmetry matrix and a translation
    vector for that symmetry element.

    Parameters
    ----------
    instruction : str
        Instruction string containing symmetry instruction for all three 
        coordinates separated by comma signs (e.g -x, -y, 0.5+z)

    Returns
    -------
    symm_matrix: np.ndarray, 
        size (3, 3) array containing the symmetry matrix for the symmetry element
    symm_vector: np.ndarray
        size (3) array containing the translation vector for the symmetry element
    """    
    instruction_strings = [val.replace(' ', '').upper() for val in instruction.split(',')]
    matrix = np.zeros((3,3), dtype=np.float64)
    vector = np.zeros(3, dtype=np.float64)
    for xyz, element in enumerate(instruction_strings):
        # search for fraction in a/b notation
        fraction1 = re.search(r'(-{0,1}\d{1,3})/(\d{1,3})(?![XYZ])', element)
        # search for fraction in 0.0 notation
        fraction2 = re.search(r'(-{0,1}\d{0,1}\.\d{1,4})(?![XYZ])', element)
        # search for whole numbers
        fraction3 = re.search(r'(-{0,1}\d)(?![XYZ])', element)
        if fraction1:
            vector[xyz] = float(fraction1.group(1)) / float(fraction1.group(2))
        elif fraction2:
            vector[xyz] = float(fraction2.group(1))
        elif fraction3:
            vector[xyz] = float(fraction3.group(1))

        symm = re.findall(r'-{0,1}[\d\.]{0,8}[XYZ]', element)
        for xyz_match in symm:
            if len(xyz_match) == 1:
                sign = 1
            elif xyz_match[0] == '-' and len(xyz_match) == 2:
                sign = -1
            else:
                sign = float(xyz_match[:-1])
            if xyz_match[-1] == 'X':
                matrix[xyz, 0] = sign
            if xyz_match[-1] == 'Y':
                matrix[xyz, 1] = sign
            if xyz_match[-1] == 'Z':
                matrix[xyz, 2] = sign
    return matrix, vector


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
        use_cols = [
            '_atom_site_label', 'atom_site_fract_x', '_atom_site_fract_y', '_atom_site_fract_z',  '_atom_site_disorder_group', '_atom_site_disorder_assembly'
        ]
        use_cols = [col for col in use_cols if col in atom_site_dict]

        original_indexes = list(range(len(atom_site_dict[use_cols[0]])))

        atoms_dict = OrderedDict((atom_site_dict['_atom_site_label'][i], {key: atom_site_dict[key][i] for key in use_cols}) for i in original_indexes)
        xyz_cols = ('_atom_site_fract_x', '_atom_site_fract_y', '_atom_site_fract_z')
        for symm_element, expanded_atoms in self.expand_positions:
            symm_mat, symm_vec = symm_to_matrix_vector(symm_element)
            for expanded_atom in expanded_atoms:
                new_atom = deepcopy(atoms_dict[expanded_atom])
                xyz = np.array([new_atom[col] for col in xyz_cols])
                new_xyz = symm_mat @ xyz + symm_vec
                new_atom['_atom_site_fract_x'] = new_xyz[0]
                new_atom['_atom_site_fract_y'] = new_xyz[1]
                new_atom['_atom_site_fract_z'] = new_xyz[2]
                new_atom['_atom_site_label'] += ':' + symm_element
                atoms_dict[new_atom['_atom_site_label']] = new_atom

        new_atom_site_dict = {
            key: [atom[key] for atom in list(atoms_dict.values())] for key in use_cols
        }
        return new_atom_site_dict
        
    def calc_f0j(
        self,
        atom_site_dict: Dict[str, List[Any]],
        cell_dict: Dict[str, Any],
        space_group_dict: Dict[str, Any],
        refln_dict: Dict[str, Any]
    ):
        if len(self.expand_positions) > 0:
            atom_site_dict_exp = self.expand_positions(atom_site_dict)
        else:
            atom_site_dict_exp = atom_site_dict

        wavefunction_path = self.density_calculator.calculate_density(atom_site_dict_exp, cell_dict)

        return self.partitioner.calc_f0j(
            list(atom_site_dict['_atom_site_label']),
            atom_site_dict_exp,
            cell_dict,
            space_group_dict,
            refln_dict,
            wavefunction_path
        )

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




