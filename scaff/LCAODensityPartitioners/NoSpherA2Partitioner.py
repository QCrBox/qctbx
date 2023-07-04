from .LCAODensityPartitionerBase import LCAODensityPartitioner
from ..conversions import cell_dict2atom_sites_dict
from ...io.tsc import TSCFile
from copy import deepcopy
import os
import re
import fractions
import subprocess
import numpy as np
from iotbx import cif
from typing import Tuple, List, Dict, Any, Optional

defaults = {
    'nosphera2_path': './NoSpherA2',
    'n_cores': 4,
    'nosphera2_accuracy': 2,
    'calc_folder': '.'
}


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

def symm_mat_vec2str(symm_mat, symm_vec):
    symm_string = ''
    for symm_parts, add in zip(symm_mat, symm_vec):
        symm_string_add = str(fractions.Fraction(add).limit_denominator(50))
        if symm_string_add != '0':
            symm_string += symm_string_add 
        for symm_part, symbol in zip(symm_parts, ['X', 'Y', 'Z']):
            if abs(symm_part) < 1e-10:
                continue
            if abs(1 - abs(symm_part)) < 1e-10:
                if symm_part > 0:
                    symm_string += f'+{symbol}'
                else:
                    symm_string += f'-{symbol}'
            else:
                fraction = fractions.Fraction(symm_part).limit_denominator(50)
                if str(fraction).startswith('-'):
                    symm_string += f'{str(fraction)}*{symbol}'
                else:
                    symm_string += f'+{str(fraction)}*{symbol}'
        symm_string += ','
    return symm_string[:-1]

def write_nospa2_cif(filename, cell_dict, space_group_dict, atom_site_dict):
    new_block = cif.model.block()
    for key, value in cell_dict.items():
        new_block[key] = value
        
    new_loop = cif.model.loop()    
    for key, value in space_group_dict.items():
        new_loop.add_column(key, list(value))
    
    new_block.add_loop(new_loop)
    
    new_loop = cif.model.loop()    
    for key, value in atom_site_dict.items():
        new_loop.add_column(key, list(value))
    
    new_block.add_loop(new_loop)

    new_model = cif.model.cif()
    new_model['nospa2'] = new_block

    with open(filename, 'w') as fo:
        fo.write(str(new_model))


def write_mock_hkl(filename, refln_dict):
    with open(filename, 'w') as fo:
        for h, k, l in zip(refln_dict['_refln_index_h'], refln_dict['_refln_index_k'], refln_dict['_refln_index_l']):
            fo.write(f'{int(h): 4d}{int(k): 4d}{int(l): 4d}{0.0: 8.2f}{0.0: 8.2f}\n')


class NoSpherA2Partitioner(LCAODensityPartitioner):
    accepts_input = ('wfn', 'wfx')

    def __init__(self, options={}):
        options = deepcopy(options)
        for key, value in defaults.items():
            if key not in options:
                options[key] = value

        self.options = options

    def run_nospherA2(
        self,
        atom_labels: List[int],
        atom_site_dict: Dict[str, List[Any]],
        cell_dict: Dict[str, Any],
        space_group_dict: Dict[str, Any],
        refln_dict: Dict[str, Any],
        density_path: str
    ):

        atom_sites_dict = cell_dict2atom_sites_dict(cell_dict)
        cell_dict['_cell_volume'] = np.linalg.det(atom_sites_dict['_atom_sites_Cartn_tran_matrix'])
        
        cleaned_sg_dict = deepcopy(space_group_dict)
        
        cleaned_sg_dict['_space_group_symop_operation_xyz'] = [
            symm_mat_vec2str(*symm_to_matrix_vector(symm_string)) for symm_string in cleaned_sg_dict['_space_group_symop_operation_xyz']
        ]

        all_atom_labels = list(atom_site_dict['_atom_site_label'])
        atom_indexes = [all_atom_labels.index(label) for label in atom_labels]     
        select_atom_site_dict = {
            key: [value[i] for i in atom_indexes] for key, value in atom_site_dict.items()
        }
        write_nospa2_cif('npa2.cif', cell_dict, cleaned_sg_dict, atom_site_dict)
        write_nospa2_cif('npa2_asym.cif', cell_dict, cleaned_sg_dict, select_atom_site_dict)
        write_mock_hkl('mock.hkl', refln_dict)

        pass_options = deepcopy(self.options)
        pass_options['density_path'] = density_path

        subprocess.check_call('{nosphera2_path} -hkl mock.hkl -wfn {density_path} -cif npa2.cif -asym_cif npa2_asym.cif -acc {nosphera2_accuracy} -cores {n_cores}'.format(**pass_options), shell=True, stdout=subprocess.DEVNULL, cwd=self.options['calc_folder'])
        
        
    def calc_f0j(
        self,
        atom_labels: List[int],
        atom_site_dict: Dict[str, List[Any]],
        cell_dict: Dict[str, Any],
        space_group_dict: Dict[str, Any],
        refln_dict: Dict[str, Any],
        density_path: str
    ):
        self.run_nospherA2(atom_labels, atom_site_dict, cell_dict, space_group_dict, refln_dict, density_path)

        tsc = TSCFile.from_file('experimental.tsc')

        f0j = np.array([
            tsc.data[(h, k, l)] if (h, k, l) in tsc.data.keys() else np.conj(tsc.data[(-h, -k, -l)]) for h, k, l in zip(refln_dict['_refln_index_h'], refln_dict['_refln_index_k'], refln_dict['_refln_index_l'])
        ]).T

        with open(os.path.join(self.options['calc_folder'], 'NoSpherA2.log'), 'r') as fo:
            content = fo.read()
        
        charge_table_match = re.search(r'Atom\s+Becke\s+Spherical\s+Hirshfeld(.*)\nTotal number of electrons', content, flags=re.DOTALL)
        
        assert charge_table_match is not None, 'Could not find charge table in NoSpherA2.log, probably unexpected format'

        charge_table = charge_table_match.group(1)

        charge_dict = {}
        for line in charge_table.split('\n')[1:]:
            name, _, _, atom_charge = line.strip().split()
            charge_dict[name] = float(atom_charge)

        return f0j, np.array([charge_dict[label] for label in atom_labels])

    def citation_strings(self) -> str:
        # TODO: Add a short string with the citation as bib and a sentence what was done
        return 'bib_string', 'sentence string'
    
    def cif_output(self) -> str:
        return 'To be implemented'