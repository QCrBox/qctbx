from typing import List, Dict, Union, Tuple, Optional, Any
from collections import OrderedDict
from copy import deepcopy
import numpy as np
import fractions
import re

def cell_dict2atom_sites_dict(
    cell_dict: Dict[str, Union[float, np.ndarray]]
) -> Dict[str, Union[str, np.ndarray]]:
    """
    Converts a cell dictionary into an atom sites dictionary that includes transformation matrix.

    The transformation matrix is generated based on the cell parameters. This matrix contains the three lattice vectors 
    as rows. It is used to convert from fractional to Cartesian coordinates.

    Args:
        cell_dict (Dict[str, Union[float, np.ndarray]]): A dictionary representing a unit cell. It should contain the 
        following keys: '_cell_length_a', '_cell_length_b', '_cell_length_c', '_cell_angle_alpha', '_cell_angle_beta', 
        '_cell_angle_gamma'. The corresponding values should be floats representing the cell lengths (a, b, c) in angstroms 
        and the cell angles (alpha, beta, gamma) in degrees.

    Returns:
        Dict[str, Union[str, np.ndarray]]: A dictionary containing the transformation matrix and its description. The keys 
        are '_atom_sites_Cartn_transform_axes' (with value being a string description of the transformation axes) and 
        '_atom_sites_Cartn_tran_matrix' (with value being a 3x3 numpy array representing the transformation matrix).
    """
    a = cell_dict['_cell_length_a']
    b = cell_dict['_cell_length_b']
    c = cell_dict['_cell_length_c']
    alpha = cell_dict['_cell_angle_alpha'] / 180.0 * np.pi
    beta = cell_dict['_cell_angle_beta'] / 180.0 * np.pi
    gamma = cell_dict['_cell_angle_gamma'] / 180.0 * np.pi
    matrix = np.array(
        [
            [
                a,
                b * np.cos(gamma),
                c * np.cos(beta)
            ],
            [
                0,
                b * np.sin(gamma),
                c * (np.cos(alpha) - np.cos(gamma) * np.cos(beta)) / np.sin(gamma)
            ],
            [
                0,
                0,
                c / np.sin(gamma) * np.sqrt(1.0 - np.cos(alpha)**2 - np.cos(beta)**2
                                            - np.cos(gamma)**2
                                            + 2 * np.cos(alpha) * np.cos(beta) * np.cos(gamma))
            ]
        ]
    )
    atom_sites_dict = {
        '_atom_sites_Cartn_transform_axes': 'a parallel to x; b in the plane of y and z',
        '_atom_sites_Cartn_tran_matrix': matrix
    }
    return atom_sites_dict


def add_sin_theta_ov_lambda(
    cell_dict: Dict[str, float],
    refln_dict: Dict[str, List[int]]
) -> Dict[str, List[Union[int, float]]]:
    """Calculate the resolution in sin(theta)/lambda for the given set of Miller
    indicees
    """
    atom_sites_dict = cell_dict2atom_sites_dict(cell_dict=cell_dict)
    cartn_tran_matrix = atom_sites_dict['_atom_sites_Cartn_tran_matrix']
    rec_cartn_tran_matrix = np.linalg.inv(cartn_tran_matrix).T
    hkl = np.array([refln_dict[f'_refln_index_{index}'] for index in ('h', 'k', 'l')]).T
    output = refln_dict.copy()
    output['_refln_sint/lambda'] = np.linalg.norm(np.einsum('xy, zy -> zx', rec_cartn_tran_matrix, hkl), axis=1) / 2 
    return output

def add_cart_pos(atom_site_dict: Dict[str, List[float]], cell_dict: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Convert fractional atomic positions to Cartesian coordinates based on the unit cell parameters.

    Parameters
    ----------
    atom_site_dict : Dict[str, List[float]]
        Dictionary containing fractional atomic positions as lists of floats
        in cif format. Needs to include: '_atom_site_fract_x',
        '_atom_site_fract_y', and '_atom_site_fract_z'.

    cell_dict : Dict[str, Any]
        Dictionary containing cell parameters in cif notation

    Returns
    -------
    atom_site_out : Dict[str, Any]
        The input dictionary with added Cartesian coordinates. These are added 
        as lists of floats with keys '_atom_site_Cartn_x', '_atom_site_Cartn_y',
        and '_atom_site_Cartn_z'.
        
    atom_sites_dict : Dict[str, Any]
        Dictionary containing the transformation matrix for conversion from 
        fractional to Cartesian coordinates and the cartesian convention in 
        as cif keys.
    """
    atom_sites_dict = cell_dict2atom_sites_dict(cell_dict)
    xyz_fract = np.array([atom_site_dict[f'_atom_site_fract_{val}'] for val in ('x', 'y', 'z')]).T
    xyz_cartn = np.einsum('xy, zy -> zx', atom_sites_dict['_atom_sites_Cartn_tran_matrix'], xyz_fract)
    atom_site_out = atom_site_dict.copy()
    atom_site_out['_atom_site_Cartn_x'] = list(xyz_cartn[:,0])
    atom_site_out['_atom_site_Cartn_y'] = list(xyz_cartn[:,1])
    atom_site_out['_atom_site_Cartn_z'] = list(xyz_cartn[:,2])
    return atom_site_out, atom_sites_dict


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


def expand_atom_site_table_symm(
    atom_site_dict: Dict[str, List[Union[str, float]]], 
    expand_positions: Dict[str, Union[str, List[str]]], 
    cell_dict: Optional[Dict[str, float]] = None, 
    check_special: bool = True
) -> Dict[str, List[Union[str, float]]]:
    """
    Expands an atom site table based on symmetry operations. 

    Args:
        atom_site_dict (Dict[str, List[Union[str, float]]]): Dictionary containing the atom site information. 
            Necessary keys are:
            - '_atom_site_label': List of strings, the labels of the atoms.
            - '_atom_site_type_symbol': List of strings, the atomic symbols.
            - '_atom_site_fract_x': List of floats, the fractional x coordinates.
            - '_atom_site_fract_y': List of floats, the fractional y coordinates.
            - '_atom_site_fract_z': List of floats, the fractional z coordinates.
            Optional keys that are also returned are:
            - '_atom_site_disorder_group': List of strings, the disorder group of the atoms (if present).
            - '_atom_site_disorder_assembly': List of strings, the disorder assembly of the atoms (if present).
            
        expand_positions (Dict[str, Union[str, List[str]]]): Dictionary containing the symmetry operations 
            and the atom labels to which these operations should be applied. The symmetry operation is the key 
            and the value can be either:
            - a string "all" indicating that the symmetry operation should be applied to all atoms.
            - a string starting with "all" followed by "skip" and a list of space separated atom labels to be skipped.
            - a list of specific atom labels to which the symmetry operation should be applied.

        cell_dict (Optional[Dict[str, float]]): Dictionary containing the unit cell parameters. 
            Default is None. Required if check_special is True.
        check_special (bool): Flag to check for special positions by distance. Default is True.

    Returns:
        new_atom_site_dict (Dict[str, List[Union[str, float]]]): New atom site dictionary after applying the symmetry operations.

    Raises:
        AssertionError: If check_special is True and cell_dict is None.
    """

    use_cols = [
        '_atom_site_label', '_atom_site_type_symbol', '_atom_site_fract_x', '_atom_site_fract_y', '_atom_site_fract_z',  '_atom_site_disorder_group', '_atom_site_disorder_assembly'
    ]
    use_cols = [col for col in use_cols if col in atom_site_dict]
    
    original_indexes = list(range(len(atom_site_dict[use_cols[0]])))
    
    atoms_dict = OrderedDict((atom_site_dict['_atom_site_label'][i], {key: atom_site_dict[key][i] for key in use_cols}) for i in original_indexes)
    xyz_cols = ('_atom_site_fract_x', '_atom_site_fract_y', '_atom_site_fract_z')
    if check_special:
        assert cell_dict is not None, 'For checking special positions by distance a cell_dict is needed'
        cell_mat_m = cell_dict2atom_sites_dict(cell_dict)['_atom_sites_Cartn_tran_matrix']
        check_present = {key: [cell_mat_m @ (np.array([value[col] for col in xyz_cols]) % 1)] for key, value in atoms_dict.items()}
    
    for symm_element, expanded_atoms in expand_positions.items():
        symm_mat, symm_vec = symm_to_matrix_vector(symm_element)
        if np.sum(np.abs(symm_mat - np.eye(3))) + np.sum(np.abs(symm_vec)) < 1e-5:
            # do not include x, y, z
            continue
        if expanded_atoms.strip().startswith('all'):
            if 'skip' in expanded_atoms:
                skipped_atoms = expanded_atom.split('skip')[1].strip().split()
                expanded_atoms = [elem for elem in atom_site_dict['_atom_site_label'] if elem not in skipped_atoms]
            else:
                expanded_atoms = list(atom_site_dict['_atom_site_label'])
        for expanded_atom in expanded_atoms:
            new_atom = deepcopy(atoms_dict[expanded_atom])
            xyz = np.array([new_atom[col] for col in xyz_cols])
            new_xyz = symm_mat @ xyz + symm_vec
            if check_special:
                new_cartn = cell_mat_m @ (new_xyz % 1)
                if any(np.linalg.norm(new_cartn - known_cartn) < 0.1 for known_cartn in check_present[new_atom['_atom_site_label']]):
                    continue
                check_present[new_atom['_atom_site_label']].append(new_cartn)
            new_atom['_atom_site_fract_x'] = new_xyz[0]
            new_atom['_atom_site_fract_y'] = new_xyz[1]
            new_atom['_atom_site_fract_z'] = new_xyz[2]
            new_atom['_atom_site_label'] += ':' + symm_element.replace(' ', '')
            atoms_dict[new_atom['_atom_site_label']] = new_atom
    
    new_atom_site_dict = {
        key: [atom[key] for atom in list(atoms_dict.values())] for key in use_cols
    }

    return new_atom_site_dict



def create_hkl_dmin(
        cell_dict: Dict[str, Union[float, np.ndarray]],
        d_min: float
    ) -> Dict[str, np.ndarray]:
    """
    Generate a dictionary of h, k, and l indices of reflections for a crystal lattice specified by cell_dict, 
    considering reflections with d-spacing greater than or equal to d_min.

    Args:
        cell_dict (Dict[str, Union[float, np.ndarray]]): A dictionary representing a unit cell. It should contain the following keys:
            '_atom_sites_Cartn_tran_matrix': a transformation matrix from fractional to Cartesian coordinates.
        d_min (float): The minimum d-spacing for the reflections in angstrom. 

    Returns:
        Dict[str, np.ndarray]: A dictionary containing the h, k, and l indices of the reflections. The keys of the dictionary are 
        '_refln_index_h', '_refln_index_k', and '_refln_index_l'. The corresponding values are 1D numpy arrays of integers.
    """
    atom_sites_dict = cell_dict2atom_sites_dict(cell_dict)
    cell_mat_m = atom_sites_dict['_atom_sites_Cartn_tran_matrix']
    cell_mat_f = np.linalg.inv(cell_mat_m).T

    a_star, b_star, c_star = np.linalg.norm(cell_mat_f, axis=1)
    hmax = int(np.ceil(1 / d_min / a_star)) + 1
    kmax = int(np.ceil(1 / d_min / b_star)) + 1
    lmax = int(np.ceil(1 / d_min / c_star)) + 1
    h, k, l = np.meshgrid(np.arange(-hmax, hmax + 1), np.arange(-kmax, kmax + 1), np.arange(-lmax, lmax + 1))
    refln_dict = {
        '_refln_index_h': h.ravel(),
        '_refln_index_k': k.ravel(),
        '_refln_index_l': l.ravel()
    }
    refln_dict = add_sin_theta_ov_lambda(cell_dict, refln_dict)
    condition = np.logical_and(refln_dict['_refln_sint/lambda'] <= 0.5 / d_min, refln_dict['_refln_sint/lambda'] > 0.5)
    return {
        '_refln_index_h': refln_dict['_refln_index_h'][condition].copy(),
        '_refln_index_k': refln_dict['_refln_index_k'][condition].copy(),
        '_refln_index_l': refln_dict['_refln_index_l'][condition].copy(),
    }

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

def split_error(string: str) -> Union[Tuple[float, float], Tuple[int, int]]:
    """Helper function to split a string containing a value with error in
    brackets to a value-esd pair

    Parameters
    ----------
    string : str
        Input string containing the value to be split

    Returns
    -------
    Union[Tuple[float, float], Tuple[int, int]]
        Pair of floats if a '.' was present in string, otherwise a pair of ints
        containing the value and its esd
    """    
    int_search = re.search(r'([\-\d]*)\((\d*)\)', string)
    search = re.search(r'(\-{0,1})([\d]*)\.(\d*)\((\d*)\)', string)
    if search is not None:
        # we have found a float
        sign, before_dot, after_dot, err = search.groups()
        if sign == '-':
            return -1 * (int(before_dot) + int(after_dot) * 10**(-len(after_dot))), int(err) * 10**(-len(after_dot))
        else:
            return int(before_dot) + int(after_dot) * 10**(-len(after_dot)), int(err) * 10**(-len(after_dot))
    elif int_search is not None:
        # we have found an int
        value, error = int_search.groups()
        return int(value), int(error)
    else:
        # no error found
        return float(string), 0.0  