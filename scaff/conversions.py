from typing import List, Dict, Union, Tuple, Optional
import numpy as np

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

def add_cart_pos(atom_site_dict, cell_dict):
    atom_sites_dict = cell_dict2atom_sites_dict(cell_dict)
    xyz_fract = np.array([atom_site_dict[f'_atom_site_fract_{val}'] for val in ('x', 'y', 'z')]).T
    xyz_cartn = np.einsum('xy, zy -> zx', atom_sites_dict['_atom_sites_Cartn_tran_matrix'], xyz_fract)
    atom_site_out = atom_site_dict.copy()
    atom_site_out['_atom_site_Cartn_x'] = list(xyz_cartn[:,0])
    atom_site_out['_atom_site_Cartn_y'] = list(xyz_cartn[:,1])
    atom_site_out['_atom_site_Cartn_z'] = list(xyz_cartn[:,2])
    return atom_site_out, atom_sites_dict


def expand_symm_unique(
        type_symbols: List[str],
        coordinates: np.ndarray,
        cell_mat_m: np.ndarray,
        symm_mats_vec: Tuple[np.ndarray, np.ndarray],
        skip_symm: Dict[str, List[int]] = {},
        magmoms: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, List[str],
               np.ndarray, Optional[np.ndarray]]:
    """Expand the type_symbols and coordinates for one complete unit cell.
    Atoms on special positions appear only once. For disorder on a special
    position use skip_symm.


    Parameters
    ----------
    type_symbols : List[str]
        size (H) array containing the calculated resolution values
    coordinates : npt.NDArray[np.float64]
        size (N, 3) array of fractional atomic coordinates
    cell_mat_m : npt.NDArray[np.float64]
        Matrix with cell vectors as column vectors, (Angstroem)
    symm_mats_vec : Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
        size (K, 3, 3) array of symmetry matrices and size (K, 3) array
        of translation vectors for all symmetry elements in the unit cell
    skip_symm : Dict[str, List[int]], optional
        Symmetry elements with indexes given the list(s) in the dictionary
        values with not be applied to the respective atoms with the atom names
        given in the key(s). Indexes need to be identical to the ones in 
        symm_mats_vec., by default {}
    magmoms : Optional[npt.NDArray[np.float64]], optional
        Magnetic Moments. The enforced symmetry might not be correcz, by default
        None

    Returns
    -------
    symm_positions: npt.NDArray[np.float64]
        size (M, 3) array of all unique atom positions within the unit cell
    symm_symbols: List[str]
        ist of length M with element symbols for the unique atom positions
        within the unit cell
    reverse_indexes: npt.NDArray[np.float64]
        size (K, N) array with indexes mapping the unique atom positions back to 
        the individual symmetry elements and atom positions in the asymmetric 
        unit
    symm_magmoms: Optional[npt.NDArray[np.float64]]]
        magnetic moments for symmetry generated atoms. Undertested!
    """
    symm_mats_r, symm_vecs_t = symm_mats_vec
    pos_frac0 = coordinates % 1
    un_positions = np.zeros((0, 3))
    n_atoms = 0
    type_symbols_symm = []
    inv_indexes = []
    if magmoms is not None:
        magmoms_symm = []
    else:
        magmoms_symm = None
    # Only check atom with itself
    for atom_index, (pos0, type_symbol) in enumerate(zip(pos_frac0, type_symbols)):
        if atom_index in skip_symm:
            use_indexes = [i for i in range(symm_mats_r.shape[0]) if i not in skip_symm[atom_index]]
        else:
            use_indexes = list(range(symm_mats_r.shape[0]))
        symm_positions = (np.einsum(
            'kxy, y -> kx',
             symm_mats_r[use_indexes, :, :], pos0) + symm_vecs_t[use_indexes, :]
        ) % 1
        _, unique_indexes, inv_indexes_at = np.unique(
            np.round(np.einsum('xy, zy -> zx', cell_mat_m, symm_positions), 3),
            axis=0,
            return_index=True,
            return_inverse=True
        )
        un_positions = np.concatenate((un_positions, symm_positions[unique_indexes]))
        type_symbols_symm += [type_symbol] * unique_indexes.shape[0]
        if magmoms is not None:
            magmoms_symm += [magmoms[atom_index]] * unique_indexes.shape[0]
        inv_indexes.append(inv_indexes_at + n_atoms)
        n_atoms += unique_indexes.shape[0]
    if magmoms_symm is not None:
        magmoms_symm = np.array(magmoms_symm)
    return un_positions.copy(), type_symbols_symm, np.array(inv_indexes, dtype=object), magmoms_symm


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
    return {
        '_refln_index_h': refln_dict['_refln_index_h'][refln_dict['_refln_sint/lambda'] <= 0.5 / d_min].copy(),
        '_refln_index_k': refln_dict['_refln_index_k'][refln_dict['_refln_sint/lambda'] <= 0.5 / d_min].copy(),
        '_refln_index_l': refln_dict['_refln_index_l'][refln_dict['_refln_sint/lambda'] <= 0.5 / d_min].copy(),
    }