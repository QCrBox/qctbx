from typing import List, Dict, Union, Tuple, Optional
import numpy as np


def cell_dict2atom_sites_dict(
    cell_dict: Dict[str, float]
):
    """Generates a matrix with the three lattice vectors as row vectors

    Returns
    -------
    matrix: np.ndarray
        size (3, 3) array containing the cell vectors as row vectors
    """
    a = cell_dict['cell_length_a']
    b = cell_dict['cell_length_b']
    c = cell_dict['cell_length_c']
    alpha = cell_dict['cell_angle_alpha'] / 180.0 * np.pi
    beta = cell_dict['cell_angle_beta'] / 180.0 * np.pi
    gamma = cell_dict['cell_angle_gamma'] / 180.0 * np.pi
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
        'atom_sites_Cartn_transform_axes': 'a parallel to x; b in the plane of y and z',
        'atom_sites_Cartn_tran_matrix': matrix
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
    cartn_tran_matrix = atom_sites_dict['atom_sites_Cartn_tran_matrix']
    rec_cartn_tran_matrix = np.linalg.inv(cartn_tran_matrix).T
    hkl = np.array([refln_dict[f'refln_index_{index}'] for index in ('h', 'k', 'l')]).T
    output = refln_dict.copy()
    output['refln_sint/lambda'] = np.linalg.norm(np.einsum('xy, zy -> zx', rec_cartn_tran_matrix, hkl), axis=1) / 2 
    return output

def add_cart_pos(atom_site_dict, cell_dict):
    atom_sites_dict = cell_dict2atom_sites_dict(cell_dict)
    xyz_fract = np.array([atom_site_dict[f'atom_site_fract_{val}'] for val in ('x', 'y', 'z')]).T
    xyz_cartn = np.einsum('xy, zy -> zx', atom_sites_dict['atom_sites_Cartn_tran_matrix'], xyz_fract)
    atom_site_out = atom_site_dict.copy()
    atom_site_out['atom_site_Cartn_x'] = list(xyz_cartn[:,0])
    atom_site_out['atom_site_Cartn_y'] = list(xyz_cartn[:,1])
    atom_site_out['atom_site_Cartn_z'] = list(xyz_cartn[:,2])
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


