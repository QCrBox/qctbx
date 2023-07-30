from io import StringIO

import numpy as np

from ..conversions import (cell_dict2atom_sites_dict, create_hkl_dmin,
                          split_error)

def cif2dicts(cif_filename, cif_dataset, complete_dmin=False):
    from iotbx import cif

    cif_model = cif.reader(cif_filename).model()
    block = cif_model[cif_dataset]

    cell_dict = {
        key: split_error(block[key])[0] for key in ('_cell_length_a', '_cell_length_b', '_cell_length_c', '_cell_angle_alpha', '_cell_angle_beta', '_cell_angle_gamma')
    }

    cell_mat_m = cell_dict2atom_sites_dict(cell_dict)['_atom_sites_Cartn_tran_matrix']
    cell_mat_f = np.linalg.inv(cell_mat_m).T

    if complete_dmin:
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

    else:
        refln_dict = {
            key: entry for key, entry in block.items() if key.startswith('_refln_')
        }

    space_group_dict = {
        '_space_group_symop_id': block.get('_space_group_symop_id', np.arange(1, len(block['_space_group_symop_operation_xyz']) + 1)),
        '_space_group_symop_operation_xyz': [val.upper() for val in block['_space_group_symop_operation_xyz']]
    }

    atom_site_keys = (
        '_atom_site_label',
        '_atom_site_type_symbol'
    )

    atom_site_dict = {
        key: list(block[key]) for key in atom_site_keys
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

    return atom_site_dict, cell_dict, space_group_dict, refln_dict

def read_settings_cif(scif_path, block_name):
    from iotbx import cif
    with open(scif_path, encoding='ASCII') as fobj:
        content = fobj.read()
    new_str = content.replace('\nsettings_', '\ndata_settings_input_')

    with StringIO(new_str) as io_obj:
        cif_data = cif.reader(file_object=io_obj).model()
    settings_cif = cif_data.blocks['settings_input_' + block_name]
    return settings_cif