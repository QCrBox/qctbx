from iotbx import cif

def write_minimal_cif(filename, cell_dict, space_group_dict, atom_site_dict):
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

    with open(filename, 'w') as fobj:
        fobj.write(str(new_model))


def write_mock_hkl(filename, refln_dict):
    with open(filename, 'w') as fobj:
        for h, k, l in zip(refln_dict['_refln_index_h'], refln_dict['_refln_index_k'], refln_dict['_refln_index_l']):
            fobj.write(f'{int(h): 4d}{int(k): 4d}{int(l): 4d}{0.0: 8.2f}{0.0: 8.2f}\n')