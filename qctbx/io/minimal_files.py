from iotbx import cif

def write_minimal_cif(
    path,
    cell_dict=None,
    space_group_dict=None,
    atom_site_dict=None,
    refln_dict=None,
    block_name='qctbx_minimal'
):
    new_block = cif.model.block()

    if cell_dict is not None:
        for key, value in cell_dict.items():
            new_block[key] = value

    if space_group_dict is not None:
        new_loop = cif.model.loop()
        for key, value in space_group_dict.items():
            new_loop.add_column(key, list(value))

        new_block.add_loop(new_loop)

    if atom_site_dict is not None:
        new_loop = cif.model.loop()
        for key, value in atom_site_dict.items():
            new_loop.add_column(key, list(value))

        new_block.add_loop(new_loop)

    if refln_dict is not None:
        new_loop = cif.model.loop()
        for key, value in refln_dict.items():
            new_loop.add_column(key, list(value))

        new_block.add_loop(new_loop)

    new_model = cif.model.cif()
    new_model[block_name] = new_block

    with open(path, 'w', encoding='ASCII') as fobj:
        fobj.write(str(new_model))

def write_mock_hkl(path, refln_dict):
    with open(path, 'w', encoding='ASCII') as fobj:
        for h, k, l in zip(refln_dict['_refln_index_h'], refln_dict['_refln_index_k'], refln_dict['_refln_index_l']):
            fobj.write(f'{int(h): 4d}{int(k): 4d}{int(l): 4d}{0.0: 8.2f}{0.0: 8.2f}\n')