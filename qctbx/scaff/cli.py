import argparse
import json
import textwrap

from ..io.cif import cif2dicts, read_settings_cif, parse_options
from ..io.minimal_files import write_minimal_cif
from ..io.tsc import TSCBFile, TSCFile
from ..conversions import expand_atom_site_table_symm
from . import (name2lcaodensity, name2lcaopartition, name2reggriddensity,
               name2reggridpartition)
from .scaff_f0j import ScaffF0jSource

known_calcs_parts = {
    'lcaowfn': ('nwchem', 'orca', 'pyscf'),
    'lcaopartition': ('nosphera2', 'horton'),
    'reggridwfn': tuple(['gpaw']),
    'reggridpartition': ('gpaw', 'python')
}

common_arguments = {
    'cif_path': {
        'required': True,
        'help': 'Path to a crystallographic information file. Needs to contain a data_<block-name> entry.'
    },
    'scif_path': {
        'required': False,
        'default': None,
        'help': 'Path to a settings crystallographic information file. Needs to contain a settings_<block-name> entry. Default: Same as cif_path.'
    },
    'block_name': {
        'required': True,
        'help': 'Name of the block to select in the data and settings cif file(s).'
    },
    'tsc_path': {
        'required': True,
        'help': 'Path for the tsc(b) file to be created. If file ending is .tscb the file will be in binary tsc format'
    }
}

def use_tsc_parser(args):
    parser = argparse.ArgumentParser(
        prog='qctbx.cli tsc',
        description='Can be used to do tsc calculations using the qctbx from the command line from an scif file containing entries for both the density calculation and a corresponding partitioning.'
    )

    parser.add_argument('--cif_path', **common_arguments['cif_path'])
    parser.add_argument('--scif_path', **common_arguments['scif_path'])
    parser.add_argument('--block_name', **common_arguments['block_name'])
    parser.add_argument('--tsc_path', **common_arguments['tsc_path'])

    pargs = parser.parse_args(args)
    if pargs.scif_path is None:
        scif_path = pargs.cif_path
    else:
        scif_path = pargs.scif_path

    f0jeval = ScaffF0jSource.from_settings_cif(scif_path, pargs.block_name)
    f0jeval.cif2tsc(pargs.cif_path, pargs.block_name, pargs.tsc_path)

def use_density_parser(args):
    parser = argparse.ArgumentParser(
        prog='qctbx.cli density',
        description='Can be used to calculate a density .cube or wavefunction file for further processing.'
    )
    parser.add_argument('--cif_path', **common_arguments['cif_path'])
    parser.add_argument('--scif_path', **common_arguments['scif_path'])
    parser.add_argument('--block_name', **common_arguments['block_name'])

    parser.add_argument(
        '--text_output_path',
        required=True,
        help='output path textfile listing the path to the wfn or cube file'
    )
    pargs = parser.parse_args(args)

    pargs = parser.parse_args(args)
    if pargs.scif_path is None:
        scif_path = pargs.cif_path
    else:
        scif_path = pargs.scif_path

    settings_cif = read_settings_cif(scif_path, pargs.block_name)
    if '_qctbx_reggridwfn_software' in settings_cif:
        calc_cls = name2reggriddensity(settings_cif['_qctbx_reggridwfn_software'])
    elif '_qctbx_lcaowfn_software' in settings_cif:
        calc_cls = name2lcaodensity(settings_cif['_qctbx_lcaowfn_software'])
    else:
        raise KeyError('Need either _qctbx_lcaowfn_software or _qctbx_reggridwfn_software in scif file.')

    calc_obj = calc_cls.from_settings_cif(scif_path, pargs.block_name)

    atom_site_dict, cell_dict, _, _ = cif2dicts(pargs.cif_path, pargs.block_name, complete_dmin=False)

    density_path = calc_obj.calculate_density(atom_site_dict, cell_dict)

    with open(pargs.text_output_path, 'w', encoding='UTF-8') as fobj:
        fobj.write(density_path)


def use_partition_parser(args):
    parser = argparse.ArgumentParser(
        prog='qctbx partition',
        description='Can be used to partition a density .cube or wavefunction file with a matching (reggrid/lcao) partitioner and calculate atomic form factors.'
    )

    parser.add_argument('--cif_path', **common_arguments['cif_path'])
    parser.add_argument('--scif_path', **common_arguments['scif_path'])
    parser.add_argument('--block_name', **common_arguments['block_name'])
    parser.add_argument(
        '--input_wfn_path',
        required=True,
        help='path to input .cube or wavefunction file. Needs to be compatible with partitioner.'
    )
    parser.add_argument(
        '--atom_labels',
        required=True,
        nargs='+',
        help='List of atom labels that should be evaluated and output in the partitioning (All atoms from cif are used for promolecule).'
    )
    parser.add_argument('--tsc_path', **common_arguments['tsc_path'])

    parser.add_argument(
        '--charge_json',
        required=False,
        help='Optional path for the output of the atomic charges as json'
    )
    pargs = parser.parse_args(args)

    pargs = parser.parse_args(args)
    if pargs.scif_path is None:
        scif_path = pargs.cif_path
    else:
        scif_path = pargs.scif_path

    settings_cif = read_settings_cif(scif_path, pargs.block_name)
    if '_qctbx_lcaopartition_software' in settings_cif:
        part_cls = name2lcaopartition(settings_cif['_qctbx_lcaopartition_software'])
    elif '_qctbx_reggridpartition_software' in settings_cif:
        part_cls = name2reggridpartition(settings_cif['_qctbx_reggridpartition_software'])
    else:
        raise KeyError('Need either _qctbx_lcaopartition_software or _qctbx_reggridpartition_software in scif file')

    part_obj = part_cls.from_settings_cif(scif_path, pargs.block_name)

    output_dicts = cif2dicts(pargs.cif_path, pargs.block_name, complete_dmin=True)
    atom_site_dict, cell_dict, space_group_dict, refln_dict = output_dicts

    f0j, charges = part_obj.calc_f0j(
        pargs.atom_labels,
        atom_site_dict,
        cell_dict,
        space_group_dict,
        refln_dict,
        pargs.input_wfn_path
    )
    if pargs.tsc_path.lower().endswith('.tscb'):
        new_tsc = TSCBFile()
    else:
        new_tsc = TSCFile()
    new_tsc.scatterers = list(pargs.atom_labels)
    hkl_zip = zip(
        refln_dict['_refln_index_h'],
        refln_dict['_refln_index_k'],
        refln_dict['_refln_index_l'],
        f0j.T
    )
    new_data = {(h, k, l): form_factors for h, k, l, form_factors in hkl_zip}
    new_tsc.header['TITLE'] = 'qctbx'
    new_tsc.data = new_data
    new_tsc.to_file(pargs.tsc_path)

    if pargs.charge_json is not None:
        with open(pargs.charge_json, 'w', encoding='UTF-8') as fobj:
            json.dump({label: value for label, value in zip(pargs.atom_labels, charges)}, fobj)


def use_check_available_parser(args):
    parser = argparse.ArgumentParser(
        prog='qctbx available',
        description='Can be used to check whether packages are available'
    )

    # parser.add_argument(**common_arguments['scif_path'])
    parser.add_argument(
        '--all',
        action='store_true',
        help='check and report on the availability of all possible density and partitioning methods'
    )
    parser.add_argument(
        '--lcaowfn',
        nargs='+',
        help='names of programs used for LCAO wavefunction calculation'
    )

    parser.add_argument(
        '--reggridwfn',
        nargs='+',
        help='names of programs used for regular grid density calculation'
    )

    parser.add_argument(
        '--lcaopartition',
        nargs='+',
        help='names of programs used for LCAO wavefunction partitioning'
    )

    parser.add_argument(
        '--reggridpartition',
        nargs='+',
        help='names of programs used for regular grid density partitioning'
    )

    parser.add_argument(
        '--scif_file',
        help='Name of scif file to check for availability'
    )

    parser.add_argument(
        '--output_json',
        required=True,
        help='name of output json file'
    )
    parser.add_argument(
         '--scif_path',
        required= False,
        help= 'Path to a settings crystallographic information file. Needs to contain a settings_<block-name> entry. Default: Same as cif_path.'
    )
    parser.add_argument(
        '--block_name',
        required=False,
        help='Name of the block to select in the settings cif file, required if running with scif option.'
    )

    pargs = parser.parse_args(args)

    collect = []

    if pargs.lcaowfn is not None:
        for entry in pargs.lcaowfn:
            collect.append(('lcaowfn', entry))

    if pargs.reggridwfn is not None:
        for entry in pargs.reggridwfn:
            collect.append(('reggridwfn', entry))

    if pargs.lcaopartition is not None:
        for entry in pargs.lcaopartition:
            collect.append(('lcaopartition', entry))

    if pargs.reggridpartition is not None:
        for entry in pargs.reggridpartition:
            collect.append(('reggridpartition', entry))

    if pargs.all:
        for calc_type, entries in known_calcs_parts.items():
            for entry in entries:
                collect.append((calc_type, entry))

    calc_type_dict = {
        'lcaowfn': name2lcaodensity,
        'reggridwfn': name2reggriddensity,
        'lcaopartition': name2lcaopartition,
        'reggridpartition': name2reggridpartition
    }

    check_available_dict = {}

    for calc_type, software in collect:
        try:
            calc_cls = calc_type_dict[calc_type](software)
        except NotImplementedError:
            check_available_dict[f'{calc_type};{software}'] = False
            continue
        check_available_dict[f'{calc_type};{software}'.lower()] = calc_cls().check_availability()

    if pargs.scif_path is not None:
        assert pargs.block_name is not None, 'Working with an scif, a block name is required.'
        settings_cif = read_settings_cif(pargs.scif_path, pargs.block_name)
        check_entries = iter((key.split('_')[-2], entry.lower()) for key, entry in settings_cif.items() if key.lower().endswith('_software'))
        for calc_type, software in check_entries:
            calc_cls = calc_type_dict[calc_type](software)
            calc_obj = calc_cls.from_settings_cif(pargs.scif_path, pargs.block_name)
            check_available_dict[f'{calc_type};{software}'] = calc_obj.check_availability()

    with open(pargs.output_json, 'w', encoding='UTF-8') as fobj:
        json.dump(check_available_dict, fobj)


def use_citations_parser(args):
    parser = argparse.ArgumentParser(
        prog='qctbx.cli citation',
        description='Can be used to generate the description and bibtex strings for the given method.'
    )

    parser.add_argument('--scif_path', **common_arguments['scif_path'])
    parser.add_argument('--block_name', **common_arguments['block_name'])
    parser.add_argument(
        '--output_json',
        required=True,
        help='name of output json file'
    )
    pargs = parser.parse_args(args)

    settings_cif = read_settings_cif(pargs.scif_path, pargs.block_name)
    if '_qctbx_reggridwfn_software' in settings_cif:
        calc_cls = name2reggriddensity(settings_cif['_qctbx_reggridwfn_software'])
    elif '_qctbx_lcaowfn_software' in settings_cif:
        calc_cls = name2lcaodensity(settings_cif['_qctbx_lcaowfn_software'])
    else:
        calc_cls = None

    if '_qctbx_lcaopartition_software' in settings_cif:
        part_cls = name2lcaopartition(settings_cif['_qctbx_lcaopartition_software'])
    elif '_qctbx_reggridpartition_software' in settings_cif:
        part_cls = name2reggridpartition(settings_cif['_qctbx_reggridpartition_software'])
    else:
        part_cls = None

    descr = ''
    bibtex = []

    if calc_cls is not None:
        calc_obj = calc_cls.from_settings_cif(pargs.scif_path, pargs.block_name)
        calc_descr, calc_bibtex = calc_obj.citation_strings()
        descr += calc_descr
        bibtex.append(calc_bibtex)

    if part_cls is not None:
        part_obj = part_cls.from_settings_cif(pargs.scif_path, pargs.block_name)
        part_descr, part_bibtex = part_obj.citation_strings()
        descr += part_descr
        bibtex.append(part_bibtex)

    with open(pargs.output_json, 'w', encoding='UTF-8') as fobj:
        json.dump({'description': descr, 'bibtex': '\n\n'.join(bibtex)}, fobj)

def use_symm_expand_parser(args):
    parser = argparse.ArgumentParser(
        prog='qctbx.cli symm_expand',
        description='Can be used to generate a cif where the crystal or other symmetry elements are used to complete a structure or unit cell. Atoms under 0.1 Ang from their symmetry equivalent atom are omitted'
    )
    parser.add_argument('--cif_path', **common_arguments['cif_path'])
    parser.add_argument('--block_name', **common_arguments['block_name'])
    parser.add_argument(
        '--to_p1',
        action='store_true',
        required=False,
        help='Expand all non-unity symmetry elements in cif file.'
    )

    parser.add_argument(
        '--symm_string',
        required=False,
        help='Possibility to provide a string in the form "-x,y,-z:C1 C2;1-x,z,y+1/2: H2 H3". Not read if to_p1 is used.'
    )

    parser.add_argument(
        '--out_cif_path',
        required=True,
        help='Path to write the expanded path to.'
    )

    parser.add_argument(
        '--txt_labels',
        required=False,
        help='Optional path for a txt file where the atom labels of asymmetric unit are output into'
    )

    parser.add_argument('--scif_path', **common_arguments['scif_path'])

    pargs = parser.parse_args(args)

    if pargs.scif_path is None:
        scif_path = pargs.cif_path
    else:
        scif_path = pargs.scif_path

    output_dicts = cif2dicts(pargs.cif_path, pargs.block_name, complete_dmin=False)
    atom_site_dict, cell_dict, space_group_dict, refln_dict = output_dicts

    expand_positions = {}
    if pargs.to_p1:
        for symm_op in space_group_dict['_space_group_symop_operation_xyz']:
            expand_positions[symm_op] = 'all'
    elif pargs.symm_string is not None:
        for group in pargs.symm_string.split(';'):
            symm_op, atom_str = group.strip().split(':')
            expand_positions[symm_op] = atom_str.split()
    else:
        settings_cif = read_settings_cif(scif_path, pargs.block_name)
        if '_qctbx_expanded_fragment' in settings_cif:
            expand_positions = parse_options(settings_cif['_qctbx_expanded_fragment'])
        else:
            raise ValueError('No values for expansion of positions in the cif_file were provided')

    new_atom_site_dict = expand_atom_site_table_symm(
        atom_site_dict, expand_positions, cell_dict
    )

    write_minimal_cif(
        pargs.out_cif_path, cell_dict, space_group_dict, new_atom_site_dict, refln_dict, pargs.block_name
    )

    if pargs.txt_labels is not None:
        labels = atom_site_dict['_atom_site_label']
        with open(pargs.txt_labels, 'w', encoding='UTF-8') as fobj:
            fobj.write(' '.join(labels))

def use_unselected_parser(args):
    parser = argparse.ArgumentParser(
        prog='qctbx',
        description=textwrap.dedent("""
            Can be used to run calculations and checks within the qctbx framework.
            Select functionality via the following mode keywords:
                tsc: calculate atomic form factors as tsc (density calculation + partitioning)\n
                density: calculate a density .cube or wavefunction file for further processing.\n
                partition: partition a density .cube or wavefunction file with a matching (reggrid/lcao) partitioner and calculate atomic form factors.\n
                available: check whether packages are available.\n
                citation: generate the description and bibtex strings for the given method.\n
                symm_expand: generate a cif where the crystal or other symmetry elements are used to complete a structure or unit cell.
        """)
    )

    parser.add_argument(
        'mode'
    )
    parser.parse_args(args)

modes = {
    'tsc': use_tsc_parser,
    'density': use_density_parser,
    'partition': use_partition_parser,
    'available': use_check_available_parser,
    'citation': use_citations_parser,
    'symm_expand': use_symm_expand_parser
}


