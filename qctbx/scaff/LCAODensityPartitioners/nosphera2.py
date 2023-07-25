import os
import re
import subprocess
from copy import deepcopy
from typing import Any, Dict, List

import numpy as np

from ...conversions import (cell_dict2atom_sites_dict, symm_mat_vec2str,
                            symm_to_matrix_vector)
from ...custom_typing import Path
from ...io.minimal_files import write_minimal_cif, write_mock_hkl
from ...io.tsc import TSCFile
from ..citations import get_partitioning_citation
from .base import LCAODensityPartitioner

defaults = {
    'method': 'hirshfeld',
    'grid_accuracy': 'medium',
    'cpu_count': 4,
    'specific_options': {},
    'calc_options':{
        'nosphera2_path': './NoSpherA2',
        'calc_folder': '.'
    }
}

nosphera2_bibtex_key = 'NoSpherA2'

nosphera2_bibtex_entry = """
@article{NoSpherA2,
    author ="Kleemiss, Florian and Dolomanov, Oleg V. and Bodensteiner, Michael and Peyerimhoff, Norbert and Midgley, Laura and Bourhis, Luc J. and Genoni, Alessandro and Malaspina, Lorraine A. and Jayatilaka, Dylan and Spencer, John L. and White, Fraser and Grundkötter-Stock, Bernhard and Steinhauer, Simon and Lentz, Dieter and Puschmann, Horst and Grabowsky, Simon",
    title  ="Accurate crystal structures and chemical properties from NoSpherA2",
    journal  ="Chem. Sci.",
    year  ="2021",
    volume  ="12",
    issue  ="5",
    pages  ="1675-1692",
    publisher  ="The Royal Society of Chemistry",
    doi  ="10.1039/D0SC05526C",
    url  ="http://dx.doi.org/10.1039/D0SC05526C"
}
""".strip()

grid_accuracy_names = ('coarse', 'medium', 'fine', 'veryfine', 'ultrafine', 'insane')

class NoSpherA2Partitioner(LCAODensityPartitioner):

    accepts_input = ('wfn', 'wfx')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.update_from_dict(defaults, update_if_present=False)

    def check_availability(self) -> bool:
        return os.path.exists(self.calc_options['nosphera2_path'])

    def run_nospherA2(
        self,
        atom_labels: List[int],
        atom_site_dict: Dict[str, List[Any]],
        cell_dict: Dict[str, Any],
        space_group_dict: Dict[str, Any],
        refln_dict: Dict[str, Any],
        density_path: Path
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
        write_minimal_cif(os.path.join(self.calc_options['calc_folder'], 'npa2.cif'), cell_dict, cleaned_sg_dict, atom_site_dict)
        write_minimal_cif(os.path.join(self.calc_options['calc_folder'], 'npa2_asym.cif'), cell_dict, cleaned_sg_dict, select_atom_site_dict)
        write_mock_hkl(os.path.join(self.calc_options['calc_folder'], 'mock.hkl'), refln_dict)

        pass_options = deepcopy(self.specific_options)
        pass_options.update(self.calc_options)
        pass_options['nosphera2_accuracy'] = grid_accuracy_names.index(self.grid_accuracy) + 1
        pass_options['cpu_count'] = self.cpu_count
        pass_options['density_path'] = os.path.abspath(density_path)

        subprocess.check_call('{nosphera2_path} -hkl mock.hkl -wfn {density_path} -cif npa2.cif -asym_cif npa2_asym.cif -acc {nosphera2_accuracy} -cores {cpu_count}'.format(**pass_options), shell=True, stdout=subprocess.DEVNULL, cwd=self.calc_options['calc_folder'])


    def calc_f0j(
        self,
        atom_labels: List[int],
        atom_site_dict: Dict[str, List[Any]],
        cell_dict: Dict[str, Any],
        space_group_dict: Dict[str, Any],
        refln_dict: Dict[str, Any],
        density_path: Path
    ):
        self.run_nospherA2(atom_labels, atom_site_dict, cell_dict, space_group_dict, refln_dict, density_path)

        tsc = TSCFile.from_file(os.path.join(self.calc_options['calc_folder'],'experimental.tsc'))

        f0j = np.array([
            tsc.data[(h, k, l)] if (h, k, l) in tsc.data.keys() else np.conj(tsc.data[(-h, -k, -l)]) for h, k, l in zip(refln_dict['_refln_index_h'], refln_dict['_refln_index_k'], refln_dict['_refln_index_l'])
        ]).T

        with open(os.path.join(self.calc_options['calc_folder'], 'NoSpherA2.log'), 'r') as fobj:
            content = fobj.read()

        charge_table_match = re.search(r'Atom\s+Becke\s+Spherical\s+Hirshfeld(.*)\nTotal number of electrons', content, flags=re.DOTALL)

        assert charge_table_match is not None, 'Could not find charge table in NoSpherA2.log, probably unexpected format'

        charge_table = charge_table_match.group(1)

        charge_dict = {}
        for line in charge_table.split('\n')[1:]:
            name, _, _, atom_charge = line.strip().split()
            charge_dict[name] = float(atom_charge)

        return f0j, np.array([charge_dict[label] for label in atom_labels])

    def citation_strings(self) -> str:
        method_bibtex_key, method_bibtex_entry = get_partitioning_citation('hirshfeld')
        description_string = (
            f'The moleculear electron density was partitioning using Hirshfeld partitioning [{method_bibtex_key}]'
            + f' with the NoSpherA2 [{nosphera2_bibtex_key}] program.'
        )
        bibtex_string = '\n\n\n'.join((method_bibtex_entry, nosphera2_bibtex_entry))
        return description_string, bibtex_string

    def cif_output(self) -> str:
        return 'To be implemented'