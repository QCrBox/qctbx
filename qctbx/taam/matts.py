from typing import Any, Dict, List
from ..F0jSourceBase import F0jSource
import os
from copy import deepcopy
import numpy as np
import subprocess
from ..conversions import symm_mat_vec2str, symm_to_matrix_vector, cell_dict2atom_sites_dict
from ..io.minimal_files import write_minimal_cif, write_mock_hkl
from ..io.tsc import TSCFile
from ..custom_typing import Path


bibtex_key = 'MATTS,DiSCaMB,iotbxcif'

bibtex_entry = """
@article{MATTS,
    title = {Extension of the transferable aspherical pseudoatom data bank for the comparison of molecular electrostatic potentials in structure–activity studies},
    volume = {75},
    issn = {2053-2733},
    url = {https://scripts.iucr.org/cgi-bin/paper?S2053273319000482},
    doi = {10.1107/S2053273319000482},
    number = {2},
    urldate = {2023-07-17},
    journal = {Acta Crystallogr A Found Adv},
    author = {Kumar, Prashant and Gruza, Barbara and Bojarowski, Sławomir Antoni and Dominiak, Paulina Maria},
    month = mar,
    year = {2019},
    pages = {398--408},
}

@article{DiSCaMB,
    title = {\textit{{DiSCaMB}} : a software library for aspherical atom model {X}-ray scattering factor calculations with {CPUs} and {GPUs}},
    volume = {51},
    issn = {1600-5767},
    shorttitle = {\textit{{DiSCaMB}}},
    url = {https://scripts.iucr.org/cgi-bin/paper?S1600576717015825},
    doi = {10.1107/S1600576717015825},
    number = {1},
    urldate = {2023-07-17},
    journal = {J Appl Crystallogr},
    author = {Chodkiewicz, Michał L. and Migacz, Szymon and Rudnicki, Witold and Makal, Anna and Kalinowski, Jarosław A. and Moriarty, Nigel W. and Grosse-Kunstleve, Ralf W. and Afonine, Pavel V. and Adams, Paul D. and Dominiak, Paulina Maria},
    month = feb,
    year = {2018},
    pages = {193--199},
}

@article{iotbxcif,
    title = {\textit{iotbx.cif} : a comprehensive {CIF} toolbox},
    volume = {44},
    issn = {0021-8898},
    shorttitle = {\textit{iotbx.cif}},
    url = {https://scripts.iucr.org/cgi-bin/paper?S0021889811041161},
    doi = {10.1107/S0021889811041161},
    number = {6},
    urldate = {2023-07-17},
    journal = {J Appl Crystallogr},
    author = {Gildea, Richard J. and Bourhis, Luc J. and Dolomanov, Oleg V. and Grosse-Kunstleve, Ralf W. and Puschmann, Horst and Adams, Paul D. and Howard, Judith A. K.},
    month = dec,
    year = {2011},
    pages = {1259--1263},
}
""".strip()


class MATTSF0jSource(F0jSource):
    def __init__(
        self, 
        discamb_path: Path, 
        work_folder: Path ='./discamb_files', 
        filebase='discamb'
    ):
        self.discamb_path = os.path.abspath(discamb_path)
        self.work_folder = work_folder
        self.filebase=filebase

        if not os.path.exists(work_folder):
            os.mkdir(work_folder)

    def calc_f0j(
            self,
            atom_site_dict: Dict[str, List[Any]],
            cell_dict: Dict[str, Any],
            space_group_dict: Dict[str, Any], 
            refln_dict: Dict[str, Any]
        ):
        atom_sites_dict = cell_dict2atom_sites_dict(cell_dict)
        cell_dict['_cell_volume'] = np.linalg.det(atom_sites_dict['_atom_sites_Cartn_tran_matrix'])
        
        cleaned_sg_dict = deepcopy(space_group_dict)
        
        cleaned_sg_dict['_space_group_symop_operation_xyz'] = [
            symm_mat_vec2str(*symm_to_matrix_vector(symm_string)) for symm_string in cleaned_sg_dict['_space_group_symop_operation_xyz']
        ]

        write_mock_hkl(os.path.join(self.work_folder, self.filebase + '.hkl'), refln_dict)
        write_minimal_cif(os.path.join(self.work_folder, self.filebase + '.cif'), cell_dict, cleaned_sg_dict, atom_site_dict)
        with open(os.path.join(self.work_folder, f'{self.filebase}_cli.out'), 'w') as fo:
            subprocess.run([self.discamb_path], cwd=self.work_folder, stdout=fo)

        tsc = TSCFile.from_file(os.path.join(self.work_folder, self.filebase + '.tsc'))

        f0j = np.array([
            tsc.data[(h, k, l)] if (h, k, l) in tsc.data.keys() else np.conj(tsc.data[(-h, -k, -l)]) for h, k, l in zip(refln_dict['_refln_index_h'], refln_dict['_refln_index_k'], refln_dict['_refln_index_l'])
        ]).T
        
        return f0j

    def citation_strings(self):
        description_string = f'Aspherical atomic form factors were generated using the MATTS interface of discamb2tsc [{bibtex_key}]'

        return description_string, bibtex_entry