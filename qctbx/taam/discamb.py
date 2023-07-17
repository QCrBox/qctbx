from typing import Any, Dict, List
from ..F0jSourceBase import F0jSource
import os
from copy import deepcopy
import numpy as np
import subprocess
from ..conversions import symm_mat_vec2str, symm_to_matrix_vector, cell_dict2atom_sites_dict
from ..io.minimal_files import write_minimal_cif, write_mock_hkl
from ..io.tsc import TSCFile

class DiscambF0jSource(F0jSource):
    def __init__(self, discamb_path, work_folder='./discamb_files', filebase='discamb'):
        self.discamb_path = discamb_path
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

        write_mock_hkl(os.path.join(self.work_folder, self.filebase + '.hkl'))
        write_minimal_cif(os.path.join(self.work_folder, self.filebase + '.cif'), cell_dict, cleaned_sg_dict, atom_site_dict)
        with open(self.work_folder, f'{self.filebase}_cli.out', 'w') as fo:
            subprocess.run([self.discamb_path], cwd=self.work_folder, stdout=fo)

        tsc = TSCFile.from_file(os.path.join(self.work_folder, self.filebase + '.tsc'))

        f0j = np.array([
            tsc.data[(h, k, l)] if (h, k, l) in tsc.data.keys() else np.conj(tsc.data[(-h, -k, -l)]) for h, k, l in zip(refln_dict['_refln_index_h'], refln_dict['_refln_index_k'], refln_dict['_refln_index_l'])
        ]).T
        
        return f0j
