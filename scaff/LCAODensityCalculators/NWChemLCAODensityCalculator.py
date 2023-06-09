import ase
from .LCAODensityCalculatorBase import LCAODensityCalculator
from ..util import dict_merge
from ..conversions import add_cart_pos
from ase.calculators.nwchem import NWChem
from typing import Dict, List, Union, Any
import os
import pathlib
import numpy as np
import subprocess

calc_defaults = {
    'label': 'nwchem',
    'work_directory': '.',
    'output_format': 'mkl'
}

qm_defaults = {
    'method': 'PBE',
    'basis_set': 'def2-SVP',
    'multiplicity': 1,
    'charge': 0,
    'n_core': 1,
    'ram': 2000,
    'ase_options': {}
}


class NWChemLCAODensityCalculator(LCAODensityCalculator):
    xyz_format = 'cartesian'
    provides_output = ('wfn')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def check_availability(self) -> bool:
        """
        Check if the nwchem executable is available in the system.

        Returns:
            bool: True if the nwchem executable is available, False otherwise.
        """
        path = pathlib.Path('nwchem')
        return path.exists()

    def calculate_density(
            self,
            atom_site_dict: Dict[str, Union[float, str]], 
            cell_dict: Dict[str, float],
            cluster_charge_dict: Dict[str, List[float]] = {}
        ):
        """
        Calculate the electronic density for a given atomic configuration using NWChem.

        Args:
            atom_site_dict (Dict[str, Union[float, str]]): Dictionary containing
                the atomic configuration information.
                Required keys: '_atom_site_type_symbol', '_atom_site_Cartn_x', 
                '_atom_site_Cartn_y', '_atom_site_Cartn_z'
            cluster_charge_dict (Dict[str, List[float]], optional): Dictionary 
                containing cluster charge information. provide a n, 3 numpy 
                array under 'positions' for the charge positions and a 
                n sized array with the charges under 'charges'.
                Defaults to an empty dict for no cluster charges.
        """
        assert len(cluster_charge_dict) == 0, 'Cluster charges are currently not supported'
        symbols = list(atom_site_dict['_atom_site_type_symbol'])
        try:
            positions = np.array([atom_site_dict[f'_atom_site_Cartn_{coord}'] for coord in ('x', 'y', 'z')]).T
        except KeyError:
            new_atom_site_dict, _ = add_cart_pos(atom_site_dict, cell_dict)
            positions = np.array([new_atom_site_dict[f'_atom_site_Cartn_{coord}'] for coord in ('x', 'y', 'z')]).T

        used_qm_options = dict_merge(qm_defaults, self.qm_options)
        used_calc_options = dict_merge(calc_defaults, self.calc_options)
        
        ase_options = used_qm_options['ase_options']
        ase_options['method'] = used_qm_options['method']
        ase_options['basis_set'] = used_qm_options['basis_set']
        ase_options['label'] = os.path.join(used_calc_options['work_directory'], used_calc_options['label'])


        nwchem = NWChem(

        )

        
        if self.qm_options['postproc_command'] is not None:
            subprocess.check_output(self.qm_options['postproc_command'], shell=True)

    def cif_output(self):
        return 'Implement me'

    def citation_strings(self) -> str:
        # TODO: Add a short string with the citation as bib and a sentence what was done
        return 'bib_string', 'sentence string'

    