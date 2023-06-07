import ase
from .LCAODensityCalculatorBase import LCAODensityCalculator
from ..conversions import add_cart_pos
from typing import Dict, List, Union, Any
import pathlib
import numpy as np
import subprocess


class AseLCAODensityCalculator(LCAODensityCalculator):
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

        atoms = ase.Atoms(symbols=symbols, positions=positions)
        atoms.set_calculator(self.qm_options['ase_calc'])
        atoms.get_potential_energy()

        if self.qm_options['postproc_command'] is not None:
            subprocess.check_output(self.qm_options['postproc_command'], shell=True)

    def cif_output(self):
        return 'Implement me'

    def citation_strings(self) -> str:
        # TODO: Add a short string with the citation as bib and a sentence what was done
        return 'bib_string', 'sentence string'

    