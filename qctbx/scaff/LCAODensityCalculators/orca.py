import os
import subprocess
from typing import Dict, List, Optional, Union

import numpy as np

from ...conversions import add_cart_pos
from ..QCCalculator.orca import ORCACalculator
from .base import LCAODensityCalculator

defaults = {
    'method': 'PBE',
    'basisset': 'def2-SVP',
    'charge': 0,
    'multiplicity': 1,
    'specific_options': {
        'keywords': [],
        'blocks': {}
    },
    'calc_options': {
        'label': 'orca',
        'work_directory': '.',
        'output_format': 'mkl',
        'ram_mb': 2000,
        'cpu_count': 1
    }
}

class ORCADensityCalculator(LCAODensityCalculator):
    """
    A specialized calculator for using the ORCA quantum chemistry package that inherits from LCAODensityCalculator.
    This class provides methods to generate input files, execute ORCA, and process the output.

    Attributes:
        provides_output (tuple): The output formats supported by the calculator.
        method: str: Either functional or orther quantum chemical method
            for the density calculation. Default: 'PBE'
        basisset: str: Basis set for the wavefunction description.
            Default: 'def2-SVP'
        multiplicity: int: spin multiplicity of the system. Default: 1
        charge: charge of the system. Default: 0
        special_options (Dict[str, Any]): Additional quantum mechanics options for the ORCA calculation.
            Keys:
                'keywords': List of additional keywords that will be added to
                    after the '!' in the ORCA input file.
                'blocks': everything that is included into the ORCA input file
                    using a % sign. If a newline is present in the included
                    string an entry will be concluded with 'end' in the input
                    file otherwise a single line entry without end will be
                    produced. If cluster charges are included, an existing
                    'pointcharges' entry will be overwritten.
        calc_options (Dict[str, Any]): Calculation options specific to the ORCA calculation. The dictionary should contain
            keys such as 'label', 'work_directory' and 'output_format'.
    """
    provides_output = ('mkl', 'wfn')
    software = 'orca'

    def __init__(
        self,
        *args,
        abs_orca_path: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the ORCADensityCalculator instance.

        Args:
            *args: Variable length argument list.
            abs_orca_path (Optional[str]): The absolute path of the ORCA
                executable. Defaults to None, in this case the absolute path
                is determined from an orca executable in PATH.
            **kwargs: Arbitrary keyword arguments.
        """

        super().__init__(*args, **kwargs)
        self._calculator = ORCACalculator(
            abs_orca_path=abs_orca_path
        )

        self.update_from_dict(defaults, update_if_present=False)

    def check_availability(self) -> bool:
        """
        Check the availability of the ORCA calculator.

        Returns:
            bool: True if ORCA calculator is available, False otherwise.
        """
        return self._calculator.check_availability()

    def calculate_density(
            self,
            atom_site_dict: Dict[str, Union[float, str]],
            cell_dict: Dict[str, float],
            cluster_charge_dict: Dict[str, List[float]]=None
        ):
        """
        Calculate the electronic density for a given atomic configuration using ORCA.

        Args:
            atom_site_dict (Dict[str, Union[float, str]]): Dictionary containing
                the atomic configuration information.
                Required keys: '_atom_site_type_symbol', '_atom_site_Cartn_x',
                '_atom_site_Cartn_y', '_atom_site_Cartn_z'
            cluster_charge_dict (Dict[str, List[float]], optional): Dictionary
                containing cluster charge information. provide a n, 3 numpy
                array under 'positions_cart' for the charge positions and a
                n sized array with the charges under 'charges'.
                Defaults to an empty dict for no cluster charges.
        """
        if cluster_charge_dict is None:
            cluster_charge_dict = {}
        self.update_from_dict(defaults, update_if_present=False)

        try:
            positions_cart = np.array([atom_site_dict[f'_atom_site_Cartn_{coord}'] for coord in ('x', 'y', 'z')]).T
        except KeyError:
            new_atom_site_dict, _ = add_cart_pos(atom_site_dict, cell_dict)
            positions_cart = np.array([new_atom_site_dict[f'_atom_site_Cartn_{coord}'] for coord in ('x', 'y', 'z')]).T

        keywords = [self.method]
        blocks = {}

        if '\n' in self.basisset:
            blocks['basis'] = self.basisset
        else:
            keywords.append(self.basisset)

        self._calculator.set_atoms(
            list(atom_site_dict['_atom_site_type_symbol']),
            positions_cart
        )

        blocks['maxcore'] = str(self.calc_options['ram_mb'] // self.calc_options['cpu_count'])
        blocks['pal'] = f"nprocs {self.calc_options['cpu_count']}"
        blocks.update(self.specific_options['blocks'])

        keywords = list(set(keywords + self.specific_options['keywords']))

        self._calculator.charge = self.charge
        self._calculator.multiplicity = self.multiplicity
        self._calculator.directory = self.calc_options['work_directory']
        self._calculator.label = self.calc_options['label']
        self._calculator.cluster_charge_dict = cluster_charge_dict
        self._calculator.blocks = blocks
        self._calculator.keywords = keywords

        self._calculator.run_calculation()

        format_standardise = self.calc_options['output_format'].lower().replace('.', '')
        if  format_standardise == 'mkl':
            subprocess.check_output(['orca_2mkl', self.calc_options['label']], cwd=self.calc_options['work_directory'])
            return os.path.join(self.calc_options['work_directory'], self.calc_options['label'] + '.mkl')
        elif format_standardise == 'wfn':
            subprocess.check_output(['orca_2aim', self.calc_options['label']], cwd=self.calc_options['work_directory'])
            return os.path.join(self.calc_options['work_directory'], self.calc_options['label'] + '.wfn')
        else:
            raise NotImplementedError('output_format from OrcaCalculator is not implemented. Choose either mkl or wfn')

    def citation_strings(self):
        self.update_from_dict(defaults, update_if_present=False)

        software_bibtex_key, sofware_bibtex_entry = self._calculator.bibtex_strings()
        software_name = 'ORCA' #TODO determine and add version
        return self.generate_description(software_name, software_bibtex_key, sofware_bibtex_entry)







