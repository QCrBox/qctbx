from .LCAODensityCalculatorBase import LCAODensityCalculator
from ..util import batched
from ..conversions import add_cart_pos
from ..QCCalculator.ORCACalculator import ORCACalculator
from ..util import dict_merge
import subprocess
import numpy as np
import os

from typing import Dict, List, Union, Optional

calc_defaults = {
    'label': 'orca',
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
    'keywords': [],
    'blocks': {}
}

class ORCADensityCalculator(LCAODensityCalculator):
    """
    A specialized calculator for using the ORCA quantum chemistry package that inherits from LCAODensityCalculator. 
    This class provides methods to generate input files, execute ORCA, and process the output.

    Attributes:
        provides_output (tuple): The output formats supported by the calculator.
        qm_options (Dict[str, Any]): Quantum mechanics options for the ORCA calculation.
            Keys:
                'method': Either functional or orther quantum chemical method
                    for the density calculation. Default: 'PBE'
                'basis_set': Basis set for the wavefunction description. 
                    Default: 'def2-SVP'
                'multiplicity': spin multiplicity of the system. Default: 1
                'charge': charge of the system. Default: 0
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
            cluster_charge_dict: Dict[str, List[float]] = {}
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
        qm_options = dict_merge(qm_defaults, self.qm_options, case_sensitive=False)

        calc_options = dict_merge(calc_defaults, self.calc_options, case_sensitive=True)

        try:
            positions_cart = np.array([atom_site_dict[f'_atom_site_Cartn_{coord}'] for coord in ('x', 'y', 'z')]).T
        except KeyError:
            new_atom_site_dict, _ = add_cart_pos(atom_site_dict, cell_dict)
            positions_cart = np.array([new_atom_site_dict[f'_atom_site_Cartn_{coord}'] for coord in ('x', 'y', 'z')]).T

        keywords = [qm_options['method']]
        blocks = {}

        if '\n' in qm_options['basis_set']:
            blocks['basis'] = qm_options['basis_set']
        else:
            keywords.append(qm_options['basis_set'])

        self._calculator.set_atoms(
            list(atom_site_dict['_atom_site_type_symbol']),
            positions_cart
        )

        blocks['maxcore'] = str(qm_options['ram'] // qm_options['n_core'])
        blocks['pal'] = f"nprocs {qm_options['n_core']}"
        blocks.update(qm_options['blocks'])

        keywords = list(set(keywords + qm_options['keywords']))

        self._calculator.charge = qm_options['charge']
        self._calculator.multiplicity = qm_options['multiplicity']
        self._calculator.directory = calc_options['work_directory']
        self._calculator.label = calc_options['label']
        self._calculator.cluster_charge_dict = cluster_charge_dict
        self._calculator.blocks = blocks
        self._calculator.keywords = keywords

        self._calculator.run_calculation()

        format_standardise = calc_options['output_format'].lower().replace('.', '')
        if  format_standardise == 'mkl':
            subprocess.check_output(['orca_2mkl', calc_options['label']], cwd=calc_options['work_directory'])
            return os.path.join(calc_options['work_directory'], calc_options['label'] + '.mkl')
        elif format_standardise == 'wfn':
            subprocess.check_output(['orca_2aim', calc_options['label']], cwd=calc_options['work_directory'])
            return os.path.join(calc_options['work_directory'], calc_options['label'] + '.wfn')
        else:
            raise NotImplementedError('output_format from OrcaCalculator is not implemented. Choose either mkl or wfn')

    def cif_output(self) -> str:
        # TODO: Implement the logic to generate a CIF output from the calculation
        return 'Someone needs to implement this before production'
    

    def citation_strings(self):
        return self._calculator.citation_strings()




        