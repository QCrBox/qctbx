from .LCAODensityCalculatorBase import LCAODensityCalculator
from ..util import batched
from ..conversions import add_cart_pos
import platform
import shutil
import pathlib
import subprocess
from typing import Dict, List, Union, Optional

calc_defaults = {
    'filebase': 'orca',
    'output_format': 'mkl'
}

qm_defaults = {
    'method': 'PBE',
    'basis_set': 'def2-SVP',
    'multiplicity': 1,
    'charge': 0,
    'keywords': [],
    'blocks': {}
}

class ORCADensityCalculator(LCAODensityCalculator):
    """
    A specialized calculator for using the ORCA quantum chemistry package that inherits from LCAODensityCalculator. 
    This class provides methods to generate input files, execute ORCA, and process the output.

    Attributes:
        xyz_format (str): The format of the atomic coordinates.
        provides_output (tuple): The output formats supported by the calculator.
        atom_site_required (tuple): The required attributes for the atom_site_dict parameter.
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
            keys such as 'filebase' and 'output_format'.
    """
    xyz_format = 'cartesian'
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
        if abs_orca_path is not None:
            self.abs_orca_path = abs_orca_path
        elif platform.system() == 'Windows':
            self.abs_orca_path = shutil.which('orca.exe')
        elif platform.system() == 'Darwin':
            self.abs_orca_path = shutil.which('orca')
        else:
            #assume linux
            self.abs_orca_path = shutil.which('orca')

    def check_availability(self) -> bool:
        """
        Check if the ORCA executable is available in the system.

        Returns:
            bool: True if the ORCA executable is available, False otherwise.
        """
        if self.abs_orca_path is not None:
            path = pathlib.Path(self.abs_orca_path)
            return path.exists()
        else:
            return False
    

    def calculate_density(
            self,
            atom_site_dict: Dict[str, Union[float, str]], 
            cell_dict,
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
                array under 'positions' for the charge positions and a 
                n sized array with the charges under 'charges'.
                Defaults to an empty dict for no cluster charges.
        """
        # Merge defaults and user-supplied options
        qm_options = qm_defaults.copy()
        
        #blocks and keyword merging needs to account for different cases
        new_lower = [key.lower() for key in self.qm_options['keywords']]
        keep_kw = list(kw for kw in qm_defaults['keywords'] if kw not in new_lower)

        qm_opt_blocks = self.qm_options.get('blocks', {})
        lower_keys = tuple(key.lower() for key in qm_opt_blocks.keys())
        keep_blocks = {key: val for key, val in qm_defaults['blocks'].items()
                        if key.lower() not in lower_keys}

        qm_options.update(self.qm_options)
        qm_options['keywords'] += keep_kw
        qm_options['blocks'].update(keep_blocks)

        calc_options = calc_defaults.copy()
        calc_options.update(self.calc_options)

        if len(cluster_charge_dict.get('charges', [])) > 0:
            cc_file = self._generate_cluster_charge_file(cluster_charge_dict)
            cc_filename = f"{calc_options['filebase']}.pc"
            with open(cc_filename, 'w') as fo:
                fo.write(cc_file)

            qm_options['blocks']['pointcharges'] = f"{cc_filename}"
 
        # Create the input file content
        input_content = self._generate_orca_input(atom_site_dict, cell_dict, qm_options)

        # Write the input file to disk
        input_filename = f"{calc_options['filebase']}.inp"
        with open(input_filename, 'w') as fo:
            fo.write(input_content)

        #Execute ORCA with the generated input file 
        out_filename = f"{calc_options['filebase']}.out"
        with open(out_filename, 'w') as fo:
            subprocess.call(
                [self.abs_orca_path, input_filename],
                stdout=fo,
                stderr=subprocess.STDOUT
            )

        format_standardise = calc_options['output_format'].lower().replace('.', '')
        if  format_standardise == 'mkl':
            subprocess.check_output(['orca_2mkl', calc_options['filebase']])
            return calc_options['filebase'] + '.mkl'
        elif format_standardise == 'wfn':
            subprocess.check_output(['orca_2aim', calc_options['filebase']])
            return calc_options['filebase'] + '.wfn'
        else:
            raise NotImplementedError('output_format from OrcaCalculator is not implemented. Choose either mkl or wfn')
        

    def _generate_cluster_charge_file(
            cluster_charge_dict: Dict[str, List[float]]
        ) -> str:
        """
        Generate the content of the ORCA cluster charge file using the given cluster charge dictionary.

        Args:
            cluster_charge_dict (Dict[str, List[float]]): Dictionary containing cluster charge information.

        Returns:
            str: The content of the cluster charge file.
        """
        position_strings = iter(
            ' '.join(f'{val: 12.8f}' for val in single_position)
            for single_position in cluster_charge_dict['positions']
        )

        charge_block = '\n'.join(
            f'{charge: 9.6f} {pos_string}' for charge, pos_string 
            in zip(cluster_charge_dict['charges'], position_strings)
        )
        
        return f"{len(cluster_charge_dict['charges'])}\n{charge_block}\n"

    def _generate_orca_input(
            self,
            atom_site_dict: Dict[str, Union[float, str]],
            cell_dict: Dict[str, float],
            qm_options: Dict[str, Union[str, int, float, List[str], Dict[str, str]]]
        ) -> str:
        """
        Generate the content of the ORCA input file using the given atom_site_dict and qm_options.

        Args:
            atom_site_dict (Dict[str, Union[float, str]]): Dictionary containing the atomic configuration information.
                Required keys: '_atom_site_type_symbol', '_atom_site_Cartn_x', '_atom_site_Cartn_y', '_atom_site_Cartn_z'
            qm_options (Dict[str, Union[str, int, float, List[str], Dict[str, str]]]): Dictionary containing the quantum mechanics options.

        Returns:
            str: The content of the ORCA input file.
        """
          
        # Set up the ORCA input file header
        header = f"! {qm_options['method']} {qm_options['basis_set']}"

        for entries in batched(qm_options['keywords'], 5):
            header += '\n!' + ' '.join(entries)      

        blocks = ''.join(
            f'\n%{key}\n{entry}\nend\n' if ' ' in entry.strip() 
            else f'\n%{key} {entry}\n'
            for key, entry in qm_options['blocks'].items() 
        )

        charge_mult = f"*xyz {qm_options['charge']} {qm_options['multiplicity']}"
        columns = (
            '_atom_site_type_symbol',
            '_atom_site_Cartn_x',
            '_atom_site_Cartn_y',
            '_atom_site_Cartn_z'
        )
        try:
            entries = [atom_site_dict[key] for key in columns]
        except KeyError:
            new_atom_site_dict, _ = add_cart_pos(atom_site_dict, cell_dict)
            entries = [new_atom_site_dict[key] for key in columns]

        # Generate the coordinates section
        coordinates = [f"{element} {x} {y} {z}" for element, x, y, z in zip(*entries)]
        coordinates_section = '\n'.join(coordinates)

        # Combine sections into a complete input file
        orca_input = f"{header}\n{blocks}\n{charge_mult}\n{coordinates_section}\n*\n"
        return orca_input
    
    def cif_output(self) -> str:
        # TODO: Implement the logic to generate a CIF output from the calculation
        return 'Someone needs to implement this before production'
    
    def citation_strings(self) -> str:
        # TODO: Add a short string with the citation as bib and a sentence what was done
        return 'bib_string', 'sentence string'



        