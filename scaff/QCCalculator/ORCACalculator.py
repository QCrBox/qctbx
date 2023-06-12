import platform
import pathlib
import shutil
import subprocess
import os
from .BaseQCCalculators import LCAOQCCalculator
from typing import Dict, List, Union, Optional
import numpy as np
from ..util import batched


class ORCACalculator(LCAOQCCalculator):
    cluster_charge_dict = {}
    keywords = []
    blocks = {}

    def __init__(
        self,
        *args, 
        abs_orca_path: Optional[str] = None,
        keywords = [],
        blocks = {},
        label = 'orca',
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

        self.keywords = keywords
        self.blocks = blocks
        self.label = label

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
        
    def run_calculation(self):
        if len(self.cluster_charge_dict.get('charges', [])) > 0:
            cc_file = self._generate_cluster_charge_file(self.cluster_charge_dict)
            cc_filename = f"{self.label}.pc"
            with open(cc_filename, 'w') as fo:
                fo.write(cc_file)

            self.blocks['pointcharges'] = f"{cc_filename}"
 
        # Create the input file content
        input_content = self._generate_orca_input()

        # Write the input file to disk
        input_filename = f"{self.label}.inp"
        with open(os.path.join(self.directory, input_filename), 'w') as fo:
            fo.write(input_content)

        #Execute ORCA with the generated input file 
        out_filename = os.path.join(self.directory, f"{self.label}.out")
        with open(out_filename, 'w') as fo:
            subprocess.call(
                [self.abs_orca_path, input_filename],
                stdout=fo,
                stderr=subprocess.STDOUT,
                cwd=self.directory
            )

    def _generate_cluster_charge_file(
            self
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
            for single_position in self.cluster_charge_dict['positions_cart']
        )

        charge_block = '\n'.join(
            f'{charge: 9.6f} {pos_string}' for charge, pos_string 
            in zip(self.cluster_charge_dict['charges'], position_strings)
        )
        
        return f"{len(self.cluster_charge_dict['charges'])}\n{charge_block}\n"

    def _generate_orca_input(
            self
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
        header = ''
        # Set up the ORCA input file header
        for entries in batched(self.keywords, 5):
            header += '\n!' + ' '.join(entries)      

        blocks = ''.join(
            f'\n%{key}\n{entry}\nend\n' if ' ' in entry.strip() 
            else f'\n%{key} {entry}\n'
            for key, entry in self.blocks.items() 
        )

        charge_mult = f"*xyz {self.charge} {self.multiplicity}"

        # Generate the coordinates section
        coordinates = [f"{element} {x} {y} {z}" for element, (x, y, z) in zip(self.symbols, self.positions_cart)]
        coordinates_section = '\n'.join(coordinates)

        # Combine sections into a complete input file
        orca_input = f"{header}\n{blocks}\n{charge_mult}\n{coordinates_section}\n*\n"
        return orca_input
    
    def citation_strings(self) -> str:
        # TODO: Add a short string with the citation as bib and a sentence what was done
        return 'bib_string', 'sentence string'

