import os
import pathlib
import platform
import shutil
import subprocess
import textwrap
from typing import Optional

from ..util import batched
from .base import LCAOQCCalculator


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
            cc_file = self._generate_cluster_charge_file()
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

    def bibtex_strings(self) -> str:
        return 'ORCA2020', textwrap.dedent(r"""
            @article{ORCA2020,
                author = {Neese, Frank and Wennmohs, Frank and Becker, Ute and Riplinger, Christoph},
                title = "{The ORCA quantum chemistry program package}",
                journal = {The Journal of Chemical Physics},
                volume = {152},
                number = {22},
                pages = {224108},
                year = {2020},
                month = {06},
                abstract = "{In this contribution to the special software-centered issue, the ORCA program package is described. We start with a short historical perspective of how the project began and go on to discuss its current feature set. ORCA has grown into a rather comprehensive general-purpose package for theoretical research in all areas of chemistry and many neighboring disciplines such as materials sciences and biochemistry. ORCA features density functional theory, a range of wavefunction based correlation methods, semi-empirical methods, and even force-field methods. A range of solvation and embedding models is featured as well as a complete intrinsic to ORCA quantum mechanics/molecular mechanics engine. A specialty of ORCA always has been a focus on transition metals and spectroscopy as well as a focus on applicability of the implemented methods to “real-life” chemical applications involving systems with a few hundred atoms. In addition to being efficient, user friendly, and, to the largest extent possible, platform independent, ORCA features a number of methods that are either unique to ORCA or have been first implemented in the course of the ORCA development. Next to a range of spectroscopic and magnetic properties, the linear- or low-order single- and multi-reference local correlation methods based on pair natural orbitals (domain based local pair natural orbital methods) should be mentioned here. Consequently, ORCA is a widely used program in various areas of chemistry and spectroscopy with a current user base of over 22 000 registered users in academic research and in industry.}",
                issn = {0021-9606},
                doi = {10.1063/5.0004608},
                url = {https://doi.org/10.1063/5.0004608},
                eprint = {https://pubs.aip.org/aip/jcp/article-pdf/doi/10.1063/5.0004608/16740678/224108\_1\_online.pdf},
            }""")


