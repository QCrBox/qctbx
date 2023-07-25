from copy import deepcopy
import os
import pathlib
import subprocess
import warnings
from typing import Dict, List, Union

import numpy as np
from ase.calculators.nwchem import NWChem

from ...conversions import add_cart_pos
from ..QCCalculator.ase import AseLCAOCalculator
from .base import LCAODensityCalculator


defaults = {
    'method': 'hcth407p',
    'basis_set': 'def2-SVP',
    'multiplicity': 1,
    'charge': 0,
    'cpu_count': 1,
    'ram': 2000,
    'specific_options': {},
    'calc_options': {
        'label': 'nwchem',
        'work_directory': '.',
        'output_format': 'wfn'
    }
}


nwchem_bibtex_key = 'nwchem'


nwchem_bibtex_entry = """
@article{NWChem,
    title = {NWChem: A comprehensive and scalable open-source solution for large scale molecular simulations},
    journal = {Computer Physics Communications},
    volume = {181},
    number = {9},
    pages = {1477-1489},
    year = {2010},
    issn = {0010-4655},
    doi = {10.1016/j.cpc.2010.04.018},
    url = {https://doi.org/10.1016/j.cpc.2010.04.018},
    author = {M. Valiev and E.J. Bylaska and N. Govind and K. Kowalski and T.P. Straatsma and H.J.J. {Van Dam} and D. Wang and J. Nieplocha and E. Apra and T.L. Windus and W.A. {de Jong}},
    keywords = {NWChem, DFT, Coupled cluster, QMMM, Plane wave methods}
}
""".strip()


molden2aimfile = 'molden= -1\nwfn= 1\nwfncheck= 1\nwfx= 1\nwfxcheck= 1\nnbo= -1\nnbocheck= -1\nwbo= -1\nprogram=0\nedftyp=0\nallmo=1\nprspin=1\nunknown=1\ncarsph=0\nnbopro=0\nnosupp=1\ntitle=0\nclear=1\nansi=0\n'


class NWChemLCAODensityCalculator(LCAODensityCalculator):
    xyz_format = 'cartesian'
    provides_output = tuple(('wfn'))

    def __init__(self, *args, nwchem_path='nwchem', molden2aimpath=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.nwchem_path = nwchem_path
        self.molden2aimpath = molden2aimpath
        self.update_from_dict(defaults, update_if_present=False)


    def check_availability(self) -> bool:
        """
        Check if the nwchem executable is available in the system.

        Returns:
            bool: True if the nwchem executable is available, False otherwise.
        """
        path = pathlib.Path(self.nwchem_path)
        return path.exists()

    def calculate_density(
        self,
        atom_site_dict: Dict[str, Union[float, str]],
        cell_dict: Dict[str, float],
        cluster_charge_dict: Dict[str, List[float]] = None
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
                array under 'positions_cart' for the charge positions and a
                n sized array with the charges under 'charges'.
                Defaults to an empty dict for no cluster charges.
        """
        if cluster_charge_dict is None:
            cluster_charge_dict = {}
        assert len(cluster_charge_dict) == 0, 'Cluster charges are currently not supported'
        try:
            positions_cart = np.array([atom_site_dict[f'_atom_site_Cartn_{coord}'] for coord in ('x', 'y', 'z')]).T
        except KeyError:
            new_atom_site_dict, _ = add_cart_pos(atom_site_dict, cell_dict)
            positions_cart = np.array([new_atom_site_dict[f'_atom_site_Cartn_{coord}'] for coord in ('x', 'y', 'z')]).T
        self.update_from_dict(defaults, update_if_present=False)

        ase_options = deepcopy(self.specific_options)
        ase_options['dft'] = {
            'xc': self.method,
        }
        ase_options['basis'] = self.basis_set
        ase_options['charge'] = self.charge
        if self.multiplicity != 1:
            ase_options['dft']['MULT'] = self.multiplicity
        ase_options['label'] = os.path.join(self.calc_options['work_directory'], self.calc_options['label'])
        if self.molden2aimpath is not None:
            ase_options['property'] = {
                'Moldenfile': None,
                'Molden_norm': 'janpa'
            }
        else:
            ase_options['property'] = {
                'Aimfile': None
            }
        ase_options['task'] = 'property'
        ase_options['memory'] = f'total {int(self.ram_mb)} mb'

        if self.cpu_count > 1:
            ase_options['command'] = f'mpirun -n {self.cpu_count} {self.nwchem_path}]'

        nwchem = NWChem(**ase_options)

        calculator = AseLCAOCalculator(
            calc=nwchem,
            positions_cart=positions_cart,
            symbols=list(atom_site_dict['_atom_site_type_symbol'])
        )

        calculator.run_calculation()

        if self.molden2aimpath is not None:
            self._write_molden2aim_ini()
            abs_path = pathlib.Path(self.molden2aimpath).resolve()
            with open(os.path.join(self.calc_options['work_directory'], 'molden2aim.log'), 'w') as fobj:
                subprocess.check_call(
                    [abs_path, '-i', f"{self.calc_options['label']}.molden"],
                    cwd=os.path.join(self.calc_options['work_directory'], self.calc_options['label']),
                    stdout=fobj
                )
        else:
            warnings.warn('The wfn output without molden2aim might not be compatible with all partitioners (NoSpherA2 should work)')

        return os.path.join(self.calc_options['work_directory'], self.calc_options['label'], f"{self.calc_options['label']}.wfn")

    def _write_molden2aim_ini(self):
        self.update_from_dict(defaults, update_if_present=False)
        with open(os.path.join(self.calc_options['work_directory'], self.calc_options['label'], 'm2a.ini'), 'w') as fobj:
            fobj.write(molden2aimfile)

    def cif_output(self):
        return 'Implement me'

    def citation_strings(self) -> str:
        self.update_from_dict(defaults, update_if_present=False)

        software_name = 'ASE/NWChem'
        ase_bibtex_key, ase_bibtex_entry = AseLCAOCalculator({}, {}).bibtex_strings()

        software_key = ','.join((ase_bibtex_key, nwchem_bibtex_key))
        software_bibtex_entry = '\n\n\n'.join((ase_bibtex_entry, nwchem_bibtex_entry))

        return self.generate_description(software_name, software_key, software_bibtex_entry)
