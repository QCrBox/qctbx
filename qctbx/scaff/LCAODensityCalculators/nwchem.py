import ase
from .base import LCAODensityCalculator
from ..QCCalculator.AseCalculator import AseLCAOCalculator
from ..util import dict_merge
from ...conversions import add_cart_pos
from ase.calculators.nwchem import NWChem
from typing import Dict, List, Union, Any
import os
import pathlib
import numpy as np
import subprocess
import warnings

calc_defaults = {
    'label': 'nwchem',
    'work_directory': '.',
    'output_format': 'wfn'
}

qm_defaults = {
    'method': 'hcth407p',
    'basis_set': 'def2-SVP',
    'multiplicity': 1,
    'charge': 0,                       
    'n_core': 1,
    'ram': 2000,
    'ase_options': {}
}
nwchem_bibtex_key = NWChem

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
    provides_output = ('wfn')

    def __init__(self, *args, nwchem_path='nwchem', molden2aimpath=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.nwchem_path = nwchem_path
        self.molden2aimpath = molden2aimpath

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
                array under 'positions_cart' for the charge positions and a 
                n sized array with the charges under 'charges'.
                Defaults to an empty dict for no cluster charges.
        """
        assert len(cluster_charge_dict) == 0, 'Cluster charges are currently not supported'
        symbols = list(atom_site_dict['_atom_site_type_symbol'])
        try:
            positions_cart = np.array([atom_site_dict[f'_atom_site_Cartn_{coord}'] for coord in ('x', 'y', 'z')]).T
        except KeyError:
            new_atom_site_dict, _ = add_cart_pos(atom_site_dict, cell_dict)
            positions_cart = np.array([new_atom_site_dict[f'_atom_site_Cartn_{coord}'] for coord in ('x', 'y', 'z')]).T

        used_qm_options = dict_merge(qm_defaults, self.qm_options)
        used_calc_options = dict_merge(calc_defaults, self.calc_options)
        
        ase_options = used_qm_options['ase_options']
        ase_options['dft'] = {
            'xc': used_qm_options['method'],
        }
        ase_options['basis'] = used_qm_options['basis_set']
        ase_options['charge'] = used_qm_options['charge']
        if used_qm_options['multiplicity'] != 1:
            ase_options['dft']['MULT'] = used_qm_options['multiplicity']
        ase_options['label'] = os.path.join(used_calc_options['work_directory'], used_calc_options['label'])
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
        ase_options['memory'] = f'total {int(used_qm_options["ram"])} mb'

        if used_qm_options['n_core'] > 1:
            ase_options['command'] = f'mpirun -n {used_qm_options["n_core"]} {self.nwchem_path}]'
     
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
            with open(os.path.join(used_calc_options['work_directory'], 'molden2aim.log'), 'w') as fo:
                subprocess.check_call(
                    [abs_path, '-i', f"{used_calc_options['label']}.molden"],
                    cwd=os.path.join(used_calc_options['work_directory'], used_calc_options['label']),
                    stdout=fo
                )
        else:
            warnings.warn('The wfn output without molden2aim might not be compatible with all partitioners (NoSpherA2 should work)')

        return os.path.join(used_calc_options['work_directory'], used_calc_options['label'], f"{used_calc_options['label']}.wfn")

    def _write_molden2aim_ini(self):
        used_calc_options = dict_merge(calc_defaults, self.calc_options)
        with open(os.path.join(used_calc_options['work_directory'], used_calc_options['label'], 'm2a.ini'), 'w') as fo:
            fo.write(molden2aimfile)

    def cif_output(self):
        return 'Implement me'

    def citation_strings(self) -> str:
        self._calc_options = dict_merge(calc_defaults, self.calc_options)
        self._qm_options = dict_merge(qm_defaults, self.qm_options)

        software_name = f'ASE/NWChem'
        ase_bibtex_key, ase_bibtex_entry = AseLCAOCalculator({}, {}).bibtex_strings()

        software_key = ','.join((ase_bibtex_key, nwchem_bibtex_key))
        software_bibtex_entry = '\n\n\n'.join((ase_bibtex_entry, nwchem_bibtex_entry))

        return self.generate_description(software_name, software_key, software_bibtex_entry)

    