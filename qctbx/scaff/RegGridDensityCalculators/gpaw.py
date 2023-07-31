from copy import deepcopy
import io
import os

from ..QCCalculator.ase import AsePBCCalculator
from .base import RegGridDensityCalculator

try:
    import gpaw
    from ase.io import write
    from ase.units import Bohr, Hartree
    from gpaw import GPAW
    from gpaw.utilities.tools import cutoff2gridspacing, gridspacing2cutoff
    import_worked = True
except ImportError:
    import_worked = False

gpaw_bibtex_key = 'GPAW1,GPAW2'

gpaw_bibtex_entry = """
@article{GPAW1,
    title = {Real-space grid implementation of the projector augmented wave method},
    author = {Mortensen, J. J. and Hansen, L. B. and Jacobsen, K. W.},
    journal = {Phys. Rev. B},
    volume = {71},
    issue = {3},
    pages = {035109},
    numpages = {11},
    year = {2005},
    month = {Jan},
    publisher = {American Physical Society},
    doi = {10.1103/PhysRevB.71.035109},
    url = {https://link.aps.org/doi/10.1103/PhysRevB.71.035109}
}

@article{GPAW2,
doi = {10.1088/0953-8984/22/25/253202},
url = {https://dx.doi.org/10.1088/0953-8984/22/25/253202},
year = {2010},
month = {jun},
publisher = {},
volume = {22},
number = {25},
pages = {253202},
author = {J Enkovaara and C Rostgaard and J J Mortensen and J Chen and M Dułak and L Ferrighi and J Gavnholt and C Glinsvad and V Haikola and H A Hansen and H H Kristoffersen and M Kuisma and A H Larsen and L Lehtovaara and M Ljungberg and O Lopez-Acevedo and P G Moses and J Ojanen and T Olsen and V Petzold and N A Romero and J Stausholm-Møller and M Strange and G A Tritsaris and M Vanin and M Walter and B Hammer and H Häkkinen and G K H Madsen and R M Nieminen and J K Nørskov and M Puska and T T Rantala and J Schiøtz and K S Thygesen and K W Jacobsen},
title = {Electronic structure calculations with GPAW: a real-space implementation of the projector
augmented-wave method},
journal = {Journal of Physics: Condensed Matter},
}""".strip()



defaults = {
    'method': 'PBE',
    'ecut_ev': 50,
    'kpoints': (1,1,1),
    'density_type': 'total',
    'specific_options': {
        'gpaw_options': {
            'convergence':{'density': 1e-6},
            'symmetry':{'symmorphic': False},
        },
        'grid_interpolation': 4,

    },
    'calc_options': {
        'label': 'gpaw',
        'work_directory': '.',
        'output_format': 'cube',
        'log_file': 'gpaw_calculation.log'
    }
}

class GPAWDensityCalculator(RegGridDensityCalculator):
    xyz_format = 'fractional'
    provides_output = ('cube')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.update_from_dict(defaults, update_if_present=False)

    def check_availability(self):
        return import_worked

    def calculate_density(
        self,
        atom_site_dict,
        cell_dict
    ):
        gpaw_options = deepcopy(self.specific_options['gpaw_options'])

        if 'xc' in gpaw_options:
            self.method = gpaw_options['xc']
            del gpaw_options['xc']
        if 'h' in gpaw_options:
            self.ecut_ev = gridspacing2cutoff(gpaw_options['h']) / Hartree
            del gpaw_options['h']

        if self.calc_options['log_file'] == '_' or self.calc_options['log_file'] is None:
            gpaw_options['txt'] = '-'
        elif isinstance(self.calc_options['log_file'], io.IOBase):
            gpaw_options['txt'] = self.calc_options['log_file']
        else:
            gpaw_options['txt'] = os.path.join(self.calc_options['work_directory'], self.calc_options['log_file'])
        ase_calc = GPAW(
            xc=self.method,
            h=cutoff2gridspacing(self.ecut_ev * Hartree),
            **gpaw_options
        )

        calculator = AsePBCCalculator(
            cell_dict=cell_dict,
            atom_site_dict=atom_site_dict,
            calc=ase_calc
        )

        atoms, calc = calculator.run_calculation()

        if self.density_type == 'total':
            density = calc.get_all_electron_density(gridrefinement=self.specific_options['grid_interpolation'])
        elif self.density_type == 'valence':
            density = calc.get_all_electron_density(skip_core=True, gridrefinement=self.specific_options['grid_interpolation'])
        else:
            raise NotImplementedError('output_type needs to be valence or total')
        path = os.path.join (self.calc_options['work_directory'], f"{self.calc_options['label']}.cube")
        write(path, atoms, data=density * Bohr**3)
        return path

    def cif_output(self):
        return 'Implement me'

    def citation_strings(self) -> str:

        gpaw_version = gpaw.__version__
        software_name = f'ASE/GPAW {gpaw_version}'
        ase_bibtex_key, ase_bibtex_entry = AsePBCCalculator({}, {}).bibtex_strings()

        software_key = ','.join((ase_bibtex_key, gpaw_bibtex_key))
        software_bibtex_entry = '\n\n\n'.join((ase_bibtex_entry, gpaw_bibtex_entry))

        return self.generate_description(software_name, software_key, software_bibtex_entry)
