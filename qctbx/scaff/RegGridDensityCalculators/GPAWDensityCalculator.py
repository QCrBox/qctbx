from . RegGridDensityCalculatorBase import RegGridDensityCalculator
from ..QCCalculator.AseCalculator import AsePBCCalculator
from ..util import dict_merge

import os
try:
    from gpaw import GPAW
    import gpaw
    from ase.io import write
    from ase.units import Bohr, eV, Hartree
    from gpaw.utilities.tools import cutoff2gridspacing, gridspacing2cutoff
    import_worked = True
except:
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

calc_defaults = {
    'label': 'gpaw',
    'work_directory': '.',
    'output_format': 'cube',
    'output_type': 'total',
    'grid_interpolation': 4,

}

qm_defaults = {
    'method': 'PBE',
    'e_cut_ev': 50,
    'kpoints': (1,1,1),
    'gpaw_options': {
        'convergence':{'density': 1e-6},
        'symmetry':{'symmorphic': False}
    }
}

class GPAWDensityCalculator(RegGridDensityCalculator):
    xyz_format = 'fractional'
    provides_output = ('cube')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._calc_options = dict_merge(calc_defaults, self.calc_options)
        self._qm_options = dict_merge(qm_defaults, self.qm_options)

    def check_availability(self):
        return import_worked
    
    def calculate_density(
        self,
        atom_site_dict,
        cell_dict
    ):  
        self._calc_options = dict_merge(calc_defaults, self.calc_options)
        self._qm_options = dict_merge(qm_defaults, self.qm_options)
        if 'xc' in self._qm_options['gpaw_options']:
            self._qm_options['method'] = self._qm_options['gpaw_options']['xc']
            del(self._qm_options['gpaw_options']['xc'])
        if 'h' in self._qm_options['gpaw_options']:
            self._qm_options['e_cut_ev'] = gridspacing2cutoff(self._qm_options['gpaw_options']['h']) / Hartree
            del(self._qm_options['gpaw_options']['h'])
        ase_calc = GPAW(
            xc=self._qm_options['method'],
            h=cutoff2gridspacing(self._qm_options['e_cut_ev'] * Hartree),
            **self._qm_options['gpaw_options']
        )

        calculator = AsePBCCalculator(
            cell_dict=cell_dict,
            atom_site_dict=atom_site_dict,
            calc=ase_calc
        )

        atoms, calc = calculator.run_calculation()

        if self._calc_options['output_type'] == 'total':
            density = calc.get_all_electron_density(gridrefinement=self._calc_options['grid_interpolation'])
        elif self._calc_options['output_type'] == 'valence':
            density = calc.get_all_electron_density(skip_core=True, gridrefinement=self._calc_options['grid_interpolation'])
        else: 
            raise NotImplementedError('output_type needs to be valence or total')
        path = os.path.join (self._calc_options['work_directory'], f" {self._calc_options['label']}.cube")
        write(path, atoms, data=density * Bohr**3)
        return path
    
    def cif_output(self):
        return 'Implement me'

    def citation_strings(self) -> str:
        self._calc_options = dict_merge(calc_defaults, self.calc_options)
        self._qm_options = dict_merge(qm_defaults, self.qm_options)

        gpaw_version = gpaw.__version__
        software_name = f'ASE/GPAW {gpaw_version}'
        ase_bibtex_key, ase_bibtex_entry = AsePBCCalculator({}, {}).bibtex_strings()

        software_key = ','.join((ase_bibtex_key, gpaw_bibtex_key))
        software_bibtex_entry = '\n\n\n'.join((ase_bibtex_entry, gpaw_bibtex_entry))

        return self.generate_description(software_name, software_key, software_bibtex_entry)


    