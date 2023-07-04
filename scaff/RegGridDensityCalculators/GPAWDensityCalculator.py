from . RegGridDensityCalculatorBase import RegGridDensityCalculator
from ..QCCalculator.AseCalculator import AsePBCCalculator
from ..util import dict_merge

import os
try:
    from gpaw import GPAW
    from ase.io import write
    from ase.units import Bohr, eV, Hartree
    from gpaw.utilities.tools import cutoff2gridspacing, gridspacing2cutoff
    import_worked = True
except:
    import_worked = False



calc_defaults = {
    'label': 'gpaw',
    'work_directory': '.',
    'output_format': 'cube',
    'output_type': 'total',
    'gridinterpolation': 4,

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

    def check_availability(self):
        return import_worked
    
    def calculate_density(
        self,
        atom_site_dict,
        cell_dict
    ):  
        calc_options = dict_merge(calc_defaults, self.calc_options)
        qm_options = dict_merge(qm_defaults, self.qm_options)
        if 'xc' in qm_options['gpaw_options']:
            qm_options['method'] = qm_options['gpaw_options']['xc']
            del(qm_options['gpaw_options']['xc'])
        if 'h' in qm_options['gpaw_options']:
            qm_options['e_cut_ev'] = gridspacing2cutoff(qm_options['gpaw_options']['h']) / Hartree
            del(qm_options['gpaw_options']['h'])
        print(cutoff2gridspacing(qm_options['e_cut_ev'] * Hartree), qm_options['e_cut_ev'])
        ase_calc = GPAW(
            xc=qm_options['method'],
            h=cutoff2gridspacing(qm_options['e_cut_ev'] * Hartree),
            **qm_options['gpaw_options']
        )

        calculator = AsePBCCalculator(
            cell_dict=cell_dict,
            atom_site_dict=atom_site_dict,
            calc=ase_calc
        )

        atoms, calc = calculator.run_calculation()

        if calc_options['output_type'] == 'total':
            density = calc.get_all_electron_density(grid_refinement=calc_options['grid_interpolation'])
        elif calc_options['output_type'] == 'valence':
            density = calc.get_all_electron_density(skip_core=True, grid_refinement=calc_options['grid_interpolation'])
        else: 
            raise NotImplementedError('output_type needs to be valence or total')
        path = os.path.join(calc_options['work_directory'], f"{calc_options['label']}.cube")
        write(path, atoms, data=density * Bohr**3)
        return path
    
    def cif_output(self):
        return 'Implement me'

    def citation_strings(self) -> str:
        # TODO: Add a short string with the citation as bib and a sentence what was done
        return 'bib_string', 'sentence string'


    