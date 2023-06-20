from . RegGridDensityCalculatorBase import RegGridDensityCalculator
from ..QCCalculator.AseCalculator import AsePBCCalculator
from ..util import dict_merge

import os
try:
    from gpaw import GPAW
    from ase.io import write
    from ase.units import Bohr, eV
    from gpaw.utilities.tools import cutoff2gridspacing
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
    'e_cut_ev': 100,
    'gpaw_options': {
        'convergence':{'density': 1e-6},
    }
}

class GPAWDensityCalculator(RegGridDensityCalculator):
    xyz_format = 'fractional'
    provides_output = ('cube')

    def __init__(*args, **kwargs):
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
        ase_calc = GPAW(
            xc=qm_options['method'],
            h=cutoff2gridspacing(qm_options['e_cut_ev'] * eV),
            symmetry= {'symmorphic': False},
            **qm_options['gpaw_options']
        )

        calculator = AsePBCCalculator(
            cell_dict=cell_dict,
            atom_site_dict=atom_site_dict,
            calc=ase_calc
        )

        atoms, calc = calculator.run_calculation()

        density = calc.get_all_electron_density()
        path = os.path.join(calc_options['work_directory'], f"{calc_options['label']}.cube")
        write(path, atoms, data=density * Bohr**3)
        return path



    