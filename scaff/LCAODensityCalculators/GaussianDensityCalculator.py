import shutil
import subprocess
import platform
import pathlib
from .LCAODensityCalculatorBase import LCAODensityCalculator
from ..QCCalculator.GaussianCalculator import GaussianCalculator


calc_defaults = {
    'filebase': 'gaussian',
    'output_format': 'wfn',
    'title': 'qctbx calculation'
}

qm_defaults = {
    'method': 'PBE',
    'basis_set': 'def2-SVP',
    'multiplicity': 1,
    'charge': 0,
    'link0': {},
    'route_section': [],
    'appendix': ''
}


class GaussianDensityCalculator(LCAODensityCalculator):
    provides_output = ('wfn', 'wfx')

    def __init__(self, *args, gauss_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._calculator = GaussianCalculator(
            gauss_path=gauss_path
        )

    def check_availability(self) -> bool:
        return self._calculator.check_availability()

    def calculate_density(self, elements, xyz, cluster_charge_dict={}):
        qm_options = qm_defaults.copy()
        qm_options.update(self.qm_options)

        calc_options = calc_defaults.copy()
        calc_options.update(self.calc_options)

        format_standardise = calc_options['output_format'].lower().replace('.', '')
        if format_standardise == 'wfn':
            #subprocess.check_output(['formchk', f"{calc_options['filebase']}.chk", f"{calc_options['filebase']}.wfn"])
            return calc_options['filebase'] + '.wfn'
        elif format_standardise == 'wfx':
            #subprocess.check_output(['formchk', '-3', f"{calc_options['filebase']}.chk", f"{calc_options['filebase']}.wfx"])
            return calc_options['filebase'] + '.wfn'
        else:
            raise NotImplementedError('output_format from GaussianDensityCalculator is not implemented. Choose wfn or wfx')


    def cif_output(self) -> str:
        # TODO: Implement the logic to generate a CIF output from the calculation
        return 'Someone needs to implement this before production'
