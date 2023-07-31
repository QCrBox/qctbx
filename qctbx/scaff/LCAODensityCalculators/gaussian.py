from typing import Dict, Union, List

from ..QCCalculator.gaussian import GaussianCalculator
from ..util import dict_merge
from .base import LCAODensityCalculator

calc_defaults = {
    'filebase': 'gaussian',
    'output_format': 'wfn',
    'title': 'qctbx calculation'
}

qm_defaults = {
    'method': 'PBE',
    'basisset': 'def2-SVP',
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
        raise NotImplementedError("This class is currently a non-working stub")
        self._calculator = GaussianCalculator(
            gauss_path=gauss_path
        )

    def check_availability(self) -> bool:
        return self._calculator.check_availability()

    def calculate_density(
            self,
            atom_site_dict: Dict[str, Union[float, str]],
            cell_dict: Dict[str, float],
            cluster_charge_dict: Dict[str, List[float]] = {}
        ):
        """
        Calculate the electronic density for a given atomic configuration using Gaussian.

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
        self._qm_options = qm_defaults.copy()
        self._qm_options.update(self.qm_options)

        self._calc_options = calc_defaults.copy()
        self._calc_options.update(self.calc_options)

        format_standardise = self._calc_options['output_format'].lower().replace('.', '')
        if format_standardise == 'wfn':
            #subprocess.check_output(['formchk', f" {self._calc_options['filebase']}.chk", f" {self._calc_options['filebase']}.wfn"])
            return self._calc_options['filebase'] + '.wfn'
        elif format_standardise == 'wfx':
            #subprocess.check_output(['formchk', '-3', f" {self._calc_options['filebase']}.chk", f" {self._calc_options['filebase']}.wfx"])
            return self._calc_options['filebase'] + '.wfn'
        else:
            raise NotImplementedError('output_format from GaussianDensityCalculator is not implemented. Choose wfn or wfx')


    def cif_output(self) -> str:

        self._calc_options = dict_merge(calc_defaults, self.calc_options)
        self._qm_options = dict_merge(qm_defaults, self.qm_options)

        software_bibtex_key, sofware_bibtex_entry = self._calculator.bibtex_strings()
        software_name = f'Gaussian {software_bibtex_key[-2:]}' #TODO determine and add version
        return self.generate_description(software_name, software_bibtex_key, sofware_bibtex_entry)
