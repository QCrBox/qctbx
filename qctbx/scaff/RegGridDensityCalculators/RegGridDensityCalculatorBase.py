from ..DensityCalculatorBase import DensityCalculator
from dataclasses import dataclass
from typing import Dict, Any
from ..citations import get_functional_citation, get_basis_citation

from abc import abstractmethod, ABC


@dataclass
class RegGridDensityCalculator(DensityCalculator):
    qm_options: Dict[str, Any]
    calc_options: Dict[str, Any]
    _qm_options = None
    _calc_options = None

    def generate_description(
        self,
        software_name,
        software_bibtex_key,
        software_bibtex_entry
    ):
        method_bibtex_key, method_bibtex_entry = get_functional_citation(self._qm_options['method'])
        basis_bibtex_key, basis_bibtex_entry = get_basis_citation(self._qm_options['basis_set'])
        if all(point == 1 for point in self._qm_options['kpoints']):
            k_string = ' at the Gamma point'
        else:
            kpts = self._qm_options['kpoints']
            k_string = f' and ({kpts[0]} {kpts[1]} {kpts[2]}) Monkhorst-Pack k-point grid'

        report_string = (
            f"The electron density was calculated using {self._qm_options['method'][{method_bibtex_key}]}"
            + f" with a grid corresponding to an energy cutoff of {self._qm_options['e_cut_ev']} eV"
            + k_string
            + f" in {software_name} [{software_bibtex_key}]"
        )
        return report_string, '\n\n\n'.join((software_bibtex_entry, method_bibtex_entry, basis_bibtex_entry))
