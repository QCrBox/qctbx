from ..DensityCalculatorBase import DensityCalculator

from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict, Any
from ..citations import get_functional_citation, get_basis_citation


@dataclass
class LCAODensityCalculator(DensityCalculator):
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
        report_string = (
            f"The wavefunction was calculated using {self._qm_options['method'][{method_bibtex_key}]}/{self._qm_options['basis_set']}[{basis_bibtex_key}]"
            + f" in {software_name} [{software_bibtex_key}]"
        )
        return report_string, '\n\n\n'.join((software_bibtex_entry, method_bibtex_entry, basis_bibtex_entry))



