from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Union, List

from ..citations import get_basis_citation, get_functional_citation
from ..density_calculator_base import DensityCalculator


@dataclass
class LCAODensityCalculator(DensityCalculator):
    qm_options: Dict[str, Any]
    calc_options: Dict[str, Any]
    _qm_options = None
    _calc_options = None

    @abstractmethod
    def calculate_density(
        self,
        atom_site_dict: Dict[str, Union[float, str]],
        cell_dict: Dict[str, float],
        cluster_charge_dict: Dict[str, List[float]] = {}
    ):
        pass

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

