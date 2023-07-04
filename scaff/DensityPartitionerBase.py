"""Refinement against Partitioned Calculated Molecular Densities"""
from typing import List, Dict, Any, Optional

from abc import abstractmethod, ABC
import numpy as np


class DensityPartitioner(ABC):
    @abstractmethod
    def check_availability(self) -> bool:
        pass

    @abstractmethod
    def calc_f0j(
        self,
        atom_labels: List[int],
        atom_site_dict: Dict[str, List[Any]],
        cell_dict: Dict[str, Any],
        space_group_dict: Dict[str, Any],
        refln_dict: Dict[str, Any],
        density_path: Optional[str] = None
    ) -> np.ndarray:
        pass

    @abstractmethod
    def cif_output(self) -> str:
        pass

    @abstractmethod
    def citation_strings(self):
        pass
