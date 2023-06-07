"""Refinement against Partitioned Calculated Molecular Densities"""
from typing import List, Dict, Any, Optional

from abc import abstractmethod, ABC


class DensityPartitioner(ABC):
    @abstractmethod
    def calc_f0j(
            self,
            atom_indexes: List[int],
            cell_dict: Dict[str, Any],
            refln_dict: Dict[str, Any],
            density_path: Optional[str] = None
        ):
        pass

    @abstractmethod
    def citation_strings(self):
        pass