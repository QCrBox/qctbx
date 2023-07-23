from abc import abstractmethod, ABC
from typing import List, Dict, Any, Union

from ..custom_typing import Path

import numpy as np

class DensityCalculator(ABC):

    @abstractmethod
    def check_availability(self) -> bool:
        pass

    @abstractmethod
    def calculate_density(
        self,
        atom_site_dict: Dict[str, Union[float, str]],
        cell_dict: Dict[str, float]
    ):
        pass

    @abstractmethod
    def cif_output(self) -> str:
        pass

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
        density_path: Path
    ) -> np.ndarray:
        pass

    @abstractmethod
    def cif_output(self) -> str:
        pass

    @abstractmethod
    def citation_strings(self):
        pass
