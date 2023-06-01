from ..DensityCalculatorBase import DensityCalculator

from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class LCAODensityCalculator(DensityCalculator):
    qm_options: Dict[str, Any]
    calc_options: Dict[str, Any]



