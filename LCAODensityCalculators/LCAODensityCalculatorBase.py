from ..DensityCalculatorBase import DensityCalculator

from abc import abstractmethod
from dataclasses import dataclass

@dataclass
class LCAODensityCalculator(DensityCalculator):
    qm_options: dict
    calc_options: dict



