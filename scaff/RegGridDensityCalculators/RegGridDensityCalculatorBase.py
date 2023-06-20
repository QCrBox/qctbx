from ..DensityCalculatorBase import DensityCalculator
from dataclasses import dataclass
from typing import Dict, Any

from abc import abstractmethod, ABC


@dataclass
class RegGridDensityCalculator(DensityCalculator):
    qm_options: Dict[str, Any]
    calc_options: Dict[str, Any]