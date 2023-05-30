from .LCAODensityPartitionerBase import LCAODensityPartitioner
from dataclasses import dataclass
import numpy as np

@dataclass
class NoSpherA2Partitioner(LCAODensityPartitioner):
    part_options = {}
    accepts_input = ('wfn', 'wfx')
    xyz_format = 'fractional'    

    def check_availability(self):
        pass

    def partition(
            self,
            filename,
            atom_positions,
            hkl,
            symm_strings
        ):
        # write an expanded mock hkl

        # write the cif file

        # write the asym cif file


        pass

