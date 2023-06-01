from .LCAODensityPartitionerBase import LCAODensityPartitioner
from dataclasses import dataclass
import numpy as np

part_defaults = {
    'nosphera2_path': './NoSpherA2',
    'grid_accuracy': 3
}

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
            atom_site_dict,
            symm_strings,
            refln_dict = None,
        ):
        # write an expanded mock hkl

        # write the cif file

        # write the asym cif file


        pass

    def parse_charges(
        self,
        nosp2_filename,

    ):
        pass

