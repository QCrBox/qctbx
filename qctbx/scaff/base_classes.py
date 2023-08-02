from abc import abstractmethod, ABC
from typing import List, Dict, Any, Union
from collections.abc import Sequence

from iotbx import cif
import numpy as np

from ..custom_typing import Path
from ..io.cif import stringify_options

def is_data_array(cif_value):
    test_data_array = any((
        isinstance(cif_value, Sequence),
        isinstance(cif_value, np.ndarray)
    ))
    return test_data_array and not isinstance(cif_value, str)


class DensityHandler(ABC):
    available_args = ()

    @abstractmethod
    def check_availability(self) -> bool:
        pass

    def to_settings_cif(self, cif_path, block_name):
        new_model = cif.model.cif()
        new_model[block_name] = self.as_cctbx_cif_block(skip_calc_options=False)
        string_output = str(new_model).replace('data_' + block_name, 'settings_' + block_name)
        with open(cif_path, 'w', encoding='ASCII') as fobj:
            fobj.write(string_output)

    def as_cctbx_cif_block(self, skip_calc_options=False):
        assert len(self.available_args) > 0, 'available_args needs to be implemented for this to work. The class definition is incomplete.'
        new_block = cif.model.block()
        cif_entry_start = '_qctbx_reggridpartition_'
        for arg in self.available_args:
            attr = getattr(self, arg)
            if attr is None:
                continue
            if skip_calc_options and arg == 'calc_options':
                continue
            if isinstance(attr, dict):
                if len(attr) == 0:
                    continue
                no_ciflike = True
                if all(is_data_array(val) for val in attr.values()):
                    val_lengths = list(len(val) for val in attr.values())
                    if all(val_length == val_lengths[0] for val_length in val_lengths):
                        no_ciflike = False
                        new_loop = cif.model.loop()
                        for key, value in attr.items():
                            new_loop.add_column(key, list(value))
                        new_block.add_loop(new_loop)
                if no_ciflike:
                    new_block[cif_entry_start + arg] = stringify_options(attr)
            else:
                new_block[cif_entry_start + arg] = attr
        return new_block

    def data_cif_output(self) -> str:
        return str(self.as_cctbx_cif_block(skip_calc_options=True))

    @abstractmethod
    def citation_strings(self):
        pass

class DensityCalculator(DensityHandler):

    @abstractmethod
    def calculate_density(
        self,
        atom_site_dict: Dict[str, Union[float, str]],
        cell_dict: Dict[str, float]
    ):
        pass

class DensityPartitioner(DensityHandler):
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

