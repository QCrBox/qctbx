from ..base_classes import DensityPartitioner
from ...conversions import parse_specific_options
from ...io.cif import read_settings_cif
from ..util import dict_merge

class LCAODensityPartitioner(DensityPartitioner):
    def __init__(
        self,
        method=None,
        grid_accuracy=None,
        cpu_count=None,
        specific_options=None,
        calc_options=None
    ):
        self.method = method
        self.grid_accuracy = grid_accuracy
        self.cpu_count = cpu_count
        if specific_options is None:
            self.specific_options = {}
        else:
            self.specific_options = specific_options
        if calc_options is None:
            self.calc_options = {}
        else:
            self.calc_options = calc_options

    def update_from_dict(self, update_dict, update_if_present=True):
        condition = (self.method is None) or update_if_present
        if condition and 'method' in update_dict:
            self.method = update_dict['method']

        condition = (self.grid_accuracy is None) or update_if_present
        if condition and 'grid_accuracy' in update_dict:
            self.grid_accuracy = update_dict['grid_accuracy']

        condition = (self.cpu_count is None) or update_if_present
        if condition and 'cpu_count' in update_dict:
            self.cpu_count = update_dict['cpu_count']

        #dictionaries are merged instead of replaced
        updates = update_dict.get('specific_options', {})
        if update_if_present:
            self.specific_options = dict_merge(self.specific_options, updates)
        else:
            self.specific_options = dict_merge(updates, self.specific_options)

        updates = update_dict.get('calc_options', {})
        if update_if_present:
            self.calc_options = dict_merge(self.calc_options, updates, case_sensitive=False)
        else:
            self.calc_options = dict_merge(updates, self.calc_options)

    @classmethod
    def from_settings_cif(cls, scif_path, block_name):
        settings_cif = read_settings_cif(scif_path, block_name)

        cif_specific_options = settings_cif.get('_qctbx_lcaopartitioning_specific_options', '').strip()
        if len(cif_specific_options) > 0:
            specific_options = parse_specific_options(cif_specific_options)
        else:
            specific_options = {}

        cif_calc_options = settings_cif.get('_qctbx_lcaopartitioning_calc_options', '').strip()
        if len(cif_calc_options) > 0:
            calc_options = parse_specific_options(cif_calc_options)
        else:
            calc_options = {}

        new_obj = cls(
            method=str(settings_cif['_qctbx_lcaopartitioning_method']),
            grid_accuracy=str(settings_cif['_qctbx_lcaopartitioning_grid_accuracy']),
            cpu_count=int(settings_cif['_qctbx_lcaopartitioning_cpu_count']),
            specific_options=specific_options,
            calc_options=calc_options
        )

        return new_obj

