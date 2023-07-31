from ..base_classes import DensityPartitioner
from ..util import dict_merge
from ...io.cif import read_settings_cif, settings_cif2kwargs

class LCAODensityPartitioner(DensityPartitioner):
    available_args = ('method', 'grid_accuracy', 'specific_options', 'calc_options')
    def __init__(
        self,
        method=None,
        grid_accuracy=None,
        specific_options=None,
        calc_options=None
    ):
        self.method = method
        self.grid_accuracy = grid_accuracy
        if specific_options is None:
            self.specific_options = {}
        else:
            self.specific_options = specific_options
        if calc_options is None:
            self.calc_options = {}
        else:
            self.calc_options = calc_options


    @classmethod
    def from_settings_cif(cls, scif_path, block_name):
        settings_cif = read_settings_cif(scif_path, block_name)

        dict_entries = ('specific_options', 'calc_options')
        type_funcs = {
            'method': str,
            'grid_accuracy': str,
        }
        cif_entry_start = '_qctbx_lcaopartition_'

        kwargs = settings_cif2kwargs(
            settings_cif,
            cif_entry_start,
            dict_entries,
            type_funcs,
            cls.available_args
        )

        return cls(**kwargs)

    def update_from_dict(self, update_dict, update_if_present=True):
        condition = (self.method is None) or update_if_present
        if condition and 'method' in update_dict:
            self.method = update_dict['method']

        condition = (self.grid_accuracy is None) or update_if_present
        if condition and 'grid_accuracy' in update_dict:
            self.grid_accuracy = update_dict['grid_accuracy']
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


