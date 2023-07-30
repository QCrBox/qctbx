from typing import Any, Dict, Tuple
import warnings

from ..citations import get_functional_citation
from ..base_classes import DensityCalculator
from ..util import dict_merge
from ...io.cif import read_settings_cif, settings_cif2kwargs

class RegGridDensityCalculator(DensityCalculator):
    available_args = ('method', 'ecut_ev', 'kpoints', 'specific_options', 'calc_options', 'density_type')

    def __init__(
        self,
        method:str=None,
        ecut_ev:float=None,
        kpoints:Tuple[int]=None,
        density_type:str=None,
        specific_options:Dict[str, Any]=None,
        calc_options:Dict[str, Any]=None
    ):
        self.method = method
        self.ecut_ev = ecut_ev
        self.kpoints = kpoints
        self.density_type = density_type

        if specific_options is not None:
            self.specific_options = specific_options
        else:
            self.specific_options = {}

        if calc_options is not None:
            self.calc_options = calc_options
        else:
            self.calc_options = {}

    @classmethod
    def from_settings_cif(cls, scif_path, block_name):
        settings_cif = read_settings_cif(scif_path, block_name)

        dict_entries = ('specific_options', 'calc_options')
        type_funcs = {
            'method': str,
            'ecut_ev': float,
            'kpoints': lambda x: tuple((int(val) for val in x.split())),
            'density_type': str,
        }
        cif_entry_start = '_qctbx_reggridwfn_'

        kwargs = settings_cif2kwargs(
            settings_cif,
            cif_entry_start,
            dict_entries,
            type_funcs,
            cls.available_args
        )

        return cls(**kwargs)

    def update_from_dict(self, update_dict, update_if_present=True):

        for key in update_dict.keys():
            if key not in self.available_args:
                warnings.warn(f'Could not find matching property for key: {key}')

        condition = (self.method is None) or update_if_present
        if condition and 'method' in update_dict:
            self.method = update_dict['method']

        condition = (self.ecut_ev is None) or update_if_present
        if condition and 'ecut_ev' in update_dict:
            self.ecut_ev = update_dict['ecut_ev']

        condition = (self.kpoints is None) or update_if_present
        if condition and 'kpoints' in update_dict:
            self.kpoints = update_dict['kpoints']

        condition = (self.density_type is None) or update_if_present
        if condition and 'density_type' in update_dict:
            self.density_type = update_dict['density_type']

        #dictionaries are merged instead of replaced
        updates = update_dict.get('specific_options', {})
        if update_if_present:
            self.specific_options = dict_merge(self.specific_options, updates)
        else:
            self.specific_options = dict_merge(updates, self.specific_options)

        updates = update_dict.get('calc_options', {})
        if update_if_present:
            self.calc_options = dict_merge(self.calc_options, updates)
        else:
            self.calc_options = dict_merge(updates, self.calc_options)

    def generate_description(
        self,
        software_name,
        software_bibtex_key,
        software_bibtex_entry
    ):
        method_bibtex_key, method_bibtex_entry = get_functional_citation(self.method)
        if all(point == 1 for point in self.kpoints):
            k_string = ' at the Gamma point'
        else:
            kpts = self.kpoints
            k_string = f' and ({kpts[0]} {kpts[1]} {kpts[2]}) Monkhorst-Pack k-point grid'

        report_string = (
            f"The electron density was calculated using {self.method}[{method_bibtex_key}]"
            + f" with a grid corresponding to an energy cutoff of {self.ecut_ev} eV"
            + k_string
            + f" in {software_name} [{software_bibtex_key}]"
        )
        return report_string, '\n\n\n'.join((software_bibtex_entry, method_bibtex_entry))
