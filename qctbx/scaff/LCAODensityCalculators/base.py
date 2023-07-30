from abc import abstractmethod
from typing import Any, Dict, Union, List

from ..citations import get_basis_citation, get_functional_citation
from ..base_classes import DensityCalculator
from ..util import dict_merge
from ...io.cif import read_settings_cif, settings_cif2kwargs

class LCAODensityCalculator(DensityCalculator):
    available_args = ('method', 'basis_set', 'charge', 'multiplicity', 'specific_options', 'calc_options')
    def __init__(
        self,
        method:str=None,
        basis_set:str=None,
        charge:int=None,
        multiplicity:int=None,
        specific_options:Dict[Any, Any]=None,
        calc_options=None
    ):
        self.method = method
        self.basis_set = basis_set
        self.charge = charge
        self.multiplicity = multiplicity
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
            'basis_set': int,
            'charge': int,
            'multiplicity': int
        }
        cif_entry_start = '_qctbx_lcaowfn_'

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

        condition = (self.basis_set is None) or update_if_present
        if condition and 'basis_set' in update_dict:
            self.basis_set = update_dict['basis_set']

        condition = (self.charge is None) or update_if_present
        if condition and 'charge' in update_dict:
            self.charge = update_dict['charge']

        condition = (self.multiplicity is None) or update_if_present
        if condition and 'multiplicity' in update_dict:
            self.multiplicity = update_dict['multiplicity']

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


    @abstractmethod
    def calculate_density(
        self,
        atom_site_dict: Dict[str, Union[float, str]],
        cell_dict: Dict[str, float],
        cluster_charge_dict: Dict[str, List[float]]=None
    ):
        pass

    def generate_description(
        self,
        software_name,
        software_bibtex_key,
        software_bibtex_entry
    ):
        method_bibtex_key, method_bibtex_entry = get_functional_citation(self.method)
        basis_bibtex_key, basis_bibtex_entry = get_basis_citation(self.basis_set)
        report_string = (
            f"The wavefunction was calculated using {self.method}[{method_bibtex_key}]/{self.basis_set}[{basis_bibtex_key}]"
            + f" in {software_name} [{software_bibtex_key}]"
        )
        return report_string, '\n\n\n'.join((software_bibtex_entry, method_bibtex_entry, basis_bibtex_entry))
