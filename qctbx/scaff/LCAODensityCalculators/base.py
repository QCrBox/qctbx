from abc import abstractmethod
from typing import Any, Dict, Union, List

from ..citations import get_basis_citation, get_functional_citation
from ..base_classes import DensityCalculator
from ..util import dict_merge, tempinput
from ...conversions import parse_specific_options

class LCAODensityCalculator(DensityCalculator):

    def __init__(
        self,
        method:str=None,
        basis_set:str=None,
        charge:int=None,
        multiplicity:int=None,
        specific_options:Dict[Any, Any]=None,
        cpu_count:int=None,
        ram_mb:float=None,
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
        self.cpu_count = cpu_count
        self.ram_mb = ram_mb
        if calc_options is None:
            self.calc_options = {}
        else:
            self.calc_options = calc_options

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

        condition = (self.cpu_count is None) or update_if_present
        if condition and 'cpu_count' in update_dict:
            self.cpu_count = update_dict['cpu_count']

        condition = (self.ram_mb is None) or update_if_present
        if condition and 'ram_mb' in update_dict:
            self.charge = update_dict['ram_mb']

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

    @classmethod
    def from_settings_cif(cls, filename, block_name):
        from iotbx import cif
        with open(filename) as fobj:
            content = fobj.read()

        new_str = content.replace('\nsettings_', '\ndata_')
        with tempinput(new_str) as named_file:
            cif_data = cif.reader(named_file).model()
            settings_cif = cif_data.blocks[block_name]

        cif_specific_options = settings_cif.get('_qctbx_lacowfn_specific_options', '').strip()
        if len(cif_specific_options) > 0:
            specific_options = parse_specific_options(cif_specific_options)
        else:
            specific_options = {}

        cif_calc_options = settings_cif.get('_qctbx_lacowfn_calc_options', '').strip()
        if len(cif_calc_options) > 0:
            calc_options = parse_specific_options(cif_calc_options)
        else:
            calc_options = {}

        new_obj = cls(
            method=str(settings_cif['_qctbx_lcaowfn_method']),
            basis_set=str(settings_cif['_qctbx_lcaowfn_basisset']),
            charge=int(settings_cif['_qctbx_lcaowfn_charge']),
            multiplicity=int(settings_cif['_qctbx_lcaowfn_multiplicity']),
            cpu_count=int(settings_cif['_qctbx_lcaowfn_cpu_count']),
            ram_mb=int(settings_cif['_qctbx_lcaowfn_ram_mb']),
            specific_options=specific_options,
            calc_options=calc_options
        )

        return new_obj









