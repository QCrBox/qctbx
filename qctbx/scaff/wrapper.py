
import inspect
import json
import os
import subprocess
import textwrap
from copy import deepcopy
from typing import Any, Dict, List

import numpy as np

from ..custom_typing import Path
from ..io.cif import cif2dicts, read_settings_cif
from ..io.minimal_files import write_minimal_cif
from . import (name2lcaodensity, name2lcaopartition, name2reggriddensity,
               name2reggridpartition)
from .LCAODensityCalculators.base import LCAODensityCalculator
from .RegGridDensityCalculators.base import RegGridDensityCalculator

defaults = {
    'calc_options': {
        'run_command': 'python',
        'dewrapped_scif_path' : 'dewrapped.scif',
        'transfer_cif_path' : 'wrapped_transfer.cif',
        'block_name': 'wrapped_qctbx',
        'density_path_sub': ('', '')
    }
}

# functions for output .py files
def density_calculation_wr(wrapped_scif_path, wrapped_block_name, wrapped_cif_path):
    calc_cls = scif2class(wrapped_scif_path, wrapped_block_name)

    calc_obj = calc_cls.from_settings_cif(wrapped_scif_path, wrapped_block_name)

    atom_site_dict, cell_dict, _, _ = cif2dicts(wrapped_cif_path, wrapped_block_name, complete_dmin=False)

    density_path = calc_obj.calculate_density(atom_site_dict, cell_dict)

    with open('density_path.txt', 'w', encoding='UTF-8') as fobj:
        fobj.write(density_path)

def partition_wr(wrapped_scif_path, wrapped_block_name, wrapped_cif_path):
    calc_cls = scif2class(wrapped_scif_path, wrapped_block_name)

    calc_obj = calc_cls.from_settings_cif(wrapped_scif_path, wrapped_block_name)

    atom_site_dict, cell_dict, space_group_dict, refln_dict = cif2dicts(wrapped_cif_path, wrapped_block_name, complete_dmin=False)

    information_dict = json.load('wrapped_part_settings.json')
    atom_labels = information_dict['atom_labels']
    density_path = information_dict['density_path']

    f0js, charges = calc_obj.calc_f0j(atom_labels, atom_site_dict, cell_dict, space_group_dict, refln_dict, density_path)

    with open('density_path.txt', 'w', encoding='UTF-8') as fobj:
        fobj.write(density_path)

##########################################################################################
###              functions + strings for partition and density wrappers                ###
##########################################################################################

minimal_header = textwrap.dedent("""
    from qctbx.scaff import name2lcaodensity, name2reggriddensity, name2lcaopartition, name2reggridpartition
    from qctbx.io.cif import read_settings_cif,cif2dicts
""")

citations_header = minimal_header + 'import json \n'

check_available_footer = textwrap.dedent("""
    if __name__ == '__main__':
        wrapped_scif_path = '{dewrapped_scif_path}'
        wrapped_block_name = '{block_name}'
        check_available_wr(wrapped_scif_path, wrapped_block_name)
""")

citations_footer =  textwrap.dedent("""
    if __name__ == '__main__':
        wrapped_scif_path = '{dewrapped_scif_path}'
        wrapped_block_name = '{block_name}'
        citation_strings_wr(wrapped_scif_path, wrapped_block_name)
""")

def scif2class(wrapped_scif_path, wrapped_block_name):
    settings_cif = read_settings_cif(wrapped_scif_path, wrapped_block_name)
    if '_qctbx_reggridwfn_software' in settings_cif:
        calc_cls = name2reggriddensity(settings_cif['_qctbx_reggridwfn_software'])
    elif '_qctbx_lcaowfn_software' in settings_cif:
        calc_cls = name2lcaodensity(settings_cif['_qctbx_lcaowfn_software'])
    elif '_qctbx_reggridpartition_software' in settings_cif:
        calc_cls = name2reggridpartition(settings_cif['_qctbx_reggridpartition_software'])
    elif '_qctbx_lcaopartition_software' in settings_cif:
        calc_cls = name2lcaopartition(settings_cif['_qctbx_lcaopartition_software'])
    else:
        raise KeyError('Need either _qctbx_lcaowfn_software or _qctbx_reggridwfn_software in scif file.')
    return calc_cls

def check_available_wr(wrapped_scif_path, wrapped_block_name):
    calc_cls = scif2class(wrapped_scif_path, wrapped_block_name)

    calc_obj = calc_cls.from_settings_cif(wrapped_scif_path, wrapped_block_name)
    avail = calc_obj.check_availability()
    with open('wrapper_available.txt', 'w', encoding='UTF-8') as fobj:
        if avail:
            fobj.write('Y')
        else:
            fobj.write('N')

def citation_strings_wr(wrapped_scif_path, wrapped_block_name):
    calc_cls = scif2class(wrapped_scif_path, wrapped_block_name)

    calc_obj = calc_cls.from_settings_cif(wrapped_scif_path, wrapped_block_name)
    citation_strings = calc_obj.citation_strings()

    with open('wrapper_citations.json', 'w', encoding='UTF-8') as fobj:
        json.dump(citation_strings, fobj)

def partitioner_wrapper_factory(base_class):
    class PartitionerWrapper(base_class):
        calc_options = {}

        def __init__(self, *args, software=None, wrapped_object=None, **kwargs):
            self.software = software
            super().__init__(*args, **kwargs)
            if wrapped_object is not None:
                assert isinstance(wrapped_object, base_class), 'DensityPartitioner type needs to match the Wrapper type'
                for attr_name, attr_value in self.__dict__.items():
                    setattr(self, attr_name, attr_value)
            self.update_from_dict(defaults, update_if_present=False)

        @classmethod
        def from_settings_cif(cls, scif_path, block_name):
            new_obj = super().from_settings_cif(scif_path, block_name)
            if 'block_name' not in new_obj.calc_options:
                new_obj.calc_options = deepcopy(new_obj.calc_options)
                new_obj.calc_options['block_name'] = block_name
            settings_cif = read_settings_cif(scif_path, block_name)
            software_entry = settings_cif[f'{cls._cif_entry_start}software']
            new_obj.software = software_entry
            return new_obj

        def to_wrapped_settings_cif(self, cif_path, block_name):
            save_calc_options = deepcopy(self.calc_options)
            saved_software = copy(self.software)
            for option in ('run_command', 'dewrapped_scif_path', 'transfer_cif_path', 'block_name'):
                del self.calc_options[option]
            self.software = ':'.join(self.software.split(':')[1:])
            self.to_settings_cif(cif_path, block_name)
            self.calc_options = save_calc_options
            self.software = saved_software

        def check_availability(self):
            self.to_wrapped_settings_cif(self.calc_options['dewrapped_scif_path'], self.calc_options['block_name'])

            with open('wrapper_check_avail.py', 'w', encoding='UTF-8') as fobj:
                fobj.write(minimal_header)
                fobj.write(inspect.getsource(scif2class))
                fobj.write(inspect.getsource(check_available_wr))
                fobj.write(check_available_footer.format(**self.calc_options))

            subprocess.call(f"{self.calc_options['run_command']} wrapper_check_avail.py", shell=True)
            with open('wrapper_available.txt', 'r', encoding='UTF-8') as fobj:
                content = fobj.read()

            os.remove('wrapper_check_avail.py')
            os.remove('wrapper_available.txt')
            return content[0] == 'Y'

        def citation_strings(self):
            self.to_wrapped_settings_cif(self.calc_options['dewrapped_scif_path'], self.calc_options['block_name'])

            with open('wrapper_citations.py', 'w', encoding='UTF-8') as fobj:
                fobj.write(citations_header)
                fobj.write(inspect.getsource(scif2class))
                fobj.write(inspect.getsource(citation_strings_wr))
                fobj.write(citations_footer.format(**self.calc_options))

            subprocess.call(f"{self.calc_options['run_command']} wrapper_citations.py", shell=True)

            with open('wrapper_citations.json', 'r', encoding='UTF-8') as fobj:
                citations = json.load(fobj)
            os.remove('wrapper_citations.py')
            os.remove('wrapper_citations.json')
            return tuple(citations)

        def calc_f0j(
            self,
            atom_labels: List[int],
            atom_site_dict: Dict[str, List[Any]],
            cell_dict: Dict[str, Any],
            space_group_dict: Dict[str, Any],
            refln_dict: Dict[str, Any],
            density_path: Path
        ) -> np.ndarray:
            self.update_from_dict(defaults, update_if_present=False)
            write_minimal_cif(
                self.calc_options['transfer_cif_path'],
                cell_dict=cell_dict,
                space_group_dict=space_group_dict,
                atom_site_dict=atom_site_dict,
                refln_dict=refln_dict,
                block_name=self.calc_options['block_name']
            )
            outside_wrap_path_part, inside_wrap_path_part = self.calc_options['density_path_sub']

            with open('wrapped_part_settings.json', 'w', encoding='UTF-8') as fobj:
                json.dump({
                    'atom_labels': atom_labels,
                    'density_path': density_path.replace(outside_wrap_path_part, inside_wrap_path_part)
                }, fobj)

    return PartitionerWrapper

#TODO: Use calc_directory to its fullest!

def density_wrapper_factory(base_class):
    class DensityWrapper(base_class):
        calc_options = {}

        def __init__(self, *args, software=None, wrapped_object=None, **kwargs):
            self.software = software
            super().__init__(*args, **kwargs)
            if wrapped_object is not None:
                assert isinstance(wrapped_object, base_class), 'Densitycalculator type needs to match the Wrapper type'
                for attr_name, attr_value in self.__dict__.items():
                    setattr(self, attr_name, attr_value)
            self.update_from_dict(defaults, update_if_present=False)

        @classmethod
        def from_settings_cif(cls, scif_path, block_name):
            new_obj = super().from_settings_cif(scif_path, block_name)
            if 'block_name' not in new_obj.calc_options:
                new_obj.calc_options = deepcopy(new_obj.calc_options)
                new_obj.calc_options['block_name'] = block_name
            settings_cif = read_settings_cif(scif_path, block_name)
            software_entry = settings_cif[f'{cls._cif_entry_start}software']
            new_obj.software = software_entry
            return new_obj

        def to_wrapped_settings_cif(self, cif_path, block_name):
            save_calc_options = deepcopy(self.calc_options)
            saved_software = copy(self.software)
            for option in ('run_command', 'dewrapped_scif_path', 'transfer_cif_path', 'block_name'):
                del self.calc_options[option]
            self.software = ':'.join(self.software.split(':')[1:])
            self.to_settings_cif(cif_path, block_name)
            self.calc_options = save_calc_options
            self.software = saved_software

        def check_availability(self):
            self.to_wrapped_settings_cif(self.calc_options['dewrapped_scif_path'], self.calc_options['block_name'])

            with open('wrapper_check_avail.py', 'w', encoding='UTF-8') as fobj:
                fobj.write(minimal_header)
                fobj.write(inspect.getsource(scif2class))
                fobj.write(inspect.getsource(check_available_wr))
                fobj.write(check_available_footer.format(**self.calc_options))

            subprocess.call(f"{self.calc_options['run_command']} wrapper_check_avail.py", shell=True)
            with open('wrapper_available.txt', 'r', encoding='UTF-8') as fobj:
                content = fobj.read()

            os.remove('wrapper_check_avail.py')
            os.remove('wrapper_available.txt')
            return content[0] == 'Y'

        def citation_strings(self):
            self.to_wrapped_settings_cif(self.calc_options['dewrapped_scif_path'], self.calc_options['block_name'])

            with open('wrapper_citations.py', 'w', encoding='UTF-8') as fobj:
                fobj.write(citations_header)
                fobj.write(inspect.getsource(scif2class))
                fobj.write(inspect.getsource(citation_strings_wr))
                fobj.write(citations_footer.format(**self.calc_options))

            subprocess.call(f"{self.calc_options['run_command']} wrapper_citations.py", shell=True)

            with open('wrapper_citations.json', 'r', encoding='UTF-8') as fobj:
                citations = json.load(fobj)
            os.remove('wrapper_citations.py')
            os.remove('wrapper_citations.json')
            return tuple(citations)

        def calculate_density(
            self,
            atom_site_dict,
            cell_dict
        ):
            self.update_from_dict(defaults, update_if_present=False)
            write_minimal_cif(
                self.calc_options['transfer_cif_path'],
                cell_dict=cell_dict,
                atom_site_dict=atom_site_dict,
                block_name=self.calc_options['block_name']
            )
            self.to_wrapped_settings_cif(self.calc_options['dewrapped_scif_path'], self.calc_options['block_name'])

            density_footer = textwrap.dedent(f"""
                if __name__ == '__main__':
                    wrapped_cif_path = '{self.calc_options['transfer_cif_path']}'
                    wrapped_scif_path = '{self.calc_options['dewrapped_scif_path']}'
                    wrapped_block_name = '{self.calc_options['block_name']}'
                    density_calculation_wr(wrapped_scif_path, wrapped_block_name, wrapped_cif_path)
            """)

            with open('wrapper_density_calculation.py', 'w', encoding='UTF-8') as fobj:
                fobj.write(minimal_header)
                fobj.write(inspect.getsource(scif2class))
                fobj.write(inspect.getsource(density_calculation_wr))
                fobj.write(density_footer)

            subprocess.call(f"{self.calc_options['run_command']} wrapper_density_calculation.py", shell=True)
            with open('density_path.txt', 'r', encoding='UTF-8') as fobj:
                density_path = fobj.read().strip()
            os.remove('wrapper_density_calculation.py')
            os.remove('density_path.txt')
            outside_wrap_path_part, inside_wrap_path_part = self.calc_options['density_path_sub']
            return density_path.replace(inside_wrap_path_part, outside_wrap_path_part)
    return DensityWrapper

WrapperRegGridDensityCalculator = density_wrapper_factory(RegGridDensityCalculator)
WrapperLCAODensityCalculator = density_wrapper_factory(LCAODensityCalculator)
