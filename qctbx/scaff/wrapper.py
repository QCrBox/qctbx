import json
import os
import subprocess
import shlex
from copy import deepcopy
from typing import Any, Dict, List

import numpy as np

from ..custom_typing import Path
from ..io.cif import read_settings_cif
from ..io.tsc import TSCBFile
from ..io.minimal_files import write_minimal_cif
from .lcao_density.base import LCAODensityCalculator
from .reggr_density.base import RegGridDensityCalculator
from .lcao_partition.base import LCAODensityPartitioner
from .reggr_partition.base import RegGridDensityPartitioner

defaults = {
    'calc_options': {
        'run_command': 'python -m',
        'base_name': 'qctbx_wrapper',
        'block_name': 'wrapped_qctbx',
        'calculation_dir' : '.',
        'inwrapped_calculation_dir': '.'
    }
}

def partitioner_wrapper_factory(base_class):
    class PartitionerWrapper(base_class):
        calc_options = {}

        def __init__(self, *args, software=None, wrapped_object=None, **kwargs):
            self.software = software
            super().__init__(*args, **kwargs)
            if wrapped_object is not None:
                assert isinstance(wrapped_object, base_class), 'DensityPartitioner type needs to match the Wrapper type'
                for attr_name, attr_value in self.__dict__.items():
                    if attr_name == 'software':
                        self.dewrapped_software = attr_value
                    else:
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

        @property
        def dewrapped_software(self):
            return ':'.join(self.software.split(':')[1:])

        @dewrapped_software.setter
        def dewrapped_software(self, value):
            self.software = 'wrapper:' + value

        def to_wrapped_settings_cif(self, cif_path, block_name):
            save_calc_options = deepcopy(self.calc_options)
            saved_software = self.software
            for option in defaults['calc_options'].keys():
                del self.calc_options[option]
            self.software = self.dewrapped_software
            self.to_settings_cif(cif_path, block_name)
            self.calc_options = save_calc_options
            self.software = saved_software

        def check_availability(self):
            scif_path = self.calc_options['base_name'] + '.scif'
            json_path = self.calc_options['base_name'] + '.json'
            calc_dir = self.calc_options['calculation_dir']
            inwr_calc_dir = self.calc_options['inwrapped_calculation_dir']
            block_name = self.calc_options['block_name']
            self.to_wrapped_settings_cif(os.path.join(calc_dir, scif_path), block_name)

            r = subprocess.call([
                *self.calc_options['run_command'].split(), 'qctbx.scaff', 'available',
                '--scif_path', os.path.join(inwr_calc_dir, scif_path),
                '--block_name', self.calc_options['block_name'],
                '--output_json', os.path.join(inwr_calc_dir, json_path)
            ])
            assert r == 0, 'Failed subprocess call in check_availability'

            with open(os.path.join(calc_dir, json_path), 'r', encoding='UTF-8') as fobj:
                check_dict = json.load(fobj)

            calc_type = self._cif_entry_start.split('_')[2]

            return check_dict[f'{calc_type},{self.dewrapped_software}']

        def citation_strings(self):
            self.update_from_dict(defaults, update_if_present=False)

            scif_path = self.calc_options['base_name'] + '.scif'
            json_path = self.calc_options['base_name'] + '.json'
            calc_dir = self.calc_options['calculation_dir']
            inwr_calc_dir = self.calc_options['inwrapped_calculation_dir']
            block_name = self.calc_options['block_name']
            self.to_wrapped_settings_cif(os.path.join(calc_dir, scif_path), block_name)

            r = subprocess.call([
                *self.calc_options['run_command'].split(), 'qctbx.scaff', 'citation',
                '--scif_path', os.path.join(inwr_calc_dir, scif_path),
                '--block_name', self.calc_options['block_name'],
                '--output_json', os.path.join(inwr_calc_dir, json_path)
            ])
            assert r == 0, 'Failed subprocess call in check_availability'

            with open(os.path.join(calc_dir, json_path), 'r', encoding='UTF-8') as fobj:
                citation_dict = json.load(fobj)
            return citation_dict['description'], citation_dict['bibtex']

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
            cif_path = self.calc_options['base_name'] + '.cif'
            scif_path = self.calc_options['base_name'] + '.scif'
            tsc_path = self.calc_options['base_name'] + '.tscb'
            json_path = self.calc_options['base_name'] + '.json'
            calc_dir = self.calc_options['calculation_dir']
            block_name = self.calc_options['block_name']
            inwr_calc_dir = self.calc_options['inwrapped_calculation_dir']
            self.to_wrapped_settings_cif(os.path.join(calc_dir, scif_path), block_name)

            density_abs_path = os.path.abspath(density_path)
            calc_dir_abs_path = os.path.abspath(calc_dir)
            if density_abs_path.startswith(calc_dir_abs_path):
                cut_path = density_abs_path[len(calc_dir_abs_path) + 1:]
                inwr_density_path = os.path.join(inwr_calc_dir, cut_path)
            else:
                inwr_density_path = os.path.join(inwr_calc_dir, density_path)

            write_minimal_cif(
                os.path.join(calc_dir, cif_path),
                cell_dict=cell_dict,
                space_group_dict=space_group_dict,
                atom_site_dict=atom_site_dict,
                refln_dict=refln_dict,
                block_name=block_name
            )
            cmd_list = shlex.split(self.calc_options['run_command'])

            process = subprocess.run(
                [
                    *cmd_list, 'qctbx.scaff', 'partition',
                    '--cif_path',  os.path.join(inwr_calc_dir, cif_path),
                    '--scif_path', os.path.join(inwr_calc_dir, scif_path),
                    '--input_wfn_path', inwr_density_path,
                    '--atom_labels', *atom_labels,
                    '--block_name', self.calc_options['block_name'],
                    '--tsc_path', os.path.join(inwr_calc_dir, tsc_path),
                    '--charge_json', os.path.join(inwr_calc_dir, json_path)
                ],
                text=True,
                capture_output=True,
                check=False
            )

            if process.returncode != 0:
                raise RuntimeError(
                    f'Error in subprocess partition runtime.\nSTDERR:\n{process.stderr}'
                    + f'\n\nSTDOUT:\n{process.stdout}')

            tsc = TSCBFile.from_file(os.path.join(calc_dir, tsc_path))

            with open(os.path.join(calc_dir, json_path), 'r', encoding='UTF-8') as fobj:
                charges_dict = json.load(fobj)

            charges = [charges_dict[atom_name] for atom_name in atom_labels]

            return np.array(list(tsc.data.values())).T, charges

    return PartitionerWrapper

def density_wrapper_factory(base_class):
    class DensityWrapper(base_class):
        calc_options = {}

        def __init__(self, *args, software=None, wrapped_object=None, **kwargs):
            self.software = software
            super().__init__(*args, **kwargs)
            if wrapped_object is not None:
                assert isinstance(wrapped_object, base_class), 'DensityPartitioner type needs to match the Wrapper type'
                for attr_name, attr_value in self.__dict__.items():
                    if attr_name == 'software':
                        self.dewrapped_software = attr_value
                    else:
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

        @property
        def dewrapped_software(self):
            return ':'.join(self.software.split(':')[1:])

        @dewrapped_software.setter
        def dewrapped_software(self, value):
            self.software = 'wrapper:' + value

        def to_wrapped_settings_cif(self, cif_path, block_name):
            save_calc_options = deepcopy(self.calc_options)
            saved_software = self.software
            for option in defaults['calc_options'].keys():
                del self.calc_options[option]
            self.software = self.dewrapped_software
            self.to_settings_cif(cif_path, block_name)
            self.calc_options = save_calc_options
            self.software = saved_software

        def check_availability(self):
            scif_path = self.calc_options['base_name'] + '.scif'
            json_path = self.calc_options['base_name'] + '.json'
            calc_dir = self.calc_options['calculation_dir']
            inwr_calc_dir = self.calc_options['inwrapped_calculation_dir']
            block_name = self.calc_options['block_name']
            self.to_wrapped_settings_cif(os.path.join(calc_dir, scif_path), block_name)

            r = subprocess.call([
                *self.calc_options['run_command'].split(), 'qctbx.scaff', 'available',
                '--scif_path', os.path.join(inwr_calc_dir, scif_path),
                '--block_name', self.calc_options['block_name'],
                '--output_json', os.path.join(inwr_calc_dir, json_path)
            ])
            assert r == 0, 'Failed subprocess call in check_availability'

            with open(os.path.join(calc_dir, json_path), 'r', encoding='UTF-8') as fobj:
                check_dict = json.load(fobj)

            calc_type = self._cif_entry_start.split('_')[2]

            return check_dict[f'{calc_type},{self.dewrapped_software}']

        def citation_strings(self):
            self.update_from_dict(defaults, update_if_present=False)

            scif_path = self.calc_options['base_name'] + '.scif'
            json_path = self.calc_options['base_name'] + '.json'
            calc_dir = self.calc_options['calculation_dir']
            inwr_calc_dir = self.calc_options['inwrapped_calculation_dir']
            block_name = self.calc_options['block_name']
            self.to_wrapped_settings_cif(os.path.join(calc_dir, scif_path), block_name)

            r = subprocess.call([
                *self.calc_options['run_command'].split(), 'qctbx.scaff', 'citation',
                '--scif_path', os.path.join(inwr_calc_dir, scif_path),
                '--block_name', self.calc_options['block_name'],
                '--output_json', os.path.join(inwr_calc_dir, json_path)
            ])
            assert r == 0, 'Failed subprocess call in check_availability'

            with open(os.path.join(calc_dir, json_path), 'r', encoding='UTF-8') as fobj:
                citation_dict = json.load(fobj)
            return citation_dict['description'], citation_dict['bibtex']

        def calculate_density(
            self,
            atom_site_dict,
            cell_dict
        ):
            self.update_from_dict(defaults, update_if_present=False)
            cif_path = self.calc_options['base_name'] + '.cif'
            scif_path = self.calc_options['base_name'] + '.scif'
            text_path = self.calc_options['base_name'] + '.txt'
            calc_dir = self.calc_options['calculation_dir']
            inwr_calc_dir = self.calc_options['inwrapped_calculation_dir']
            self.to_wrapped_settings_cif(os.path.join(calc_dir, scif_path), self.calc_options['block_name'])


            write_minimal_cif(
                os.path.join(calc_dir, cif_path),
                cell_dict=cell_dict,
                atom_site_dict=atom_site_dict,
                block_name=self.calc_options['block_name']
            )

            r = subprocess.call([
                *self.calc_options['run_command'].split(), 'qctbx.scaff', 'density',
                '--cif_path',  os.path.join(inwr_calc_dir, cif_path),
                '--scif_path', os.path.join(inwr_calc_dir, scif_path),
                '--block_name', self.calc_options['block_name'],
                '--text_output_path', os.path.join(inwr_calc_dir, text_path)
            ])
            assert r == 0, 'Failed subprocess call in density'


            with open(os.path.join(calc_dir, text_path), 'r', encoding='UTF-8') as fobj:
                density_path = fobj.read().strip()
            return density_path.replace(inwr_calc_dir, calc_dir)
    return DensityWrapper

WrapperRegGridDensityCalculator = density_wrapper_factory(RegGridDensityCalculator)
WrapperLCAODensityCalculator = density_wrapper_factory(LCAODensityCalculator)

WrapperRegGridDensityPartitioner = partitioner_wrapper_factory(RegGridDensityPartitioner)
WrapperLCAODensityPartitioner = partitioner_wrapper_factory(LCAODensityPartitioner)
