
import inspect
import json
import os
import subprocess
import textwrap
from copy import deepcopy

from ..io.cif import cif2dicts, read_settings_cif
from ..io.minimal_files import write_minimal_cif
from . import name2lcaodensity, name2reggriddensity
from .LCAODensityCalculators.base import LCAODensityCalculator
from .RegGridDensityCalculators.base import RegGridDensityCalculator

defaults = {
    'calc_options': {
        'run_command': 'python',
        'dewrapped_scif_path' : 'dewrapped.scif',
        'transfer_cif_path' : 'wrapped_transfer.cif',
        'block_name': 'wrapped_qctbx'
    }
}

# functions for output .py files
def density_calculation_wr(wrapped_scif_path, wrapped_block_name, wrapped_cif_path):
    settings_cif = read_settings_cif(wrapped_scif_path, wrapped_block_name)
    if '_qctbx_reggridwfn_software' in settings_cif:
        calc_cls = name2reggriddensity(settings_cif['_qctbx_reggridwfn_software'])
    elif '_qctbx_lcaowfn_software' in settings_cif:
        calc_cls = name2lcaodensity(settings_cif['_qctbx_lcaowfn_software'])
    else:
        raise KeyError('Need either _qctbx_lcaowfn_software or _qctbx_reggridwfn_software in scif file.')

    calc_obj = calc_cls.from_settings_cif(wrapped_scif_path, wrapped_block_name)

    atom_site_dict, cell_dict, _, _ = cif2dicts(wrapped_cif_path, wrapped_block_name, complete_dmin=False)

    density_path = calc_obj.calculate_density(atom_site_dict, cell_dict)

    with open('density_path.txt', 'w', encoding='UTF-8') as fobj:
        fobj.write(density_path)

def check_available_wr(wrapped_scif_path, wrapped_block_name):
    settings_cif = read_settings_cif(wrapped_scif_path, wrapped_block_name)
    if '_qctbx_reggridwfn_software' in settings_cif:
        calc_cls = name2reggriddensity(settings_cif['_qctbx_reggridwfn_software'])
    elif '_qctbx_lcaowfn_software' in settings_cif:
        calc_cls = name2lcaodensity(settings_cif['_qctbx_lcaowfn_software'])
    else:
        raise KeyError('Need either _qctbx_lcaowfn_software or _qctbx_reggridwfn_software in scif file.')

    calc_obj = calc_cls.from_settings_cif(wrapped_scif_path, wrapped_block_name)
    avail = calc_obj.check_availability()
    with open('wrapper_available.txt', 'w', encoding='UTF-8') as fobj:
        if avail:
            fobj.write('Y')
        else:
            fobj.write('N')

def citation_strings_wr(wrapped_scif_path, wrapped_block_name):
    settings_cif = read_settings_cif(wrapped_scif_path, wrapped_block_name)
    if '_qctbx_reggridwfn_software' in settings_cif:
        calc_cls = name2reggriddensity(settings_cif['_qctbx_reggridwfn_software'])
    elif '_qctbx_lcaowfn_software' in settings_cif:
        calc_cls = name2lcaodensity(settings_cif['_qctbx_lcaowfn_software'])
    else:
        raise KeyError('Need either _qctbx_lcaowfn_software or _qctbx_reggridwfn_software in scif file.')

    calc_obj = calc_cls.from_settings_cif(wrapped_scif_path, wrapped_block_name)
    citation_strings = calc_obj.citation_strings()

    with open('wrapper_citations.json', 'w', encoding='UTF-8') as fobj:
        json.dump(citation_strings, fobj)


def density_wrapper_factory(base_class):
    class DensityWrapper(base_class):
        calc_options = {}

        def __init__(self, *args, software, wrapped_object=None, **kwargs):
            self.software = software
            super().__init__(*args, **kwargs)
            if wrapped_object is not None:
                assert isinstance(wrapped_object, base_class), 'Densitycalculator type needs to match the Wrapper type'
                for attr_name, attr_value in self.__dict__.items():
                    setattr(self, attr_name, attr_value)
            self.update_from_dict(defaults, update_if_present=False)

        def check_availability(self):
            self.to_settings_cif(self.calc_options['dewrapped_scif_path'], self.calc_options['block_name'])
            check_available_header = textwrap.dedent("""
                from qctbx.scaff import name2reggriddensity
                from qctbx.io.cif import read_settings_cif
            """)
            check_available_func_str = inspect.getsource(check_available_wr)

            check_available_footer = textwrap.dedent(f"""
                if __name__ == '__main__':
                    wrapped_scif_path = '{self.calc_options['dewrapped_scif_path']}'
                    wrapped_block_name = '{self.calc_options['block_name']}'
                    check_available_wr(wrapped_scif_path, wrapped_block_name)
            """)

            with open('wrapper_check_avail.py', 'w', encoding='UTF-8') as fobj:
                fobj.write(check_available_header)
                fobj.write(check_available_func_str)
                fobj.write(check_available_footer)

            subprocess.call(f"{self.run_command} wrapper_check_avail.py", shell=True)
            with open('wrapper_available.txt', 'r', encoding='UTF-8') as fobj:
                content = fobj.read()

            os.remove('wrapper_check_avail.py')
            os.remove('wrapper_available.txt')
            return content[0] == 'Y'

        def calculate_density(
            self,
            atom_site_dict,
            cell_dict
        ):
            write_minimal_cif(
                self.calc_options['transfer_cif_path'],
                cell_dict=cell_dict,
                atom_site_dict=atom_site_dict,
                block_name=self.calc_options['block_name']
            )
            self.to_settings_cif(self.calc_options['dewrapped_scif_path'], self.calc_options['block_name'])
            density_header = textwrap.dedent("""
                from qctbx.io.cif import read_settings_cif, cif2dicts
                from qctbx.scaff import name2reggriddensity
            """)
            density_func_str = inspect.getsource(density_calculation_wr)
            density_footer = textwrap.dedent(f"""
                if __name__ == '__main__':
                    wrapped_cif_path = '{self.calc_options['transfer_cif_path']}'
                    wrapped_scif_path = '{self.calc_options['dewrapped_scif_path']}'
                    wrapped_block_name = '{self.calc_options['block_name']}'
                    density_calculation_wr(wrapped_scif_path, wrapped_block_name, wrapped_cif_path)
            """)

            with open('wrapper_density_calculation.py', 'w', encoding='UTF-8') as fobj:
                fobj.write(density_header)
                fobj.write(density_func_str)
                fobj.write(density_footer)

            subprocess.call(f"{self.run_command} wrapper_density_calculation.py", shell=True)
            with open('density_path.txt', 'r', encoding='UTF-8') as fobj:
                density_path = fobj.read().strip()
            os.remove('wrapper_density_calculation.py')
            os.remove('density_path.txt')
            return density_path

        def to_settings_cif(self, cif_path, block_name):
            save_calc_options = deepcopy(self.calc_options)
            for option in ('run_command', 'dewrapped_scif_path', 'transfer_cif_path', 'block_name'):
                del self.calc_options[option]
            super().to_settings_cif(cif_path, block_name)
            self.calc_options = save_calc_options

        def citation_strings(self):
            self.to_settings_cif(self.calc_options['dewrapped_scif_path'], self.calc_options['block_name'])
            citations_header = textwrap.dedent("""
                import json
                from qctbx.scaff import name2reggriddensity
                from qctbx.io.cif import read_settings_cif
            """)

            citations_func_str = inspect.getsource(citation_strings_wr)

            citations_footer =  textwrap.dedent(f"""
                if __name__ == '__main__':
                    wrapped_scif_path = '{self.calc_options['dewrapped_scif_path']}'
                    wrapped_block_name = '{self.calc_options['block_name']}'
                    citation_strings_wr(wrapped_scif_path, wrapped_block_name)
            """)
            with open('wrapper_citations.py', 'w', encoding='UTF-8') as fobj:
                fobj.write(citations_header)
                fobj.write(citations_func_str)
                fobj.write(citations_footer)

            subprocess.call(f"{self.run_command} wrapper_citations.py", shell=True)

            with open('wrapper_citations.json', 'r', encoding='UTF-8') as fobj:
                citations = json.load(fobj)
            os.remove('wrapper_citations.py')
            os.remove('wrapper_citations.json')
            return tuple(citations)

        @classmethod
        def from_settings_cif(cls, scif_path, block_name):
            new_obj = super().from_settings_cif(scif_path, block_name)
            if 'block_name' not in new_obj.calc_options:
                new_obj.calc_options = deepcopy(new_obj.calc_options)
                new_obj.calc_options['block_name'] = block_name
            settings_cif = read_settings_cif(scif_path, block_name)
            software_entry = settings_cif[f'{cls._cif_entry_start}software']
            assert software_entry.lower().startswith('wrapper'), f'{cls._cif_entry_start}software does not start with a wrapper entry and a semicolon'
            try:
                new_obj.software = ':'.join(software_entry.split(':')[1:])
            except IndexError as exc:
                raise IndexError(f'{cls._cif_entry_start}software does not start with a wrapper entry and a semicolon' ) from exc
            assert len(new_obj.software) > 0, f'{cls._cif_entry_start}software does not start with a wrapper entry and a semicolon'
            return new_obj
    return DensityWrapper

WrapperRegGridDensityCalculator = density_wrapper_factory(RegGridDensityCalculator)
WrapperLCAODensityCalculator = density_wrapper_factory(LCAODensityCalculator)
