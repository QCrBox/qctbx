import shutil
import subprocess
import platform
import pathlib
from .LCAODensityCalculatorBase import LCAODensityCalculator


calc_defaults = {
    'filebase': 'gaussian',
    'output_format': 'wfn',
    'title': 'qctbx calculation'
}

qm_defaults = {
    'method': 'PBE',
    'basis_set': 'def2-SVP',
    'multiplicity': 1,
    'charge': 0,
    'link0': {},
    'route_section': [],
    'appendix': ''
}


class GaussianDensityCalculator(LCAODensityCalculator):
    xyz_format = 'cartesian'
    provides_output = ('wfn', 'wfx')

    def __init__(self, *args, abs_g16_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        if abs_g16_path is not None:
            self.abs_g16_path = abs_g16_path
        elif platform.system() == 'Windows':
            self.abs_g16_path = shutil.which('g16.exe')
        elif platform.system() == 'Darwin':
            self.abs_g16_path = shutil.which('g16')
        else:
            # assume linux
            self.abs_g16_path = shutil.which('g16')

    def check_availability(self) -> bool:
        if self.abs_g16_path is not None:
            path = pathlib.Path(self.abs_g16_path)
            return path.exists()
        else:
            return False

    def calculate_density(self, elements, xyz, cluster_charge_dict={}):
        qm_options = qm_defaults.copy()
        qm_options.update(self.qm_options)

        calc_options = calc_defaults.copy()
        calc_options.update(self.calc_options)

        input_content = self._generate_gaussian_input(elements, xyz, qm_options, cluster_charge_dict)

        input_filename = f"{calc_options['filebase']}.com"
        with open(input_filename, 'w') as fo:
            fo.write(input_content)

        out_filename = f"{calc_options['filebase']}.log"
        #with open(out_filename, 'w') as fo:
        #    subprocess.call(
        #        [self.abs_g16_path, input_filename],
        #        stdout=fo,
        #        stderr=subprocess.STDOUT
        #    )

        format_standardise = calc_options['output_format'].lower().replace('.', '')
        if format_standardise == 'wfn':
            #subprocess.check_output(['formchk', f"{calc_options['filebase']}.chk", f"{calc_options['filebase']}.wfn"])
            return calc_options['filebase'] + '.wfn'
        elif format_standardise == 'wfx':
            #subprocess.check_output(['formchk', '-3', f"{calc_options['filebase']}.chk", f"{calc_options['filebase']}.wfx"])
            return calc_options['filebase'] + '.wfn'
        else:
            raise NotImplementedError('output_format from GaussianDensityCalculator is not implemented. Choose wfn or wfx')


    def _generate_gaussian_input(self, elements, xyz, qm_options, cluster_charge_dict):
        charge_mult = f"{qm_options['charge']} {qm_options['multiplicity']}"

        if len(cluster_charge_dict.get('charges', [])) > 0:
            ecp_lines = [f"{charge} {x} {y} {z}" for charge, (x, y, z) in zip(cluster_charge_dict['charges'], cluster_charge_dict['positions'])]
            ecp_section = "\n".join(ecp_lines)
            ecp_input = f"\n{ecp_section}\n"
        else:
            ecp_input = ""

        coordinates = [f"{element} {x} {y} {z}" for element, (x, y, z) in zip(elements, xyz)]
        coordinates_section = '\n'.join(coordinates)


        route_section = f"# {qm_options['method']} {qm_options['basis_set']} "
        for keyword in qm_options['keywords']:
            route_section += f"{keyword} "
        route_section += "Geom=Connectivity\n"

        chk_name = f"{calc_defaults['filebase']}.chk"
        qm_options['link0']['chk'] = chk_name
        link0_section = '\n'.join(f'%{key}={val}' for key, val in qm_options['link0'].items())

        input_header = f"{link0_section}\n{route_section}\n\n{calc_defaults['title']}\n\n{charge_mult}\n"

        gaussian_input = f"{input_header}{coordinates_section}{ecp_input}\n\n"
        return gaussian_input

    def cif_output(self) -> str:
        # TODO: Implement the logic to generate a CIF output from the calculation
        return 'Someone needs to implement this before production'
