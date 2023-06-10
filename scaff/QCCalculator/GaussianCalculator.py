from .BaseQCCalculators import LCAOQCCalculator
import platform
import shutil
import pathlib
import subprocess

class GaussianCalculator(LCAOQCCalculator):
    link0 = {}
    route_section = []
    appendix = ''

    def __init__(
        self,
        *args,
        gauss_path=None,
        link0 = {},
        route_section = [],
        appendix='',
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        if gauss_path is not None:
            self.gauss_path = gauss_path
        else:
            for year in range(30, 8, -1):
                if platform.system() == 'Windows' and pathlib.Path(f'g{year}.exe'.exists()):
                    self.gauss_path = f'g{year}.exe'
                    break
                elif pathlib.Path(f'g{year}'.exists()):
                    self.gauss_path = f'g{year}'
                    break
            else:
                self.gauss_path = None
        
        self.link0 = link0
        self.route_section = route_section
        self.appendix = appendix

    def check_availability(self) -> bool:
        if self.gauss_path is not None:
            path = pathlib.Path(self.gauss_path)
            return path.exists()
        else:
            return False
        

    def run_calculation(self):
        input_content = self._generate_gaussian_input()

        input_filename = f"{self.label}.com"
        with open(input_filename, 'w') as fo:
            fo.write(input_content)

        out_filename = f"{self.label}.log"
        with open(out_filename, 'w') as fo:
            subprocess.call(
                [self.abs_g16_path, input_filename],
                stdout=fo,
                stderr=subprocess.STDOUT
            )
        

    def _generate_gaussian_input(self):
        charge_mult = f"{self.charge} {self.multiplicity}"

        if self.cluster_charge_dict.get('charges', []) > 0:
            ecp_lines = [f"{charge} {x} {y} {z}" for charge, (x, y, z) in zip(self.cluster_charge_dict['charges'], self.cluster_charge_dict['positions'])]
            ecp_section = "\n".join(ecp_lines)
            ecp_input = f"\n{ecp_section}\n"
        else:
            ecp_input = ""

        coordinates = [f"{element} {x} {y} {z}" for element, (x, y, z) in zip(self.symbols, self.positions)]
        coordinates_section = '\n'.join(coordinates)

        route_section = f"# "
        for keyword in self.route_section:
            route_section += f"{keyword} "

        chk_name = f"{self.label}.chk"
        self.link0['chk'] = chk_name
        link0_section = '\n'.join(f'%{key}={val}' for key, val in self.link0.items())

        input_header = f"{link0_section}\n{route_section}\n\n{self.label}\n\n{charge_mult}\n"

        gaussian_input = f"{input_header}{coordinates_section}{ecp_input}\n\n"
        return gaussian_input

    def cif_output(self) -> str:
        # TODO: Implement the logic to generate a CIF output from the calculation
        return 'Someone needs to implement this before production'

    