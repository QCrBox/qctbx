from .base import LCAOWrapper
import platform
import shutil
import pathlib
import subprocess
import textwrap
import warnings
from ..util import dict_merge


class GaussianWrapper(LCAOWrapper):
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
        raise NotImplementedError('This Calculator is not really testd at all and therefore should be seen as a stub for development')
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

        input_path = f"{self.label}.com"
        with open(input_path, 'w') as fobj:
            fobj.write(input_content)

        out_path = f"{self.label}.log"
        with open(out_path, 'w') as fobj:
            subprocess.call(
                [self.abs_g16_path, input_path],
                stdout=fobj,
                stderr=subprocess.STDOUT
            )


    def _generate_gaussian_input(self):
        charge_mult = f"{self.charge} {self.multiplicity}"

        if self.cluster_charge_dict.get('charges', []) > 0:
            ecp_lines = [f"{charge} {x} {y} {z}" for charge, (x, y, z) in zip(self.cluster_charge_dict['charges'], self.cluster_charge_dict['positions_cart'])]
            ecp_section = "\n".join(ecp_lines)
            ecp_input = f"\n{ecp_section}\n"
        else:
            ecp_input = ""

        coordinates = [f"{element} {x} {y} {z}" for element, (x, y, z) in zip(self.symbols, self.positions_cart)]
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

    def bibtex_strings(self) -> str:
        # TODO make this version agnostic
        if self.gauss_path.endswith('.exe'):
            gname = self.gauss_path[:-4].split('/')[-1].splt('\\')[-1].lower()
        else:
            gname = self.gauss_path.split('/')[-1].splt('\\')[-1].lower()
        if gname == 'g16':
            return 'g16', textwrap.dedent("""
                @misc{g16,
                    author={M. J. Frisch and G. W. Trucks and H. B. Schlegel and G. E. Scuseria and M. A. Robb and J. R. Cheeseman and G. Scalmani and V. Barone and G. A. Petersson and H. Nakatsuji and X. Li and M. Caricato and A. V. Marenich and J. Bloino and B. G. Janesko and R. Gomperts and B. Mennucci and H. P. Hratchian and J. V. Ortiz and A. F. Izmaylov and J. L. Sonnenberg and D. Williams-Young and F. Ding and F. Lipparini and F. Egidi and J. Goings and B. Peng and A. Petrone and T. Henderson and D. Ranasinghe and V. G. Zakrzewski and J. Gao and N. Rega and G. Zheng and W. Liang and M. Hada and M. Ehara and K. Toyota and R. Fukuda and J. Hasegawa and M. Ishida and T. Nakajima and Y. Honda and O. Kitao and H. Nakai and T. Vreven and K. Throssell and Montgomery, {Jr.}, J. A. and J. E. Peralta and F. Ogliaro and M. J. Bearpark and J. J. Heyd and E. N. Brothers and K. N. Kudin and V. N. Staroverov and T. A. Keith and R. Kobayashi and J. Normand and K. Raghavachari and A. P. Rendell and J. C. Burant and S. S. Iyengar and J. Tomasi and M. Cossi and J. M. Millam and M. Klene and C. Adamo and R. Cammi and J. W. Ochterski and R. L. Martin and K. Morokuma and O. Farkas and J. B. Foresman and D. J. Fox},
                    title={Gaussian˜16},
                    year={2016},
                    note={Gaussian Inc. Wallingford CT}
                }""")
        elif gname == 'g09':
            return 'g09', textwrap.dedent("""
                @misc{g09,
                    author={M. J. Frisch and G. W. Trucks and H. B. Schlegel and G. E. Scuseria and M. A. Robb and J. R. Cheeseman and G. Scalmani and V. Barone and G. A. Petersson and H. Nakatsuji and X. Li and M. Caricato and A. V. Marenich and J. Bloino and B. G. Janesko and R. Gomperts and B. Mennucci and H. P. Hratchian and J. V. Ortiz and A. F. Izmaylov and J. L. Sonnenberg and D. Williams-Young and F. Ding and F. Lipparini and F. Egidi and J. Goings and B. Peng and A. Petrone and T. Henderson and D. Ranasinghe and V. G. Zakrzewski and J. Gao and N. Rega and G. Zheng and W. Liang and M. Hada and M. Ehara and K. Toyota and R. Fukuda and J. Hasegawa and M. Ishida and T. Nakajima and Y. Honda and O. Kitao and H. Nakai and T. Vreven and K. Throssell and Montgomery, {Jr.}, J. A. and J. E. Peralta and F. Ogliaro and M. J. Bearpark and J. J. Heyd and E. N. Brothers and K. N. Kudin and V. N. Staroverov and T. A. Keith and R. Kobayashi and J. Normand and K. Raghavachari and A. P. Rendell and J. C. Burant and S. S. Iyengar and J. Tomasi and M. Cossi and J. M. Millam and M. Klene and C. Adamo and R. Cammi and J. W. Ochterski and R. L. Martin and K. Morokuma and O. Farkas and J. B. Foresman and D. J. Fox},
                    title={Gaussian˜09},
                    year={2009},
                    note={Gaussian Inc. Wallingford CT}
                }""")
        else:
            warnings.warn('Bibtex string for this Gaussian Version not implemented, you need to add the version manually')
            return gname, ''

