from .base import LCAODensityCalculator

from ...conversions import cell_dict2atom_sites_dict, add_cart_pos

import os
import textwrap
from typing import List, Dict, Union

try:
    from pyscf import gto, dft, qmmm
    from pyscf.tools import wfn_format
    from pyscf.lib.misc import num_threads
except ImportError:
    _pyscf_imported = False
else:
    _pyscf_imported = True

defaults = {
    'method': 'PBE',
    'basisset': 'def2-SVP',
    'charge': 0,
    'multiplicity': 1,
    'specific_options': {},
    'calc_options': {
        'label': 'pyscf',
        'work_directory': '.',
        'output_format': 'wfn',
        'ram_mb': 2000,
        'cpu_count': 1
    }
}

pyscf_bibtex_keys = 'PySCF1,PySCF2'

pyscf_bibtex = textwrap.dedent("""
    @article{PySCF1,
        author = {Sun, Qiming and Berkelbach, Timothy C. and Blunt, Nick S. and Booth, George H. and Guo, Sheng and Li, Zhendong and Liu, Junzi and McClain, James D. and Sayfutyarova, Elvira R. and Sharma, Sandeep and Wouters, Sebastian and Chan, Garnet Kin-Lic},
        title = {PySCF: the Python-based simulations of chemistry framework},
        journal = {WIREs Computational Molecular Science},
        volume = {8},
        number = {1},
        pages = {e1340},
        doi = {https://doi.org/10.1002/wcms.1340},
        url = {https://wires.onlinelibrary.wiley.com/doi/abs/10.1002/wcms.1340},
        eprint = {https://wires.onlinelibrary.wiley.com/doi/pdf/10.1002/wcms.1340},
        year = {2018}
    }

    @article{PySCF2,
        author = {Sun, Qiming and Zhang, Xing and Banerjee, Samragni and Bao, Peng and Barbry, Marc and Blunt, Nick S. and Bogdanov, Nikolay A. and Booth, George H. and Chen, Jia and Cui, Zhi-Hao and Eriksen, Janus J. and Gao, Yang and Guo, Sheng and Hermann, Jan and Hermes, Matthew R. and Koh, Kevin and Koval, Peter and Lehtola, Susi and Li, Zhendong and Liu, Junzi and Mardirossian, Narbe and McClain, James D. and Motta, Mario and Mussard, Bastien and Pham, Hung Q. and Pulkin, Artem and Purwanto, Wirawan and Robinson, Paul J. and Ronca, Enrico and Sayfutyarova, Elvira R. and Scheurer, Maximilian and Schurkus, Henry F. and Smith, James E. T. and Sun, Chong and Sun, Shi-Ning and Upadhyay, Shiv and Wagner, Lucas K. and Wang, Xiao and White, Alec and Whitfield, James Daniel and Williamson, Mark J. and Wouters, Sebastian and Yang, Jun and Yu, Jason M. and Zhu, Tianyu and Berkelbach, Timothy C. and Sharma, Sandeep and Sokolov, Alexander Yu. and Chan, Garnet Kin-Lic},
        title = "{Recent developments in the PySCF program package}",
        journal = {The Journal of Chemical Physics},
        volume = {153},
        number = {2},
        pages = {024109},
        year = {2020},
        month = {07},
        issn = {0021-9606},
        doi = {10.1063/5.0006074},
        url = {https://doi.org/10.1063/5.0006074},
        eprint = {https://pubs.aip.org/aip/jcp/article-pdf/doi/10.1063/5.0006074/16722275/024109\_1\_online.pdf},
    }
""").strip()

class PyScfLCAOCalculator(LCAODensityCalculator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.update_from_dict(defaults, update_if_present=False)


    def check_availability(self) -> bool:
        return _pyscf_imported
    
    def calculate_density(
        self,
        atom_site_dict: Dict[str, Union[float, str]],
        cell_dict: Dict[str, float],
        cluster_charge_dict: Dict[str, List[float]]=None
    ):
        self.update_from_dict(defaults, update_if_present=False)
        atom_site_dict_cartn, _ = add_cart_pos(atom_site_dict, cell_dict)

        num_threads(n=self.calc_options['cpu_count'])

        mol = gto.Mole()
        zipped = (atom_site_dict_cartn[f'_atom_site_{col}'] for col in ('type_symbol', 'Cartn_x', 'Cartn_y', 'Cartn_z'))
        mol.atom = [[elem, (x, y, z)] for elem, x, y, z in zip(*zipped)]
        mol.basis = self.basisset
        mol.multiplicity = self.multiplicity
        mol.charge = self.charge
        mol.max_memory = self.calc_options['ram_mb']
        mol.build()

        rks = dft.RKS(mol)
        rks.xc = self.method
        if cluster_charge_dict is not None:
            coords = [tuple(xyz) for xyz in cluster_charge_dict['positions_cart']]
            charges = cluster_charge_dict['charges']
            mf = qmmm.mm_charge(rks, coords, charges)
            mf.kernel()
        else:
            rks.kernel()
        wfn_path = os.path.join(self.calc_options['work_directory'], self.calc_options['label'] + '.wfn')
        with open(wfn_path, 'w', encoding='UTF-8') as fobj:
            wfn_format.write_mo(fobj, mol, rks.mo_coeff, rks.mo_energy, rks.mo_occ)

        return wfn_path

    def citation_strings(self):
        software_name = 'PySCF'
        self.update_from_dict(defaults, update_if_present=False)
        return self.generate_description(software_name, pyscf_bibtex_keys, pyscf_bibtex)