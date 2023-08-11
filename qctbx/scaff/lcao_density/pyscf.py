import os
import textwrap
from typing import Dict, List, Union

import numpy as np

from ...conversions import add_cart_pos
from .base import LCAODensityCalculator

try:
    from pyscf import dft, gto, lib, qmmm, scf
    from pyscf.lib.misc import num_threads
    from pyscf.x2c import x2c
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

pyscf_bibtex = textwrap.dedent(r"""
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

TYPE_MAP = [
    [1],  # S
    [2, 3, 4],  # P
    [5, 8, 9, 6, 10, 7],  # D
    [11,14,15,17,20,18,12,16,19,13],  # F
    [21,24,25,30,33,31,26,34,35,28,22,27,32,29,23],  # G
    [56,55,54,53,52,51,50,49,48,47,46,45,44,43,42,41,40,39,38,37,36],  # H
]

def write_wfn(fobj, mol, mo_coeff, mo_energy, mo_occ, tot_ener):
    """
    NoSpherA2 function for writing a wfn file even when the calculation is not closed shell
    """

    mol, ctr = x2c._uncontract_mol(mol, True, 0.)
    mo_coeff = np.dot(ctr, mo_coeff)

    nmo = mo_coeff.shape[1]
    mo_cart = []
    centers = []
    types = []
    exps = []
    p0 = 0
    for ib in range(mol.nbas):
        ia = mol.bas_atom(ib)
        l = mol.bas_angular(ib)
        es = mol.bas_exp(ib)
        c = mol._libcint_ctr_coeff(ib)
        n_p, n_c = c.shape
        nd = n_c*(2*l+1)
        mosub = mo_coeff[p0:p0+nd].reshape(-1,n_c,nmo)
        c2s = gto.cart2sph(l)
        mosub = np.einsum('yki,cy,pk->pci', mosub, c2s, c)
        mo_cart.append(mosub.transpose(1,0,2).reshape(-1,nmo))

        for t in TYPE_MAP[l]:
            types.append([t]*n_p)
        ncart = mol.bas_len_cart(ib)
        exps.extend([es]*ncart)
        centers.extend([ia+1]*(n_p*ncart))
        p0 += nd
    mo_cart = np.vstack(mo_cart)
    centers = np.hstack(centers)
    types = np.hstack(types)
    exps = np.hstack(exps)
    nprim, nmo = mo_cart.shape

    fobj.write('From PySCF\n')
    fobj.write('GAUSSIAN %14d MOL ORBITALS %6d PRIMITIVES %8d NUCLEI\n'%(mo_cart.shape[1], mo_cart.shape[0], mol.natm))
    for ia in range(mol.natm):
        x, y, z = mol.atom_coord(ia)
        fobj.write('%3s%8d (CENTRE%3d) %12.8f%12.8f%12.8f  CHARGE = %4.1f\n'%(mol.atom_pure_symbol(ia), ia+1, ia+1, x, y, z, mol.atom_charge(ia)))
    for i0, i1 in lib.prange(0, nprim, 20):
        fobj.write('CENTRE ASSIGNMENTS  %s\n'% ''.join('%3d'%x for x in centers[i0:i1]))
    for i0, i1 in lib.prange(0, nprim, 20):
        fobj.write('TYPE ASSIGNMENTS    %s\n'% ''.join('%3d'%x for x in types[i0:i1]))
    for i0, i1 in lib.prange(0, nprim, 5):
        fobj.write('EXPONENTS  %s\n'% ' '.join('%13.7E'%x for x in exps[i0:i1]))

    for k in range(nmo):
        mo = mo_cart[:,k]
        fobj.write('MO  %-12d          OCC NO = %12.8f ORB. ENERGY = %12.8f\n'%(k+1, mo_occ[k], mo_energy[k]))
        for i0, i1 in lib.prange(0, nprim, 5):
            fobj.write(' %s\n' % ' '.join('%15.8E'%x for x in mo[i0:i1]))
    fobj.write('END DATA\n')
    fobj.write(' THE SCF ENERGY =%20.12f THE VIRIAL(-V/T)=   0.00000000\n'%tot_ener)


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
        mol.output = os.path.join(self.calc_options['work_directory'], self.calc_options['label'] + '.log')
        mol.verbose = 4
        zipped = (atom_site_dict_cartn[f'_atom_site_{col}'] for col in ('type_symbol', 'Cartn_x', 'Cartn_y', 'Cartn_z'))
        mol.atom = [[elem, (x, y, z)] for elem, x, y, z in zip(*zipped)]
        mol.basis = self.basisset
        mol.multiplicity = self.multiplicity
        mol.charge = self.charge
        mol.max_memory = self.calc_options['ram_mb']
        mol.build()

        rks = dft.RKS(mol)
        rks.xc = self.method
        rks = rks.density_fit()
        rks.grids.radi_method = dft.gauss_chebyshev
        rks.grids.level = 0
        rks.with_df.auxbasis = 'def2-tzvp-jkfit'
        rks.diis_space = 19
        rks.conv_tol = 0.0033
        rks.conv_tol_grad = 1e-2
        rks.level_shift = 0.25
        rks.damp = 0.600000
        if cluster_charge_dict is not None:
            coords = [tuple(xyz) for xyz in cluster_charge_dict['positions_cart']]
            charges = cluster_charge_dict['charges']
            used = qmmm.mm_charge(rks, coords, charges)
        else:
            used = rks

        used.kernel()

        rks.conv_tol = 1e-9
        rks.conv_tol_grad = 1e-5
        rks.level_shift = 0.0
        rks.damp = 0.0
        rks = scf.newton(rks)
        if cluster_charge_dict is not None:
            coords = [tuple(xyz) for xyz in cluster_charge_dict['positions_cart']]
            charges = cluster_charge_dict['charges']
            used = qmmm.mm_charge(rks, coords, charges)
        else:
            used = rks
        used.kernel()
        wfn_path = os.path.join(self.calc_options['work_directory'], self.calc_options['label'] + '.wfn')
        with open(wfn_path, 'w', encoding='UTF-8') as fobj:
            write_wfn(fobj, mol, rks.mo_coeff, rks.mo_energy, rks.mo_occ, rks.e_tot)

        return wfn_path

    def citation_strings(self):
        software_name = 'PySCF'
        self.update_from_dict(defaults, update_if_present=False)
        return self.generate_description(software_name, pyscf_bibtex_keys, pyscf_bibtex)