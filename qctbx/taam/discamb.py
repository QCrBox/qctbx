from typing import Any, Dict, List
from ..F0jSourceBase import F0jSource
import os
from copy import deepcopy
import numpy as np
import subprocess
from ..conversions import symm_mat_vec2str, symm_to_matrix_vector, cell_dict2atom_sites_dict
from ..io.minimal_files import write_minimal_cif, write_mock_hkl
from ..io.tsc import TSCFile

bibtex_key = 'MATTS,DiSCaMB,iotbxcif'

bibtex_entry = """
@article{MATTS,
	title = {Extension of the transferable aspherical pseudoatom data bank for the comparison of molecular electrostatic potentials in structure–activity studies},
	volume = {75},
	issn = {2053-2733},
	url = {https://scripts.iucr.org/cgi-bin/paper?S2053273319000482},
	doi = {10.1107/S2053273319000482},
	abstract = {The transferable aspherical pseudoatom data bank, UBDB2018, is extended with over 130 new atom types present in small and biological molecules of great importance in biology and chemistry. UBDB2018 can be applied either as a source of aspherical atomic scattering factors in a standard X-ray experiment (
              d
              min
              ≃ 0.8 Å) instead of the independent atom model (IAM), and can therefore enhance the final crystal structure geometry and refinement parameters; or as a tool to reconstruct the molecular charge-density distribution and derive the electrostatic properties of chemical systems for which 3D structural data are available. The extended data bank has been extensively tested, with the focus being on the accuracy of the molecular electrostatic potential computed for important drug-like molecules, namely the HIV-1 protease inhibitors. The UBDB allows the reconstruction of the reference B3LYP/6-31G** potentials, with a root-mean-squared error of 0.015 e bohr
              −1
              computed for entire potential grids which span values from
              ca
              200 e bohr
              −1
              to
              ca
              −0.1 e bohr
              −1
              and encompass both the inside and outside regions of a molecule. UBDB2018 is shown to be applicable to enhancing the physical meaning of the molecular electrostatic potential descriptors used to construct predictive quantitative structure–activity relationship/quantitative structure–property relationship (QSAR/QSPR) models for drug discovery studies. In addition, it is suggested that electron structure factors computed from UBDB2018 may significantly improve the interpretation of electrostatic potential maps measured experimentally by means of electron diffraction or single-particle cryo-EM methods.},
	number = {2},
	urldate = {2023-07-17},
	journal = {Acta Crystallogr A Found Adv},
	author = {Kumar, Prashant and Gruza, Barbara and Bojarowski, Sławomir Antoni and Dominiak, Paulina Maria},
	month = mar,
	year = {2019},
	pages = {398--408},
}

@article{DiSCaMB,
	title = {\textit{{DiSCaMB}} : a software library for aspherical atom model {X}-ray scattering factor calculations with {CPUs} and {GPUs}},
	volume = {51},
	issn = {1600-5767},
	shorttitle = {\textit{{DiSCaMB}}},
	url = {https://scripts.iucr.org/cgi-bin/paper?S1600576717015825},
	doi = {10.1107/S1600576717015825},
	abstract = {It has been recently established that the accuracy of structural parameters from X-ray refinement of crystal structures can be improved by using a bank of aspherical pseudoatoms instead of the classical spherical model of atomic form factors. This comes, however, at the cost of increased complexity of the underlying calculations. In order to facilitate the adoption of this more advanced electron density model by the broader community of crystallographers, a new software implementation called
              DiSCaMB
              , `densities in structural chemistry and molecular biology', has been developed. It addresses the challenge of providing for high performance on modern computing architectures. With parallelization options for both multi-core processors and graphics processing units (using CUDA), the library features calculation of X-ray scattering factors and their derivatives with respect to structural parameters, gives access to intermediate steps of the scattering factor calculations (thus allowing for experimentation with modifications of the underlying electron density model), and provides tools for basic structural crystallographic operations. Permissively (MIT) licensed,
              DiSCaMB
              is an open-source C++ library that can be embedded in both academic and commercial tools for X-ray structure refinement.},
	number = {1},
	urldate = {2023-07-17},
	journal = {J Appl Crystallogr},
	author = {Chodkiewicz, Michał L. and Migacz, Szymon and Rudnicki, Witold and Makal, Anna and Kalinowski, Jarosław A. and Moriarty, Nigel W. and Grosse-Kunstleve, Ralf W. and Afonine, Pavel V. and Adams, Paul D. and Dominiak, Paulina Maria},
	month = feb,
	year = {2018},
	pages = {193--199},
}

@article{iotbxcif,
	title = {\textit{iotbx.cif} : a comprehensive {CIF} toolbox},
	volume = {44},
	issn = {0021-8898},
	shorttitle = {\textit{iotbx.cif}},
	url = {https://scripts.iucr.org/cgi-bin/paper?S0021889811041161},
	doi = {10.1107/S0021889811041161},
	abstract = {iotbx.cif
              is a new software module for the development of applications that make use of the CIF format. Comprehensive tools are provided for input, output and validation of CIFs, as well as for interconversion with high-level
              cctbx
              [Grosse-Kunstleve, Sauter, Moriarty \& Adams (2002).
              J. Appl. Cryst.
              35
              , 126–136] crystallographic objects. The interface to the library is written in Python, whilst parsing is carried out using a compiled parser, combining the performance of a compiled language (C++) with the benefits of using an interpreted language.},
	number = {6},
	urldate = {2023-07-17},
	journal = {J Appl Crystallogr},
	author = {Gildea, Richard J. and Bourhis, Luc J. and Dolomanov, Oleg V. and Grosse-Kunstleve, Ralf W. and Puschmann, Horst and Adams, Paul D. and Howard, Judith A. K.},
	month = dec,
	year = {2011},
	pages = {1259--1263},
}
""".strip()


class MATTSF0jSource(F0jSource):
    def __init__(self, discamb_path, work_folder='./discamb_files', filebase='discamb'):
        self.discamb_path = os.path.abspath(discamb_path)
        self.work_folder = work_folder
        self.filebase=filebase

        if not os.path.exists(work_folder):
            os.mkdir(work_folder)

    def calc_f0j(
            self,
            atom_site_dict: Dict[str, List[Any]],
            cell_dict: Dict[str, Any],
            space_group_dict: Dict[str, Any], 
            refln_dict: Dict[str, Any]
        ):
        atom_sites_dict = cell_dict2atom_sites_dict(cell_dict)
        cell_dict['_cell_volume'] = np.linalg.det(atom_sites_dict['_atom_sites_Cartn_tran_matrix'])
        
        cleaned_sg_dict = deepcopy(space_group_dict)
        
        cleaned_sg_dict['_space_group_symop_operation_xyz'] = [
            symm_mat_vec2str(*symm_to_matrix_vector(symm_string)) for symm_string in cleaned_sg_dict['_space_group_symop_operation_xyz']
        ]

        write_mock_hkl(os.path.join(self.work_folder, self.filebase + '.hkl'), refln_dict)
        write_minimal_cif(os.path.join(self.work_folder, self.filebase + '.cif'), cell_dict, cleaned_sg_dict, atom_site_dict)
        with open(os.path.join(self.work_folder, f'{self.filebase}_cli.out'), 'w') as fo:
            subprocess.run([self.discamb_path], cwd=self.work_folder, stdout=fo)

        tsc = TSCFile.from_file(os.path.join(self.work_folder, self.filebase + '.tsc'))

        f0j = np.array([
            tsc.data[(h, k, l)] if (h, k, l) in tsc.data.keys() else np.conj(tsc.data[(-h, -k, -l)]) for h, k, l in zip(refln_dict['_refln_index_h'], refln_dict['_refln_index_k'], refln_dict['_refln_index_l'])
        ]).T
        
        return f0j

    def citation_strings(self):
        description_string = 'Aspherical atomic form factors were generated using the MATTS interface of discamb2tsc [{bibtex_key}]'

        return description_string, bibtex_entry