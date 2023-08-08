import atexit
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np

from ...conversions import cell_dict2atom_sites_dict
from ...custom_typing import Path
from ..citations import get_partitioning_citation
from ..constants import ANGSTROM_PER_BOHR
from ..util import batched
from .base import LCAODensityPartitioner

try:
    import horton
except ImportError:
    _horton_imported = False
else:
    _horton_imported = True

defaults = {
    'method': 'mbis',
    'grid_accuracy': 'medium',
    'specific_options': {},
    'calc_options':{
        'hkl_batch_size': 2000,
        'log_file': 'horton.log',
        'work_directory': '.'
    }
}

horton_bibtex_key = 'HORTON'

horton_bibtex_entry = """
@misc{HORTON,
    author={Toon Verstraelen and Pawel Tecmer and Farnaz Heidar-Zadeh and Cristina E. González-Espinoza and Matthew Chan and Taewon D. Kim and Katharina Boguslawski and Stijn Fias and Steven Vandenbrande and Diego Berrocal and Paul W. Ayers},
    title={HORTON 2.1.1},
    year={2017},
    url={http://theochem.github.com/horton/}
}""".strip()

class HortonPartitioner(LCAODensityPartitioner):
    wpart = None
    _log_fo = None

    accepts_input = ('mkl', 'wfn')
    software = 'horton'

    def __init__(self, *args, **kwargs):
        """
        Initialize HortonPartitioner with given options. Default options will be used if not provided.

        Args:
            options (Dict[str, Any], optional): Options for HortonPartitioner. Defaults to {}.
        """
        super().__init__(*args, **kwargs)
        self.update_from_dict(defaults, update_if_present=False)
        if self.calc_options['log_file'] is not None and _horton_imported:
            atexit.unregister(horton.log.print_footer)
            self._log_fo = open(os.path.join(self.calc_options['work_directory'], self.calc_options['log_file']), 'a')
            horton.log._file = self._log_fo

        else:
            self._log_fo = None

        if self.method.lower().startswith('hirshfeld'):
            assert 'atomdb_path' in self.calc_options, 'The Hirshfeld methods need a valid path to an an h5 file generated with horton-atomdb.py under the keyword "atomdb_path"'

    def __del__(self):
        if self._log_fo is not None and _horton_imported:
            horton.log._file = sys.stdout
            if not self._log_fo.closed:
                self._log_fo.close()

    def check_availability(self) -> bool:
        """
        Check if the HORTON is available in the system.

        Returns:
            bool: True if HORTON is available, False otherwise.
        """
        return _horton_imported

    def partition(self, density_path: Path):
        """
        Perform partitioning using HORTON's methods given a path to the density
        data.

        Args:
            density_path (str): Path to the density data file.
        """
        self.update_from_dict(defaults, update_if_present=False)

        assert density_path is not None, 'So far density has not been partitioned, so a path is needed'

        mol = horton.IOData.from_file(density_path)
        grid = horton.BeckeMolGrid(mol.coordinates, mol.numbers, mol.pseudo_numbers, agspec=self.grid_accuracy, mode='keep')
        moldens = mol.obasis.compute_grid_density_dm(mol.get_dm_full(), grid.points)

        if self.method.lower() == 'hirshfeld':
            atomdb = horton.ProAtomDB.from_file(self.calc_options['atomdb_path'])
            wpart = horton.HirshfeldWPart(mol.coordinates, mol.numbers, mol.pseudo_numbers, grid, moldens, atomdb, **self.specific_options)
        elif self.method.lower() == 'hirshfeld-i':
            atomdb = horton.ProAtomDB.from_file(self.calc_options['atomdb_path'])
            wpart = horton.HirshfeldIWPart(mol.coordinates, mol.numbers, mol.pseudo_numbers, grid, moldens, atomdb, **self.specific_options)
        elif self.method.lower() == 'iterative-stockholder':
            wpart = horton.IterativeStockholderWPart(mol.coordinates, mol.numbers, mol.pseudo_numbers, grid, moldens, **self.specific_options)
        elif self.method.lower() == 'mbis':
            wpart = horton.MBISWPart(mol.coordinates, mol.numbers, mol.pseudo_numbers, grid, moldens, **self.specific_options)
        else:
            raise NotImplementedError('Partitioning method not implemented. Use either Hirshfeld, Hirshfeld-I, Iterative-Stockholder or MBIS')
        wpart.do_partitioning()

        self.wpart = wpart

    def calc_f0j(
        self,
        atom_labels: List[int],
        atom_site_dict: Dict[str, List[Any]],
        cell_dict: Dict[str, Any],
        space_group_dict: Dict[str, Any],
        refln_dict: Dict[str, Any],
        density_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Calculate atomic structure factors given atom indexes and cell dictionary.
        Perform partitioning first if not done before. If the par

        Args:
            atom_labels (List[int]): Labels of atoms to be included in the
                partitioning.
            atom_site_dict (Dict[str, Any]): Dictionary of atom site entries,
                needs to contain _atom_site_label
            cell_dict (Dict[str, Any]): Cell dictionary containing the unit cell
                parameters in with the keys being the CIF format entries.
            refln_dict (Dict[str, Any]): Miller indicees as refln_index keyed
                lists or numpy arrays
            density_path (Optional[str], optional): Path to the density data file.
                Defaults to None. Needed if partitioning was not done before

        Returns:
            np.ndarray: Calculated atomic structure factors.
            np.ndarray: Calculated atomic charges
        """
        self.partition(density_path)

        all_atom_labels = list(atom_site_dict['_atom_site_label'])

        atom_indexes = [all_atom_labels.index(label) for label in atom_labels]

        index_vec_h = np.array((
            refln_dict['_refln_index_h'],
            refln_dict['_refln_index_k'],
            refln_dict['_refln_index_l']
        )).T

        cell_mat_m = cell_dict2atom_sites_dict(cell_dict)['_atom_sites_Cartn_tran_matrix']
        cell_mat_f = np.linalg.inv(cell_mat_m).T
        f0j = np.zeros((len(atom_indexes), index_vec_h.shape[0]), dtype=np.complex128)
        vec_s = np.einsum('xy, zy -> zx', cell_mat_f, index_vec_h)

        for atom_index in atom_indexes:
            #print(atom_index, end='/ ')
            at_grid = self.wpart.get_grid(atom_index)
            coordinates = (at_grid.points - at_grid.center) * ANGSTROM_PER_BOHR
            for f0j_index, vec_s_slice in enumerate(batched(vec_s, self.calc_options['hkl_batch_size'])):
                #print(f0j_index, end=' ')
                vec_s_array = np.array(vec_s_slice)
                phase_factors = np.exp(2j * np.pi * np.einsum('ax, bx -> ab', vec_s_array, coordinates))
                start = f0j_index * self.calc_options['hkl_batch_size']
                if vec_s_array.shape[0] == self.calc_options['hkl_batch_size']:
                    end = (f0j_index + 1) * self.calc_options['hkl_batch_size']
                else:
                    end = f0j.shape[1]
                f0j[atom_index, start:end] = np.sum(
                    (at_grid.weights * self.wpart[('at_weights', atom_index)] * self.wpart.get_moldens(atom_index))[None,:] * phase_factors,
                    axis=1)
        self.wpart.do_charges()

        return f0j, np.array([self.wpart['charges'][idx] for idx in atom_indexes])

    def citation_strings(self) -> str:
        if self.method.lower() == 'hirshfeld':
            method_bibtex_key, method_bibtex_entry = get_partitioning_citation('hirshfeld')
            method_string = f'Hirshfeld partitioning [{method_bibtex_key}]'
        elif self.method.lower() == 'hirshfeld-i':
            method_bibtex_key, method_bibtex_entry = get_partitioning_citation('hirshfeld-i')
            method_string = 'Iterative Hirshfeld partitioning  [{method_bibtex_key}]'
        elif self.method.lower() == 'iterative-stockholder':
            method_bibtex_key, method_bibtex_entry = get_partitioning_citation('iterstockholder')
            method_string = 'Iterative Stockholder partitioning  [{method_bibtex_key}]'
        elif self.method.lower() == 'iterative-stockholder':
            method_bibtex_key, method_bibtex_entry = get_partitioning_citation('mbis')
            method_string = 'Minimal Basis Iterative Stockholder partitioning  [{method_bibtex_key}]'

        description_string = (
            f'The moleculear electron density was partitioning using {method_string}'
            + f' with the HORTON python library [{horton_bibtex_key}].'
        )
        bibtex_string = '\n\n\n'.join((method_bibtex_entry, horton_bibtex_entry))
        return description_string, bibtex_string
