from .LCAODensityPartitionerBase import LCAODensityPartitioner
try:
    import horton
except:
    horton = None
    
import numpy as np
import sys
from contextlib import redirect_stdout
from ..conversions import cell_dict2atom_sites_dict
from ..constants import ANGSTROM_PER_BOHR
from ..util import batched
from typing import Dict, Any, List, Optional

defaults = {
    'method': 'mbis',
    'method_options': {},
    'hkl_batch_size': 2000,
    'log_file': 'horton.log'
}

class HortonPartitioner(LCAODensityPartitioner):
    wpart = None

    accepts_input = ('mkl', 'wfn')
    
    def __init__(self, options: Dict[str, Any] = {}):
        """
        Initialize HortonPartitioner with given options. Default options will be used if not provided.

        Args:
            options (Dict[str, Any], optional): Options for HortonPartitioner. Defaults to {}.
        """
        super().__init__()
        for key, value in defaults.items():
            if key not in options:
                options[key] = value
        
        if options['method'].lower().startswith('hirshfeld'):
            assert 'atomdb_path' in options, 'The Hirshfeld methods need a valid path to an an h5 file generated with horton-atomdb.py under the keyword "atomdb_path"'
        self.options = options
        if options['log_file'] is not None:
            self._log_fo = open('horton.log', 'a')
            horton.log._file = self._log_fo
        else:
            self._log_fo = None


    def __del__(self):
        if self._log_fo is not None:
            self._log_fo.close()
            horton.log._file = sys.stdout
    
    def check_availability(self) -> bool:
        """
        Check if the HORTON is available in the system.

        Returns:
            bool: True if HORTON is available, False otherwise.
        """
        return horton is not None
    
    def partition(self, density_path: str):
        """
        Perform partitioning using HORTON's methods given a path to the density 
        data. 

        Args:
            density_path (str): Path to the density data file.
        """
        assert density_path is not None, 'So far density has not been partitioned, so a path is needed'    
   
        mol = horton.IOData.from_file(density_path)
        grid = horton.BeckeMolGrid(mol.coordinates, mol.numbers, mol.pseudo_numbers, mode='keep')
        moldens = mol.obasis.compute_grid_density_dm(mol.get_dm_full(), grid.points)
        
        method = self.options['method']

        if method.lower() == 'hirshfeld':
            atomdb = horton.ProAtomDB.from_file(self.options['atomdb_path'])
            wpart = horton.HirshfeldWPart(mol.coordinates, mol.numbers, mol.pseudo_numbers, grid, moldens, atomdb, **self.options['method_options'])
        elif method.lower() == 'hirshfeld-i':
            atomdb = horton.ProAtomDB.from_file(self.options['atomdb_path'])
            wpart = horton.HirshfeldIWPart(mol.coordinates, mol.numbers, mol.pseudo_numbers, grid, moldens, atomdb, **self.options['method_options'])
        elif method.lower() == 'iterative-stockholder':
            wpart = horton.IterativeStockholderWPart(mol.coordinates, mol.numbers, mol.pseudo_numbers, grid, moldens, **self.options['method_options'])
        elif method.lower() == 'mbis':
            wpart = horton.MBISWPart(mol.coordinates, mol.numbers, mol.pseudo_numbers, grid, moldens, **self.options['method_options'])
        else:
            raise NotImplementedError('Partitioning method not implemented. Use either Hirshfeld, Hirshfeld-I, Iterative-Stockholder or MBIS')
        wpart.do_partitioning()
                
        self.wpart = wpart
    
    def calc_f0j(
            self,
            atom_indexes: List[int],
            cell_dict: Dict[str, Any],
            refln_dict: Dict[str, Any],
            density_path: Optional[str] = None
        ) -> np.ndarray:
        """
        Calculate atomic structure factors given atom indexes and cell dictionary.
        Perform partitioning first if not done before. If the par

        Args:
            atom_indexes (List[int]): List of atom indexes.
            cell_dict (Dict[str, Any]): Cell dictionary containing the unit cell
                parameters in with the keys being the CIF format entries.
            refln_dict (Dict[str, Any]): Miller indicees as refln_index keyed
                lists or numpy arrays
            density_path (Optional[str], optional): Path to the density data file. 
                Defaults to None. Needed if partitioning was not done before

        Returns:
            np.ndarray: Calculated atomic structure factors.
        """
        if self.wpart is None:
            self.partition(density_path)
        
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
            for f0j_index, vec_s_slice in enumerate(batched(vec_s, self.options['hkl_batch_size'])):
                #print(f0j_index, end=' ')
                vec_s_array = np.array(vec_s_slice)
                phase_factors = np.exp(2j * np.pi * np.einsum('ax, bx -> ab', vec_s_array, coordinates))
                start = f0j_index * self.options['hkl_batch_size']
                if vec_s_array.shape[0] == self.options['hkl_batch_size']:
                    end = (f0j_index + 1) * self.options['hkl_batch_size']
                else:
                    end = f0j.shape[1]
                f0j[atom_index, start:end] = np.sum(
                    (at_grid.weights * self.wpart[('at_weights', atom_index)] * self.wpart.get_moldens(atom_index))[None,:] * phase_factors, 
                    axis=1)
            #print('')
        return f0j
    
    def calc_charges(self, atom_indexes: List[int], density_path: Optional[str] = None) -> np.ndarray:
        """
        Calculate atomic charges of given atom indexes. Perform partitioning 
        first if not done before.

        Args:
            atom_indexes (List[int]): List of atom indexes.
            density_path (Optional[str], optional): Path to the density data file. 
                Defaults to None. Needed if partitioning was not done before

        Returns:
            np.ndarray: Calculated charges for each atom index.
        """        
        if self.wpart is None:
            self.partition(density_path)
            
        return np.array([self.wpart['charges'][idx] for idx in atom_indexes])
    

    def add_citation_strings(self) -> str:
        # TODO: Add a short string with the citation as bib and a sentence what was done
        return 'bib_string', 'sentence string'
    