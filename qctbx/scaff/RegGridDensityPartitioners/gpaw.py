from gpaw.density import RealSpaceDensity
from gpaw.mpi import world
from gpaw.io.logger import GPAWLogger
from gpaw.lfc import BasisFunctions
from gpaw.setup import Setups
from gpaw.utilities.partition import AtomPartition
from gpaw.xc import XC

from ase.spacegroup import crystal
from gpaw import GPAW

from ...conversions import cell_dict2atom_sites_dict, expand_atom_site_table_symm
from ..constants import ANGSTROM_PER_BOHR, ATOMIC_N_ELEC
from .cubetools import read_cube

from copy import deepcopy
import numpy as np
import warnings
from typing import List, Dict, Any


from .base import RegGridDensityPartitioner, calc_f0j_core
from ..RegGridDensityCalculators.gpaw import gpaw_bibtex_key, gpaw_bibtex_entry
from ..citations import get_partitioning_citation
from ..util import dict_merge

defaults = {
    'partition': 'valence',
    'gpaw_options' : {
        'xc': 'PBE',
        'txt': 'gpaw_partition.txt'
    },
    'gridinterpolation': 2
}

class HirshfeldDensity(RealSpaceDensity):
    """Density as sum of atomic densities."""

    def __init__(self, calculator, log=None):
        self.calculator = calculator
        dens = calculator.density
        try:
            RealSpaceDensity.__init__(self, dens.gd, dens.finegd,
                                    dens.nspins, collinear=True, charge=0.0,
                                    stencil=dens.stencil, 
                                    redistributor=dens.redistributor)
        except:
            RealSpaceDensity.__init__(self, dens.gd, dens.finegd,
                                    dens.nspins, collinear=True, charge=0.0,
                                    stencil=2, 
                                    redistributor=dens.redistributor)
        self.log = GPAWLogger(world=world)
        if log is None:
            self.log.fd = None
        else:
            self.log.fd = log

    def set_positions(self, spos_ac, atom_partition):
        """HirshfeldDensity builds a hack density object to calculate
        all electron density
        of atoms. This methods overrides the parallel distribution of
        atomic density matrices
        in density.py"""
        self.atom_partition = atom_partition
        self.atomdist = self.redistributor.get_atom_distributions(spos_ac)
        self.nct.set_positions(spos_ac)
        self.ghat.set_positions(spos_ac)
        self.mixer.reset()
        # self.nt_sG = None
        self.nt_sg = None
        self.nt_g = None
        self.rhot_g = None
        self.Q_aL = None
        self.nct_G = self.gd.zeros()
        self.nct.add(self.nct_G, 1.0 / self.nspins)

    def get_density(self, atom_indices=None, gridrefinement=2, skip_core=False):
        """Get sum of atomic densities from the given atom list.

        Parameters
        ----------
        atom_indices : list_like
            All atoms are taken if the list is not given.
        gridrefinement : 1, 2, 4
            Gridrefinement given to get_all_electron_density

        Returns
        -------
        type
             spin summed density, grid_descriptor
        """

        all_atoms = self.calculator.get_atoms()
        if atom_indices is None:
            atom_indices = range(len(all_atoms))

        # select atoms
        atoms = self.calculator.get_atoms()[atom_indices]
        spos_ac = atoms.get_scaled_positions()
        Z_a = atoms.get_atomic_numbers()

        par = self.calculator.parameters
        setups = Setups(Z_a, par.setups, par.basis,
                        XC(par.xc),
                        world=self.calculator.wfs.world)

        # initialize
        self.initialize(setups,
                        self.calculator.timer,
                        np.zeros((len(atoms), 3)), False)
        self.set_mixer(None)
        rank_a = self.gd.get_ranks_from_positions(spos_ac)
        self.set_positions(spos_ac, AtomPartition(self.gd.comm, rank_a))
        try:
            # older GPAW versions
            basis_functions = BasisFunctions(self.gd,
                                            [setup.phit_j
                                            for setup in self.setups],
                                            cut=True)
        except:
            # newer GPAW versions
            basis_functions = BasisFunctions(self.gd,
                                            [setup.basis_functions_J
                                            for setup in self.setups],
                                            cut=True)
        basis_functions.set_positions(spos_ac)
        self.initialize_from_atomic_densities(basis_functions)

        aed_sg, gd = self.get_all_electron_density(atoms,
                                                   gridrefinement,
                                                   skip_core=skip_core)
        return aed_sg.sum(axis=0), gd
    

class GPAWDensityPartitioner(RegGridDensityPartitioner):

    def __init__(self, options):
        super().__init__()
        self.options = options

    def check_availability(self) -> bool:
        return True

    def calc_f0j(
        self,
        atom_labels: List[int],
        atom_site_dict: Dict[str, List[Any]],
        cell_dict: Dict[str, Any],
        space_group_dict: Dict[str, Any],
        refln_dict: Dict[str, Any],
        density_path: str
    ) -> np.ndarray:
        options = dict_merge(defaults, self.options)

        cell_mat_m = cell_dict2atom_sites_dict(cell_dict)['_atom_sites_Cartn_tran_matrix']

        fract_xyz = np.stack([atom_site_dict[f'_atom_site_fract_{coord}'] for coord in ('x', 'y', 'z')], axis=1)

        atoms = crystal(
            symbols=atom_site_dict['_atom_site_type_symbol'],
            basis=fract_xyz % 1,
            cell=cell_mat_m.T
        )

        cube = read_cube(density_path)
        density = cube[0] / ANGSTROM_PER_BOHR**3 * np.linalg.det(np.stack(tuple((cube[1]['xvec'], cube[1]['yvec'], cube[1]['zvec']))) * ANGSTROM_PER_BOHR)

        assert np.all((np.round((np.array(density.shape) / options['gridinterpolation']), 10) % 1.0) == 0.0), 'gridinterpolation produces a remainder for size of cube file density grid'

        coarse_grid_size = tuple(int(val / options['gridinterpolation']) for val in density.shape)

        calc = GPAW(gpts=coarse_grid_size, **options['gpaw_options'])
        atoms.set_calculator(calc)
        calc.initialize(atoms)
        calc.set_positions(atoms)
        hdens_obj = HirshfeldDensity(calc)
        if options['partition'] == 'valence':
            skip_core = True
            splines = {setup.symbol: (setup.get_partial_waves()[2], setup.rgd.r_g) for setup in calc.density.setups}
            qubox_density_atomic_dicts = {
                symbol: {
                    '_qubox_density_atomic_rgrid': rgrid * ANGSTROM_PER_BOHR**3,
                    '_qubox_density_atomic_core': spline.map(rgrid) / ANGSTROM_PER_BOHR**3
                } for symbol, (spline, rgrid) in splines.items()
            }
            
            f0j_core_dict, n_elec_core = calc_f0j_core(cell_dict, refln_dict, qubox_density_atomic_dicts)
        elif options['partition'] == 'total':
            skip_core = False
        else:
            raise NotImplementedError('partition setting in options needs to either valence or total.')

        all_atom_weights = 1.0 / hdens_obj.get_density(
            gridrefinement=options['gridinterpolation'], 
            skip_core=skip_core
        )[0]
        all_atom_weights[np.logical_not(np.isfinite(all_atom_weights))] = 0.0

        atom_indexes = iter(atom_site_dict['_atom_site_label'].index(label) for label in atom_labels)
        atom_types = atom_site_dict['_atom_site_type_symbol']
        h = np.array(refln_dict['_refln_index_h'], dtype=np.int64)
        k = np.array(refln_dict['_refln_index_k'], dtype=np.int64)
        l = np.array(refln_dict['_refln_index_l'], dtype=np.int64)

        f0j = np.empty((len(atom_labels), h.shape[0]), dtype=np.complex128)
        charges = np.empty(len(atom_labels), dtype=np.float64)
        for atom_index in atom_indexes:
            frac_position = fract_xyz[atom_index]
            atom_type = atom_types[atom_index]
            phase_to_zero = np.exp(-2j * np.pi * (frac_position[0] * h + frac_position[1] * k + frac_position[2] * l))
            atom_weight = hdens_obj.get_density([atom_index], gridrefinement=options['gridinterpolation'], skip_core=skip_core)[0]
            f0j_atom = np.fft.ifftn(density * atom_weight * all_atom_weights) * np.prod(density.shape)
            f0j[atom_index] = f0j_atom[h, k, l] * phase_to_zero
            charges[atom_index] = ATOMIC_N_ELEC[atom_type] - np.real(f0j_atom[0, 0, 0])
            if skip_core:
                f0j[atom_index] += f0j_core_dict[atom_type]
                charges[atom_index] -= n_elec_core[atom_type]
        return f0j, charges
    
    def citation_strings(self) -> str:
        method_bibtex_key, method_bibtex_entry = get_partitioning_citation('hirshfeld')
        description_string = (
            f'The moleculear electron density was partitioning using Hirshfeld partitioning [{method_bibtex_key}]'
            + f'using the Hirshfeld partitioning as implemented in GPAW [{gpaw_bibtex_key}]'
        )
        bibtex_entry = '\n\n\n'.join((method_bibtex_entry, gpaw_bibtex_entry))
        return description_string, bibtex_entry
    
    def cif_output(self) -> str:
        return 'To be implemented'

        

