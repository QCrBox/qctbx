from copy import deepcopy
import functools
from itertools import product
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import minimize_scalar

from ...conversions import cell_dict2atom_sites_dict
from ...custom_typing import Path
from ..citations import get_partitioning_citation
from ..constants import ANGSTROM_PER_BOHR
from .base import RegGridDensityPartitioner, calc_f0j_core
from .cubetools import read_cube

defaults = {
    'partition': 'valence',
    'atomic_densities_dict': {},
    'missing_e_atomic_max': 1e-4
}

def target_with_e_missing(cutoff, spline, spline_limit, missing_e):
    return (spline.integral(cutoff, spline_limit) - missing_e)**2


class PythonRegGridPartitioner(RegGridDensityPartitioner):
    """
    PythonRegGridPartitioner is a class for partitioning atomic densities on a
    regular grid.

    Attributes:
        atom_splines (dict): Storage for interpolated atomic density splines for
            each atomic element.
        atom_n_elec (dict): Storage for total number of electrons within the
            cutoff distance for each atomic element.
        charges (np.ndarray): Storage for the resulting charges after
            partitioning.

    Args:
        options (Dict[str, Any]): A dictionary of options for partitioning.
            - 'atomic_densities_dict': (dict) Atomic densities in the ciflike
                format of AtomicDensityReaders. It is not optional.
                Each atomic element is a key, and the value is another dictionary
                with the following structure:
                    - '_qubox_density_atomic_rgrid': (list or np.ndarray)
                        Radial grid for atomic densities.
                    - '_qubox_density_atomic_valence': (list or np.ndarray)
                        Valence atomic density, only needed for valence partitioning
                    - '_qubox_density_atomic_total': (list or np.ndarray)
                        Total atomic density, only needed for total partitioning
                    - '_qubox_density_atomic_core': (list or np.ndarray)
                        Core atomic density, only needed for valence partitioning
            - 'partition': (str, optional) Type of atomic density partition.
                Needs to match the density in the provided cube file.
                It can be either 'valence', where the valence density is in the
                cube file and is partitioned, while the core density is
                evaluated separately on the spherical grid of the atomic density
                and added to the atom completely or 'total', which means the
                cube file contains the all electron density. Default is
                'valence'.
            - 'missing_e_atomic_max': (float, optional) Maximum allowed missing
                electrons for determining the distance cutoff of the atomic
                densities. Default is 1e-4.
    """

    atom_splines: Dict[str, Any]
    atom_n_elec: Dict[str, float]
    charges: Optional[np.ndarray]

    def check_availability(self) -> bool:
        """
        If qctbx is installed this should be available

        Returns:
            bool: True
        """
        return True

    def __init__(self, options: Dict[str, Any]):
        """
        Constructor for the PythonRegGridPartitioner class.

        Args:
            options (Dict[str, Any]): A dictionary containing various options
            for partitioning. See class docstring for explanation.
        """
        super().__init__()

        assert 'atomic_densities_dict' in options, 'atomic densities in the ciflike format of AtomicDensityReaders is not an optional input in options'

        options = deepcopy(options)
        for key, value in defaults.items():
            if key not in options:
                options[key] = value

        self.options = options

        self.generate_splines()


    def generate_splines(self):
        """
        Generates splines for atomic densities.
        """
        self.atom_splines = {}
        self.atom_n_elec = {}

        if self.options['partition'] == 'valence':
            density_type = 'valence'
        elif self.options['partition'] == 'total':
            density_type = 'total'
        else:
            raise NotImplementedError('partition can be either valence or total')

        for element, atomic_entries in self.options['atomic_densities_dict'].items():
            atom_density_spline = InterpolatedUnivariateSpline(
                atomic_entries['_qubox_density_atomic_rgrid'],
                atomic_entries[f'_qubox_density_atomic_{density_type}'],
                ext=1
            )
            atom_shell_spline = InterpolatedUnivariateSpline(
                atomic_entries['_qubox_density_atomic_rgrid'],
                4 * np.pi * np.array(atomic_entries['_qubox_density_atomic_rgrid'])**2 * atomic_entries[f'_qubox_density_atomic_{density_type}'],
                ext=1
            )
            used_target = functools.partial(
                target_with_e_missing,
                spline=atom_density_spline,
                spline_limit=atomic_entries['_qubox_density_atomic_rgrid'][-1],
                missing_e=self.options['missing_e_atomic_max']
            )
            cutoff = minimize_scalar(used_target).x
            atom_density_spline.cutoff = cutoff
            self.atom_splines[element] = atom_density_spline
            self.atom_n_elec[element] = atom_shell_spline.integral(0.0, atomic_entries['_qubox_density_atomic_rgrid'][-1])

    def calc_f0j(
        self,
        atom_labels: List[int],
        atom_site_dict: Dict[str, List[Any]],
        cell_dict: Dict[str, Any],
        space_group_dict: Dict[str, Any],
        refln_dict: Dict[str, Any],
        density_path: Path
    ) -> np.ndarray:

        if self.options['partition'] == 'valence':
            qubox_density_atom_dict = self.options['atomic_densities_dict']
            f0j_core_dict, _ = calc_f0j_core(cell_dict, refln_dict, qubox_density_atom_dict)
        elif self.options['partition'] != 'total':
            raise NotImplementedError('Only options for partition are valence (core is transformed separately) or total')

        cell_mat_m = cell_dict2atom_sites_dict(cell_dict)['_atom_sites_Cartn_tran_matrix']
        cell_lengths = np.array([cell_dict[f'_cell_length_{axis}'] for axis in ('a', 'b', 'c')])

        cube = read_cube(density_path)
        density = cube[0] / ANGSTROM_PER_BOHR**3 * np.linalg.det(np.stack(tuple((cube[1]['xvec'], cube[1]['yvec'], cube[1]['zvec']))) * ANGSTROM_PER_BOHR)
        linspaces = (np.linspace(0.0, 1.0, npoints, endpoint=False) for npoints in density.shape)
        xyz_cart_cell = np.einsum('xy, abcy -> abcx', cell_mat_m, np.stack(np.meshgrid(*linspaces, indexing='ij'), axis=-1))

        atom_site_pivot = [{key: atom_site_dict[key][index] for key in atom_site_dict.keys()} for index in range(len(atom_site_dict['_atom_site_type_symbol']))]
        partioned_atoms = [entry for entry in atom_site_pivot if entry['_atom_site_label'] in atom_labels]

        # we need 1 / sum(promol) but only for points where there is density of an evaluated atom otherwise the weight of all evaluated atoms has to be zero

        all_atom_weights = np.zeros_like(density)
        for atom_site in partioned_atoms:
            spline = self.atom_splines[atom_site['_atom_site_type_symbol']]
            xyz_cart = cell_mat_m[0] * (atom_site['_atom_site_fract_x'] % 1) + cell_mat_m[1] * (atom_site['_atom_site_fract_y'] % 1)+ cell_mat_m[2] * (atom_site['_atom_site_fract_z'] % 1)
            n_supercell = np.ceil(spline.cutoff / cell_lengths).astype(np.int64)
            for x_add, y_add, z_add in product(*[np.arange(-n, n+1, 1) for n in n_supercell]):
                xyz_cart_dash = xyz_cart + cell_mat_m[0] * x_add + cell_mat_m[1] * y_add + cell_mat_m[2] * z_add
                distances = np.linalg.norm(xyz_cart_dash[None, None, None, :] - xyz_cart_cell, axis=-1)
                all_atom_weights[distances < spline.cutoff] += spline(distances[distances < spline.cutoff])

        eval_xyz_cart_cell = xyz_cart_cell[all_atom_weights != 0]
        eval_all_atom_weights = np.zeros_like(all_atom_weights[all_atom_weights != 0])

        non_partitioned_atoms = [entry for entry in atom_site_pivot if entry['_atom_site_label'] not in atom_labels]
        for atom_site in non_partitioned_atoms:
            spline = self.atom_splines[atom_site['_atom_site_type_symbol']]
            xyz_cart = cell_mat_m[0] * (atom_site['_atom_site_fract_x'] % 1) + cell_mat_m[1] * (atom_site['_atom_site_fract_y'] % 1)+ cell_mat_m[2] * (atom_site['_atom_site_fract_z'] % 1)
            n_supercell = np.ceil(spline.cutoff / cell_lengths).astype(np.int64)
            for x_add, y_add, z_add in product(*[np.arange(-n, n+1, 1) for n in n_supercell]):
                xyz_cart_dash = xyz_cart + cell_mat_m[0] * x_add + cell_mat_m[1] * y_add + cell_mat_m[2] * z_add
                distances = np.linalg.norm(xyz_cart_dash[None, :] - eval_xyz_cart_cell, axis=-1)
                eval_all_atom_weights[distances < spline.cutoff] += spline(distances[distances < spline.cutoff])
        all_atom_weights[all_atom_weights != 0] += eval_all_atom_weights
        all_atom_weights[all_atom_weights != 0] = 1 / all_atom_weights[all_atom_weights != 0]

        all_atom_labels = [atom_site['_atom_site_label'] for atom_site in atom_site_pivot]

        h = np.array(refln_dict['_refln_index_h'], dtype=np.int64)
        k = np.array(refln_dict['_refln_index_k'], dtype=np.int64)
        l = np.array(refln_dict['_refln_index_l'], dtype=np.int64)
        f0j = np.zeros((len(atom_labels), h.shape[0]), dtype=np.complex128)
        charges = np.zeros(len(atom_labels))
        for atom_index, atom_label in enumerate(atom_labels):
            atom_weight = np.zeros_like(density)
            atom_site = atom_site_pivot[all_atom_labels.index(atom_label)]
            spline = self.atom_splines[atom_site['_atom_site_type_symbol']]
            xyz_frac = np.array([atom_site['_atom_site_fract_x'] % 1, atom_site['_atom_site_fract_y'] % 1, atom_site['_atom_site_fract_z'] % 1])
            xyz_cart = cell_mat_m @ xyz_frac
            n_supercell = np.ceil(spline.cutoff / cell_lengths).astype(np.int64)
            for x_add, y_add, z_add in product(*[np.arange(-n, n+1, 1) for n in n_supercell]):
                xyz_cart_dash = xyz_cart + cell_mat_m[0] * x_add + cell_mat_m[1] * y_add + cell_mat_m[2] * z_add
                distances = np.linalg.norm(xyz_cart_dash[None, None, None, :] - xyz_cart_cell, axis=-1)
                atom_weight[distances < spline.cutoff] += spline(distances[distances < spline.cutoff])
            #print(atom_label, elem_num2type[atom_site['_atom_site_type_symbol']] - np.sum(density * atom_weight * all_atom_weights)  )

            phase_to_zero = np.exp(-2j * np.pi * (xyz_frac[0] * h + xyz_frac[1] * k + xyz_frac[2] * l))
            f0j_atom = np.fft.ifftn(density * atom_weight * all_atom_weights) * np.prod(density.shape)
            charges[atom_index] = self.atom_n_elec[atom_site['_atom_site_type_symbol']] - np.real(f0j_atom[0,0,0])
            f0j[atom_index, :] = f0j_atom[h, k, l] * phase_to_zero
            if self.options['partition'] == 'valence':
                f0j[atom_index, :] += f0j_core_dict[atom_site['_atom_site_type_symbol']]

        return f0j, charges

    def citation_strings(self) -> str:
        method_bibtex_key, method_bibtex_entry = get_partitioning_citation('hirshfeld')
        description_string = (
            f'The moleculear electron density was partitioning using Hirshfeld partitioning [{method_bibtex_key}]'
            + 'using the buildin routine of qctbx'
        )
        return description_string, method_bibtex_entry

    def cif_output(self) -> str:
        return 'To be implemented'
