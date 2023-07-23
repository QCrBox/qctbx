from typing import Dict, List, Any, Union
from .RegGridDensityCalculators.base import RegGridDensityCalculator
from .density_calculator_base import DensityCalculator
from .density_partitioner_base import DensityPartitioner
from ..conversions import expand_atom_site_table_symm
from ..f0j_source_base import F0jSource

class ScaffF0jSource(F0jSource):
    """
    Source for calculating aspherical atomic form factors using a given density calculator and partitioner.

    Attributes
    ----------
    density_calculator : DensityCalculator
        Calculator for electron density.
    partitioner : DensityPartitioner
        Partitioner for dividing the electron density into atomic contributions. Needs to fit the
        type (RegGrid or LCAO) of the density_calculator
    expand_positions : Dict[str, str]
        Dictionary to determine which symmetry operations to apply to each atom.
        Keys are symmetry operation strings, values are atom labels or 'all'.
    use_charges : bool
        Whether to use charges in the calculation. Currently not implemented.
    """
    _charge_dict = {}

    def __init__(
        self,
        density_calculator: DensityCalculator,
        partitioner: DensityPartitioner,
        expand_positions: Dict[str, Union[str, List[str]]] = {},
        use_charges: bool=False
    ):
        """
        Initializes the ScaffF0jSource instance.

        Parameters
        ----------
        density_calculator : DensityCalculator
            Calculator for electron density.
        partitioner : DensityPartitioner
            Partitioner for dividing the electron density into atomic contributions. Needs to fit the
            type (RegGrid or LCAO) of the density_calculator
        expand_positions : Dict[str, str]
            Dictionary to determine which symmetry operations to apply to each atom.
            Keys are symmetry operation strings, values are atom labels or 'all'.
        use_charges : bool
            Whether to use charges in the calculation. Currently not implemented.

        Raises
        ------
        NotImplementedError
            If use_charges is True, as cluster charge calculations are not implemented yet.
        """
        self.density_calculator = density_calculator
        self.partitioner = partitioner
        self.expand_positions = expand_positions
        self.use_charges = use_charges
        if use_charges:
            raise NotImplementedError('cluster charge calculations are not implemented yet')

    def calc_f0j(
        self,
        atom_site_dict: Dict[str, List[Any]],
        cell_dict: Dict[str, Any],
        space_group_dict: Dict[str, Any],
        refln_dict: Dict[str, Any]
    ):
        """
        Calculate "Specifically calculated atomic form factors" using the specified density
        calculation and partitioning methods.

        Parameters
        ----------
        atom_site_dict : Dict[str, List[Any]]
            Dictionary containing information about the atomic sites.
            Required keys: '_atom_site_label', '_atom_site_type_symbol',
            '_atom_site_fract_x', '_atom_site_fract_y', '_atom_site_fract_z'.
        cell_dict : Dict[str, Union[int, float]]
            A dictionary representing cell parameters.
            Required keys: '_cell_length_a', '_cell_length_b', '_cell_length_c',
            '_cell_angle_alpha', '_cell_angle_beta', '_cell_angle_gamma'.
        space_group_dict : Dict[str, Union[int, str, List[str]]]
            A dictionary representing space group parameters. Needs to contain key:
            '_space_group_symop_operation_xyz
        refln_dict : Dict[str, Union[int, float, List[int]]]
            A dictionary representing reflection parameters.
            Required keys: '_refln_index_h', '_refln_index_k', '_refln_index_l'.

        Returns
        -------
        np.ndarray
            The calculated aspherical atomic form factors.
        """
        if isinstance(self.density_calculator, RegGridDensityCalculator):
            expand_positions = {op: 'all' for op in space_group_dict['_space_group_symop_operation_xyz']}
            atom_site_dict_exp = expand_atom_site_table_symm(atom_site_dict, expand_positions, cell_dict, check_special=True)
        elif len(self.expand_positions) > 0:
            atom_site_dict_exp = expand_atom_site_table_symm(atom_site_dict, self.expand_positions, check_special=False)
        else:
            atom_site_dict_exp = atom_site_dict

        wavefunction_path = self.density_calculator.calculate_density(atom_site_dict_exp, cell_dict)

        f0j, charges = self.partitioner.calc_f0j(
            list(atom_site_dict['_atom_site_label']),
            atom_site_dict_exp,
            cell_dict,
            space_group_dict,
            refln_dict,
            wavefunction_path
        )
        if charges is not None:
            self._charge_dict = {atom_name: value for atom_name, value in zip(atom_site_dict['_atom_site_label'], charges)}

        return f0j

    def citation_strings(self):
        """
        Generate strings describing the approach to calculation and BibTeX formatted citations.

        Returns
        -------
        Tuple[str, str]
            A tuple containing the description and the bibtex entries required.
        """
        density_str, density_bib = self.density_calculator.citation_strings()
        part_str, part_bib = self.partitioner.citation_strings()

        return ' '.join((density_str, part_str)), '\n\n\n'.join((density_bib, part_bib))




