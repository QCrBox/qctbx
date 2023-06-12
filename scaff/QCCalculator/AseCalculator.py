from .BaseQCCalculators import LCAOQCCalculator, RegGrQCCalculator
from ..conversions import cell_dict2atom_sites_dict
import ase
import numpy as np
from ase.spacegroup import crystal

class AseLCAOCalculator(LCAOQCCalculator):
    calc = None

    def __init__(
        self,
        *args,
        calc = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.calc = calc

    @property
    def charge(self):
        raise ValueError('The overall charge is set in the calculator object in ase')

    @charge.setter
    def charge(self, value):
        raise ValueError('The overall charge is set in the calculator object in ase')
    
    @property
    def multiplicity(self):
        raise ValueError('The multiplicity is set in the calculator object in ase')
    
    @multiplicity.setter
    def multiplicity(self, value):
        raise ValueError('The multiplicity is set in the calculator object in ase')
    
    def run_calculation(self):

        atoms = ase.Atoms(
            symbols=self.symbols,
            positions_cart=self.positions_cart,
        )
        atoms.set_calculator(self.calc)
        atoms.get_potential_energy()


class AsePBCCalculator(RegGrQCCalculator):
    calc = None
    def __init__(self, *args, calc = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.calc = calc

    def run_calculation(self):
        atom_sites_dict = cell_dict2atom_sites_dict({
            '_cell_length_a': self.cell_parameters[0],
            '_cell_length_b': self.cell_parameters[1],
            '_cell_length_c': self.cell_parameters[2],
            '_cell_angle_alpha': self.cell_parameters[3],
            '_cell_angle_beta': self.cell_parameters[4],
            '_cell_angle_gamma': self.cell_parameters[5],
        })

        atoms = crystal(
            symbols=self.symbols,
            positions_cart=self.positions_cart,
            cell=np.array(atom_sites_dict['_atom_sites_Cartn_tran_matrix'])
        )

        atoms.set_calculator(self.calc)
        atoms.get_potential_energy()

        return atoms, self.calc