from .BaseLCAOQCCalculator import LCAOQCCalculator
import ase

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
            positions=self.positions,
        )
        atoms.set_calculator(self.calc)
        atoms.get_potential_energy()
