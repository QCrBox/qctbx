try:
    import ase
    from ase.spacegroup import crystal
except ImportError:
    _ase_imported = False
else:
    _ase_imported = True

from .base import LCAOWrapper, RegGrWrapper

ase_bibtex_string = """
@article{ase-paper,
  author={Ask Hjorth Larsen and Jens Jørgen Mortensen and Jakob Blomqvist and Ivano E Castelli and Rune Christensen and Marcin Dułak and Jesper Friis and Michael N Groves and Bjørk Hammer and Cory Hargus and Eric D Hermes and Paul C Jennings and Peter
Bjerre Jensen and James Kermode and John R Kitchin and Esben Leonhard Kolsbjerg and Joseph Kubal and Kristen
Kaasbjerg and Steen Lysgaard and Jón Bergmann Maronsson and Tristan Maxson and Thomas Olsen and Lars Pastewka and Andrew
Peterson and Carsten Rostgaard and Jakob Schiøtz and Ole Schütt and Mikkel Strange and Kristian S Thygesen and Tejs
Vegge and Lasse Vilhelmsen and Michael Walter and Zhenhua Zeng and Karsten W Jacobsen},
  title={The atomic simulation environment—a Python library for working with atoms},
  journal={Journal of Physics: Condensed Matter},
  volume={29},
  number={27},
  pages={273002},
  url={http://stacks.iop.org/0953-8984/29/i=27/a=273002},
  year={2017},
  abstract={The atomic simulation environment (ASE) is a software package written in the Python programming language with the aim of setting up, steering, and analyzing atomistic simulations. In ASE, tasks are fully scripted in Python. The powerful syntax of Python combined with the NumPy array library make it possible to perform very complex simulation tasks. For example, a sequence of calculations may be performed with the use of a simple ‘for-loop’ construction. Calculations of energy, forces, stresses and other quantities are performed through interfaces to many external electronic structure codes or force fields using a uniform interface. On top of this calculator interface, ASE provides modules for performing many standard simulation tasks such as structure optimization, molecular dynamics, handling of constraints and performing nudged elastic band calculations.}
}"""

class AseLCAOWrapper(LCAOWrapper):
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
            positions=self.positions_cart,
        )
        atoms.calc = self.calc
        atoms.get_potential_energy()

    def bibtex_strings(self) -> str:
        return 'ASE', ase_bibtex_string


class AsePBCWrapper(RegGrWrapper):
    calc = None
    def __init__(self, *args, calc = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.calc = calc

    def run_calculation(self):
        atoms = crystal(
            symbols=self.symbols,
            basis=self.positions_frac,
            cell=self._cell_mat_m.T
        )

        atoms.calc = self.calc
        atoms.get_potential_energy()

        return atoms, self.calc

    def bibtex_strings(self) -> str:
        return 'ASE', ase_bibtex_string