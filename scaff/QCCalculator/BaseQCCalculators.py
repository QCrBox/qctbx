import numpy as np
import os

class BaseQCCalculator:
    _positions_cart = np.empty(0)
    symbols = []
    _directory = '.'

    def __init__(
        self,
        positions_cart=None,
        symbols=None,
        directory=None,
        **kwargs
    ):
        if symbols is not None:
            self.symbols = symbols
        if positions_cart is not None:
            self.positions_cart = positions_cart
        if directory is not None:
            self._directory = directory

    @property
    def positions_cart(self):
        return self._positions_cart

    @positions_cart.setter
    def positions_cart(self, value):
        pos = np.array(value)
        assert value.shape[1] == 3, 'The positions_cart need to have 3 entries for every atom'
        self._positions_cart = pos

    @property
    def directory(self):
        return self._directory
    
    @directory.setter
    def directory(self, path):
        assert os.path.exists(path), 'The set directory does not exist: ' + str(path)
        self._directory = path

    def set_atoms(self, symbols, positions_cart):
        pos = np.array(positions_cart)
        assert len(symbols) == pos.shape[0], ' The number of entries in positions_cart and elements needs to be identical'
        self.positions_cart = pos
        self.symbols = symbols


class LCAOQCCalculator(BaseQCCalculator):
    _charge = 0
    _multiplicity = 1

    def __init__(
        self,
        *args,
        charge=None,
        multiplicity=None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        if charge is not None:
            self.charge = charge
        if multiplicity is not None:
            self.multiplicity = multiplicity

    @property
    def multiplicity(self):
        return self._multiplicity

    @multiplicity.setter
    def multiplicity(self, value):
        assert int(value) == value, 'The multiplicity can only be integer'
        self._multiplicity = value
    
    @property
    def charge(self):
        return self._charge

    @charge.setter
    def charge(self, value):
        assert int(value) == value, 'The overall charge can only be integer'
        self._charge = value


class RegGrQCCalculator(BaseQCCalculator):
    _cell_parameters = np.zeros(6)
    def __init__(
        self,
        *args, 
        cell_parameters,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.cell_parameters = cell_parameters

    @property
    def cell_parameters(self):
        return self._cell_parameters
    
    @cell_parameters.setter
    def cell_parameters(self, value):
        assert len(value) == 6, 'There are always six cell parameters'
        self._cell_parameters = np.array(value)
    
    
        