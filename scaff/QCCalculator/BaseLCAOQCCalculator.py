import numpy as np
import os

class LCAOQCCalculator:
    _positions = np.empty(0)
    symbols = []
    _charge = 0
    _multiplicity = 1
    _directory = '.'

    def __init__(
        self,
        symbols=None,
        positions=None,
        charge=None,
        multiplicity=None,
        directory=None,
    ):
        if symbols is not None:
            self.symbols = symbols
        if positions is not None:
            self.positions = positions
        if charge is not None:
            self.charge = charge
        if multiplicity is not None:
            self.multiplicity = multiplicity
        if directory is not None:
            self._directory = directory

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
    
    @property
    def positions(self):
        return self._positions

    @positions.setter
    def positions(self, value):
        pos = np.array(value)
        assert value.shape[1] == 3, 'The positions need to have 3 entries for every atom'
        self._positions = pos

    @property
    def directory(self):
        return self._directory
    
    @directory.setter
    def directory(self, path):
        assert os.path.exists(path), 'The set directory does not exist: ' + str(path)
        self._directory = path

    def set_atoms(self, symbols, positions):
        pos = np.array(positions)
        assert len(symbols) == pos.shape[0], ' The number of entries in positions and elements needs to be identical'
        self.positions = pos
        self.symbols = symbols
    
    
        