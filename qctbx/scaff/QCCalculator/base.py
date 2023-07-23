import os
from abc import abstractmethod

import numpy as np

from ...conversions import cell_dict2atom_sites_dict


class BaseQCCalculator:
    _positions_cart = np.empty((0,3))
    symbols = []
    _directory = '.'

    def __init__(
        self,
        positions_cart=None,
        symbols=None,
        directory=None,
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
        assert value.shape[1] == 3, 'The positions need to have 3 entries for every atom'
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

    @abstractmethod
    def bibtex_strings(self):
        "Method need to return a string containing a bibtex naming key and the corresponding complete bibtex entry as string"


class LCAOQCCalculator(BaseQCCalculator):
    _charge = 0
    _multiplicity = 1

    def __init__(
        self,
        *args,
        charge=None,
        multiplicity=None,
        atom_site_dict=None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        if charge is not None:
            self.charge = charge
        if multiplicity is not None:
            self.multiplicity = multiplicity
        if atom_site_dict is not None:
            self.atom_site_dict = atom_site_dict

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
    def atom_site_dict(self):
        return {
            '_atom_site_type_symbol': self.symbols,
            '_atom_site_Cartn_x': self.positions_cart[:,0],
            '_atom_site_Cartn_y': self.positions_cart[:,1],
            '_atom_site_Cartn_z': self.positions_cart[:,2],
        }

    @atom_site_dict.setter
    def atom_site_dict(self, value):
        assert '_atom_site_Cartn_x' in value, 'Atom site dict needs to contain positions in cartesian coordinates'
        assert '_atom_site_type_symbol' in value, 'Atom site dict needs to contain type symbols'
        self.positions_cart = np.stack(
            tuple(np.array(value[f'_atom_site_Cartn_{coord}']) for coord in ('x', 'y', 'z')), axis=-1
        )
        self.symbols = list(value['_atom_site_type_symbol'])


class RegGrQCCalculator(BaseQCCalculator):
    _cell_parameters = np.empty(6)
    _cell_mat_m = np.empty((3,3))
    def __init__(
        self,
        *args,
        cell_parameters=None,
        cell_dict=None,
        atom_site_dict=None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        if cell_parameters is not None:
            self.cell_parameters = cell_parameters
        if cell_dict is not None:
            self.cell_dict = cell_dict
        if atom_site_dict is not None:
            self.atom_site_dict = atom_site_dict

    @property
    def cell_parameters(self):
        return self._cell_parameters

    @cell_parameters.setter
    def cell_parameters(self, value):
        assert len(value) == 6, 'There are always six cell parameters'
        self._cell_parameters = np.array(value)
        self._cell_mat_m = cell_dict2atom_sites_dict(self.cell_dict)['_atom_sites_Cartn_tran_matrix']

    @property
    def cell_dict(self):
        keys = ('_cell_length_a', '_cell_length_b', '_cell_length_c', '_cell_angle_alpha', '_cell_angle_beta', '_cell_angle_gamma')
        return {key: value for key, value in zip(keys, self.cell_parameters)}

    @cell_dict.setter
    def cell_dict(self, value):
        keys = ('_cell_length_a', '_cell_length_b', '_cell_length_c', '_cell_angle_alpha', '_cell_angle_beta', '_cell_angle_gamma')
        self.cell_parameters = np.array([value[key] for key in keys])

    @property
    def positions_frac(self):
        return np.einsum('xy, zy -> zx', np.linalg.inv(self._cell_mat_m), self.positions_cart)

    @positions_frac.setter
    def positions_frac(self, value):
        self.positions_cart = np.einsum('xy, zy -> zx', self._cell_mat_m, value)

    @property
    def atom_site_dict(self):
        pos_frac = self.positions_frac
        return {
            '_atom_site_type_symbol': self.symbols,
            '_atom_site_fract_x': pos_frac[:, 0],
            '_atom_site_fract_y': pos_frac[:, 1],
            '_atom_site_fract_z': pos_frac[:, 2],
            '_atom_site_Cartn_x': self.positions_cart[:,0],
            '_atom_site_Cartn_y': self.positions_cart[:,1],
            '_atom_site_Cartn_z': self.positions_cart[:,2]
        }

    @atom_site_dict.setter
    def atom_site_dict(self, new_dict):
        cartn_keys = tuple(f'_atom_site_Cartn_{coord}' for coord in ('x', 'y', 'z'))
        fract_keys = tuple(f'_atom_site_fract_{coord}' for coord in ('x', 'y', 'z'))

        if all(key in new_dict for key in cartn_keys):
            self.positions_cart = np.stack(tuple(np.array(new_dict[key]) for key in cartn_keys), axis=-1)
        elif all(key in new_dict for key in fract_keys):
            self.positions_frac = np.stack(tuple(np.array(new_dict[key]) for key in fract_keys), axis=-1)
        else:
            raise KeyError('atomic positions need to be present using either the _atom_site_Cartn or the _atom_site_fract keys')

        self.symbols = list(new_dict['_atom_site_type_symbol'])


