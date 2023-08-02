import os
from typing import Union

from ..constants import ANGSTROM_PER_BOHR
from .base import RegGrQCCalculator

#TODO Check for kpts 1,1,1. Exchange with explicit gamma

def qe_entry_string(
    name: str,
    value: Union[str, float, int, bool],
    string_sign: bool = True
) -> str:
    """Creates a formatted string for output in a quantum-espresso input file

    Parameters
    ----------
    name : str
        Name of the option
    value : Union[str, float, int, bool]
        The value of the option
    string_sign : bool, optional
        If the value is a string this value determines, whether the entry,
        will have '' as an indicator of the type, by default True

    Returns
    -------
    str
        Formatted string

    Raises
    ------
    NotImplementedError
        The type of value is currently not implemented
    """
    if isinstance(value, str):
        if string_sign:
            entry_str = f"'{value}'"
        else:
            entry_str = value
    elif isinstance(value, float):
        entry_str = f'{value}'
    elif isinstance(value, int):
        entry_str = f'{value}'
    elif isinstance(value, bool):
        if value:
            entry_str = '.TRUE.'
        else:
            entry_str = '.FALSE.'
    else:
        print(value, type(value))
        raise NotImplementedError(f'{type(value)} is not implemented')
    return f'    {name} = {entry_str}'

class BaseQECalculator(RegGrQCCalculator):
    _mpi_cores = 1
    _omp_numthreads = 1

    def __init__(self, input_dict, mpi_cores, omp_numthreads):
        self.input_dict = input_dict
        self.mpi_cores = mpi_cores
        self.omp_numthreads = omp_numthreads

    @property
    def mpi_cores(self):
        return self._mpi_cores

    @mpi_cores.setter
    def mpi_cores(self, value):
        if value == 'auto':
            self._mpi_cores = os.cpu_count()
        elif value is None:
            self._mpi_cores = 1
        else:
            self._mpi_cores = int(value)

    @property
    def omp_numthreads(self):
        return self._omp_numthreads

    @omp_numthreads.setter
    def omp_numthreads(self, value):
        if value == 'auto':
            self._omp_numthreads = os.cpu_count() // self.mpi_cores
        elif value == 0:
            self._omp_numthreads = 1
        else:
            self._omp_numthreads = int(value)


class QEPWCalculator(BaseQECalculator):
    def __init__(self, *args, paw_pot_files, kpoints, **kwargs):
        super().__init__(*args, **kwargs)
        self.paw_pot_files = paw_pot_files
        self.kpoints = kpoints

    def run_calculation(self):
        pass



class QEPPCalculator(BaseQECalculator):
    pass