from .BaseQCCalculators import RegGrQCCalculator
from ..constants import ANGSTROM_PER_BOHR, atomic_masses
from typing import Union

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
    if type(value) is str:
        if string_sign:
            entry_str = f"'{value}'"
        else:
            entry_str = value
    elif type(value) is float:
        entry_str = f'{value}'
    elif type(value) is int:
        entry_str = f'{value}'
    elif type(value) is bool:
        if value:
            entry_str = '.TRUE.'
        else:
            entry_str = '.FALSE.'
    else:
        print(value, type(value))
        raise NotImplementedError(f'{type(value)} is not implemented')
    return f'    {name} = {entry_str}'