"""Collects all functions and logic to calculate 'Specifically Calculated
Atomic form factors' (SCAFF) for structures. This includes but is not limited
to atomic form factors generated by Hirshfeld partitioning"""

def name2lcaodensity(name):
    if name.lower() == 'orca':
        from .lcao_density.orca import ORCADensityCalculator
        return ORCADensityCalculator
    elif name.lower() == 'nwchem':
        from .lcao_density.nwchem import NWChemLCAODensityCalculator
        return NWChemLCAODensityCalculator
    elif name.lower() == 'pyscf':
        from lcao_density.pyscf import PyScfLCAOCalculator
        return PyScfLCAOCalculator
    else:
        raise NotImplementedError(f'LCAO Density calculator "{name}" not found.')

def name2reggriddensity(name):
    if name.lower() == 'gpaw':
        from .reggr_density.gpaw import GPAWDensityCalculator
        return GPAWDensityCalculator
    else:
        raise NotImplementedError(f'Regular Grid Density calculator "{name}" not found.')

def name2lcaopartition(name):
    if name.lower() == 'nosphera2':
        from .lcao_partition.nosphera2 import NoSpherA2Partitioner
        return NoSpherA2Partitioner
    elif name.lower() == 'horton':
        from .lcao_partition.horton import HortonPartitioner
        return HortonPartitioner
    else:
        raise NotImplementedError(f'LCAO Density partitioner "{name}" not found.')

def name2reggridpartition(name):
    if name.lower() == 'python':
        from .reggr_partition.python import PythonRegGridPartitioner
        return PythonRegGridPartitioner
    elif name.lower() == 'gpaw':
        from .reggr_partition.gpaw import GPAWDensityPartitioner
        return GPAWDensityPartitioner
    else:
        raise NotImplementedError(f'Regular Grid Density partitioner "{name}" not found.')
