from ..constants import ATOMIC_N_ELEC, ANGSTROM_PER_BOHR
import h5py
import numpy as np
from horton.part.proatomdb import ProAtomRecord
from horton.grid.radial import RadialGrid
from horton.grid import RTransform

def hortondb2atomic_density(horton_db_path, element, charge=0):
    horton_key = f'/Z={ATOMIC_N_ELEC[element]}_Q={charge:+d}'
    atom_db = h5py.File(horton_db_path)
    group = atom_db[horton_key]
    grid_name, rmin_str, rmax_str, npoint_str  = group.attrs['rtransform'].split()
    rmin = float(rmin_str)
    rmax = float(rmax_str)
    npoint = int(npoint_str)
    power = np.log(rmax/rmin)/np.log(npoint)
    if charge == 0:
        atom_types = [element] * npoint
    else:
        atom_types = [f'{element}{charge:+d}'] * npoint
    return {
        '_qubox_density_atomic_atom_type': atom_types,
        '_qubox_density_atomic_rgrid': list(rmin*np.arange(1, npoint+1)**power * ANGSTROM_PER_BOHR),
        '_qubox_density_atomic_total': np.array(group['rho']) / ANGSTROM_PER_BOHR**3
    }

def dens_dict2proatomrecord(dens_dict):
    grid_points = np.array(dens_dict['_qubox_density_atomic_rgrid'])/ANGSTROM_PER_BOHR
    for possible_transform in ('ExpRTransform', 'PowerRTransform', 'LinearRTransform'):
        try:
            rtransform = RTransform.from_string(f'{possible_transform} {grid_points.min()} {grid_points.max()} {len(grid_points)}')
            if np.all(np.abs((rtransform.get_radii() / grid_points) - 1) < 1e-7):
                break
        except ValueError:
            pass
    else:
        raise ValueError('Coud not find a suitable equivalent grid for horton')

    atom_type = dens_dict['_qubox_density_atomic_atom_type'][0]
    return ProAtomRecord(
        number=ATOMIC_N_ELEC[atom_type],
        charge=0,
        energy=1e10,
        rgrid=RadialGrid(rtransform),
        rho=np.array(dens_dict['_qubox_density_atomic_total'])*ANGSTROM_PER_BOHR**3,
    )