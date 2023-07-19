import numpy as np

# from https://physics.nist.gov/cgi-bin/cuu/Value?bohrrada0
ANGSTROM_PER_BOHR = 0.5291772109

# from https://physics.nist.gov/cgi-bin/cuu/Value?rydhcev
EV_PER_RYDBERG = 13.6056931230

# from https://doi.org/10.1515/pac-2019-0603

ATOMIC_MASSES = {
     'H': 1.0080, 'He': 4.0026, 'Li': 6.94, 'Be': 9.0122, 'B': 10.81, 'C': 12.011, 
     'N': 14.007, 'O': 15.999, 'F': 18.998, 'Ne': 20.180, 'Na': 22.990, 'Mg': 24.305, 
     'Al': 26.982, 'Si': 28.085, 'P': 30.974, 'S': 32.06, 'Cl': 35.45, 'Ar': 39.95, 
     'K': 39.098, 'Ca': 40.078, 'Sc': 44.956, 'Ti': 47.867, 'V': 50.942, 'Cr': 51.996, 
     'Mn': 54.938, 'Fe': 55.845, 'Co': 58.933, 'Ni': 58.693, 'Cu': 63.546, 'Zn': 65.38, 
     'Ga': 69.723, 'Ge': 72.630, 'As': 74.922, 'Se': 78.971, 'Br': 79.904, 'Kr': 83.798, 
     'Rb': 85.468, 'Sr': 87.62, 'Y': 88.906, 'Zr': 91.224, 'Nb': 92.906, 'Mo': 95.95, 
     'Tc': np.nan, 'Ru': 101.07, 'Rh': 102.91, 'Pd': 106.42, 'Ag': 107.87, 'Cd': 112.41, 
     'In': 114.82, 'Sn': 118.71, 'Sb': 121.76, 'Te': 127.60, 'I': 126.90, 'Xe': 131.29, 
     'Cs': 132.91, 'Ba': 137.33, 'La': 138.91, 'Ce': 140.12, 'Pr': 140.91, 'Nd': 144.24, 
     'Pm': np.nan, 'Sm': 150.36, 'Eu': 151.96, 'Gd': 157.25, 'Tb': 158.93, 'Dy': 162.50, 
     'Ho': 164.93, 'Er': 167.26, 'Tm': 168.93, 'Yb': 173.05, 'Lu': 174.97, 'Hf': 178.49, 
     'Ta': 180.95, 'W': 183.84, 'Re': 186.21, 'Os': 190.23, 'Ir': 192.22, 'Pt': 195.08, 
     'Au': 196.97, 'Hg': 200.59, 'Tl': 204.38, 'Pb': 207.2, 'Bi': 208.98, 'Po': np.nan, 
     'At': np.nan, 'Rn': np.nan, 'Fr': np.nan, 'Ra': np.nan, 'Ac': np.nan, 'Th': 232.04, 
     'Pa': 231.04, 'U': 238.03, 'Np': np.nan, 'Pu': np.nan, 'Am': np.nan, 'Cm': np.nan, 
     'Bk': np.nan, 'Cf': np.nan, 'Es': np.nan, 'Fm': np.nan, 'Md': np.nan, 'No': np.nan, 
     'Lr': np.nan, 'Rf': np.nan, 'Db': np.nan, 'Sg': np.nan, 'Bh': np.nan, 'Hs': np.nan, 
     'Mt': np.nan, 'Ds': np.nan, 'Rg': np.nan, 'Cn': np.nan, 'Nh': np.nan, 'Fl': np.nan, 
     'Mc': np.nan, 'Lv': np.nan, 'Ts': np.nan, 'Og': np.nan
}

ATOMIC_N_ELEC = {symbol: index + 1 for index, symbol in enumerate(ATOMIC_MASSES.keys())}