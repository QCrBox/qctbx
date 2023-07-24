import json
import os
import shutil

from qctbx.io.cif import cif2dicts
from qctbx.scaff.LCAODensityCalculators.nwchem import NWChemLCAODensityCalculator
from ase.calculators.calculator import CalculationFailed
folder = 'test_lcao_density_minimal'

def test_water_nwchem_runs():
    with open(os.path.join(folder, 'test_settings.json'), 'r', encoding='UTF-8') as fobj:
        test_settings = json.load(fobj)
    atom_site_dict, cell_dict, *_ = cif2dicts(
        os.path.join(folder, test_settings["cif_path"]),
        test_settings["cif_dataset"]
    )

    nwchem_calc = NWChemLCAODensityCalculator.from_settings_cif(
        os.path.join(folder, test_settings["settings_path"]),
        test_settings["cif_dataset"]
    )
    work_dir = os.path.join('temp_calculation_dir')
    nwchem_calc.calc_options['work_directory'] = work_dir

    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    os.mkdir(work_dir)
    nwchem_calc.calculate_density(atom_site_dict, cell_dict)
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)

    assert True

