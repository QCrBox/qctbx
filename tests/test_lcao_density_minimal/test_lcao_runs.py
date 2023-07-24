import json
import os
import shutil

import pytest

from qctbx.io.cif import cif2dicts
from qctbx.scaff.LCAODensityCalculators.nwchem import NWChemLCAODensityCalculator
from qctbx.scaff.LCAODensityCalculators.orca import ORCADensityCalculator
folder = 'test_lcao_density_minimal'

@pytest.mark.parametrize('calculator, settings_cif_path, cif_path, cif_dataset', [
    (
        NWChemLCAODensityCalculator,
        './settings_nwchem.scif',
        '../datasets/minimal_tests/Water.cif',
        'Water'
    ),
    (
        ORCADensityCalculator,
        './settings_orca.scif',
        '../datasets/minimal_tests/Water.cif',
        'Water'
    )
])
def test_water_runs(calculator, settings_cif_path, cif_path, cif_dataset):
    atom_site_dict, cell_dict, *_ = cif2dicts(
        os.path.join(folder, cif_path),
        cif_dataset
    )

    nwchem_calc = calculator.from_settings_cif(
        os.path.join(folder, settings_cif_path),
        cif_dataset
    )
    work_dir = os.path.join('temp_calculation_dir')
    nwchem_calc.calc_options['work_directory'] = work_dir

    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    os.mkdir(work_dir)
    output_file = nwchem_calc.calculate_density(atom_site_dict, cell_dict)

    assert os.path.exists(output_file)

    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)

