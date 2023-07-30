import os
import shutil

import pytest

from qctbx.io.cif import cif2dicts
from qctbx.scaff.LCAODensityCalculators.nwchem import NWChemLCAODensityCalculator
from qctbx.scaff.LCAODensityCalculators.orca import ORCADensityCalculator

@pytest.mark.parametrize('calculator, settings_cif_path, cif_path, cif_dataset', [
    (
        NWChemLCAODensityCalculator,
        './scaff_tests/lcao_density_settings/settings_nwchem.scif',
        './datasets/minimal_tests/Water.cif',
        'Water'
    ),
    (
        ORCADensityCalculator,
        './scaff_tests/lcao_density_settings/settings_orca.scif',
        './datasets/minimal_tests/Water.cif',
        'Water'
    )
])
def test_water_runs(calculator, settings_cif_path, cif_path, cif_dataset):
    atom_site_dict, cell_dict, *_ = cif2dicts(
        cif_path,
        cif_dataset
    )

    chosen_calc = calculator.from_settings_cif(
        settings_cif_path,
        cif_dataset
    )
    work_dir = os.path.join('temp_calculation_dir')
    chosen_calc.calc_options['work_directory'] = work_dir

    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    os.mkdir(work_dir)
    output_file = chosen_calc.calculate_density(atom_site_dict, cell_dict)

    assert os.path.exists(output_file)

    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)

