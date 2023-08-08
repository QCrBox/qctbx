import os
import shutil

import pytest

from qctbx.io.cif import cif2dicts
from qctbx.scaff.lcao_density.nwchem import NWChemLCAODensityCalculator
from qctbx.scaff.lcao_density.orca import ORCADensityCalculator

@pytest.mark.density_runs
@pytest.mark.parametrize('calculator, settings_cif_path, cif_path, cif_dataset', [
    (
        NWChemLCAODensityCalculator,
        './tests/scaff_tests/lcao_density_settings/settings_nwchem.scif',
        './tests/datasets/minimal_tests/Water.cif',
        'Water'
    ),
    (
        ORCADensityCalculator,
        './tests/scaff_tests/lcao_density_settings/settings_orca.scif',
        './tests/datasets/minimal_tests/Water.cif',
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

# TODO: Test that the output from fully set up in python and scif is equal

