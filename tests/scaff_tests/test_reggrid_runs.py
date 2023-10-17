import os
import shutil

import pytest

from qctbx.io.cif import cif2dicts
from qctbx.scaff.reggr_density.gpaw import GPAWDensityCalculator

@pytest.mark.density_runs
@pytest.mark.parametrize('calculator, settings_cif_path, cif_path, cif_dataset', [
    (
        GPAWDensityCalculator,
        './tests/scaff_tests/reggrid_density_settings/settings_gpaw.scif',
        './tests/datasets/minimal_tests/Water.cif',
        'Water'
    )
])
def test_water_runs(calculator, settings_cif_path, cif_path, cif_dataset, tmp_path):
    atom_site_dict, cell_dict, *_ = cif2dicts(
        cif_path,
        cif_dataset
    )

    chosen_calc = calculator.from_settings_cif(
        settings_cif_path,
        cif_dataset
    )
    chosen_calc.calc_options['work_directory'] = tmp_path

    output_file = chosen_calc.calculate_density(atom_site_dict, cell_dict)

    assert os.path.exists(output_file)
