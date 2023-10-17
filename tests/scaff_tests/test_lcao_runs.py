import os
import shutil

import pytest

from qctbx.io.cif import cif2dicts
from qctbx.scaff.lcao_density.nwchem import NWChemLCAODensityCalculator
from qctbx.scaff.lcao_density.orca import ORCADensityCalculator

@pytest.mark.work
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


# TODO: Test that the output from fully set up in python and scif is equal

