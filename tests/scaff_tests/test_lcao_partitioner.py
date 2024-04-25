import os
import shutil
from pathlib import Path

import pytest

from qctbx.io.cif import cif2dicts
from qctbx.scaff.lcao_density.orca import ORCADensityCalculator
from qctbx.scaff.lcao_partition.horton import HortonPartitioner
from qctbx.scaff.lcao_partition.nosphera2 import NoSpherA2Partitioner

from ..helper_funcs import new_scif_with_workdir

@pytest.mark.partitioner_runs
@pytest.mark.parametrize('part_base, settings_cif_path', [
    (HortonPartitioner, Path('./tests/scaff_tests/lcao_partitioner_settings/settings_horton.scif')),
    (NoSpherA2Partitioner, Path('./tests/scaff_tests/lcao_partitioner_settings/settings_nosphera2.scif'))
])
def test_water_runs(part_base, settings_cif_path, tmp_path):
    cif_path = './tests/datasets/minimal_tests/Water.cif'
    cif_dataset = 'Water'

    new_scif_path = tmp_path / 'settings.scif'
    new_scif_with_workdir(settings_cif_path, tmp_path, new_scif_path)

    atom_site_dict, cell_dict, space_group_dict, refln_dict = cif2dicts(
        cif_path,
        cif_dataset,
        complete_dmin=True
    )

    calc = ORCADensityCalculator.from_settings_cif(
        new_scif_path,
        cif_dataset
    )

    calc.calc_options['work_directory'] = tmp_path
    density_path =  calc.calculate_density(atom_site_dict, cell_dict)

    part = part_base.from_settings_cif(new_scif_path, cif_dataset)
    part.calc_options['work_directory'] = tmp_path

    f0j, charges = part.calc_f0j(
        atom_labels=list(atom_site_dict['_atom_site_label']),
        atom_site_dict=atom_site_dict,
        cell_dict=cell_dict,
        space_group_dict=space_group_dict,
        refln_dict=refln_dict,
        density_path=density_path
    )

    assert sum(charges) == pytest.approx(0, abs=1e-2)

