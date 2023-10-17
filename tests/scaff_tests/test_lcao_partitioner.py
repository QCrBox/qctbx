import os
import shutil

import pytest

from qctbx.io.cif import cif2dicts
from qctbx.scaff.lcao_density.orca import ORCADensityCalculator
from qctbx.scaff.lcao_partition.horton import HortonPartitioner
from qctbx.scaff.lcao_partition.nosphera2 import NoSpherA2Partitioner

def new_scif_with_workdir(input_scif_path, work_dir, output_scif_path):
    work_dir_str = str(work_dir)
    with open(input_scif_path, 'r', encoding='ASCII') as fo:
        scif_content = fo.read()

    with open(output_scif_path, 'w', encoding='ASCII') as fo:
        fo.write(scif_content.replace("$WORKDIRPLACEHOLDER", work_dir_str))

@pytest.mark.partitioner_runs
@pytest.mark.parametrize('part_base, part_settings', [
    (HortonPartitioner, './tests/scaff_tests/lcao_partitioner_settings/settings_horton.scif'),
    (NoSpherA2Partitioner, './tests/scaff_tests/lcao_partitioner_settings/settings_nosphera2.scif')
])
def test_water_runs(part_base, part_settings, tmp_path):
    #calc_settings = './test_lcao_density_minimal/settings_orca.scif'
    cif_path = './tests/datasets/minimal_tests/Water.cif'
    cif_dataset = 'Water'
    if isinstance(part, HortonPartitioner):
        with open(part_settings, 'r', encoding='ASCII') as fo
            part_settings_content = fo.read()
        with open()

        part.calc_options['log_file'] = tmp_path / 'horton.log'


    atom_site_dict, cell_dict, space_group_dict, refln_dict = cif2dicts(
        cif_path,
        cif_dataset,
        complete_dmin=True
    )

    calc = ORCADensityCalculator.from_settings_cif(
        part_settings,
        cif_dataset
    )

    calc.calc_options['work_directory'] = tmp_path
    density_path =  calc.calculate_density(atom_site_dict, cell_dict)

    part = part_base.from_settings_cif(part_settings, cif_dataset)
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

