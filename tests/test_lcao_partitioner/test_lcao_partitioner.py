import os
import shutil

import pytest

from qctbx.io.cif import cif2dicts
from qctbx.scaff.LCAODensityCalculators.orca import ORCADensityCalculator
from qctbx.scaff.LCAODensityPartitioners.horton import HortonPartitioner
from qctbx.scaff.LCAODensityPartitioners.nosphera2 import NoSpherA2Partitioner



@pytest.mark.parametrize('part_base, part_settings', [
    (HortonPartitioner, './test_lcao_partitioner/settings_horton.scif'),
    (NoSpherA2Partitioner, './test_lcao_partitioner/settings_nosphera2.scif')

])
def test_water_runs(part_base, part_settings):
    #calc_settings = './test_lcao_density_minimal/settings_orca.scif'
    cif_path = './datasets/minimal_tests/Water.cif'
    cif_dataset = 'Water'

    atom_site_dict, cell_dict, space_group_dict, refln_dict = cif2dicts(
        cif_path,
        cif_dataset,
        complete_dmin=True
    )

    calc = ORCADensityCalculator.from_settings_cif(
        part_settings,
        cif_dataset
    )
    work_dir = os.path.join('temp_calculation_dir')
    calc.calc_options['work_directory'] = work_dir

    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    os.mkdir(work_dir)
    density_path =  calc.calculate_density(atom_site_dict, cell_dict)

    part = part_base.from_settings_cif(part_settings, cif_dataset)

    f0j, charges = part.calc_f0j(
        atom_labels=list(atom_site_dict['_atom_site_label']),
        atom_site_dict=atom_site_dict,
        cell_dict=cell_dict,
        space_group_dict=space_group_dict,
        refln_dict=refln_dict,
        density_path=density_path
    )

    assert sum(charges) == pytest.approx(0, abs=1e-2)

    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)


