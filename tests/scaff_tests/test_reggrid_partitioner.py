import pytest

from qctbx.io.cif import cif2dicts
from qctbx.scaff.reggr_density.gpaw import GPAWDensityCalculator
from qctbx.scaff.reggr_partition.python import PythonRegGridPartitioner
from qctbx.scaff.reggr_partition.gpaw import GPAWDensityPartitioner


@pytest.mark.partitioner_runs
@pytest.mark.parametrize('part_base, part_settings', [
    (PythonRegGridPartitioner, './tests/scaff_tests/reggrid_partitioner_settings/settings_python.scif'),
    (GPAWDensityPartitioner, './tests/scaff_tests/reggrid_partitioner_settings/settings_gpaw.scif')

])
def test_water_runs(part_base, part_settings, tmp_path):
    #calc_settings = './test_lcao_density_minimal/settings_orca.scif'
    cif_path = './tests/datasets/minimal_tests/Water.cif'
    cif_dataset = 'Water'

    atom_site_dict, cell_dict, space_group_dict, refln_dict = cif2dicts(
        cif_path,
        cif_dataset,
        complete_dmin=True
    )

    calc = GPAWDensityCalculator.from_settings_cif(
        part_settings,
        cif_dataset
    )
    calc.calc_options['work_directory'] = tmp_path

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

