import pytest

from qctbx.io.cif import cif2dicts
from qctbx.scaff.reggr_density.gpaw import GPAWDensityCalculator
from qctbx.scaff.reggr_partition.python import PythonRegGridPartitioner
from qctbx.scaff.reggr_partition.gpaw import GPAWDensityPartitioner

from ..helper_funcs import new_scif_with_workdir

@pytest.mark.partitioner_runs
@pytest.mark.parametrize('part_base, settings_cif_path', [
    (PythonRegGridPartitioner, './tests/scaff_tests/reggrid_partitioner_settings/settings_python.scif'),
    (GPAWDensityPartitioner, './tests/scaff_tests/reggrid_partitioner_settings/settings_gpaw.scif')

])
def test_water_runs(part_base, settings_cif_path, tmp_path):
    new_scif_path = tmp_path / 'settings.scif'
    new_scif_with_workdir(settings_cif_path, tmp_path, new_scif_path)

    cif_path = './tests/datasets/minimal_tests/Water.cif'
    cif_dataset = 'Water'

    atom_site_dict, cell_dict, space_group_dict, refln_dict = cif2dicts(
        cif_path,
        cif_dataset,
        complete_dmin=True
    )

    calc = GPAWDensityCalculator.from_settings_cif(
        new_scif_path,
        cif_dataset
    )

    density_path =  calc.calculate_density(atom_site_dict, cell_dict)

    part = part_base.from_settings_cif(new_scif_path, cif_dataset)

    f0j, charges = part.calc_f0j(
        atom_labels=list(atom_site_dict['_atom_site_label']),
        atom_site_dict=atom_site_dict,
        cell_dict=cell_dict,
        space_group_dict=space_group_dict,
        refln_dict=refln_dict,
        density_path=density_path
    )

    assert sum(charges) == pytest.approx(0, abs=1e-2)
