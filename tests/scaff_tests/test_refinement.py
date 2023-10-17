import os

from iotbx import cif

import pytest

from smtbx.refinement import restraints

from qctbx.scaff.scaff_f0j import ScaffF0jSource
from qctbx.refine.basic_refinement import basic_refinement

@pytest.mark.refine
@pytest.mark.parametrize('scif_path, partitioning_overwrite', [
    ('./tests/scaff_tests/refinement_settings/settings_nosphera2.scif', 'hirshfeld'),
    ('./tests/scaff_tests/refinement_settings/settings_horton.scif', 'hirshfeld'),
    ('./tests/scaff_tests/refinement_settings/settings_horton.scif', 'hirshfeld-i'),
    ('./tests/scaff_tests/refinement_settings/settings_horton.scif', 'iterative-stockholder'),
    ('./tests/scaff_tests/refinement_settings/settings_horton.scif', 'mbis'),
    ('./tests/scaff_tests/refinement_settings/settings_python.scif', 'hirshfeld'),
    ('./tests/scaff_tests/refinement_settings/settings_gpaw.scif', 'hirshfeld'),
])
def test_refinement(scif_path, partitioning_overwrite, tmp_path):

    test_id = scif_path.split('_')[-1][:-4]
    if partitioning_overwrite is not None:
        test_id += partitioning_overwrite

    cif_path = './tests/datasets/crystal_data/epoxide.cif'
    block_name = 'epoxide'

    with open(cif_path, "r", encoding='ASCII') as f:
        ciftbx_object = cif.reader(file_object=f)
    structure = ciftbx_object.build_crystal_structures()[block_name]

    miller_array = ciftbx_object.as_miller_arrays(
        data_block_name=block_name,
        crystal_symmetry=structure.crystal_symmetry(),
    )[0]

    restraints_manager = restraints.manager()

    for sc in structure.scatterers():
        if sc.scattering_type != 'H':
            sc.flags.set_use_u_iso(False)
            sc.flags.set_use_u_aniso(True)
            sc.flags.set_grad_site(True)
            sc.flags.set_grad_u_aniso(True)
        else:
            sc.flags.set_use_u_iso(True)
            sc.flags.set_use_u_aniso(False)
            sc.flags.set_grad_site(True)
            sc.flags.set_grad_u_iso(True)

    f0jeval = ScaffF0jSource.from_settings_cif(scif_path, block_name)
    if partitioning_overwrite is not None:
        f0jeval.partitioner.method = partitioning_overwrite
    f0jeval.density_calculator.calc_options['work_directory'] = tmp_path
    f0jeval.partitioner.calc_options['work_directory'] = tmp_path

    har_convergence_conditions = {
        'position(abs)': 5e-4,
        'position/esd' : 5e-2,
        'uij(abs)': 1e-4,
        'uij/esd': 1e-1,
        'max(cycles)': 20
    }
    tsc_path = os.path.join(tmp_path, 'qctbx.tscb')

    structure, normal_eqns = basic_refinement(
        structure,
        miller_array,
        f0jeval,
        har_convergence_conditions,
        [],
        restraints_manager,
        tsc_path=tsc_path,
        update_weights=False
    )

    assert normal_eqns.wR2() < 0.12, 'too large wR2'
