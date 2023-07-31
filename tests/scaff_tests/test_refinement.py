import os
import shutil

import random, string

from iotbx import cif

import pytest

from smtbx.refinement import restraints
from smtbx.refinement import least_squares
from smtbx.refinement import constraints
from cctbx.array_family import flex

from qctbx.scaff.scaff_f0j import ScaffF0jSource
from qctbx.refine.basic_refinement import basic_refinement

def randomword(length):
   letters = string.ascii_lowercase
   return ''.join(random.choice(letters) for i in range(length))

@pytest.mark.parametrize('scif_path, partitioning_overwrite', [
    ('./scaff_tests/refinement_settings/settings_nosphera2.scif', None),
    ('./scaff_tests/refinement_settings/settings_horton.scif', 'hirshfeld'),
    ('./scaff_tests/refinement_settings/settings_horton.scif', 'hirshfeld-i'),
    ('./scaff_tests/refinement_settings/settings_horton.scif', 'iterative-stockholder'),
    ('./scaff_tests/refinement_settings/settings_horton.scif', 'mbis'),
    ('./scaff_tests/refinement_settings/settings_python.scif', None),
    ('./scaff_tests/refinement_settings/settings_gpaw.scif', None),
])
def test_refinement(scif_path, partitioning_overwrite):

    cif_path = './datasets/crystal_data/epoxide.cif'
    block_name = 'epoxide'

    with open(cif_path, "r") as f:
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

    work_dir = 'temp_' + randomword(10)
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    os.mkdir(work_dir)

    f0jeval = ScaffF0jSource.from_settings_cif(scif_path, block_name)
    if partitioning_overwrite is not None:
        f0jeval.partitioner.method = partitioning_overwrite
    f0jeval.density_calculator.calc_options['work_directory'] = work_dir
    f0jeval.partitioner.calc_options['work_directory'] = work_dir

    har_convergence_conditions = {
        'position(abs)': 5e-4,
        'position/esd' : 5e-2,
        'uij(abs)': 1e-4,
        'uij/esd': 1e-1,
        'max(cycles)': 20
    }
    tsc_path = os.path.join(work_dir, 'qctbx.tscb')

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

    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)