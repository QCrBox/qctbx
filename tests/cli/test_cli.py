import json
import os
import subprocess

from qctbx.scaff.cli import known_calcs_parts

import pytest

cif_path = './tests/datasets/crystal_data/epoxide.cif'
block_name = 'epoxide'

def new_scif_with_workdir(input_scif_path, work_dir, output_scif_path):
    work_dir_str = str(work_dir)
    with open(input_scif_path, 'r', encoding='ASCII') as fo:
        scif_content = fo.read()

    with open(output_scif_path, 'w', encoding='ASCII') as fo:
        fo.write(scif_content.replace("$WORKDIRPLACEHOLDER", work_dir_str))

@pytest.mark.density_runs
@pytest.mark.parametrize('test_id, scif_path, checks', [
    ('horton', './tests/cli/cli_settings/settings_horton.scif', {'lcaowfn;orca': True, 'lcaopartition;horton': True}),
    ('nosphera2', './tests/cli/cli_settings/settings_nosphera2.scif', {'lcaowfn;orca': True, 'lcaopartition;nosphera2': True}),
    ('gpaw', './tests/cli/cli_settings/settings_gpaw.scif', {'reggridwfn;gpaw': True, 'reggridpartition;gpaw': True}),
    ('python', './tests/cli/cli_settings/settings_python.scif', {'reggridwfn;gpaw': True, 'reggridpartition;python': True})
])
def test_cli_available_scif(test_id, scif_path, checks, tmp_path):
    out_json_path = os.path.join(tmp_path, 'available.json')
    use_scif_path = os.path.join(tmp_path, f'{test_id}.scif')
    new_scif_with_workdir(scif_path, tmp_path, use_scif_path)

    r = subprocess.call(f'python -m qctbx.scaff available --scif_path {use_scif_path} --block_name {block_name} --output_json {out_json_path}', shell=True)

    assert r == 0, 'Error in subprocess runtime'
    with open(out_json_path, 'r', encoding='UTF-8') as fobj:
        results_dict = json.load(fobj)

    assert all(results_dict[check_key] == check_result for check_key, check_result in checks.items()), 'Output does not match expectations'


@pytest.mark.cli
def test_cli_available_options(tmp_path):
    out_json_path = os.path.join(tmp_path, 'available.json')
    flags = ''
    for flag, options in known_calcs_parts.items():
        flags += '--' + flag + ' ' + ' '.join(options) + ' dummy '

    r = subprocess.call(f'python -m qctbx.scaff available {flags} --output_json {out_json_path}', shell=True)
    assert r == 0, 'Error in subprocess runtime'

    with open(out_json_path, 'r', encoding='UTF-8') as fobj:
        results_dict = json.load(fobj)
    for flag, options in known_calcs_parts.items():
        for option in options:
            assert results_dict[f'{flag};{option}'], f'the option with following flag is not available: {flag};{option}'

        assert not results_dict[f'{flag};dummy'], f'the option with following flag is not available: {flag};dummy'

@pytest.mark.cli
@pytest.mark.parametrize('test_id, scif_path', [
    ('horton', './tests/cli/cli_settings/settings_horton.scif'),
    ('nosphera2', './tests/cli/cli_settings/settings_nosphera2.scif'),
    ('gpaw', './tests/cli/cli_settings/settings_gpaw.scif'),
    ('python', './tests/cli/cli_settings/settings_python.scif')
])
def test_cli_citations(test_id, scif_path, tmp_path):
    out_json_path = os.path.join(tmp_path, 'available.json')
    use_scif_path = os.path.join(tmp_path, f'{test_id}.scif')
    new_scif_with_workdir(scif_path, tmp_path, use_scif_path)

    r = subprocess.call(f'python -m qctbx.scaff citation --scif_path {use_scif_path} --block_name {block_name} --output_json {out_json_path}', shell=True)

    assert r == 0, 'Error in subprocess runtime'
    with open(out_json_path, 'r', encoding='UTF-8') as fobj:
        results_dict = json.load(fobj)
    assert 'author' in results_dict['bibtex']
    assert len(results_dict['description']) > 0


@pytest.mark.cli
@pytest.mark.parametrize('test_id, scif_path, periodic', [
    ('horton', './tests/cli/cli_settings/settings_horton.scif', False),
    ('nosphera2', './tests/cli/cli_settings/settings_nosphera2.scif', False),
    ('gpaw', './tests/cli/cli_settings/settings_gpaw.scif', True),
    ('python', './tests/cli/cli_settings/settings_python.scif', True)
])
def test_cli_density_partition(test_id, scif_path, periodic, tmp_path):
    density_text_path = os.path.join(tmp_path, 'density_path.txt')
    tsc_path = os.path.join(tmp_path, 'qctbx.tscb')

    use_scif_path = os.path.join(tmp_path, f'{test_id}.scif')
    new_scif_with_workdir(scif_path, tmp_path, use_scif_path)

    if periodic:
        used_cif_path = os.path.join(tmp_path, 'expanded.cif')
        r = subprocess.call(f'python -m qctbx.scaff symm_expand --cif_path {cif_path} --block_name {block_name} --to_p1 --out_cif_path {used_cif_path}', shell=True)
        assert r == 0, 'Error in subprocess symm_expand runtime'

    else:
        used_cif_path = cif_path

    r = subprocess.call(f'python -m qctbx.scaff density --cif_path {used_cif_path} --scif_path {use_scif_path} --block_name {block_name} --text_output_path {density_text_path}', shell=True)

    assert r == 0, 'Error in subprocess density runtime'

    with open(density_text_path, 'r', encoding='UTF-8') as fobj:
        density_path = fobj.read().strip()

    atom_labels_str = ' '.join(['O1', 'C2', 'H2a', 'H2b', 'C3', 'H3a', 'H3b'])
    r = subprocess.call(f'python -m qctbx.scaff partition --cif_path {cif_path} --scif_path {use_scif_path} --block_name {block_name} --input_wfn_path {density_path} --atom_labels {atom_labels_str} --tsc_path {tsc_path}', shell=True)

    assert r == 0, 'Error in subprocess partition runtime'

    assert os.path.exists(tsc_path)


@pytest.mark.cli
@pytest.mark.parametrize('test_id, scif_path', [
    ('horton', './tests/cli/cli_settings/settings_horton.scif'),
    ('nosphera2', './tests/cli/cli_settings/settings_nosphera2.scif'),
    ('gpaw', './tests/cli/cli_settings/settings_gpaw.scif'),
    ('python', './tests/cli/cli_settings/settings_python.scif')
])
def test_cli_tsc(test_id, scif_path, tmp_path):
    tsc_path = os.path.join(tmp_path, 'qctbx.tscb')

    use_scif_path = os.path.join(tmp_path, f'{test_id}.scif')
    new_scif_with_workdir(scif_path, tmp_path, use_scif_path)

    r = subprocess.call(f'python -m qctbx.scaff tsc --cif_path {cif_path} --scif_path {use_scif_path} --block_name {block_name} --tsc_path {tsc_path}', shell=True)

    assert r == 0, 'Error in subprocess tsc runtime'

    assert os.path.exists(tsc_path)

