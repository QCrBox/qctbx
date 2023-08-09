import json
import os
import shutil
import subprocess

from qctbx.cli import known_calcs_parts

import pytest

cif_path = './tests/datasets/crystal_data/epoxide.cif'
block_name = 'epoxide'

@pytest.mark.cli
@pytest.mark.parametrize('test_id, scif_path, checks', [
    ('horton', './tests/cli/cli_settings/settings_horton.scif', {'lcaowfn;orca': True, 'lcaopartition;horton': True}),
    ('nosphera2', './tests/cli/cli_settings/settings_nosphera2.scif', {'lcaowfn;orca': True, 'lcaopartition;nosphera2': True}),
    ('gpaw', './tests/cli/cli_settings/settings_gpaw.scif', {'reggridwfn;gpaw': True, 'reggridpartition;gpaw': True}),
    ('python', './tests/cli/cli_settings/settings_python.scif', {'reggridwfn;gpaw': True, 'reggridpartition;python': True})
])
def test_cli_available_scif(test_id, scif_path, checks):
    work_dir = f'./tests/temp_scifavailable_{test_id}'
    out_json_path = os.path.join(work_dir, 'available.json')
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    os.mkdir(work_dir)

    r = subprocess.call(f'python -m qctbx available --scif_path {scif_path} --block_name {block_name} --output_json {out_json_path}', shell=True)

    assert r == 0, 'Error in subprocess runtime'
    with open(out_json_path, 'r', encoding='UTF-8') as fobj:
        results_dict = json.load(fobj)

    assert all(results_dict[check_key] == check_result for check_key, check_result in checks.items()), 'Output does not match expectations'

    shutil.rmtree(work_dir)

@pytest.mark.cli
def test_cli_available_options():
    work_dir = './tests/temp_available'
    out_json_path = os.path.join(work_dir, 'available.json')
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    os.mkdir(work_dir)
    flags = ''
    for flag, options in known_calcs_parts.items():
        flags += '--' + flag + ' ' + ' '.join(options) + ' dummy '

    r = subprocess.call(f'python -m qctbx available {flags} --output_json {out_json_path}', shell=True)
    assert r == 0, 'Error in subprocess runtime'

    with open(out_json_path, 'r', encoding='UTF-8') as fobj:
        results_dict = json.load(fobj)
    for flag, options in known_calcs_parts.items():
        for option in options:
            assert results_dict[f'{flag};{option}']

        assert not results_dict[f'{flag};dummy']

    shutil.rmtree(work_dir)

@pytest.mark.cli
@pytest.mark.parametrize('test_id, scif_path', [
    ('horton', './tests/cli/cli_settings/settings_horton.scif'),
    ('nosphera2', './tests/cli/cli_settings/settings_nosphera2.scif'),
    ('gpaw', './tests/cli/cli_settings/settings_gpaw.scif'),
    ('python', './tests/cli/cli_settings/settings_python.scif')
])
def test_cli_citations(test_id, scif_path):
    work_dir = f'./tests/temp_cititations_{test_id}'
    out_json_path = os.path.join(work_dir, 'available.json')
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    os.mkdir(work_dir)

    r = subprocess.call(f'python -m qctbx citation --scif_path {scif_path} --block_name {block_name} --output_json {out_json_path}', shell=True)

    assert r == 0, 'Error in subprocess runtime'
    with open(out_json_path, 'r', encoding='UTF-8') as fobj:
        results_dict = json.load(fobj)
    assert 'author' in results_dict['bibtex']
    assert len(results_dict['description']) > 0

    shutil.rmtree(work_dir)

@pytest.mark.cli
@pytest.mark.parametrize('test_id, scif_path, periodic', [
    ('horton', './tests/cli/cli_settings/settings_horton.scif', False),
    ('nosphera2', './tests/cli/cli_settings/settings_nosphera2.scif', False),
    ('gpaw', './tests/cli/cli_settings/settings_gpaw.scif', True),
    ('python', './tests/cli/cli_settings/settings_python.scif', True)
])
def test_cli_density_partition(test_id, scif_path, periodic):
    work_dir = f'./tests/temp_denspart_{test_id}'
    density_text_path = os.path.join(work_dir, 'density_path.txt')
    tsc_path = os.path.join(work_dir, 'qctbx.tscb')
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    os.mkdir(work_dir)

    if periodic:
        used_cif_path = os.path.join(work_dir, 'expanded.cif')
        r = subprocess.call(f'python -m qctbx symm_expand --cif_path {cif_path} --block_name {block_name} --to_p1 --out_cif_path {used_cif_path}', shell=True)
        assert r == 0, 'Error in subprocess symm_expand runtime'

    else:
        used_cif_path = cif_path

    r = subprocess.call(f'python -m qctbx density --cif_path {used_cif_path} --scif_path {scif_path} --block_name {block_name} --text_output_path {density_text_path}', shell=True)

    assert r == 0, 'Error in subprocess density runtime'

    with open(density_text_path, 'r', encoding='UTF-8') as fobj:
        density_path = fobj.read().strip()

    atom_labels_str = ' '.join(['O1', 'C2', 'H2a', 'H2b', 'C3', 'H3a', 'H3b'])
    r = subprocess.call(f'python -m qctbx partition --cif_path {cif_path} --scif_path {scif_path} --block_name {block_name} --input_wfn_path {density_path} --atom_labels {atom_labels_str} --tsc_path {tsc_path}', shell=True)

    assert r == 0, 'Error in subprocess partition runtime'

    assert os.path.exists(tsc_path)

    shutil.rmtree(work_dir)

@pytest.mark.cli
@pytest.mark.parametrize('test_id, scif_path', [
    ('horton', './tests/cli/cli_settings/settings_horton.scif'),
    ('nosphera2', './tests/cli/cli_settings/settings_nosphera2.scif'),
    ('gpaw', './tests/cli/cli_settings/settings_gpaw.scif'),
    ('python', './tests/cli/cli_settings/settings_python.scif')
])
def test_cli_tsc(test_id, scif_path):
    work_dir = f'./tests/temp_denspart_{test_id}'
    tsc_path = os.path.join(work_dir, 'qctbx.tscb')
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    os.mkdir(work_dir)

    r = subprocess.call(f'python -m qctbx tsc --cif_path {cif_path} --scif_path {scif_path} --block_name {block_name} --tsc_path {tsc_path}', shell=True)

    assert r == 0, 'Error in subprocess tsc runtime'

    assert os.path.exists(tsc_path)

    shutil.rmtree(work_dir)
