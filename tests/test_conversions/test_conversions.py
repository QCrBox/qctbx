import numpy as np
import pytest

from qctbx.conversions import symm_mat_vec2str, symm_to_matrix_vector


@pytest.mark.parametrize('test_input, mat_result, vec_result', [
        ('x,y,z' , np.eye(3), np.zeros(3)),
        ('X,Y,Z' , np.eye(3), np.zeros(3)),
        (' x, y, z' , np.eye(3), np.zeros(3)),
        ('-x,-Y,-z' , -np.eye(3), np.zeros(3)),
        ('1/2+x,1/2+y,1/2+z' , np.eye(3), 0.5 * np.ones(3)),
        ('x+0.5,1/2+y,z-1/2' , np.eye(3), np.array([0.5, 0.5, -0.5])),
])
def test_symm_to_matrix_vector(test_input, mat_result, vec_result):
    symm_mat, symm_vec = symm_to_matrix_vector(test_input)
    assert np.sum(np.abs(symm_mat - mat_result)) == 0.0
    assert np.sum(np.abs(symm_vec - vec_result)) == 0.0

@pytest.mark.parametrize('matrix, vector', [
        (np.eye(3), np.zeros(3)),
        (-np.eye(3), np.zeros(3)),
        (np.eye(3), 0.5 * np.ones(3)),
        (np.eye(3), np.array([0.5, 0.5, -0.5])),
        (np.rot90(np.eye(3)), np.array([1/6, 1/8, 1/4]))
])
def test_symm_back_and_forth(matrix, vector):
    string = symm_mat_vec2str(matrix, vector)
    new_mat, new_vec = symm_to_matrix_vector(string)
    assert np.sum(np.abs(matrix - new_mat)) == 0.0
    assert np.sum(np.abs(vector - new_vec)) == 0.0
