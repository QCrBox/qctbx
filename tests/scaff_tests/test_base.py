import numpy as np
import pytest
from scitbx.array_family.flex import double

from qctbx.scaff.base_classes import is_data_array

@pytest.mark.small_funcs
def test_is_data_array():
    test_np_array = np.random.random(20 * 2)

    assert is_data_array(test_np_array)
    assert is_data_array(list(test_np_array))
    assert is_data_array(tuple(test_np_array))
    assert is_data_array(double(test_np_array))
    assert not is_data_array('somestring')
    assert not is_data_array({'a': 5})

