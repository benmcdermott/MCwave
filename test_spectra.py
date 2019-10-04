## tests for module spectra

import numpy as np
import pytest
import spectra

def test_sort_spec():
    k = np.array([0.0,4.0,2.2,4.5,1.1])   # sorted order [0,4,2,1,3]
    e = np.array([0.0,4.0,2.0,1.0,3.0])

    etest = np.array([0.0,1.0,2.0,3.0,4.0])

    ksort, esort = spectra.sort_spec(k, e)

    # need to think of a good way to test that e is sorted by the indices of k

    assert abs(round(ksort[0] - 1e-8)) <= 1e-8, 'wavenumbers should start from zero'
    assert all(ksort[i] <= ksort[i+1] for i in range(len(ksort-1))), \
        'wavenumbers not sorted in ascending order'
    assert all(esort == etest), 'energy not sorted by wavenumber'


def test_bin_spec():
    k = np.array([0.0,0.1,0.2,2.0,2.1,2.2,3.0,3.1,3.2])
    e = np.linspace(1.0,9.0,9)

    test_spec = np.array([6.0,15.0,22.0])

    spec = spectra.bin_spec(k, e, 2*len(test_spec))

    assert all(spec == test_spec), 'spectrum binned incorrectly'



