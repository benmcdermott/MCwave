#### spectra module ####
# to contain all functions for calculating various 2D/3D spectra

import numpy as np


# to sort wavenumbers, and sort spectrum in order of increasing wavenumber
def sort_spec(kmag, E):
    kflat = np.ndarray.flatten(kmag)
    Eflat = np.ndarray.flatten(E)
    isort = np.argsort(kflat)
    ksort = kflat[isort]
    Esort = Eflat[isort]
    return ksort, Esort


# to bin the spectrum into rings of width dk
def bin_spec(ksort, Esort, nx):
    ind = int(nx / 2)
    k = np.arange(0.0, nx / 2, 1.0)
    dk = k[1] - k[0]
    spec = np.zeros((ind, 1))
    for i in range(ind):
    # sum energy in each 2D ring at distance |k|, with width dk
        # np.where finds the indices in ksort where k >= k[i] > k+dk
        # then we sum the energy in that ring of width dk
        spec[i] = np.sum(Esort[np.where((ksort >= k[i]) & (ksort < k[i]+dk))])
    return spec


# 2D scalar spectrum in a horizontal plane
def scalar_spec2d(s, nx):

    # want a 2D slice
    if len(np.shape(s)) > 2:
        raise ValueError('Input array must be 2D')

    # need number of rows=number of columns
    if np.shape(s)[0] != np.shape(s)[1]:
        raise ValueError('Dimensions of input array must be equal')

    factor = 1.0 / nx / nx  # fft normalisation factor
    ind = int(nx / 2)
    dk = 1.0
    k = np.arange(0.0, nx / 2 - 1 + dk, 1.0)  # positive horizontal wavenumbers
    s_hat = abs(np.fft.fft2(s))  # 2D fft
    s_hat = factor * s_hat[0:ind, 0:ind]  # normalise and cut

    kmag = np.zeros((ind, ind))
    E = np.zeros((ind, ind))
    for i in range(ind):
        for j in range(ind):
            kmag[i, j] = np.sqrt(k[i] ** 2 + k[j] ** 2)
            E[i, j] = np.pi * kmag[i, j] * s_hat[i, j] * np.conj(s_hat[i, j])

    ksort, Esort = sort_spec(kmag, E)

    spec = bin_spec(ksort, Esort, nx)

    return k, spec
