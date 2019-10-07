#### spectra module ####
# contains all functions for calculating 2D spatial spectra (x,y) -> k_perp
# and 3D spatial spectra (x,y,z) -> k

import numpy as np


## to sort wavenumbers, and sort spectrum in order of increasing wavenumber
def sort_spec(kmag, E):
    kflat = np.ndarray.flatten(kmag)
    Eflat = np.ndarray.flatten(E)
    isort = np.argsort(kflat)
    ksort = kflat[isort]
    Esort = Eflat[isort]
    return ksort, Esort


## to bin the spectrum into rings of width dk
def bin_spec(ksort, Esort, nx):
    ind = int(nx / 2)
    k = np.arange(0.0, nx / 2, 1.0)
    dk = k[1] - k[0]
    spec = np.zeros((ind))

    # sum energy in each 2D ring at distance |k|, with width dk
    # np.where finds the indices in ksort where k >= k[i] > k+dk
    # then we sum the energy in that ring of width dk

    for i in range(ind):
        spec[i] = np.sum(Esort[np.where((ksort >= k[i]) & (ksort < k[i]+dk))])

    return spec


## 2D scalar spectrum in a horizontal plane
# s is a horizontal slice (size [nx,nx]) of some scalar quantity
# outputs:  k   - wavenumbers perpendicular to the rotation vector
#          spec - spectrum of s as a function of k
def scalar_spec2d(s, nx):

    # want a 2D slice
    if len(np.shape(s)) > 2:
        raise ValueError('Input array must be 2D')

    # need number of rows = number of columns
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

## 2D kinetic/ magnetic energy spectrum
# fx and fy (size [nx,nx]) are 2D slices of x,y components of a vector field
# outputs: k - perpendicular wavenumbers
#         spec - perpendicular spectrum
def spec2d(fx,fy,nx):

## some checks on input arrays
    # need size(fx) = size(fy)
    assert np.shape(fx) == np.shape(fy), 'input arrays must have same shape'
    assert (np.shape(fx)[0] and np.shape(fx)[1]) == int(nx), 'input arrays must be size [nx,nx]'

    # want a 2D slice
    if (len(np.shape(fx)) or len(np.shape(fy))) > 2:
        raise ValueError('Input arrays must be 2D slices')

    # need number of rows = number of columns
    if not ((np.shape(fx)[0] == np.shape(fx)[1]) and not (np.shape(fy)[0] == np.shape(fy)[1])):
        raise ValueError('Dimensions of input arrays must be equal')

    # form spectrum from 2D transforms of arrays
    factor = 1.0 / nx / nx  # fft normalisation factor
    ind = int(nx / 2)
    dk = 1.0
    k = np.arange(0.0, nx / 2 - 1 + dk, 1.0)  # positive horizontal wavenumbers
    fx_hat = abs(np.fft.fft2(fx))  # 2D ffts
    fy_hat = abs(np.fft.fft2(fy))
    fx_hat = factor * fx_hat[0:ind, 0:ind]  # normalise and cut
    fy_hat = factor * fy_hat[0:ind, 0:ind]

    kmag = np.zeros((ind, ind))
    E = np.zeros((ind, ind))
    for i in range(ind):
        for j in range(ind):
            kmag[i, j] = np.sqrt(k[i] ** 2 + k[j] ** 2)
            E[i, j] = np.pi * kmag[i, j] * ( fx_hat[i, j] * np.conj(fx_hat[i, j]) +
                                             fy_hat[i, j] * np.conj(fy_hat[i, j]) )


    ksort, Esort = sort_spec(kmag, E)

    spec = bin_spec(ksort, Esort, nx)

    return k, spec

## 2D spectrum in a plane with 3 components of the vector field
# see 'spec2d' for details
def spec2d_3comp(fx,fy,fz,nx):

    assert np.shape(fx) == np.shape(fy) == np.shape(fz), 'input arrays must have same shape'
    assert (np.shape(fx)[0] and np.shape(fx)[1]) == int(nx), 'input arrays must be size [nx,nx]'

    if (len(np.shape(fx)) or len(np.shape(fy)) or len(np.shape(fz))) > 2:
        raise ValueError('Input arrays must be 2D slices')

    if not (np.shape(fx)[0] == np.shape(fx)[1]) or (np.shape(fy)[0] == np.shape(fy)[1]) or (
            np.shape(fz)[0] == np.shape(fz)[1]):
        raise ValueError('Dimensions of input arrays must be equal')

    factor = 1.0 / nx / nx
    ind = int(nx / 2)
    dk = 1.0
    k = np.arange(0.0, nx / 2 - 1 + dk, 1.0)
    fx_hat = abs(np.fft.fft2(fx))
    fy_hat = abs(np.fft.fft2(fy))
    fz_hat = abs(np.fft.fft2(fz))
    fx_hat = factor * fx_hat[0:ind, 0:ind]
    fy_hat = factor * fy_hat[0:ind, 0:ind]
    fz_hat = factor * fz_hat[0:ind, 0:ind]

    kmag = np.zeros((ind, ind))
    E = np.zeros((ind, ind))
    for i in range(ind):
        for j in range(ind):
            kmag[i, j] = np.sqrt(k[i] ** 2 + k[j] ** 2)
            E[i, j] = np.pi * kmag[i, j] * ( fx_hat[i, j] * np.conj(fx_hat[i, j]) +
                                             fy_hat[i, j] * np.conj(fy_hat[i, j]) +
                                             fz_hat[i, j] * np.conj(fz_hat[i, j]) )

    ksort, Esort = sort_spec(kmag, E)
    spec = bin_spec(ksort, Esort, nx)

    return k, spec

## Helicity spectra
#  Here we don't take the absolute value, as helicity can take positive
#  or negative values. Warning, we must be careful as the helicity
#  spectrum will be in general be complex - we take the real part

def helicity_spec(ax,ay,az,bx,by,bz,nx):

    assert (np.shape(ax) == np.shape(ay) == np.shape(az) ==
            np.shape(bx) == np.shape(by) == np.shape(bz)), \
            'Input arrays must be the same shape: [nx,nx]'

    assert all(x == nx for x in ax.shape),'Input arrays must be size [nx,nx]'

    factor = 1.0 / nx / nx
    ind = int(nx / 2)
    dk = 1.0
    k = np.arange(0.0, nx / 2 - 1 + dk, 1.0)

    ax_hat = np.fft.fft2(ax)
    ay_hat = np.fft.fft2(ay)
    az_hat = np.fft.fft2(az)
    ax_hat = factor * ax_hat[0:ind, 0:ind]
    ay_hat = factor * ay_hat[0:ind, 0:ind]
    az_hat = factor * az_hat[0:ind, 0:ind]

    bx_hat = np.fft.fft2(bx)
    by_hat = np.fft.fft2(by)
    bz_hat = np.fft.fft2(bz)
    bx_hat = factor * bx_hat[0:ind, 0:ind]
    by_hat = factor * by_hat[0:ind, 0:ind]
    bz_hat = factor * bz_hat[0:ind, 0:ind]

    kmag = np.zeros((ind, ind))
    H = np.zeros((ind, ind))
    for i in range(ind):
        for j in range(ind):
            kmag[i, j] = np.sqrt(k[i] ** 2 + k[j] ** 2)
            H[i, j] = np.pi * kmag[i, j] * ( ax_hat[i, j] * np.conj(bx_hat[i, j]) +
                                             ay_hat[i, j] * np.conj(by_hat[i, j]) +
                                             az_hat[i, j] * np.conj(bz_hat[i, j]) )

    ksort, Hsort = sort_spec(kmag, H)
    spec = bin_spec(ksort, Hsort, nx)

    return k, spec