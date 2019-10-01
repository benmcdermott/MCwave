#### spectra module ####
# to contain all functions for calculating various 2D/3D spectra

import numpy as np

# to sort wavenumbers, and sort specturm in order of increasing wavenumber
def sort_spec(kmag,E):
    kflat = np.ndarray.flatten(kmag)
    Eflat = np.ndarray.flatten(np.real(E))
    isort = np.argsort(kflat)
    ksort = kflat[isort]
    Esort = Eflat[isort]
    return ksort, Esort

# to bin the spectrum into rings of width dk
def bin_spec(ksort,Esort,Nspec):
    k = np.arange(0.0,Nspec,1.0)
    dk = k[1]-k[0]
    spec = np.zeros((Nspec,1))
    for i in range(Nspec):
        for j in range(len(ksort)):
            if ksort[j] >= k[i]:
                if ksort[j] < k[i]+dk:
                    spec[i] += Esort[j]
    return spec

# 2D scalar spectrum in a horizontal plane
def scalar_spec2d(s):
    if len(np.shape(s)) > 2:
        raise ValueError('Input array must be 2D')
    if np.shape(s)[0] != np.shape(s)[1]:
        raise ValueError('Dimensions of input array must be equal')

    nx = np.shape(s)[0]
    factor = float(1.0/nx/nx)                                  # fft normalisation factor
    ind = int(nx/2)
    dk = 1.0
    k = np.arange(0.0,nx/2-1+dk,1.0)                         # positive horizontal wavenumbers
    s_hat = abs(np.fft.fft2(s))                              # 2D fft
    s_hat = factor*s_hat[0:ind,0:ind]                   # normalise and cut

    kmag = np.zeros((ind,ind))
    E = np.zeros((ind,ind))
    for i in range(ind):
        for j in range(ind):
            kmag[i,j] = np.sqrt(k[i]**2 + k[j]**2)
            E[i,j] = np.pi*kmag[i,j]*s_hat[i,j]*np.conj(s_hat[i,j])

    ksort, Esort = sort_spec(kmag,E)

    spec = bin_spec(ksort,Esort,ind)
