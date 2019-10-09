# Plotting module - functions for making nice plots

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from helicity import *

# axial velocity magnitude controls transparency
# relative helicity controls colour
def vz_coloredby_hr(sub,t,case):

    plt.style.use('benmc')

    # path to binary files
    path = '/vistabella/BRM35_4T/hirm_long/'
    outpath = '/home/brm35/work/hirm/cloud/long/'

    # Box size normalised by blob size
    delta = 0.125
    Lx = 10.0*np.pi / delta
    Ly = 2.0*np.pi / delta

    # Spatial resolution
    NX = 2048
    NY = 512
    NZ = NY
    shape = (NX,NY,NZ)

    # Spatial grid vectors
    x = np.linspace(-Lx/2.0,Lx/2.0,NX)
    y = np.linspace(-Ly/2.0,Ly/2.0,NY)

    # Load in velocity data and calculate relative helicity
    str1 = str(t).zfill(4)
    vz = np.fromfile(path+sub+'vx.'+str1+'.out',dtype=np.float32).reshape(shape,order='F')

    if case < 3:
        hr = relative_helicity(path+sub,11,shape,case)
    elif case == 3:
        hr = np.abs(relative_helicity(path+sub,11,shape,case))
    elif case == 4:
        hr = emfx_norm(path+sub, t, shape)

    # take x-z slice
    cut = int(NY/2)
    vz_slice = vz[:,:,cut]
    hr_slice = hr[:,:,cut]

    # convert hr/emf to a colormap
    hmax = 0.8
    if case == 3:       # cross helicity magnitude
        colors = Normalize(0.0, hmax, clip=True)(hr_slice)
        cmap = plt.cm.inferno
    else:
        colors = Normalize(-hmax, hmax, clip=True)(hr_slice)
        cmap = plt.cm.seismic

    colors = cmap(colors)

    # convert vz to transparency
    vmax = 0.5*np.max(abs(vz_slice[:]))
    alphas = Normalize(0.0, vmax, clip=True)(np.abs(vz_slice))

    # set alpha channel of hr colours
    colors[..., -1] = alphas

    # plot
    plt.figure(figsize=[2.67,8])
    plt.imshow(colors, origin='lower', extent=(-25,25,-125,125))
    ax=plt.gca()
    ax.set_aspect('equal')
    plt.xlabel(r'$x/\delta$')
    plt.ylabel(r'$z/\delta$')
    plt.xlim(-25,25)
    plt.ylim(-100,100)

    # set output file and save
    if case == 1:
        outfile = outpath+sub+'vzxz_hk_'+str(10*((t-1)/2))+'.png'
    elif case == 2:
        outfile = outpath+sub+'vzxz_hm_'+str(10*((t-1)/2))+'.png'
    elif case == 3:
        outfile = outpath+sub+'vzxz_hc_'+str(10*((t-1)/2))+'.png'
    elif case == 4:
        outfile = outpath+sub+'vzxz_emfx_'+str(10*((t-1)/2))+'.png'

    plt.savefig(outfile, dpi=600, bbox_inches='tight')

