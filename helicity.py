## module containing functions to compute helicity and electromotive force (emf)

import numpy as np

# computes helicity density
# path is the path to the binary files
# t is the numeric label of binary file
# shape is the shape of the data arrays
# case = 1 : kinetic helicity
# case = 2 : magnetic helicity
# case = 3 : cross helicity

def helicity(path,t,shape,case):

    # kinetic helicity
    if case == 1:
        a = 'v'     # velocity
        b = 'w'     # vorticity
    # magnetic helicity
    elif case == 2:
        a = 'a'     # magnetic potential
        b = 'b'     # magnetic field
    # cross helicity
    elif case == 3:
        a = 'v'     # velocity
        b = 'b'     # magnetic field
    else:
        ValueError('hk: case = 1, hm: case = 2, hc: case = 3')

    str1 = str(t).zfill(4)

    ax = np.fromfile(path+a+'y.'+str1+'.out',dtype=np.float32).reshape(shape,order='F')
    ay = np.fromfile(path+a+'z.'+str1+'.out',dtype=np.float32).reshape(shape,order='F')
    az = np.fromfile(path+a+'x.'+str1+'.out',dtype=np.float32).reshape(shape,order='F')

    bx = np.fromfile(path+b+'y.'+str1+'.out',dtype=np.float32).reshape(shape,order='F')
    by = np.fromfile(path+b+'z.'+str1+'.out',dtype=np.float32).reshape(shape,order='F')
    bz = np.fromfile(path+b+'x.'+str1+'.out',dtype=np.float32).reshape(shape,order='F')

    assert np.isfortran(ax) == np.isfortran(ay) == np.isfortran(az) == np.isfortran(bx) \
        == np.isfortran(by) == np.isfortran(bz) , 'Binary files loaded incorrectly, must be Fortran ordered'

    return ax*bx + ay*by + az*bz

# computes relative helicity density
# helicity normalised by |a||b|

def relative_helicity(path,t,shape,case):

    # kinetic helicity
    if case == 1:
        a = 'v'     # velocity
        b = 'w'     # vorticity
    # magnetic helicity
    elif case == 2:
        a = 'a'     # magnetic potential
        b = 'b'     # magnetic field
    # cross helicity
    elif case == 3:
        a = 'v'     # velocity
        b = 'b'     # magnetic field
    else:
        ValueError('hk: case = 1, hm: case = 2, hc: case = 3')

    str1 = str(t).zfill(4)

    ax = np.fromfile(path+a+'y.'+str1+'.out',dtype=np.float32).reshape(shape,order='F')
    ay = np.fromfile(path+a+'z.'+str1+'.out',dtype=np.float32).reshape(shape,order='F')
    az = np.fromfile(path+a+'x.'+str1+'.out',dtype=np.float32).reshape(shape,order='F')

    bx = np.fromfile(path+b+'y.'+str1+'.out',dtype=np.float32).reshape(shape,order='F')
    by = np.fromfile(path+b+'z.'+str1+'.out',dtype=np.float32).reshape(shape,order='F')
    bz = np.fromfile(path+b+'x.'+str1+'.out',dtype=np.float32).reshape(shape,order='F')

    assert np.isfortran(ax) == np.isfortran(ay) == np.isfortran(az) == np.isfortran(bx) \
        == np.isfortran(by) == np.isfortran(bz) , 'Binary files loaded incorrectly, must be Fortran ordered'

    denom = np.sqrt(ax*ax + ay*ay + az*az) * np.sqrt(bx*bx + by*by + bz*bz)

    return np.divide(ax*bx + ay*by + az*bz, denom)


# emf in the x-direction - (u x b)_x
def emfx(path,t,shape):

    str1 = str(t).zfill(4)

    vy = np.fromfile(path+'vz.'+str1+'.out',dtype=np.float32).reshape(shape,order='F')
    vz = np.fromfile(path+'vx.'+str1+'.out',dtype=np.float32).reshape(shape,order='F')

    by = np.fromfile(path+'bz.'+str1+'.out',dtype=np.float32).reshape(shape,order='F')
    bz = np.fromfile(path+'bx.'+str1+'.out',dtype=np.float32).reshape(shape,order='F')

    assert np.isfortran(vy) == np.isfortran(vz) == np.isfortran(by) == np.isfortran(bz) , \
        'Binary files loaded incorrectly, must be Fortran ordered'

    return vy*bz - vz*by


# emf in the x-direction normalised by total |u x b|
def emfx_norm(path,t,shape):

    str1 = str(t).zfill(4)

    vx = np.fromfile(path+'vy.'+str1+'.out',dtype=np.float32).reshape(shape,order='F')
    vy = np.fromfile(path+'vz.'+str1+'.out',dtype=np.float32).reshape(shape,order='F')
    vz = np.fromfile(path+'vx.'+str1+'.out',dtype=np.float32).reshape(shape,order='F')

    bx = np.fromfile(path+'by.'+str1+'.out',dtype=np.float32).reshape(shape,order='F')
    by = np.fromfile(path+'bz.'+str1+'.out',dtype=np.float32).reshape(shape,order='F')
    bz = np.fromfile(path+'bx.'+str1+'.out',dtype=np.float32).reshape(shape,order='F')

    assert np.isfortran(vx) == np.isfortran(vy) == np.isfortran(vz) == np.isfortran(bx) \
        == np.isfortran(by) == np.isfortran(bz) , 'Binary files loaded incorrectly, must be Fortran ordered'

    # emf = u x b
    emfx = vy*bz - vz*by
    emfy = vz*bx - vx*bz
    emfz = vx*by - vy*bx

    E = np.sqrt(emfx*emfx + emfy*emfy + emfz*emfz)

    return np.divide(emfx, E)


