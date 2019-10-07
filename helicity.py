## helicity module
import numpy as np

## computes helicity density
# path is the path to the binary files
# t is the numeric label of binary file
# shape is the shape of the data arrays
# case = 1 : kinetic helicity
# case = 2 : magnetic helicity
# case = 3 : cross helicity

def helicity(path,t,shape,case):

    if case == 1:
        a = 'v'     # velocity
        b = 'w'     # vorticity
    elif case == 2:
        a = 'a'     # magnetic potential
        b = 'b'     # magnetic field
    elif case == 3:
        a = 'v'     # velocity
        b = 'b'     # magnetic field
    else:
        ValueError('hk: case = 1, hm: case = 2, hc: case = 3')

    s = str(t)
    str1 = s.rjust(4, '0')

    ax = np.fromfile(path+a+'y.'+str1+'.out',dtype=np.float32).reshape(shape,order='F')
    ay = np.fromfile(path+a+'z.'+str1+'.out',dtype=np.float32).reshape(shape,order='F')
    az = np.fromfile(path+a+'x.'+str1+'.out',dtype=np.float32).reshape(shape,order='F')

    bx = np.fromfile(path+b+'y.'+str1+'.out',dtype=np.float32).reshape(shape,order='F')
    by = np.fromfile(path+b+'z.'+str1+'.out',dtype=np.float32).reshape(shape,order='F')
    bz = np.fromfile(path+b+'x.'+str1+'.out',dtype=np.float32).reshape(shape,order='F')

    return ax*bx * ay*by * az*bz

## computes relative helicity density
# helicity normalised by |a||b|

def relative_helicity(path,t,shape,case):

    if case == 1:
        a = 'v'     # velocity
        b = 'w'     # vorticity
    elif case == 2:
        a = 'a'     # magnetic potential
        b = 'b'     # magnetic field
    elif case == 3:
        a = 'v'     # velocity
        b = 'b'     # magnetic field
    else:
        ValueError('hk: case = 1, hm: case = 2, hc: case = 3')

    s = str(t)
    str1 = s.rjust(4, '0')

    ax = np.fromfile(path+a+'y.'+str1+'.out',dtype=np.float32).reshape(shape,order='F')
    ay = np.fromfile(path+a+'z.'+str1+'.out',dtype=np.float32).reshape(shape,order='F')
    az = np.fromfile(path+a+'x.'+str1+'.out',dtype=np.float32).reshape(shape,order='F')

    bx = np.fromfile(path+b+'y.'+str1+'.out',dtype=np.float32).reshape(shape,order='F')
    by = np.fromfile(path+b+'z.'+str1+'.out',dtype=np.float32).reshape(shape,order='F')
    bz = np.fromfile(path+b+'x.'+str1+'.out',dtype=np.float32).reshape(shape,order='F')

    return (ax*bx * ay*by * az*bz) / np.sqrt(ax*ax + ay*ay + az*az) / np.sqrt(bx*bx + by*by + bz*bz)
