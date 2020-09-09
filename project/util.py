from math import pi

import numpy as np


def mix_angles(theta1, theta2, a, rads=False):
    half_period = pi if rads else 180
    # Get difference in [-half_period, half_period)
    diff = (theta2 - theta1 + half_period) % (2*half_period) - half_period

    return theta1 + a*diff


def ang2vec(angs, rads=False):
    scale = 1 if rads else pi/180
    angs = np.array(angs, copy=False)
    vecs = np.empty(angs.shape+(2,))
    vecs[...,0] = np.cos(angs*scale)
    vecs[...,1] = np.sin(angs*scale)
    return vecs


def rot2(theta, rads=False):
    scale = 1 if rads else pi/180
    theta = np.array(theta, copy=False)
    res = np.empty(theta.shape+(2,2))
    c, s = np.cos(theta*scale), np.sin(theta*scale)
    res[...,0,0] = c
    res[...,0,1] = -s
    res[...,1,0] = s
    res[...,1,1] = c
    return res


def quat_exp(axes):
    '''
    Maps imaginary quats to SO3. Specifically, (x,y,z) is mapped to the rotation about
    normalize(x,y,z) by 2*norm(x,y,z) radians.
    '''
    norms = np.sqrt((axes**2).sum(axis=-1, keepdims=True))
    axes = axes / norms
    thetas = 2*norms

    c, s = np.cos(thetas)[...,None], np.sin(thetas)[...,None]
    rots = c * np.eye(3) + (1 - c) * axes[...,None] @ axes[...,None,:]
    rots[...,1,2] -= s[...,0,0] * axes[...,0]
    rots[...,0,2] += s[...,0,0] * axes[...,1]
    rots[...,0,1] -= s[...,0,0] * axes[...,2]
    rots[...,2,1] += s[...,0,0] * axes[...,0]
    rots[...,2,0] -= s[...,0,0] * axes[...,1]
    rots[...,1,0] += s[...,0,0] * axes[...,2]
    return rots
