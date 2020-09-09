import sys
import numpy as np
import scipy.ndimage.filters as filters

sys.path.append('motion')

import BVH as BVH
import Animation as Animation
from Quaternions import Quaternions
from Pivots import Pivots
from Learning import RBF

data_terrain = [
    'data/animations/LocomotionFlat01_000.bvh',
    'data/animations/LocomotionFlat02_000.bvh',
    'data/animations/LocomotionFlat02_001.bvh',
    'data/animations/LocomotionFlat03_000.bvh',
    'data/animations/LocomotionFlat04_000.bvh',
    'data/animations/LocomotionFlat05_000.bvh',
    'data/animations/LocomotionFlat06_000.bvh',
    'data/animations/LocomotionFlat06_001.bvh',
    'data/animations/LocomotionFlat07_000.bvh',
    'data/animations/LocomotionFlat08_000.bvh',
    'data/animations/LocomotionFlat08_001.bvh',
    'data/animations/LocomotionFlat09_000.bvh',
    'data/animations/LocomotionFlat10_000.bvh',
    'data/animations/LocomotionFlat11_000.bvh',
    'data/animations/LocomotionFlat12_000.bvh',

    'data/animations/LocomotionFlat01_000_mirror.bvh',
    'data/animations/LocomotionFlat02_000_mirror.bvh',
    'data/animations/LocomotionFlat02_001_mirror.bvh',
    'data/animations/LocomotionFlat03_000_mirror.bvh',
    'data/animations/LocomotionFlat04_000_mirror.bvh',
    'data/animations/LocomotionFlat05_000_mirror.bvh',
    'data/animations/LocomotionFlat06_000_mirror.bvh',
    'data/animations/LocomotionFlat06_001_mirror.bvh',
    'data/animations/LocomotionFlat07_000_mirror.bvh',
    'data/animations/LocomotionFlat08_000_mirror.bvh',
    'data/animations/LocomotionFlat08_001_mirror.bvh',
    'data/animations/LocomotionFlat09_000_mirror.bvh',
    'data/animations/LocomotionFlat10_000_mirror.bvh',
    'data/animations/LocomotionFlat11_000_mirror.bvh',
    'data/animations/LocomotionFlat12_000_mirror.bvh',

    'data/animations/WalkingUpSteps01_000.bvh',
    'data/animations/WalkingUpSteps02_000.bvh',
    'data/animations/WalkingUpSteps03_000.bvh',
    'data/animations/WalkingUpSteps04_000.bvh',
    'data/animations/WalkingUpSteps04_001.bvh',
    'data/animations/WalkingUpSteps05_000.bvh',
    'data/animations/WalkingUpSteps06_000.bvh',
    'data/animations/WalkingUpSteps07_000.bvh',
    'data/animations/WalkingUpSteps08_000.bvh',
    'data/animations/WalkingUpSteps09_000.bvh',
    'data/animations/WalkingUpSteps10_000.bvh',
    'data/animations/WalkingUpSteps11_000.bvh',
    'data/animations/WalkingUpSteps12_000.bvh',

    'data/animations/WalkingUpSteps01_000_mirror.bvh',
    'data/animations/WalkingUpSteps02_000_mirror.bvh',
    'data/animations/WalkingUpSteps03_000_mirror.bvh',
    'data/animations/WalkingUpSteps04_000_mirror.bvh',
    'data/animations/WalkingUpSteps04_001_mirror.bvh',
    'data/animations/WalkingUpSteps05_000_mirror.bvh',
    'data/animations/WalkingUpSteps06_000_mirror.bvh',
    'data/animations/WalkingUpSteps07_000_mirror.bvh',
    'data/animations/WalkingUpSteps08_000_mirror.bvh',
    'data/animations/WalkingUpSteps09_000_mirror.bvh',
    'data/animations/WalkingUpSteps10_000_mirror.bvh',
    'data/animations/WalkingUpSteps11_000_mirror.bvh',
    'data/animations/WalkingUpSteps12_000_mirror.bvh',

    'data/animations/NewCaptures01_000.bvh',
    'data/animations/NewCaptures02_000.bvh',
    'data/animations/NewCaptures03_000.bvh',
    'data/animations/NewCaptures03_001.bvh',
    'data/animations/NewCaptures03_002.bvh',
    'data/animations/NewCaptures04_000.bvh',
    'data/animations/NewCaptures05_000.bvh',
    'data/animations/NewCaptures07_000.bvh',
    'data/animations/NewCaptures08_000.bvh',
    'data/animations/NewCaptures09_000.bvh',
    'data/animations/NewCaptures10_000.bvh',
    'data/animations/NewCaptures11_000.bvh',

    'data/animations/NewCaptures01_000_mirror.bvh',
    'data/animations/NewCaptures02_000_mirror.bvh',
    'data/animations/NewCaptures03_000_mirror.bvh',
    'data/animations/NewCaptures03_001_mirror.bvh',
    'data/animations/NewCaptures03_002_mirror.bvh',
    'data/animations/NewCaptures04_000_mirror.bvh',
    'data/animations/NewCaptures05_000_mirror.bvh',
    'data/animations/NewCaptures07_000_mirror.bvh',
    'data/animations/NewCaptures08_000_mirror.bvh',
    'data/animations/NewCaptures09_000_mirror.bvh',
    'data/animations/NewCaptures10_000_mirror.bvh',
    'data/animations/NewCaptures11_000_mirror.bvh',
]

to_meters = 5.644

YUP_TO_ZUP = np.array([
    [1,  0,  0],
    [0,  0, -1],
    [0,  1,  0]
])

ZUP_TO_YUP = np.array([
    [1,  0,  0],
    [0,  0,  1],
    [0, -1,  0]
])

YUP_TO_ZUP4 = np.array([
    [1,  0,  0, 0],
    [0,  0, -1, 0],
    [0,  1,  0, 0],
    [0,  0,  0, 1]
])

ZUP_TO_YUP4 = np.array([
    [1,  0,  0, 0],
    [0,  0,  1, 0],
    [0, -1,  0, 0],
    [0,  0,  0, 1]
])

def load_anim(i):
    anim, names, _ = BVH.load(data_terrain[i])
    anim.offsets *= to_meters
    anim.positions *= to_meters
    anim = anim[::2]

    gait = np.loadtxt(data_terrain[i].replace('.bvh', '.gait'))[::2]
    gait = np.concatenate([
        gait[:,0:1],
        gait[:,1:2],
        gait[:,2:3] + gait[:,3:4],
        gait[:,4:5] + gait[:,6:7],
        gait[:,5:6],
        gait[:,7:8]
    ], axis=-1)

    global_xforms = Animation.transforms_global(anim)
    print(global_xforms[99,0])
    print(global_xforms[100,0])
    global_xforms = global_xforms / global_xforms[:,:,3:,3:]
    global_positions = global_xforms[:,:,:3,3:].copy()  # (F, J, 3, 1)
    global_xforms[:,:,:3,3:] -= global_positions[:,:1]
    global_positions = (YUP_TO_ZUP @ global_positions).squeeze()  # (F, J, 3)

    global_xforms = YUP_TO_ZUP4 @ global_xforms @ ZUP_TO_YUP4
    global_xforms = np.swapaxes(global_xforms, 2, 3)

    root_pos = global_positions[:,0]

    across = ((global_positions[:,18] - global_positions[:,25]) +
              (global_positions[:,2] - global_positions[:,7]))
    across /= np.sqrt((across**2).sum(axis=-1, keepdims=True))
    forward = filters.gaussian_filter1d(np.cross(across, np.array([[0,0,1]])), 20, axis=0, mode='nearest')
    forward /= np.sqrt((forward**2).sum(axis=-1, keepdims=True))

    return root_pos, global_xforms, forward, gait
