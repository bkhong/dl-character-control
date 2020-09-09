import numpy as np


NUM_JOINTS = 31

JOINT_NAMES = [
    'Hips',
    'LHipJoint',
    'LeftUpLeg',
    'LeftLeg',
    'LeftFoot',
    'LeftToeBase',
    'RHipJoint',
    'RightUpLeg',
    'RightLeg',
    'RightFoot',
    'RightToeBase',
    'LowerBack',
    'Spine',
    'Spine1',
    'Neck',
    'Neck1',
    'Head',
    'LeftShoulder',
    'LeftArm',
    'LeftForeArm',
    'LeftHand',
    'LeftFingerBase',
    'LeftHandIndex1',
    'LThumb',
    'RightShoulder',
    'RightArm',
    'RightForeArm',
    'RightHand',
    'RightFingerBase',
    'RightHandIndex1',
    'RThumb'
]

JOINT_PARENTS = [-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 13, 14, 15, 13, 17, 18, 19,
                 20, 21, 20, 13, 24, 25, 26, 27, 28, 27]

PARENT_JOINTS = sorted(list(set(JOINT_PARENTS[1:])))


# PFNN

PFNN_XMEAN = np.load('project/data/pfnn/Xmean.npy')
PFNN_XSTD = np.load('project/data/pfnn/Xstd.npy')
PFNN_YMEAN = np.load('project/data/pfnn/Ymean.npy')
PFNN_YSTD = np.load('project/data/pfnn/Ystd.npy')

PFNN_XMEAN_ORIGINAL = np.load('project/data/pfnn/Xmean_original.npy')
PFNN_XSTD_ORIGINAL = np.load('project/data/pfnn/Xstd_original.npy')
PFNN_YMEAN_ORIGINAL = np.load('project/data/pfnn/Ymean_original.npy')
PFNN_YSTD_ORIGINAL = np.load('project/data/pfnn/Ystd_original.npy')

PFNN_EXAMPLE_Y = np.load('project/data/pfnn/y0.npy')
