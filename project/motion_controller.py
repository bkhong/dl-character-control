'''
Add subclasses of MotionController to this file in order to implement new experiments/models.
'''

from math import pi, sin, cos
from abc import ABC, abstractmethod

import numpy as np

from . import networks
from .util import mix_angles, ang2vec, rot2, quat_exp
from .data import PFNN_EXAMPLE_Y


def load_motion_controller(name, hmap, *args):
    return globals()[name](hmap, *args)


class MotionController(ABC):
    def __init__(self, hmap):
        self._heightmap = hmap  # should only be used to query get_height
        self.traj_positions = np.zeros((120, 3), dtype=np.float32)
        self.traj_positions[:,2] = hmap.get_height(0, 0)
        self.traj_directions = np.full(120, 90, dtype=np.float32)
        self.traj_gaits = np.empty((120, 6), dtype=np.float32)
        self.traj_gaits[:] = [1, 0, 0, 0, 0, 0]
        self._prev_traj_pos = self.traj_positions.copy()
        self._prev_traj_dirs = self.traj_directions.copy()
        self._prev_traj_gaits = self.traj_gaits.copy()

    def get_height(self, x, y):
        return self._heightmap.get_height(x, y)

    def get_trajectory(self):
        '''
        :return:
            (positions, directions), where positions and directions are numpy arrays of
            shapes (120, 3) and (120, 2) respectively
        '''
        return self.traj_positions, ang2vec(self.traj_directions)

    def _shift_trajectory(self):
        self._prev_traj_pos[:] = self.traj_positions
        self._prev_traj_dirs[:] = self.traj_directions
        self._prev_traj_gaits[:] = self.traj_gaits

        self.traj_positions[:-1] = self._prev_traj_pos[1:]
        self.traj_directions[:-1] = self._prev_traj_dirs[1:]
        self.traj_gaits[:-1] = self._prev_traj_gaits[1:]

    def reset(self, hmap):
        '''
        Resets to the demo's original state, with the character standing still at the origin.
        '''
        self._heightmap = hmap

        self.traj_positions[:] = 0
        self.traj_positions[:,2] = self._heightmap.get_height(0, 0)
        self.traj_directions[:] = 90
        self.traj_gaits[:] = [1, 0, 0, 0, 0, 0]

        self._prev_traj_pos[:] = self.traj_positions
        self._prev_traj_dirs[:] = self.traj_directions
        self._prev_traj_gaits[:] = self.traj_gaits

    @abstractmethod
    def frame(self, target_vel, target_dir, gaits, strafe_amount):
        '''
        :param target_vel:
            A 2-vector giving the direction (in the xy-plane) that the character should
            move in according to player input.
        :param target_dir:
            A float giving the angle (in the xy-plane, measured in degress from the x-axis)
            the character should face in, according to player input. This is different from
            the angle of target_vel when strafing.
        :param gaits:
            An array of 6 floats between 0 and 1, giving the gait information (just as
            it is passed into the PFNN in the original paper). Order:
                stand, walk, run, crouch, jump, bump
        :param strafe_amount:
            A float between 0 and 1 indicating how much the character is strafing (this
            should not be a network input; it's just used when predicting the future
            trajectory).
        :return:
            (pos, xforms), where pos is a 3-vector (numpy) holding the character's root
            position for this frame, and xforms is a numpy array of shape (31, 4, 4)
            holding this frame's transforms for each joint. Both are in panda format,
            meaning the coordinate system is z-up, and matrices are in row-vector
            compatible format.
        '''
        raise NotImplementedError


class PfnnMotionController(MotionController):
    '''
    Implements the experimental setup from the original paper.
    '''

    def __init__(self, hmap, model_version=None):
        super().__init__(hmap)

        self.network = networks.FastPFNN.load_version(model_version)
        # self.network = networks.StockPFNN.load_version(model_version)
        # self.network = networks.PhaseFunctionedNet.load_version(model_version)
        self.network.eval()

        self.phase, self.next_phase = 0, 0
        self.root_pos = np.zeros(3, dtype=np.float32)
        self.root_pos[2] = hmap.get_height(0, 0)
        self.root_rot = 90
        self.joint_positions = np.zeros((31, 3), dtype=np.float32)
        self.joint_velocities = np.zeros((31, 3), dtype=np.float32)
        self.joint_xforms = np.zeros((31, 4, 4), dtype=np.float32)
        self._update_joints(PFNN_EXAMPLE_Y)

        self._inference_time, self._time_sample_count = 0, 0

    def reset(self, hmap):
        super().reset(hmap)
        self.phase, self.next_phase = 0, 0
        self.root_pos[:] = [0, 0, self._heightmap.get_height(0, 0)]
        self.root_rot = 90
        self._update_joints(PFNN_EXAMPLE_Y)

    def frame(self, target_vel, target_dir, gaits, strafe_amount):
        target_vel = np.array(target_vel, dtype=np.float32)

        ## Advance frame

        self._shift_trajectory()
        self.phase = self.next_phase
        prev_root_pos = self.root_pos.copy()
        prev_root_rot = self.root_rot
        prev_joint_pos = self.joint_positions
        prev_joint_vels = self.joint_velocities

        ## Record current gaits

        # The future gaits don't add any information, so they can probably be removed as
        # network inputs
        self.traj_gaits[60:] = gaits

        ## Predict future trajectory

        # Blend future trajectory predicted by network with the one derived from player input
        # (see section 6 of paper).
        bias_pos, bias_dir = 0.5 + 0.5 * strafe_amount, 2.0 - 1.5 * strafe_amount
        time_fracs = 1 - np.arange(1, 60, dtype=np.float32) / 60
        target_pos_weights = 1 - time_fracs ** bias_pos
        target_dir_weights = 1 - time_fracs ** bias_dir

        # print(target_vel)
        # print(self.traj_positions[61:120:10,:2])
        dpos_predicted = self.traj_positions[61:120,:2] - self.traj_positions[60:119,:2]
        dpos = (1-target_pos_weights)[:,None]*dpos_predicted + target_pos_weights[:,None]*target_vel
        # dpos = np.tile(target_vel, (59,1))
        self.traj_positions[61:120,:2] = self.traj_positions[60,:2] + np.cumsum(dpos, axis=0)
        # print(self.traj_positions[61:120:10,:2])
        self.traj_directions[61:120] = mix_angles(self.traj_directions[61:120], target_dir,
                                                  target_dir_weights)
        # self.traj_directions[61:120] = target_dir

        # Compute heights
        self.traj_positions[60:120,2] = self.get_height(self.traj_positions[60:120,0],
                                                        self.traj_positions[60:120,1])

        ## Compute root position and rotation

        self.root_pos[:2] = self.traj_positions[60,:2]
        self.root_pos[2] = self.traj_positions[::10,2].mean()
        self.root_rot = self.traj_directions[60] - 90

        ## Build network input

        # Note: to compensate for the different coordinate systems, we always negate x and
        # switch y and z before sending inputs to the network, as well as after receiving
        # outputs.

        x = np.empty(342, dtype=np.float32)

        # Trajectory positions and directions
        pos = rot2(-self.root_rot) @ (self.traj_positions[::10,:2] - self.root_pos[:2]).T
        drc = ang2vec(self.traj_directions[::10] - self.root_rot).T
        x[0:12] = -pos[0]
        x[12:24] = pos[1]
        x[24:36] = -drc[0]
        x[36:48] = drc[1]

        # Gaits
        x[48:120] = self.traj_gaits[::10].T.flatten()

        # Last frame's joint positions and velocities
        jpos = prev_joint_pos.copy()
        jpos[:,:2] = (jpos[:,:2] - prev_root_pos[:2]) @ rot2(-prev_root_rot).T
        jvel = prev_joint_vels.copy()
        jvel[:,:2] = jvel[:,:2] @ rot2(-prev_root_rot).T
        x[120:213:3] = -jpos[:,0]
        x[121:213:3] = jpos[:,2] - self.root_pos[2]
        x[122:213:3] = jpos[:,1]
        x[213:306:3] = -jvel[:,0]
        x[214:306:3] = jvel[:,2]
        x[215:306:3] = jvel[:,1]

        # Trajectory heights (r and l are switched here for consistency with paper)
        positions_r = self.traj_positions[::10,:2] + 25*ang2vec(self.traj_directions[::10] + 90)
        positions_l = self.traj_positions[::10,:2] + 25*ang2vec(self.traj_directions[::10] - 90)
        x[306:318] = self.get_height(positions_r[:,0], positions_r[:,1])
        x[318:330] = self.get_height(positions_l[:,0], positions_l[:,1])
        x[330:342] = self.traj_positions[::10,2]
        # Subtract off root position height
        x[306:342] -= self.root_pos[2]

        ## Run inference

        t0 = globalClock.getRealTime()

        y = self.network.inference(self.phase, x)

        t1 = globalClock.getRealTime()
        self._inference_time += t1 - t0
        self._time_sample_count += 1
        if self._time_sample_count == 500:
            avg_time = 1000 * self._inference_time / self._time_sample_count
            print(f'Inference time: {avg_time:.5f} ms')
            self._inference_time, self._time_sample_count = 0, 0

        ## Compute joint positions, velocities, rotations, and transforms

        self._update_joints(y, prev_joint_pos)

        ## Update future trajectory with network predictions

        move_amount = (1 - gaits[0]) ** 0.25

        # Update next frame based on predicted translational and rotational velocities
        traj_vel = np.dot(rot2(self.traj_directions[60] - 90), [-y[0], y[1]])
        self.traj_positions[61,:2] = self.traj_positions[60,:2] + move_amount*traj_vel
        # print(y[2] * 180 / pi)
        self.traj_directions[61] = self.traj_directions[60] - move_amount*y[2] * 180 / pi
        # self.traj_directions[61] = mix_angles(target_dir, self.traj_directions[61], 0.9)

        # Update future predictions by interpolating predicted trajectory
        predicted_tpos_x, predicted_tpos_y = -y[8:14], y[14:20]
        predicted_tdir = np.arctan2(y[26:32], -y[20:26]) * 180 / pi
        interp_frames, pred_frames = np.arange(62, 120), np.arange(61, 120, 10)
        self.traj_positions[62:120,0] = np.interp(interp_frames, pred_frames, predicted_tpos_x)
        self.traj_positions[62:120,1] = np.interp(interp_frames, pred_frames, predicted_tpos_y)
        self.traj_directions[62:120] = self._interp_dirs(predicted_tdir)

        # Decode local coords
        next_root_rot_mat = rot2(self.traj_directions[61] - 90)
        self.traj_positions[62:120,:2] = self.traj_positions[62:120,:2] @ next_root_rot_mat.T
        self.traj_positions[62:120,:2] += self.traj_positions[61,:2]
        self.traj_directions[62:120] += self.traj_directions[61] - 90

        ## Update phase

        self.next_phase = self.phase + (0.1 + 0.9*move_amount) * y[3]
        self.next_phase %= 1.0

        return self.root_pos, self.joint_xforms

    # np.interp counterpart for angles, assuming xp = arange(61, 120, 10) and
    # x = arange(62, 120)
    def _interp_dirs(self, fp):
        interp_amounts = ((np.arange(62, 120) - 61) / 10) % 1.0
        theta1 = fp.repeat(10)[1:-1]
        theta2 = np.concatenate((fp[1:], fp[-1:])).repeat(10)[1:-1]
        return mix_angles(theta1, theta2, interp_amounts)

    def _update_joints(self, y, prev_joint_pos=None):
        root_rot_mat = rot2(self.root_rot)

        pos = np.empty((31,3))
        pos[:,0] = -y[32:125:3]
        pos[:,1] = y[34:125:3]
        pos[:,2] = y[33:125:3]
        pos[:,:2] = pos[:,:2] @ root_rot_mat.T
        pos += self.root_pos

        vel = np.empty((31,3))
        vel[:,0] = -y[125:218:3]
        vel[:,1] = y[127:218:3]
        vel[:,2] = y[126:218:3]
        vel[:,:2] = vel[:,:2] @ root_rot_mat.T

        # maybe remove this smoothing
        # self.joint_positions[:] = (pos if prev_joint_pos is None else
        #                            0.5*(prev_joint_pos + vel) + 0.5*pos)
        self.joint_positions[:] = pos
        self.joint_velocities[:] = vel

        rotq = np.empty((31,3))
        rotq[:,0] = -y[218:311:3]
        rotq[:,1] = y[220:311:3]
        rotq[:,2] = y[219:311:3]
        rot = quat_exp(rotq)
        rot[:,:2,:] = root_rot_mat @ rot[:,:2,:]
        self.joint_xforms[:,:3,:3] = np.swapaxes(rot, 1, 2)
        self.joint_xforms[:,3,:3] = pos - self.root_pos  # (?) why not use smoothed self.joint_positions
        self.joint_xforms[:,:,3] = [0, 0, 0, 1]

        # It seems the coordinate system for the character model is rotated 180 degrees from
        # what it should be, so correct here.
        self.joint_xforms[:,:2] *= -1
