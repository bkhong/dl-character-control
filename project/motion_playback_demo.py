import numpy as np

import panda3d.core as core

from .demobase import DemoBase
from .load_animations import load_anim
from .util import mix_angles, ang2vec, rot2, quat_exp
from .data import PFNN_XMEAN, PFNN_XSTD

BUILD_NETWORK_INPUTS = True


class MotionPlaybackDemo(DemoBase):

    def __init__(self):
        super().__init__()

        self.anim = load_anim(0)

    def _update_character(self, t, dt):
        root_pos, global_xforms, forward, gaits = self.anim

        # Interpolate pos and xform
        frame = min(60 * t, root_pos.shape[0]-2)
        f_num, f_off = int(frame), frame % 1.0
        pos = root_pos[f_num] * (1 - f_off) + root_pos[f_num+1] * f_off
        xforms = global_xforms[f_num] * (1 - f_off) + global_xforms[f_num+1] * f_off

        self.char_pos = core.LVector3(*pos)
        self.joint_global_xforms = xforms

        # Update displayed trajectory
        nearest_frame = min(root_pos.shape[0]-60, max(60, round(60 * t)))
        if nearest_frame > 100: return
        positions = root_pos[nearest_frame-60:nearest_frame+60].copy()
        positions[:,2] = self.heightmap.get_height(positions[:,0], positions[:,1])
        directions = forward[nearest_frame-60:nearest_frame+60, :2]
        self.heightmap.update_trajectory(positions, directions)

        if nearest_frame == 100: print('---------')
        if BUILD_NETWORK_INPUTS and nearest_frame == 100:

            td_angles = np.arctan2(directions[:,1], directions[:,0]) * 180 / np.pi
            prev_joint_pos = global_xforms[nearest_frame-1,:,3,:3]
            prev_joint_vels = global_xforms[nearest_frame,:,3,:3] - prev_joint_pos
            prev_joint_vels += root_pos[nearest_frame] - root_pos[nearest_frame-1]
            prev_joint_pos[:,2] += root_pos[nearest_frame-1,2]
            gts = gaits[nearest_frame-60:nearest_frame+60]

            rpos = np.empty(3)
            rpos[:2] = positions[60,:2]
            rpos[2] = positions[::10,2].mean()
            rrot = td_angles[60] - 90

            x = np.empty(342, dtype=np.float32)

            # Trajectory positions and directions
            pos = rot2(-rrot) @ (positions[::10,:2] - rpos[:2]).T
            drc = ang2vec(td_angles[::10] - rrot).T
            x[0:12] = -pos[0]
            x[12:24] = pos[1]
            x[24:36] = -drc[0]
            x[36:48] = drc[1]

            # Gaits
            x[48:120] = gts[::10].T.flatten()

            # Last frame's joint positions and velocities
            jpos = prev_joint_pos
            jpos[:,:2] = jpos[:,:2] @ rot2(90 - td_angles[59]).T
            jvel = prev_joint_vels.copy()
            jvel[:,:2] = jvel[:,:2] @ rot2(90 - td_angles[59]).T
            x[120:213:3] = -jpos[:,0]
            x[121:213:3] = jpos[:,2] - self.rheight
            x[122:213:3] = jpos[:,1]
            x[213:306:3] = -jvel[:,0]
            x[214:306:3] = jvel[:,2]
            x[215:306:3] = jvel[:,1]

            # Trajectory heights (r and l are switched here for consistency with paper)
            positions_r = positions[::10,:2] + 25*ang2vec(td_angles[::10] + 90)
            positions_l = positions[::10,:2] + 25*ang2vec(td_angles[::10] - 90)
            x[306:318] = self.heightmap.get_height(positions_r[:,0], positions_r[:,1])
            x[318:330] = self.heightmap.get_height(positions_l[:,0], positions_l[:,1])
            x[330:342] = positions[::10,2]
            # Subtract off root position height
            x[306:342] -= rpos[2]

            x -= PFNN_XMEAN
            x /= PFNN_XSTD

            np.save('12345', x)

        self.rheight = positions[::10,2].mean()


MotionPlaybackDemo().run()
