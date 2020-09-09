from math import pi, sin, cos
from abc import ABC, abstractmethod

import numpy as np

import panda3d.core as core
from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from direct.actor.Actor import Actor
from direct.gui.OnscreenImage import OnscreenImage

from .heightmap import Heightmap
from .data import NUM_JOINTS, JOINT_NAMES, JOINT_PARENTS, PARENT_JOINTS

PAD_SENSITIVITY = 1/250  # drag 250 pixels to get max pad response


class DemoBase(ShowBase, ABC):

    def __init__(self):
        super().__init__()

        # Allow panda to synthesize shaders. Make sure hardware-animated-vertices is set
        # to true in panda config.
        render.setShaderAuto()

        # Show FPS
        self.setFrameRateMeter(True)

        # Load character
        self.character = Actor('data/character/character')
        self.character.reparentTo(self.render)
        self.joints = []
        for name in JOINT_NAMES:
            j = self.character.controlJoint(None, 'modelRoot', name)
            j.reparentTo(self.render)
            self.joints.append(j)

        # Add lights
        dlight = core.DirectionalLight('DirectionalLight')
        dlight.setDirection(core.LVector3(2, 0, -1))
        dlight.setColor(core.LColor(1, 1, 1, 1))
        dlnp = self.render.attachNewNode(dlight)
        self.render.setLight(dlnp)

        alight = core.AmbientLight('AmbientLight')
        alight.setColor(core.LColor(0.6, 0.6, 0.6, 1))
        alnp = self.render.attachNewNode(alight)
        self.render.setLight(alnp)

        # Camera angle in xy-plane (degrees)
        self.camera_angle = 0

        # Camera control
        self.cam_left_pressed, self.cam_right_pressed = 0, 0
        self.accept('a', self.set_cam_left_pressed, [1])
        self.accept('a-up', self.set_cam_left_pressed, [0])
        self.accept('d', self.set_cam_right_pressed, [1])
        self.accept('d-up', self.set_cam_right_pressed, [0])

        # Pad display
        self.pad_radius = 60
        self.pad_outline = OnscreenImage('project/data/pad/pad_outline.png', (0, 0, 0))
        self.pad_outline.setTransparency(core.TransparencyAttrib.MAlpha)
        self.pad_outline.hide()
        self.pad_response_circle = OnscreenImage('project/data/pad/pad_response_indicator.png',
                                                 (0, 0, 0))
        self.pad_response_circle.setTransparency(core.TransparencyAttrib.MAlpha)
        self.pad_response_circle.hide()
        self.accept('window-event', self.handle_window_event)

        # Pad control
        self.mouse1_pressed, self.controlling_pad = False, False
        self.accept('mouse1', self.set_mouse1_pressed, [True])
        self.accept('mouse1-up', self.set_mouse1_pressed, [False])

        # Load terrain
        self.heightmaps = {}
        self.set_heightmap('hmap2', 'project/data/heightmaps/hmap2.npy')

        # Heightmap choice
        self.accept('1', self.set_heightmap, ['hmap1', 'project/data/heightmaps/hmap1.npy'])
        self.accept('2', self.set_heightmap, ['hmap2', 'project/data/heightmaps/hmap2.npy'])
        self.accept('3', self.set_heightmap, ['hmap3', 'project/data/heightmaps/hmap3.npy'])
        self.accept('4', self.set_heightmap, ['hmap4', 'project/data/heightmaps/hmap4.npy'])
        self.accept('5', self.set_heightmap, ['hmap5', 'project/data/heightmaps/hmap5.npy'])

        # Tasks
        self.taskMgr.add(self.update_pad, 'UpdatePadTask', sort=1)
        self.last_update_char_time = 0
        self.taskMgr.add(self.update_character, 'UpdateCharacterTask', sort=2)
        self.last_update_cam_time = 0
        self.taskMgr.add(self.update_camera, 'UpdateCameraTask', sort=3)

    def set_cam_left_pressed(self, val):
        self.cam_left_pressed = val

    def set_cam_right_pressed(self, val):
        self.cam_right_pressed = val

    def set_mouse1_pressed(self, val):
        self.mouse1_pressed = val
        if self.mouse1_pressed and self.mouseWatcherNode.hasMouse():
            mpos = self.screen_coords_to_pixels(self.mouseWatcherNode.getMouse())
            diff = mpos - self.pad_center
            self.controlling_pad = diff.lengthSquared() <= self.pad_radius**2
        else:
            self.controlling_pad = False

    def handle_window_event(self, window):
        self.window_w, self.window_h = window.getSize()
        min_dim = min(self.window_w, self.window_h)

        self.pad_radius = min_dim * 0.1

        if self.window_w < 1300:
            self.pad_center = core.LVector2(min_dim * 0.15, min_dim * 0.15)
        else:
            self.pad_center = core.LVector2(self.window_w * 0.75, self.window_h * 0.5)

        self.pad_outline.setPos(self.pixels_to_im_coords(self.pad_center))
        self.pad_outline.setScale(2*self.pad_radius / min_dim)
        self.pad_outline.show()

        self.pad_response_circle.setScale(0.2*self.pad_radius / min_dim)

        # Propagate event
        self.windowEvent(window)

    # Converts from screen coords in the [-1, 1] range to pixel coords
    def screen_coords_to_pixels(self, point):
        x_frac, y_frac = (point + 1.0) / 2.0
        return core.LVector2(x_frac * self.window_w, (1 - y_frac) * self.window_h)

    # Converts from pixel coords to coords that can be passed in to OnscreenImage.setPos()
    def pixels_to_im_coords(self, point):
        px, py = point
        px, py = px - self.window_w / 2, py - self.window_h / 2
        return core.LVector3(px, 0, -py) * (2.0 / min(self.window_w, self.window_h))

    def update_pad(self, task):
        if self.controlling_pad and self.mouseWatcherNode.hasMouse():
            mpos = self.screen_coords_to_pixels(self.mouseWatcherNode.getMouse())

            pad_input = (mpos - self.pad_center) * PAD_SENSITIVITY
            if pad_input.lengthSquared() > 1:
                pad_input.normalize()

            resp_circle_pos = self.pad_center + pad_input * (self.pad_radius * 0.9)
            self.pad_response_circle.setPos(self.pixels_to_im_coords(resp_circle_pos))
            self.pad_response_circle.show()
        else:
            pad_input = core.LVector2()
            self.pad_response_circle.hide()

        pad_input.setY(-pad_input.getY())  # flip y
        self.pad_input = pad_input
        return Task.cont

    def set_joint_global_xforms(self, global_xforms):
        # Invert global transforms
        inverse_global_xforms = [None]*NUM_JOINTS
        for i in PARENT_JOINTS:
            inverse_global_xforms[i] = core.LMatrix4()
            inverse_global_xforms[i].invertAffineFrom(global_xforms[i])

        # Use each joint's global xform and the inverse of its parent's global xform to
        # compute and apply its local xform
        self.joints[0].setMat(global_xforms[0])
        for i in range(1, NUM_JOINTS):
            local_xform = global_xforms[i] * inverse_global_xforms[JOINT_PARENTS[i]]
            self.joints[i].setMat(local_xform)

    def update_character(self, task):
        dt = task.time - self.last_update_char_time

        self._update_character(task.time, dt)
        self.character.setPos(self.char_pos)
        self.set_joint_global_xforms([
            core.LMatrix4(*m) for m in self.joint_global_xforms.reshape(31, 16)
        ])

        self.last_update_char_time = task.time
        return Task.cont

    @abstractmethod
    def _update_character(self, t, dt):
        '''Should set self.char_pos and self.joint_global_xforms.'''
        raise NotImplementedError()

    def update_camera(self, task):
        dt = task.time - self.last_update_cam_time

        # Update camera angle
        self.camera_angle += 90 * (self.cam_right_pressed - self.cam_left_pressed) * dt

        # Reposition camera
        ang_rads = self.camera_angle * pi / 180
        cam_offset = core.LVector3(300 * cos(ang_rads), 300 * sin(ang_rads), 320)
        self.camera.setPos(self.char_pos + cam_offset)
        self.camera.lookAt(self.char_pos + core.LVector3(0, 0, 80), core.LVector3(0, 0, 1))

        self.last_update_cam_time = task.time
        return Task.cont

    def set_heightmap(self, name, fname):
        if name not in self.heightmaps:
            self.heightmaps[name] = Heightmap(name, fname, self.render)

        reset = False
        if hasattr(self, 'heightmap'):
            self.heightmap.model_np.stash()
            reset = True

        self.heightmap = self.heightmaps[name]
        self.heightmap.model_np.unstash()
        if reset: self.reset_level()

    def reset_level(self):
        self.camera_angle = 0
        self.char_pos = core.LVector3()
