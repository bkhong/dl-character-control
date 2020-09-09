'''
This is the main character motion synthesis demo. Run with
  python ./demo.py [ControllerName]
where [ControllerName] is the class name of one of the motion controllers defined in
motion_controller.py .
'''

import sys
from math import pi, atan2

import numpy as np

import panda3d.core as core

from .demobase import DemoBase
from .motion_controller import load_motion_controller
from .util import mix_angles


class CharControlDemo(DemoBase):

    def __init__(self):
        super().__init__()

        # Unnecessary if using vsync with 60 FPS monitor, otherwise uncomment
        # globalClock.setMode(core.ClockObject.M_limited)
        # globalClock.setFrameRate(60)

        self.motion_controller = load_motion_controller(sys.argv[1], self.heightmap, *sys.argv[2:])
        self.target_vel = core.LVector2()
        self.target_dir = 90
        self.strafe_amount = 0
        self.crouch_amount = 0
        self.gaits = np.zeros(6)

        self.strafe_pressed, self.crouch_pressed, self.run_pressed = 0, 0, 0
        self.accept('lshift', self.set_strafe_pressed, [1])
        self.accept('lshift-up', self.set_strafe_pressed, [0])
        self.accept('c', self.set_crouch_pressed, [1])
        self.accept('c-up', self.set_crouch_pressed, [0])
        self.accept('r', self.set_run_pressed, [1])
        self.accept('r-up', self.set_run_pressed, [0])

    def set_strafe_pressed(self, val):
        self.strafe_pressed = val

    def set_crouch_pressed(self, val):
        self.crouch_pressed = val

    def set_run_pressed(self, val):
        self.run_pressed = val

    def _update_character(self, t, dt):
        # Compute motion inputs
        input_vel = core.LVector3(self.pad_input, 0)
        input_vel = core.LMatrix3.rotateMat(90 + self.camera_angle).xform(input_vel)
        input_vel = (input_vel * 7.5).getXy()
        if input_vel.length() < 1.5: input_vel.set(0, 0)  # deadzone
        self.target_vel = self.target_vel*0.1 + input_vel*0.9

        self.strafe_amount = self.strafe_amount*0.1 + self.strafe_pressed*0.9

        input_dir = (atan2(self.target_vel[1], self.target_vel[0]) * 180 / pi
                     if self.target_vel.length() > 1e-5 else self.target_dir)
        input_dir = mix_angles(input_dir, 180 + self.camera_angle, self.strafe_amount)
        self.target_dir = mix_angles(self.target_dir, input_dir, 0.9)

        # Compute gait
        self.crouch_amount = self.crouch_amount*0.1 + self.crouch_pressed*0.9
        if self.target_vel.length() < 0.1:
            gaits = [1 - 10*self.target_vel.length(), 0, 0, 0, 0, 0]
        elif self.crouch_amount > 0.1:
            gaits = [0, 0, 0, self.crouch_amount, 0, 0]
        elif self.run_pressed:
            gaits = [0, 0, 1, 0, 0, 0]
        else:
            gaits = [0, 1, 0, 0, 0, 0]
        self.gaits = self.gaits*0.9 + np.array(gaits)*0.1

        # Run motion controller
        pos, xforms = self.motion_controller.frame(self.target_vel, self.target_dir,
                                                   self.gaits, self.strafe_amount)
        self.char_pos = core.LVector3(*pos)
        self.joint_global_xforms = xforms

        self.heightmap.update_trajectory(*self.motion_controller.get_trajectory())

    def reset_level(self):
        super().reset_level()
        self.target_vel.set(0, 0)
        self.target_dir = 90
        self.strafe_amount = 0
        self.crouch_amount = 0
        self.gaits = np.zeros(6)
        self.motion_controller.reset(self.heightmap)


CharControlDemo().run()
