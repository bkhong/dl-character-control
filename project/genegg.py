'''
This file was used to convert the binary character model data provided by the original
paper into the EGG format readable by Panda3d.
'''

import struct
import numpy as np
import panda3d.core as core
import panda3d.egg as egg

from .data import JOINT_NAMES

VERTICES_FILE = 'demo/network/character_vertices.bin'
TRIANGLES_FILE = 'demo/network/character_triangles.bin'
PARENTS_FILE = 'demo/network/character_parents.bin'
TRANSFORMS_FILE = 'demo/network/character_xforms.bin'

OUT_FILE = 'project/data/character/character.egg'


# Load data
with open(VERTICES_FILE, 'rb') as f:
    vert_bytes = f.read()
with open(TRIANGLES_FILE, 'rb') as f:
    tri_bytes = f.read()
with open(PARENTS_FILE, 'rb') as f:
    parents_bytes = f.read()
with open(TRANSFORMS_FILE, 'rb') as f:
    xforms_bytes = f.read()

verts = list(f[0] for f in struct.iter_unpack('f', vert_bytes))
tris = list(i[0] for i in struct.iter_unpack('I', tri_bytes))
parents = list(int(f[0]) for f in struct.iter_unpack('f', parents_bytes))
xforms = list(f[0] for f in struct.iter_unpack('f', xforms_bytes))

verts = np.array(verts, dtype=np.float32).reshape(-1, 15)
tris = np.array(tris, dtype=np.uint32).reshape(-1, 3)
xforms = np.array(xforms, dtype=np.float32).reshape(-1, 4, 4)
xforms = xforms.swapaxes(1, 2).reshape(-1, 16)


# Create EggData object representing the root of the EGG file
root = egg.EggData()
root.setCoordinateSystem(core.CS_yup_right)

# Create character group
char = egg.EggGroup('character')
char.setDartType(egg.EggGroup.DT_default)

mesh_grp = egg.EggGroup('character_mesh')

# Create vertex pool defining character's vertices
vert_pool = egg.EggVertexPool('character_mesh.verts')
for i, v in enumerate(verts):
    ev = egg.EggVertex()
    ev.setPos(core.LPoint3d(*v[0:3]))
    ev.setNormal(core.LVector3d(*v[3:6]))
    ev.setColor(core.LVecBase4f(0.7, 0.5, 0.94, 1.0))
    vert_pool.addVertex(ev, i)
mesh_grp.addChild(vert_pool)

# Add character's triangles
for t in tris:
    et = egg.EggPolygon()
    et.addVertex(vert_pool.getVertex(int(t[0])))
    et.addVertex(vert_pool.getVertex(int(t[1])))
    et.addVertex(vert_pool.getVertex(int(t[2])))
    mesh_grp.addChild(et)

char.addChild(mesh_grp)

# Add joints
joints = []
for i, name in enumerate(JOINT_NAMES):
    j = egg.EggGroup(name)
    j.setGroupType(egg.EggGroup.GT_joint)
    j.setTransform3d(core.LMatrix4d(*xforms[i]))
    joints.append(j)
    char.addChild(j)

# Organize joints in proper hierarchy
for j, pi in zip(joints, parents):
    if pi != -1:
        joints[pi].addChild(j)

# Add vertex refs to joints indicating skinning weights
for i, v in enumerate(verts):
    ev = vert_pool.getVertex(i)
    for j_weight, j_idx in zip(v[7:11], v[11:15]):
        if j_weight > 0:
            joints[int(j_idx)].refVertex(ev, j_weight)
            if j_idx == 18:
                ev.setColor(core.LVecBase4f(0.7, 0.5, 0.94, 1.0)*(1-j_weight) + core.LVecBase4f(1.0, 0.5, 0.5, 1.0)*j_weight)


root.addChild(char)

root.writeEgg(OUT_FILE)
