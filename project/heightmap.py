__all__ = ['Heightmap']

from math import cos, sin, pi

import numpy as np

import panda3d.core as core


class Heightmap:
    GROUND_COLORS = [0.8, 0.65]

    # Mysterious scaling value from original demo code
    HSCALE = 3.937007874

    def __init__(self, name, fname, render):
        self.name = name
        self.heights = np.load(fname)
        self.w, self.h = self.heights.shape

        # Calculate vertex positions
        x_coords = (np.arange(self.w, dtype=np.float32) - self.w/2) * self.HSCALE
        y_coords = (self.h/2 - np.arange(self.h, dtype=np.float32)) * self.HSCALE
        x_coords = np.broadcast_to(x_coords[:,None], self.heights.shape)
        y_coords = np.broadcast_to(y_coords[None,:], self.heights.shape)
        self.positions = np.stack((x_coords, y_coords, self.heights), axis=-1)
        self.positions = self.positions.astype(np.float32)

        # Calculate normals
        self.normals = np.zeros((self.w, self.h, 1)) + np.array([0, 0, 1])
        edge1 = self.positions[2:  , 1:-1] - self.positions[1:-1,1:-1]
        edge2 = self.positions[1:-1,  :-2] - self.positions[1:-1,1:-1]
        edge3 = self.positions[ :-2, 1:-1] - self.positions[1:-1,1:-1]
        edge4 = self.positions[1:-1, 2:  ] - self.positions[1:-1,1:-1]
        self.normals[1:-1,1:-1] += np.cross(edge1, edge2)
        self.normals[1:-1,1:-1] += np.cross(edge2, edge3)
        self.normals[1:-1,1:-1] += np.cross(edge3, edge4)
        self.normals[1:-1,1:-1] += np.cross(edge4, edge1)
        self.normals[1:-1,1:-1] /= np.sqrt((self.normals[1:-1,1:-1]**2).sum(axis=2, keepdims=True))
        self.normals = self.normals.astype(np.float32)

        # Calculate colors
        grid_mask = (self.positions[..., 0] // 100 + self.positions[..., 1] // 100) % 2
        self.colors = (1-grid_mask) * self.GROUND_COLORS[0] + grid_mask * self.GROUND_COLORS[1]
        self.colors = self.colors.astype(np.float32)

        # Create model with terrain geometry
        self.model_np = render.attachNewNode(core.ModelNode('Heightmap_'+self.name))
        self._init_terrain()
        self._init_trajectory()

    def _init_terrain(self):
        vertex_data = core.GeomVertexData(
            'Terrain',
            core.GeomVertexFormat.getV3n3c4(),
            core.Geom.UH_static
        )

        pos = core.GeomVertexWriter(vertex_data, 'vertex')
        nrm = core.GeomVertexWriter(vertex_data, 'normal')
        col = core.GeomVertexWriter(vertex_data, 'color')

        for i in range(self.w):
            for j in range(self.h):
                pos.addData3f(*self.positions[i,j])
                nrm.addData3f(*self.normals[i,j])
                col.addData4f(self.colors[i,j], self.colors[i,j], self.colors[i,j], 1)

        triangles = core.GeomTriangles(core.Geom.UH_static)
        for i in range(self.w-1):
            for j in range(self.h-1):
                triangles.addVertices( i   *self.h +  j   ,  i   *self.h + (j+1), (i+1)*self.h +  j   )
                triangles.addVertices((i+1)*self.h + (j+1), (i+1)*self.h +  j,     i   *self.h + (j+1))

        geom = core.Geom(vertex_data)
        geom.addPrimitive(triangles)

        geom_node = core.GeomNode('Terrain')
        geom_node.addGeom(geom)
        self.terrain_np = self.model_np.attachNewNode(geom_node)

    def _create_dot_geom(self, dot_rad, n_points):
        vertex_data = core.GeomVertexData(
            'DotsRad'+str(dot_rad),
            core.GeomVertexFormat.getV3(),
            core.Geom.UH_static
        )

        pos = core.GeomVertexWriter(vertex_data, 'vertex')
        pos.addData3f(0, 0, 0)
        for i in range(n_points):
            pos.addData3f(dot_rad * cos(2*pi*i/n_points), dot_rad * sin(2*pi*i/n_points), 0)

        triangles = core.GeomTriangles(core.Geom.UH_static)
        for i in range(1,n_points):
            triangles.addVertices(0, i, i+1)
        triangles.addVertices(0, n_points, 1)

        geom = core.Geom(vertex_data)
        geom.addPrimitive(triangles)
        return geom

    def _create_arrow_geom(self, headsize):
        vertex_data = core.GeomVertexData(
            'Arrows',
            core.GeomVertexFormat.getV3(),
            core.Geom.UH_static
        )

        pos = core.GeomVertexWriter(vertex_data, 'vertex')
        pos.addData3f(0, 0, 0)
        pos.addData3f(0, 1, 0)
        pos.addData3f(headsize, 1-headsize, 0)
        pos.addData3f(-headsize, 1-headsize, 0)

        lines = core.GeomLines(core.Geom.UH_static)
        lines.addVertices(0, 1)
        lines.addVertices(1, 2)
        lines.addVertices(1, 3)

        geom = core.Geom(vertex_data)
        geom.addPrimitive(lines)
        return geom

    def _init_trajectory(self):
        # Create dots on ground
        self.traj_dots_np = []

        small_dot = self._create_dot_geom(1, 20)
        big_dot = self._create_dot_geom(2, 50)

        for i in range(12):
            frame_idx = 10*(i-6)
            geom_node = core.GeomNode('TrajPosFrame'+str(frame_idx))
            geom_node.addGeom(big_dot)
            nodepath = self.model_np.attachNewNode(geom_node)
            nodepath.set_color(0.7, 0.4, 0.4)
            nodepath.hide()
            self.traj_dots_np.append(nodepath)
            for j in range(1,10):
                frame_idx = 10*(i-6) + j
                geom_node = core.GeomNode('TrajPosFrame'+str(frame_idx))
                geom_node.addGeom(small_dot)
                nodepath = self.model_np.attachNewNode(geom_node)
                nodepath.set_color(0.7, 0.6, 0.6)
                nodepath.hide()
                self.traj_dots_np.append(nodepath)

        # Create arrows
        self.traj_arrs_np = []

        arrow = self._create_arrow_geom(0.15)

        for i in range(12):
            frame_idx = 10*(i-6)
            geom_node = core.GeomNode('TrajDirFrame'+str(frame_idx))
            geom_node.addGeom(arrow)
            nodepath = self.model_np.attachNewNode(geom_node)
            nodepath.set_render_mode_thickness(3)
            nodepath.set_color(0.7, 0.4, 0.4)
            nodepath.hide()
            self.traj_arrs_np.append(nodepath)

    def get_height(self, x, y):
        # Transform to array coordinates
        x, y = x / self.HSCALE + self.w / 2, -y / self.HSCALE + self.h / 2

        # Clip to array size
        x = np.clip(x, 0, self.w-2)
        y = np.clip(y, 0, self.h-2)

        # Bilinear interpolation
        x_off, i = np.modf(x)
        y_off, j = np.modf(y)
        i, j = i.astype('i'), j.astype('i')
        return ((1 - x_off) * (1 - y_off) * self.heights[i,  j  ] +
                     x_off  * (1 - y_off) * self.heights[i+1,j  ] +
                (1 - x_off) *      y_off  * self.heights[i,  j+1] +
                     x_off  *      y_off  * self.heights[i+1,j+1])

    # positions has shape (120, 3) and holds trajectory (x,y,z) positions where z should
    #     equal get_height(x,y)  (passing this as an arg saves computation in the main demo)
    # directions has shape (120, 2) and holds normalized trajectory dirs
    def update_trajectory(self, positions, directions):
        for i in range(120):
            x, y, z = positions[i]
            z += 1.9 if i % 10 else 2.0
            self.traj_dots_np[i].setPos(x, y, z)
            self.traj_dots_np[i].show()

        for arr, pos, drc in zip(self.traj_arrs_np, positions[::10], directions[::10]):
            x0, y0, z0 = pos
            x1, y1 = pos[:2] + 15*drc
            z1 = self.get_height(x1, y1)
            z0, z1 = z0 + 2.0, z1 + 2.0
            arr.setPos(x0, y0, z0)
            arr.lookAt(x1, y1, z1)
            arr.setScale(15)
            arr.show()
