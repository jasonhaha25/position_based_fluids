import taichi as ti
import numpy as np

series_prefix = "./sample/pbf3d.ply"

class PLYWriter:
    def __init__(self, num_particles, positions):
        self.writer = ti.PLYWriter(num_vertices=num_particles)
        self.num_particles = num_particles
        self.positions = positions
        self.out_frame = 0


    def write(self, frame):
        np_pos = self.positions.to_numpy()
        self.writer.add_vertex_pos(np_pos[:, 0], np_pos[:, 1], np_pos[:, 2])
        self.writer.export_frame_ascii(self.out_frame, series_prefix)
        print("recorded frame ", frame)
        self.out_frame += 1
