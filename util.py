import taichi as ti
import numpy as np
import os
with open('temp.txt', 'r') as the_file:
    dim = int(the_file.readline())
os.unlink("temp.txt")

boundary = (100, 50) if dim == 2 else (20, 10, 10)
max_frame = 1000

# constants
g = ti.Vector([0.0, -9.8]) if dim == 2 else ti.Vector([0.0, -9.8, 0.0])
dt = 1. / 30.
h = 1.1
h6 = h ** 6
h9 = h ** 9

# Particle
num_particles = 1000 if dim == 2 else 2500
particle_radius = 0.4 if dim == 2 else 0.1

# Neighbour logic
cell_size = 2
gridX = int(boundary[0] / cell_size)
gridY = int(boundary[1] / cell_size)
gridZ = int(boundary[2] / cell_size) if dim == 3 else 0
num_max_particles_in_cell = 100
num_max_neighbours = 50  # maximum number of neighbours taken
interest_zone_radius = 1.5

# density constraint Constants
num_solver_itr = 5
epsilon = 5e-6
mass = 1.0
rho0 = 1.0
CFM_epsilon = 100

@ti.data_oriented
class Util:
    @ti.func
    def pos_to_cell(position):
        return int(position / cell_size)

    @ti.func
    def in_bound(cell):
        return 0 <= cell[0] < gridX and 0 <= cell[1] < gridY

    @ti.func
    def W_poly6(r):
        # 0 <= r < h ? 315 / (64*pi*h^9) * (h^2 - r^2)^3 : 0
        # Eq (20) Müller D, et al. (2003)
        w = 0.0
        if 0.0 <= r <= h:
            hr = h * h - r * r
            w = 315. / (64. * np.pi * h9) * (hr * hr * hr)
        return w

    @ti.func
    def W_spiky_gradient(r):
        # 0 <= r < h ? 15 / (pi*h^6) * (h - r)^3 : 0
        # Eq (21) Müller D, et al. (2003)
        # take derivative respect to r to get g_norm = -45 / (pi*h^6) * (h - r)^2
        w = ti.Vector([0.0, 0.0])
        r_norm = r.norm()
        if 0 <= r_norm <= h:
            hr = h - r_norm
            gradient_norm = -45.0 / (np.pi * h6) * hr * hr
            w = r / r_norm * gradient_norm
        return w

    @ti.func
    def W_spiky_gradient_3d(r):
        # 0 <= r < h ? 15 / (pi*h^6) * (h - r)^3 : 0
        # Eq (21) Müller D, et al. (2003)
        # take derivative respect to r to get g_norm = -45 / (pi*h^6) * (h - r)^2
        w = ti.Vector([0.0, 0.0, 0.0])
        r_norm = r.norm()
        if 0 <= r_norm <= h:
            hr = h - r_norm
            gradient_norm = -45.0 / (np.pi * h6) * hr * hr
            w = r / r_norm * gradient_norm
        return w

    @ti.func
    def s_corr(r):
        # Eq(13) with k=0.1, n=4, delta_q=0.3h
        tmp = Util.W_poly6(r.norm()) / Util.W_poly6(0.2 * h)
        return (-0.001) * (tmp ** 4)

    @ti.func
    def euclidean(i, j, positions):
        return (positions[i] - positions[j]).norm()
