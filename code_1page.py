import numpy as np
import taichi as ti
import math

ti.init(arch=ti.gpu)

# All color
color_bg = 0x1f4677
color_particle = 0xc7dcf9
color_particle2 = 0xff0000
color_particle3 = 0x00ff00

# Constants
dim = 2
g = ti.Vector([0.0, -9.8])
dt = 1. / 30.
epsilon = 5e-6
h = 1.1
h6 = h ** 6
h9 = h ** 9

num_solver_itr = 5

mass = 1.0
rho0 = 1.0
CFM_epsilon = 100

# Screen Size
screen = (1000, 500)
screen_world_ratio = 10.
boundary = (screen[0] / screen_world_ratio,
            screen[1] / screen_world_ratio)

# Particle
num_particles = 1000
particle_radius = 0.4
particle_radius_render = particle_radius * screen_world_ratio

# Position/Velocity
previous_positions = ti.Vector.field(dim, ti.f32, shape=num_particles)
positions = ti.Vector.field(dim, ti.f32, shape=num_particles)
velocities = ti.Vector.field(dim, ti.f32, shape=num_particles)

lambdas = ti.field(ti.f32, shape=num_particles)
delta_p = ti.Vector.field(dim, ti.f32, shape=num_particles)

# Cell logic
num_max_particles_in_cell = 100
cell_size = 2
gridX = int(boundary[0] / cell_size)
gridY = int(boundary[1] / cell_size)

particles_in_cell = ti.field(ti.i32, shape=(gridX, gridY, num_max_particles_in_cell))
num_particles_in_cell = ti.field(ti.i32, shape=(gridX, gridY))

# Neighbour logic
num_max_neighbours = 50  # maximum number of neighbours taken
interest_zone_radius = h * 1.5
num_neighbours = ti.field(ti.i32, shape=num_particles)
neighbours = ti.field(ti.i32, shape=(num_particles, num_max_neighbours))


@ti.kernel
def particle_init():
    start = (20, 10)
    offsetX = 1
    offsetY = 1
    for i in range(40):
        for j in range(25):
            positions[i * 25 + j] = ti.Vector([start[0] + i * offsetX, start[1] + j * offsetY])


@ti.func
def fix_boundary():
    for i in range(num_particles):
        for d in ti.static(range(dim)):
            lower_bound, upper_bound = particle_radius, boundary[d] - particle_radius
            if positions[i][d] < lower_bound:
                positions[i][d] = lower_bound + epsilon * ti.random()
            if positions[i][d] > upper_bound:
                positions[i][d] = upper_bound - epsilon * ti.random()


def render(gui):
    gui.clear(color_bg)
    pos = positions.to_numpy()
    pos[:, 0] *= screen_world_ratio / screen[0]
    pos[:, 1] *= screen_world_ratio / screen[1]

    gui.circles(pos, radius=particle_radius_render, color=color_particle)


    # print(neighbours.to_numpy())
    # print(num_neighbours.to_numpy()[0])
    # print(neighbours.to_numpy()[0])
    # print(num_particles_in_cell.to_numpy()[:, 0])

    gui.circles(np.array([pos[0]]), radius=particle_radius_render, color=color_particle2)
    bbb = np.array([pos[t] for t in neighbours.to_numpy()[0] if t != -1])
    if bbb.size != 0:
        gui.circles(bbb, radius=particle_radius_render, color=color_particle3)

    gui.show()


@ti.kernel
def algo1():
    save_previous_positions()
    apply_gravity()
    fix_boundary()

    find_neighborhood()

    for _ in ti.static(range(num_solver_itr)):
        calculate_lambda()
        calculate_delta_p()
        update_position()
        fix_boundary()

    update_velocity()


@ti.func
def save_previous_positions():
    for i in range(num_particles):
        previous_positions[i] = positions[i]


@ti.func
def apply_gravity():
    for i in range(num_particles):
        velocities[i] += dt * g
        positions[i] += dt * velocities[i]


@ti.func
def pos_to_cell(position):
    return int(position / cell_size)


@ti.func
def in_bound(cell):
    return 0 <= cell[0] < gridX and 0 <= cell[1] < gridY


@ti.func
def euclidean(i, j):
    return (positions[i] - positions[j]).norm()


@ti.func
def find_neighborhood():
    # clear neighbour
    for i in ti.grouped(num_particles_in_cell):
        num_particles_in_cell[i] = 0
    for i in ti.grouped(particles_in_cell):
        particles_in_cell[i] = 0
    for i in ti.grouped(num_neighbours):
        num_neighbours[i] = 0
    for i in ti.grouped(neighbours):
        neighbours[i] = -1

    # put particle into neighbourhood
    for i in range(num_particles):
        particle_position = positions[i]
        cur_cell = pos_to_cell(particle_position)
        cur_num_index = ti.atomic_add(num_particles_in_cell[cur_cell], 1)
        particles_in_cell[cur_cell, cur_num_index] = i

    # calculate neighbourhood
    for i in range(num_particles):
        particle_position = positions[i]
        cur_cell = pos_to_cell(particle_position)
        for off_x, off_y in ti.ndrange((-1, 2), (-1, 2)):
            check_offset = ti.Vector([off_x, off_y])
            pot_cell = cur_cell + check_offset
            if in_bound(pot_cell):
                # check all particle in potential cell

                for j_index in range(num_particles_in_cell[pot_cell]):
                    j = particles_in_cell[pot_cell, j_index]
                    if j != i and euclidean(i, j) < interest_zone_radius:
                        cur_num_index = ti.atomic_add(num_neighbours[i], 1)
                        neighbours[i, cur_num_index] = j
                    if num_neighbours[i] >= num_max_neighbours:
                        break
            if num_neighbours[i] >= num_max_neighbours:
                break


@ti.func
def W_poly6(r, h):
    # 0 <= r < h ? 315 / (64*pi*h^9) * (h^2 - r^2)^3 : 0
    # Eq (20) Müller D, et al. (2003)
    w = 0.0
    if 0.0 <= r <= h:
        hr = h * h - r * r
        w = 315. / (64. * math.pi * h9) * (hr * hr * hr)
    return w


@ti.func
def W_spiky_gradient(r, h):
    # 0 <= r < h ? 15 / (pi*h^6) * (h - r)^3 : 0
    # Eq (21) Müller D, et al. (2003)
    # take derivative respect to r to get g_norm = -45 / (pi*h^6) * (h - r)^2
    w = ti.Vector([0.0, 0.0])
    r_norm = r.norm()
    if 0 <= r_norm <= h:
        hr = h - r_norm
        gradient_norm = -45.0 / (math.pi * h6) * hr * hr
        w = r / r_norm * gradient_norm
    return w


@ti.func
def calculate_lambda():
    for i in range(num_particles):
        pos_i = positions[i]

        rho_i = 0.0
        grad_pi_Ci = ti.Vector([0.0, 0.0])
        sum_grad_sph_sqr = 0.0

        for j_index in range(num_neighbours[i]):
            j = neighbours[i, j_index]
            pos_j = positions[j]
            pos_ij = pos_i - pos_j

            # Eq(2)
            rho_i += W_poly6(pos_ij.norm(), h)

            # Eq(7)
            grad_pj_Ci = W_spiky_gradient(pos_ij, h)
            grad_pi_Ci += grad_pj_Ci
            sum_grad_sph_sqr += grad_pj_Ci.dot(grad_pj_Ci)

        # Eq(1)
        c_i = (mass * rho_i / rho0) - 1.0

        # Eq(11)
        sum_grad_sph_sqr *= 1.0 / rho0
        sum_grad_sph_sqr += 1.0 / rho0 * grad_pi_Ci.dot(grad_pi_Ci)

        lambdas[i] = (-c_i) / (sum_grad_sph_sqr + CFM_epsilon)


@ti.func
def s_corr(r, h):
    # Eq(13) with k=0.1, n=4, delta_q=0.3h
    tmp = W_poly6(r.norm(), h) / W_poly6(0.2 * h, h)
    return (-0.001) * (tmp ** 4)


@ti.func
def calculate_delta_p():
    # Eq(14)
    for i in range(num_particles):
        pos_i = positions[i]
        lambda_i = lambdas[i]
        delta_p[i] = ti.Vector([0.0, 0.0])
        for j_index in range(num_neighbours[i]):
            j = neighbours[i, j_index]
            pos_j = positions[j]
            lambda_j = lambdas[j]
            pos_ij = pos_i - pos_j
            delta_p[i] += (lambda_i + lambda_j + s_corr(pos_ij, h)) * W_spiky_gradient(pos_ij, h)

        delta_p[i] *= 1.0 / rho0


@ti.func
def update_position():
    for i in range(num_particles):
        positions[i] += delta_p[i]


@ti.func
def update_velocity():
    for i in positions:
        velocities[i] = 1.0 / dt * (positions[i] - previous_positions[i])


if __name__ == "__main__":
    gui = ti.GUI('PBF2D', screen)
    particle_init()

    while gui.running and not gui.get_event(gui.ESCAPE):
        algo1()
        render(gui)
