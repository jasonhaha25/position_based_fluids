from Gui import PBF_Gui
with open('temp.txt', 'w') as the_file:
    the_file.write('3')
from util import *
from PLYWriter import PLYWriter

ti.init(arch=ti.gpu)


@ti.data_oriented
class PBF3D:
    def __init__(self, dim=3):
        self.gui = PBF_Gui()
        self.dim = dim

        # Position/Velocity
        self.previous_positions = ti.Vector.field(dim, ti.f32, shape=num_particles)
        self.positions = ti.Vector.field(dim, ti.f32, shape=num_particles)
        self.velocities = ti.Vector.field(dim, ti.f32, shape=num_particles)

        self.lambdas = ti.field(ti.f32, shape=num_particles)
        self.delta_p = ti.Vector.field(dim, ti.f32, shape=num_particles)

        # Neighbour logic
        self.particles_in_cell = ti.field(ti.i32, shape=(gridX, gridY, gridZ, num_max_particles_in_cell))
        self.num_particles_in_cell = ti.field(ti.i32, shape=(gridX, gridY, gridZ))

        self.num_neighbours = ti.field(ti.i32, shape=num_particles)
        self.neighbours = ti.field(ti.i32, shape=(num_particles, num_max_neighbours))

        # initial particle positions
        self.particle_3d_init()

        #visual
        self.boundary = ti.Vector.field(3, ti.f32, shape=1)
        for i in range(1):
            self.boundary[i][0] = boundary[0]
            self.boundary[i][1] = boundary[1]
            self.boundary[i][2] = boundary[2]
        self.b_amount = 0.1

    @ti.kernel
    def algo1(self):
        # implement paper algorithm
        self.save_previous_positions()
        self.apply_gravity()
        self.fix_boundary()

        self.find_neighborhood()

        for _ in ti.static(range(num_solver_itr)):
            self.calculate_lambda()
            self.calculate_delta_p()
            self.update_position()
            self.fix_boundary()

        self.update_velocity()

    @ti.kernel
    def particle_3d_init(self):
        start = (10, 5, 5)
        offsetX = 0.1
        offsetY = 0.1
        offsetZ = 0.1
        for i in range(4):
            for j in range(25):
                for k in range(25):
                    self.positions[i * 25 + j * 25 + k] = ti.Vector([start[0] + i * offsetX,
                                                                     start[1] + j * offsetY,
                                                                     start[2] + k * offsetZ])

    @ti.func
    def fix_boundary(self):
        for i in range(num_particles):
            for d in ti.static(range(self.dim)):
                lower_bound, upper_bound = particle_radius, self.boundary[0][d] - particle_radius
                if self.positions[i][d] < lower_bound:
                    self.positions[i][d] = lower_bound + epsilon * ti.random()
                if self.positions[i][d] > upper_bound:
                    self.positions[i][d] = upper_bound - epsilon * ti.random()

    @ti.func
    def save_previous_positions(self):
        for i in range(num_particles):
            self.previous_positions[i] = self.positions[i]

    @ti.func
    def apply_gravity(self):
        for i in range(num_particles):
            self.velocities[i] += dt * g
            self.positions[i] += dt * self.velocities[i]

    @ti.func
    def find_neighborhood(self):
        # clear neighbour
        for i in ti.grouped(self.num_particles_in_cell):
            self.num_particles_in_cell[i] = 0
        for i in ti.grouped(self.particles_in_cell):
            self.particles_in_cell[i] = 0
        for i in ti.grouped(self.num_neighbours):
            self.num_neighbours[i] = 0
        for i in ti.grouped(self.neighbours):
            self.neighbours[i] = -1

        # put particle into neighbourhood
        for i in range(num_particles):
            particle_position = self.positions[i]
            cur_cell = Util.pos_to_cell(particle_position)
            cur_num_index = ti.atomic_add(self.num_particles_in_cell[cur_cell], 1)
            self.particles_in_cell[cur_cell, cur_num_index] = i

        # calculate neighbourhood
        for i in range(num_particles):
            particle_position = self.positions[i]
            cur_cell = Util.pos_to_cell(particle_position)
            for off_x, off_y, off_z in ti.ndrange((-1, 2), (-1, 2), (-1, 2)):
                check_offset = ti.Vector([off_x, off_y, off_z])
                pot_cell = cur_cell + check_offset
                if Util.in_bound(pot_cell):
                    # check all particle in potential cell

                    for j_index in range(self.num_particles_in_cell[pot_cell]):
                        j = self.particles_in_cell[pot_cell, j_index]
                        if j != i and Util.euclidean(i, j, self.positions) < interest_zone_radius:
                            cur_num_index = ti.atomic_add(self.num_neighbours[i], 1)
                            self.neighbours[i, cur_num_index] = j
                        if self.num_neighbours[i] >= num_max_neighbours:
                            break
                if self.num_neighbours[i] >= num_max_neighbours:
                    break

    @ti.func
    def calculate_lambda(self):
        for i in range(num_particles):
            pos_i = self.positions[i]

            rho_i = 0.0
            grad_pi_Ci = ti.Vector([0.0, 0.0, 0.0])
            sum_grad_sph_sqr = 0.0

            for j_index in range(self.num_neighbours[i]):
                j = self.neighbours[i, j_index]
                pos_j = self.positions[j]
                pos_ij = pos_i - pos_j

                # Eq(2)
                rho_i += Util.W_poly6(pos_ij.norm())

                # Eq(7)
                grad_pj_Ci = Util.W_spiky_gradient_3d(pos_ij)
                grad_pi_Ci += grad_pj_Ci
                sum_grad_sph_sqr += grad_pj_Ci.dot(grad_pj_Ci)

            # Eq(1)
            c_i = (mass * rho_i / rho0) - 1.0

            # Eq(11)
            sum_grad_sph_sqr *= 1.0 / rho0
            sum_grad_sph_sqr += 1.0 / rho0 * grad_pi_Ci.dot(grad_pi_Ci)

            self.lambdas[i] = (-c_i) / (sum_grad_sph_sqr + CFM_epsilon)

    @ti.func
    def calculate_delta_p(self):
        # Eq(14)
        for i in range(num_particles):
            pos_i = self.positions[i]
            lambda_i = self.lambdas[i]
            self.delta_p[i] = ti.Vector([0.0, 0.0, 0.0])
            for j_index in range(self.num_neighbours[i]):
                j = self.neighbours[i, j_index]
                pos_j = self.positions[j]
                lambda_j = self.lambdas[j]
                pos_ij = pos_i - pos_j
                self.delta_p[i] += (lambda_i + lambda_j + Util.s_corr(pos_ij)) * Util.W_spiky_gradient_3d(pos_ij)

            self.delta_p[i] *= 1.0 / rho0

    @ti.func
    def update_position(self):
        for i in range(num_particles):
            self.positions[i] += self.delta_p[i]

    @ti.func
    def update_velocity(self):
        for i in self.positions:
            self.velocities[i] = 1.0 / dt * (self.positions[i] - self.previous_positions[i])

    @ti.kernel
    def move(self, amount: ti.template()):
        for i in ti.static(range(1)):
            self.boundary[i][0] += amount

    def run(self):
        frame = 0
        w = PLYWriter(num_particles, self.positions)
        while self.gui.is_running() and self.gui.not_escape() and frame <= max_frame:
            if frame % 50 == 0:
                self.b_amount *= -1
            self.move(self.b_amount)
            self.algo1()
            # if self.dim == 2:
            #     self.gui.render(self.positions, self.neighbours)
            # elif self.dim == 3:

            if frame % 10 == 0:
                w.write(frame)
            frame += 1




if __name__ == "__main__":
    pbf = PBF3D(3)
    pbf.run()
