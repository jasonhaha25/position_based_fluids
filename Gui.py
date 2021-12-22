import taichi as ti
import numpy as np

# Screen Size
boundary = (100, 50)
screen_world_ratio = 10.
SCREEN_WIDTH = int(boundary[0] * screen_world_ratio)
SCREEN_HEIGHT = int(boundary[1] * screen_world_ratio)
SCREEN_SIZE = (SCREEN_WIDTH, SCREEN_HEIGHT)

# Particle Render Size
particle_radius = 0.4
particle_radius_render = particle_radius * screen_world_ratio

# All color
color_bg = 0x1f4677
color_particle = 0xc7dcf9
color_particle2 = 0xff0000
color_particle3 = 0x00ff00
color_pusher = 0xffc0cb

@ti.data_oriented
class PBF_Gui:
    def __init__(self):
        self.screen = SCREEN_SIZE
        self.gui = ti.GUI('PBF2D', self.screen)

    def render(self, positions, neighbours, boundary):
        self.gui.clear(color_bg)
        pos = positions.to_numpy()
        pos[:, 0] *= screen_world_ratio / SCREEN_SIZE[0]
        pos[:, 1] *= screen_world_ratio / SCREEN_SIZE[1]

        self.gui.circles(pos, radius=particle_radius_render, color=color_particle)

        # print(neighbours.to_numpy())
        # print(num_neighbours.to_numpy()[0])
        # print(neighbours.to_numpy()[0])
        # print(num_particles_in_cell.to_numpy()[:, 0])

        self.gui.circles(np.array([pos[0]]), radius=particle_radius_render, color=color_particle2)
        bbb = np.array([pos[t] for t in neighbours.to_numpy()[0] if t != -1])
        if bbb.size != 0:
            self.gui.circles(bbb, radius=particle_radius_render, color=color_particle3)

        temp = boundary.to_numpy()[0][0]
        self.gui.line(np.array([temp/100, 0]), np.array([temp/100, 1]), color=color_pusher, radius=3)

        self.gui.show()

    def is_running(self):
        return self.gui.running

    def not_escape(self):
        return not self.gui.get_event(self.gui.ESCAPE)

