import pygame
import math
import random
from pygame.locals import *

class Master():
    def __init__(self) -> None:
        self.particles = []
        self.particle_generation_cooldown = 100
        self.particle_generation_last_update = 0

    def add_particles(self,x_list, y_list):
        self.particles.append(Particles(random.randint(x_list[0] ,x_list[1] ), random.randint(y_list[0] ,y_list[1] ), 5))

    def recursive_call(self, time, display, scroll, dt, x_list, y_list):
        if self.particles != []:
            for pos, particle in sorted(enumerate(self.particles), reverse=True):
                particle.move(time, dt)
                particle.draw(display, scroll)
                if not particle.alive:
                    self.particles.pop(pos)
        if time - self.particle_generation_last_update > self.particle_generation_cooldown:
            self.particle_generation_last_update = time
            self.add_particles(x_list, y_list)


class Particles():
    def __init__(self, x, y, speed) -> None:
        self.x = x
        self.y = y
        self.speed = speed
        self.gravity = 5
        self.alive = True
        self.angle = random.randint(0,360)
        self.angle_change_cooldown = 100
        self.angel_change_last_update = 0
        self.radius_last_update = 0
        self.radius_cooldown = 400
        self.radius = 7

    def move(self, time, dt):
        if time - self.angel_change_last_update > self.angle_change_cooldown:
            
            self.angel_change_last_update = time
            self.angle += random.randint(0,10)
            if self.angle > 360:
                self.angle = 0
        if time - self.radius_last_update > self.radius_cooldown:
            if self.radius > 0:
                self.radius -= 0.5
            else:
                self.alive = False
            self.radius_last_update = time
        self.x += math.sin(math.radians(self.angle)) * dt
        self.y += 0.5 * dt
        if self.x > 10000 or self.y > 800:
            self.alive = False

    def draw(self, display, scroll):
        pygame.draw.circle(display, (135, 53, 85), (self.x - scroll[0], self.y - scroll[1]), self.radius)
        self.radius *= 2
        display.blit(self.circle_surf(), (int(self.x- self.radius) - scroll[0], int(self.y - self.radius) - scroll[1]), special_flags=BLEND_RGBA_ADD)
        self.radius /= 2

    def circle_surf(self):
        surf = pygame.Surface((self.radius * 2, self.radius * 2))
        pygame.draw.circle(surf, (20, 20, 20), (self.radius, self.radius), self.radius)
        surf.set_colorkey((0, 0, 0))
        return surf
