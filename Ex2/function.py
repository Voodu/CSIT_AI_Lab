# %%
import numpy as np
from random import random

class Particle:
	def init():
		Particle.best_position = np.random.rand(2) * 9 - 4.5
		Particle.best_value = Particle.test_function(Particle.best_position)

	def test_function(position):
		x = position[0]
		y = position[1]
		z = (1.5 - x - x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2
		# z = 5 + 3*x - 4*y - x**2 + x*y - y**2; x,y,z = 2/3, -5/3, 28/3
		return z

	def __init__(self, inertia=1, own_trust=1, swarm_trust=1):
		self.position = np.random.rand(2) * 9 - 4.5
		self.intertia = inertia
		self.own_trust = own_trust
		self.swarm_trust = swarm_trust
		self.best_position = np.random.rand(2) * 9 - 4.5
		self.best_value = Particle.test_function(self.best_position)
		self.velocity = np.random.rand(2) * 2 - 1 # -1.0 to +1.0

	def new_velo(self):
		r1, r2 = random() * 2, random() * 2
		own_speed = self.intertia * self.velocity
		own_best = (r1 * self.own_trust) * (self.best_position - self.position)
		swarm_best = (r2 * self.swarm_trust) * (Particle.best_position - self.position)
		self.velocity = own_speed + own_best + swarm_best

	def move(self, drag=0):
		new_position = self.position + self.velocity * (1-drag)
		if np.any(np.fabs(new_position) > 4.5):
			new_position = self.position - self.velocity * (1-drag)

		self.position = new_position
		cur_value = Particle.test_function(self.position)
		if cur_value < self.best_value: # CHANGE DIRECTION FOR MINIMIZING!
			self.best_position = self.position
			self.best_value = cur_value 
		if self.best_value < Particle.best_value: # CHANGE DIRECTION FOR MINIMIZING!
			Particle.best_value = self.best_value
			Particle.best_position = self.position

# %%
# %%
def run_swarm(swarm_size = 200, time = 200, inertia = 1, own_trust = 1, swarm_trust = 2):
	Particle.init()
	swarm = np.array([Particle(inertia,own_trust, swarm_trust) 	for _ in range(swarm_size)])
	for t in range(time):
		for p in swarm:
			p.move(t/time - 0.3)
			p.new_velo()

	return (Particle.best_position, Particle.best_value)

params = {
	"swarm_size": 200,
	"time": 200,
	"inertia": 1,
	"own_trust": 1,
	"swarm_trust": 2
}

result = run_swarm(**params)
print("best_position", result[0])
print("best_value", result[1])
# %%
