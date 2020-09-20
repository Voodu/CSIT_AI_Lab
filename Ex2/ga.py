# %%
from importlib import resources
import pandas as pd
import numpy as np
np.random.seed(0)

class Task:
	def __init__(self, id, resources, time):
		self.id = id
		self.resources = resources
		self.time = time
		self.current_step_index = 0
		self.__current_finish_time = 0
	
	def __repr__(self):
		return f'''
		Task ID: {self.id}
		Resources: {self.resources}
		Time: {self.time}'''

	def current_step(self):
		'''
		Output:
			tuple with currently required (resource, time)
		'''
		return (self.resources[self.current_step_index] - 1, self.time[self.current_step_index]) if self.current_step_index <= 9 else None

	def do_current_step(self, start_time):
		if self.current_step_index <= 9:
			self.__current_finish_time = start_time + self.time[self.current_step_index]
			self.current_step_index += 1

	def current_finish_time(self):
		return self.__current_finish_time

	def reset(self):
		self.current_step_index = 0
		self.__current_finish_time = 0


# %%
data = pd.read_excel('GA_task.xlsx', skiprows=1)
tasks = np.array([Task(x+1, np.zeros(shape=(data.shape[0]), dtype="int"), np.zeros(shape=(data.shape[0]), dtype="int")) for x in range(len(data.columns)//2)])
# %%
for ir, row in data.iterrows():
	for ic, column in enumerate(row):
		if ic % 2 == 0:
			tasks[ic//2].resources[ir] = column
		else:
			tasks[ic//2].time[ir] = column
# %%
test_set = tasks[:5]
test_set

# %%
class Scheduler:
	def __init__(self, tasks, geno=None):
		self.tasks = tasks[:]
		self.geno = self.__build_geno() if geno is None else geno 
		self.id = self.geno#str(uuid.uuid4())[:6]
		self.readable_schedule = [[] for x in range(10)]
		#[1, 0, 0, 1, 4, 2, 2, 0, 2, 3, 0, 0, 3, 3, 0, 3, 0, 1, 0, 2, 1, 3, 1, 2, 4, 2, 0, 1, 3, 1, 4, 4, 2, 1, 4, 4, 3, 3, 2, 1, 4, 2, 1, 4, 0, 3, 2, 4, 3, 4]
		# print("Geno:", self.geno)
	
	def __build_geno(self):
		g = np.array([np.arange(len(self.tasks)) for x in range(10)])
		g = g.flatten()
		np.random.shuffle(g)
		return g

	def create_schedule(self):
		for task in self.tasks:
			task.reset()
		schedule = [[] for x in range(10)]
		self.readable_schedule = [[] for x in range(10)]
		for g in self.geno:
			t = self.tasks[g]
			cs = t.current_step() # cur is task from geno
			res_occupied_til = sum(schedule[cs[0]])
			gap = max(t.current_finish_time() - res_occupied_til, 0)
			occup_time = gap + cs[1]
			start_time = res_occupied_til + gap
			schedule[cs[0]].append(occup_time)
			self.readable_schedule[cs[0]].append((f"T:{g+1}.{t.current_step_index}", f"S:", start_time, f"E:", start_time+cs[1], f"D:", cs[1])) # Task.Step, Start, End, Duration
			t.do_current_step(start_time)
		for task in self.tasks:
			assert task.current_step() is None

		for m in schedule:
			while len(m) < 500:
				m.append(0)
		return np.array(schedule)
	
	def cross(self, other):
		counts = [0 for x in range(len(self.tasks))]
		child = []
		i = 0
		for x,y in zip(self.geno, other.geno):
			if i % 2 == 0:
				if counts[x] < 10:
					child.append(x)
					counts[x] += 1
				elif counts[y] < 10:
					child.append(y)
					counts[y] += 1
			else:
				if counts[y] < 10:
					child.append(y)
					counts[y] += 1
				elif counts[x] < 10:
					child.append(x)
					counts[x] += 1
			i += 1
		
		for i, c in enumerate(counts):
			while c < 10:
				child.insert(np.random.randint(len(child), size=1)[0], i)
				c += 1

		return Scheduler(self.tasks, np.array(child))

	def mutate(self, percent = 25):
		newGeno = self.geno[:]
		shuffle_total = len(newGeno) * (percent/100)
		indexes = np.random.choice(len(newGeno), int(shuffle_total), replace=False)
		values = newGeno[indexes]
		np.random.shuffle(values)
		newGeno[indexes] = values

		return Scheduler(self.tasks, newGeno)



# %%
def adapt_function(schedule): #schedule row = 1 machine
	return schedule.sum(axis=1).max()


# %%
# Create initial population
population_size = 15
population = [Scheduler(test_set) for _ in range(population_size)]

# %%
n_generations = 500
for i in range(n_generations):
	# Check the outcome with the current population
	results = np.array([(i, adapt_function(x.create_schedule())) for i, x in enumerate(population)])

	# Find the best results
	best = sorted(results, key=lambda it: it[1])[:4]
	if i%20 == 0:
		print(f"{i} - Lowest time: {best[0][1]}")

	# Create new population
	new_pop = []
	new_pop.append(population[best[0][0]])
	new_pop.append(population[best[1][0]])
	new_pop.append(population[best[2][0]])
	new_pop.append(population[best[3][0]])
	new_pop.append(new_pop[0].cross(new_pop[1]))
	new_pop.append(new_pop[1].cross(new_pop[2]))
	new_pop.append(new_pop[2].cross(new_pop[3]))
	new_pop.append(new_pop[3].cross(new_pop[0]))
	new_pop.append(new_pop[-4].mutate())
	new_pop.append(new_pop[-3].mutate())
	new_pop.append(new_pop[-2].mutate(50))
	new_pop.append(new_pop[-1].mutate(75))
	new_pop.append(Scheduler(test_set))
	new_pop.append(Scheduler(test_set))
	new_pop.append(Scheduler(test_set))

	population = new_pop[:]

	

# %%
print(f"Lowest time: {best[0][1]}")
print("Best schedule")
print(population[0].readable_schedule)
# %%
