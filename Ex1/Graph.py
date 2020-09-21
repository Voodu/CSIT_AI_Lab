from State import State
from Node import Node
import numpy as np
from queue import PriorityQueue

class Graph:
	def __init__(self, n_nodes=10, directional=False, full=True, fixed=True):
		if fixed:
			n1, n2, n3, n4 = Node(), Node(), Node(), Node()
			n1.x, n1.y, n1.z = 2, 1, 0
			n2.x, n2.y, n2.z = -4, -5, 6
			n3.x, n3.y, n3.z = 0, 0, 0
			n4.x, n4.y, n4.z = 1, 1, 1
			self.nodes = [n1, n2, n3, n4]
		else:
			self.nodes = [Node() for x in range(n_nodes)]
		self.__init__affinity_matrix(directional, full)

	def nodes_count(self):
		return len(self.matrix)

	def get_distance(self, n1, n2):
		return self.matrix[n1][n2]

	def set_distance(self, n1, n2, dist):
		self.matrix[n1][n2] = dist

	def is_connected(self, n1, n2):
		return not (np.isinf(self.get_distance(n1,n2)) or self.get_distance(n1,n2) == 0)

	def neighbors(self, node):
		n = []
		for i in range(len(self.nodes)):
			if self.is_connected(node, i):
				n.append(i)
		return n

	def get_all_edges(self):
		edges = []
		for i in range(len(self.matrix)):
			for j in range(len(self.matrix[i])):
				distance = self.get_distance(i, j)
				if i != j and distance != np.inf:
					edges.append((i, j, distance))
		
		return edges

	def partial_path_cost(self, path):
		sum = 0
		for i in range(1, len(path)):
			sum += self.get_distance(path[i-1], path[i])
		return sum
	
	def full_path_cost(self, path):
		if len(path) != self.nodes_count():
			return np.inf
		cost = self.partial_path_cost(path)
		cost += self.get_distance(path[-1], path[0]) # closing the cycle
		return cost

	def dfs(self, start):
		paths, stack = [], [[start]]
		while stack:
			c = stack.pop()
			for n in self.neighbors(c[-1]):
				if n not in c:
					stack.append([*c, n])
			# print(stack)
			if stack and len(stack[-1]) == self.nodes_count():
				paths.append(stack[-1])
		return paths

	def bfs(self, start):
		paths, queue = [], [[start]]
		while queue:
			c = queue.pop()
			for n in self.neighbors(c[-1]):
				if n not in c:
					queue.insert(0, [*c, n])
			# print(queue)
			if queue and len(queue[-1]) == self.nodes_count():
				paths.append(queue[-1])
		return paths

	def nn(self, start):
		path = [start]
		while len(path) != self.nodes_count():
			cur = path[-1]
			cur_neighbors = [x for x in self.neighbors(cur) if x not in path]
			if not cur_neighbors:
				break
			city = self.__closest_city_naive(cur, cur_neighbors)
			path.append(city)
		return self.full_path_cost(path), path

	def __closest_city_naive(self, start, cities):
		closest_city, closest_distance = start, np.inf
		for city in cities:
			distance = self.get_distance(start, city)
			if distance < closest_distance:
				closest_city, closest_distance = city, distance
		return closest_city

	def a_star(self, start, heuristic):
		states = PriorityQueue()
		states.put((0, State([start], self.neighbors(start))))
		current = states.get()
		
		while len(current[1].path) != self.nodes_count():
			for neighbor in current[1].neighbors:
				if neighbor not in current[1].path:
					past_path = current[1].path + [neighbor]
					c_path = past_path + [start]
					g = self.partial_path_cost(c_path)
					h = heuristic(c_path, self)
					states.put((g + h, State(past_path, self.neighbors(neighbor))))
			current = states.get()
		
		return current[0], current[1].path

	def dijkstra(self, start):
		return self.a_star(start, lambda a, b: 0)

	def aco(self, start, n_iter=20, n_ants=20):
		def ant_path(start, pher_map, alpha, beta):
			path = [start]
			vis_map = np.divide(1, self.matrix)
			# repeat until path does not cover all the nodes
			while len(path) != self.nodes_count():
				# get all neighbors of the current node which are not yet visited
				neighbors = [n for n in self.neighbors(path[-1]) if n not in path]
				# if there are none, then it is dead end
				if not neighbors:
					return np.inf, path 
				# calculate distance to all the neighbors
				visibilities = [vis_map[path[-1]][n] for n in neighbors]
				# calculate pheromone of all the neighbors
				pheromones = [pher_map[path[-1]][n] for n in neighbors]
				# calculate probabilites of all the neighbors
				probabilities = np.array([(vis**alpha)*(prob**beta) for vis, prob in zip(visibilities, pheromones)])
				probabilities = probabilities / probabilities.sum()
				# pick one using specified probabilities
				
				picked = np.random.choice(a=neighbors, p=probabilities)
				# go to it and update path
				path.append(picked)
			
			# closing the cycle is done in self.path_cost
			return self.full_path_cost(path), path

		def update_map(pher_map, history, evapo_rate):
			# history[i] = (route, cost) -> ([4, 5, 1, 3, 2], 43)
			# evaporate some pheromone
			pher_map = pher_map * (1-evapo_rate)
			# add pheromone to the edges based on the route length
			for entry in history:
				for node_i in range(len(entry[1])-1):
					pher_map[node_i][node_i + 1] += 1/entry[0]
			return pher_map

		
		alpha = 1
		beta = 2
		evapo_rate = .5
		pher_map = np.ones((self.nodes_count(),self.nodes_count()))
		best_record = (np.inf, [])
		for it in range(n_iter):
			history = []
			for ant in range(n_ants):
				cost, path = ant_path(start, pher_map, alpha, beta)
				history.append((cost, path))
				if cost < best_record[0]:
					best_record = (cost, path)
			pher_map = update_map(pher_map, history, evapo_rate)
		
		return best_record

	def __init__affinity_matrix(self, directional, full):
		self.__init_simple_matrix(directional)
		if not full:
			drop_pairs = self.__get_drop_pairs()
			self.__drop_edges(drop_pairs)

	def __get_drop_pairs(self):
		mx_size = len(self.nodes)**2
		drop_ix1 = np.random.choice(range(len(self.nodes)), int(0.2 * mx_size))
		drop_ix2 = np.random.choice(range(len(self.nodes)), int(0.2 * mx_size))
		# It is slightly more than 80%, because A-A drops are also included
		drop_node_pairs = [x for x in zip(drop_ix1, drop_ix2)]
		return drop_node_pairs

	def __drop_edges(self, nodes):
		for n1, n2 in nodes:
			self.set_distance(n1, n2, np.inf)

	def __init_simple_matrix(self, directional):
		nodes = self.nodes
		self.matrix = np.zeros((len(nodes), len(nodes)))
		for i, node1 in enumerate(nodes):
			for j, node2 in enumerate(nodes):
				if directional:
					self.set_distance(i, j, node1.distance_directed(node2))
				else:
					self.set_distance(i, j, node1.distance(node2))
