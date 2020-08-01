# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
# %%


class City:
    __city_count = 0

    def __init__(self):
        self.id = City.__city_count
        self.x, self.y = np.random.randint(-100, 101, 2)
        self.z = np.random.randint(0, 51)
        City.__city_count += 1

    def __str__(self):
        return f"[{self.id}, ({self.x}, {self.y}, {self.z})]"

    def __distance(self, x_diff, y_diff, z_diff):
        diff = np.array([x_diff, y_diff, z_diff])
        return np.sqrt(np.sum(diff**2))

    def distance(self, other):
        diff = [self.x-other.x, self.y-other.y, self.z-other.z]
        return self.__distance(*diff)

    def distance_directed(self, other):
        diff = [self.x-other.x, self.y-other.y, self.z-other.z]
        if self.z - other.z > 0:  # going down
            diff[2] *= 0.9
        elif self.z - other.z < 0:  # going up
            diff[2] *= 1.1
        return self.__distance(*diff)


# %%
c1, c2, c3, c4 = City(), City(), City(), City()
c1.x, c1.y, c1.z = 2, 1, 0
c2.x, c2.y, c2.z = -4, -5, 6
c3.x, c3.y, c3.z = 0, 0, 0
c4.x, c4.y, c4.z = 1, 1, 1
cities = [c1, c2, c3, c4]
mx_size = len(cities)**2
drop_ix1 = np.random.choice(
    range(len(cities)), int(0.2 * mx_size), replace=False)
drop_ix2 = np.random.choice(
    range(len(cities)), int(0.2 * mx_size), replace=False)
# It is slightly more than 80%, because A-A drops are also included
drop_edges_ix = [x for x in zip(drop_ix1, drop_ix2)]
drop_edges_ix


# %%
sym_aff_mx = np.zeros((len(cities), len(cities)))
for i, city1 in enumerate(cities):
    for j, city2 in enumerate(cities):
        sym_aff_mx[i][j] = city1.distance(city2)

sym_aff_mx

# %%
asym_aff_mx = np.zeros((len(cities), len(cities)))
for i, city1 in enumerate(cities):
    for j, city2 in enumerate(cities):
        asym_aff_mx[i][j] = city1.distance_directed(city2)

asym_aff_mx

# %%


def drop_edges(aff_mx, edges):
    drop_aff_mx = np.copy(aff_mx)
    for c1, c2 in zip(drop_ix1, drop_ix2):
        drop_aff_mx[c1][c2] = np.inf
    return drop_aff_mx


# %%
drop_sym_aff_mx = drop_edges(sym_aff_mx, drop_edges_ix)
drop_sym_aff_mx

# %%
drop_asym_aff_mx = drop_edges(asym_aff_mx, drop_edges_ix)
drop_asym_aff_mx

# %%


def is_connected(c1, c2, aff_mx):
    return not (np.isinf(aff_mx[c1][c2]) or aff_mx[c1][c2] == 0)


def neighbors(city, aff_mx):
    for i in range(len(cities)):
        if is_connected(city, i, aff_mx):
            yield i


# %%
mapdict = {i: chr(v) for i, v in enumerate(range(ord('A'), ord('Z')+1))}

# recursive
# def dfs(start, aff_mx, visited=[], cur_path=[], paths=[]):
#     cur_path.append(start)
#     visited.append(start)
#     for n in neighbors(start, aff_mx):
#         if n not in visited:
#             dfs(n, aff_mx, visited[:], cur_path[:], paths)
#     if len(cur_path) == len(aff_mx):
#         paths.append(cur_path)
#     return paths


def dfs(start, aff_mx):
    paths, stack = [], [[start]]
    while stack:
        c = stack.pop()
        for n in neighbors(c[-1], aff_mx):
            if n not in c:
                stack.append([*c, n])
        # print(stack)
        if stack and len(stack[-1]) == len(aff_mx):
            paths.append(stack[-1])
    return paths


def bfs(start, aff_mx):
    paths, queue = [], [[start]]
    while queue:
        c = queue.pop()
        for n in neighbors(c[-1], aff_mx):
            if n not in c:
                queue.insert(0, [*c, n])
        # print(queue)
        if queue and len(queue[-1]) == len(aff_mx):
            paths.append(queue[-1])
    return paths

# %%


def path_cost(path, aff_mx):
    sum = 0
    for i in range(1, len(path)):
        sum += aff_mx[path[i-1]][path[i]]
    sum += aff_mx[path[-1]][path[0]] # closing the cycle
    return sum


def shortest_path(paths, aff_mx):
    min_distance = np.inf
    min_path = []
    for path in paths:
        distance = path_cost(path, aff_mx)
        if distance < min_distance:
            min_distance, min_path = distance, path
    return min_distance, min_path


# %%
def closest_city(start, cities, aff_mx):
    closest_city, closest_distance = start, np.inf
    for city in cities:
        distance = aff_mx[start][city]
        if distance < closest_distance:
            closest_city, closest_distance = city, distance
    return closest_city


def nn(start, aff_mx):
    path = [start]
    while len(path) != len(aff_mx):
        cur = path[-1]
        cur_neighbors = [x for x in neighbors(cur, aff_mx) if x not in path]
        city = closest_city(cur, cur_neighbors, aff_mx)
        path.append(city)
    return path

# %%
unvisited = list(True for x in cities)
def dijkstra(source, aff_mx):
    global unvisited
    unvisited[source] = False
    q, dist = [], []
    # prev = []
    for i in range(len(aff_mx)):
        dist.append(np.inf)
        # prev.append(np.nan)
        q.append(i)
    dist[source] = 0
    
    while q:
        u = q.pop(np.argmin(dist))
        for v in neighbors(u, aff_mx):
            if v not in q:
                continue
            alt = dist[u] + aff_mx[u][v]
            if alt < dist[v]:
                dist[v] = alt
                # prev[v] = u
    return dist


matrix = sym_aff_mx
# path = []
# while True in unvisited:
dijkstra(1, matrix)
    
# shortest_path(dfs(0, matrix), matrix)


# %%
