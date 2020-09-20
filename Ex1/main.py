# %%
from Graph import Graph
import numpy as np
np.random.seed(0)
# %%

g_undir_full = Graph()
g_undir_partial = Graph(full=False)
g_dir_full = Graph(directional=True)
g_dir_partial = Graph(full=False, directional=True)

# %%
def shortest_path(paths, g):
    min_distance = np.inf
    min_path = []
    for path in paths:
        distance = g.path_cost(path)
        if distance < min_distance:
            min_distance, min_path = distance, path
    return min_distance, min_path

# %%
graph = g_dir_partial
starting_city = 0

print("Dijkstra", graph.dijkstra(starting_city))
print("NN", graph.nn(starting_city))
print("DFS", shortest_path(graph.dfs(starting_city), graph))
print("BFS", shortest_path(graph.bfs(starting_city), graph))
print("ACO", graph.aco(starting_city))



# %%
from queue import PriorityQueue

def raw_a_star(start, goal, g, h):
    q = PriorityQueue()
    q.put((0, start))
    visited = set({start})
    unvisited = set(range(len(g.matrix)))
    unvisited.remove(start)
    g_vals = [np.inf for _ in range(len(g.matrix))]
    g_vals[start] = 0
    f_vals = [np.inf for _ in range(len(g.matrix))]
    f_vals[start] = h(start, goal, g)

    while not q.empty():
        _, current = q.get()
        visited.add(current)
        if current == goal:
            return f_vals[goal] # CHECK IT
        cur_neighbors = g.neighbors(current)
        for n in cur_neighbors:
            checked = g_vals[current] + g.matrix[current][n]
            if checked < g_vals[n]:
                g_vals[n] = checked
                f_vals[n] = g_vals[n] + h(n, goal, g)
                if n not in visited:
                    q.put((f_vals[n], n))
    return np.inf

def euc_dist(a, b, g):
    return g.nodes[a].distance(g.nodes[b])

def h_tsp(current, start_city, unvisited_cities, aff_mx):
    d1, d2, d3 = np.inf, np.inf, np.inf
    # D1: distance to the nearest unvisited city from the current city
    for city in unvisited_cities:
        if aff_mx[current][city] < d1:
            d1 = aff_mx[current][city]
    # D2: estimated distance to travel all the unvisited cities (MST heuristic used here)
    
    # D3: nearest distance from an unvisited city to the start city
    for city in unvisited_cities:
        if aff_mx[city][start_city] < d3:
            d3 = aff_mx[city][start_city]

    return d1 + d2 + d3

# %%
raw_a_star(0, 3, graph, euc_dist)

# %%