# %%
from State import State
from os import curdir
from Graph import Graph
import numpy as np
np.random.seed(4)

# %%
g_undir_full = Graph(fixed=False)
g_undir_partial = Graph(fixed=True, full=False)
g_dir_full = Graph(fixed=True, directional=True)
g_dir_partial = Graph(fixed=True, full=False, directional=True)

# %%
def shortest_path(paths, g):
    min_distance = np.inf
    min_path = []
    for path in paths:
        distance = g.full_path_cost(path)
        if distance < min_distance:
            min_distance, min_path = distance, path
    return min_distance, min_path

# %%
graph = g_undir_full
starting_city = 0

# inadmissible - average of left edges * number of left steps
# admissible - min of left edges * number of left steps

def get_unvisited_edges(path, graph):
    all_edges = graph.get_all_edges()
    left_edges = []    
    for edge in all_edges:
        if edge[1] == path[0]: #entering start
            if edge[0] not in path:
                left_edges.append(edge)
                continue
        if edge[0] == path[-2]: #exiting the last
            if edge[1] not in path:
                left_edges.append(edge)
                continue
        if edge[0] not in path and edge[1] not in path:
            left_edges.append(edge)
    return left_edges

def h_inadm(path, graph):
    left_edges = get_unvisited_edges(path, graph)
    if not left_edges:
        return 0
    mean = np.mean(left_edges, axis=0)[-1] # mean of the distances
    return mean * (graph.nodes_count() - len(path) + 1)

def h_adm(path, graph):
    left_edges = get_unvisited_edges(path, graph)
    if not left_edges:
        return 0
    minimum = np.min(left_edges, axis=0)[-1] # min of the distances
    return minimum * (graph.nodes_count() - len(path) + 1)

# %%
print("DFS", shortest_path(graph.dfs(starting_city), graph))
print("BFS", shortest_path(graph.bfs(starting_city), graph))
print("Dijkstra", graph.dijkstra(starting_city))
print("NN", graph.nn(starting_city))
print("ACO", graph.aco(starting_city, 40, 40))
print("A* adm", graph.a_star(starting_city, h_adm))
print("A* inadm", graph.a_star(starting_city, h_inadm))

# %%
