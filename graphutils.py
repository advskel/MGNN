import random

from incidence_graph import IncidenceGraph
import numpy as np
from tqdm import tqdm
import heapq
from unionfind import UnionFind


# TODO import existing graph based on adj matrix or edge list
def grid(rows, cols):
    g = IncidenceGraph()

    def index(r, c):
        return r * cols + c

    def reverse_index(index):
        return index // cols, index % cols

    for v in tqdm(range(rows * cols), desc='Generating grid'):
        r, c = reverse_index(v)
        if c + 1 < cols:
            g.put_simplex((index(r, c), index(r, c + 1)), 1)
        if r + 1 < rows:
            g.put_simplex((index(r, c), index(r + 1, c)), 1)
        if c - 1 >= 0:
            g.put_simplex((index(r, c), index(r, c - 1)), 1)
        if r - 1 >= 0:
            g.put_simplex((index(r, c), index(r - 1, c)), 1)
    return g, index, reverse_index


def plane(v, e, d, min_x, max_x, dist_metric, var, connected=True):
    # todo custom seed
    # default no self loops, no multiple connections, undirected

    if e > v * (v - 1) / 2:
        raise ValueError(f'Too many edges, max is {v * (v - 1) / 2} for {v} vertices')
    if connected and e < v - 1:
        raise ValueError(f'Not enough edges for a connected graph, min is {v - 1} for {v} vertices')

    g = IncidenceGraph()
    for i in range(v):
        coords = np.random.uniform(low=min_x, high=max_x, size=d)
        g.put_simplex(i, data=coords)

    def data_gen(xs):
        dist = dist_metric(xs[0], xs[1])
        return dist + abs(random.normalvariate(0, dist * var))

    available = list(range(v))
    if connected:
        uf = UnionFind(range(v))
    for i in tqdm(range(e), desc='Generating edges'):
        if len(available) < 2:
            break
        if connected and uf.n_comps > 1:
            comps = uf.components()
            cs = random.sample(range(len(comps)), 2)
            v_i = random.choice(list(comps[cs[0]]))
            v_j = random.choice(list(comps[cs[1]]))
            _, node_i = g._IncidenceGraph__get((v_i,))
            _, node_j = g._IncidenceGraph__get((v_j,))
            uf.union(v_i, v_j)
        else:
            v_i = random.choice(available)
            _, node_i = g._IncidenceGraph__get((v_i,))
            i_avail = node_i.non_adjacency_set(0)
            v_j = random.choice(list(i_avail))
            _, node_j = g._IncidenceGraph__get((v_j,))

        g.put_simplex((v_i, v_j), data_gen=data_gen)

        if len(node_i.neighbors) >= v - 1:
            available.remove(v_i)
        if len(node_j.neighbors) >= v - 1:
            available.remove(v_j)

    return g


def astar(graph, i, j, heuristic=None, ver_heur=False):
    if heuristic is None:
        heuristic = lambda x_i, x, y_i, y: 0.0

    pq = []
    heapq.heappush(pq, (0.0, 0.0, i))
    iters = 0
    dists = [-1] * graph.size(0)
    j_data = graph.get(j)
    while len(pq) > 0:
        h, dist, v = heapq.heappop(pq)
        if v == j:
            if ver_heur and h > dist:
                raise ValueError('Inadmissible/inconsistent heuristic')
            return dist, iters
        if ver_heur and h < dist:
            raise ValueError('Inadmissible/inconsistent heuristic')
        if dists[v] != -1:
            if ver_heur and dist < dists[v]:
                raise ValueError('Inadmissible/inconsistent heuristic')
            continue

        dists[v] = dist

        _, node = graph._IncidenceGraph__get((v,))
        for neighbor in node.neighbors:
            edge = graph.get((v, neighbor))
            if edge < 0:
                raise ValueError('Negative edge weight')
            new_dist = dist + edge
            if dists[neighbor] != -1:
                if ver_heur and new_dist < dists[neighbor]:
                    raise ValueError('Inadmissible/inconsistent heuristic')
                continue
            n_data = graph.get(neighbor)

            new_h = heuristic(neighbor, n_data, j, j_data)
            if ver_heur and new_h < 0:
                raise ValueError('Inadmissible/inconsistent heuristic')
            # if new_h != heuristic(j_data, n_data):
            #     raise ValueError('Non-symmetric heuristic')

            heapq.heappush(pq, (new_h + new_dist, new_dist, neighbor))

        iters += 1

    return None, iters


def dijkstra(graph, i):
    pq = []
    heapq.heappush(pq, (0.0, 0, i))
    dists = [-1] * graph.size(0)
    edges = [-1] * graph.size(0)
    while len(pq) > 0:
        dist, e, v = heapq.heappop(pq)
        if dists[v] != -1:
            continue
        dists[v] = dist
        edges[v] = e

        _, node = graph._IncidenceGraph__get((v,))
        for neighbor in node.neighbors:
            if dists[neighbor] != -1:
                continue
            edge = graph.get((v, neighbor))
            if edge < 0:
                raise ValueError('Negative edge weight')
            heapq.heappush(pq, (dist + edge, e + 1, neighbor))

    return dists, edges


def all_pairs_sp(graph):
    dists = []
    edges = []
    for i in range(graph.size(0)):
        dists.append(dijkstra(graph, i)[0])
        edges.append(dijkstra(graph, i)[1])
    return dists, edges


def graph_heuristic(graphrep, dist_metric):
    def heuristic(x_i, x, y_i, y):
        return dist_metric(graphrep[x_i], graphrep[y_i])

    return heuristic


def dist_heuristic(dist_metric):
    def heuristic(x_i, x, y_i, y):
        return dist_metric(x, y)

    return heuristic
