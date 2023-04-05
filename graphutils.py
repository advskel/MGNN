import random
from typing import Tuple, Callable, List, Any, Optional

from incidence_graph import IncidenceGraph
import numpy as np
from tqdm import tqdm
import heapq
from unionfind import UnionFind


def osm_import_directed(location):
    import osmnx as ox

    print('Importing graph from OSM...')
    G = ox.utils_graph.get_digraph(ox.graph_from_place(location, network_type="drive"))
    converted = IncidenceGraph()

    mapping = dict()
    i = 0
    print('Converting graph to IncidenceGraph...')
    for u, d in G.nodes(data=True):
        mapping[u] = i
        converted.put_simplex(vertices=i, data=np.array([d['x'], d['y']], dtype=np.float32))
        i += 1
    for u, v, d in G.edges(data=True):
        if mapping[u] == mapping[v]:
            continue
        converted.put_incidence_relation(src_list=[mapping[u]], dest_list=[mapping[v]], data=d['length'])
    return converted

def osm_import_undirected(location):
    import osmnx as ox

    G = ox.utils_graph.get_digraph(ox.graph_from_place(location, network_type="drive"))
    converted = IncidenceGraph()

    mapping = dict()
    i = 0
    for u, d in G.nodes(data=True):
        mapping[u] = i
        converted.put_simplex(vertices=i, data=np.array([d['x'], d['y']], dtype=np.float32))
        i += 1
    for u, v, d in G.edges(data=True):
        if mapping[u] == mapping[v]:
            continue
        converted.put_simplex(vertices=(mapping[u], mapping[v]), data=d['length'])
    return converted


# TODO import existing graph based on adj matrix or edge list
def grid(rows: int, cols: int) -> Tuple[IncidenceGraph, Callable[[int, int], int], Callable[[int], Tuple[int, int]]]:
    # TODO create option to connect diagonally
    """
    Generates a grid graph with the given number of rows and columns.

    Args:
        rows: number of rows
        cols: number of columns

    Returns:
        An incidence graph representing the grid and two utility functions: the first maps (integer) coordinates to
        vertex indices, and the second maps vertex indices to coordinates.
    """
    g = IncidenceGraph()

    def index(r, c):
        return r * cols + c

    def reverse_index(index):
        return index // cols, index % cols

    for v in tqdm(range(rows * cols), desc='Generating grid'):
        r, c = reverse_index(v)
        if c + 1 < cols:
            g.put_simplex((index(r, c), index(r, c + 1)), data=1)
        if r + 1 < rows:
            g.put_simplex((index(r, c), index(r + 1, c)), data=1)
        if c - 1 >= 0:
            g.put_simplex((index(r, c), index(r, c - 1)), data=1)
        if r - 1 >= 0:
            g.put_simplex((index(r, c), index(r - 1, c)), data=1)
    return g, index, reverse_index


def plane(v: int, e: int, d: int, min_x: int | float, max_x: int | float,
          dist_metric: Callable[[Any, Any], int | float], var: float=0.0, connected: bool=True):
    """
    Generates a random graph on an R^d plane, in which the edge lengths between two vertices must be at least the
    distance between them, plus some positive random value drawn from a normal distribution. The graph contains no
    self-loops, no repeated connections, and is undirected.

    Args:
        v: The number of vertices to generate.
        e: The number of edges to generate.
        d: The integer dimension of the vertex coordinates.
        min_x: The minimum coordinate value.
        max_x: The maximum coordinate value.
        dist_metric: A function that takes as input two d-dimensional vectors and returns the distance between them.
        var: A positive float value. Each edge length will be at least the distance between two vertices, plus a
            positive random value drawn from a normal distribution with mean 0 and standard deviation var * dist, where
            `dist` is the distance between the vertices. Therefore, `var` is a proportion of the distance between the
            vertices, not directly the standard deviation.
        connected: Whether to ensure that the graph is connected.

    Returns:
        An incidence graph representing the generated graph.

    Raises:
        ValueError: If there are too many requested edges, or if not enough edges are requested to ensure connectivity
            (if `connected` is True).
    """
    # todo custom seed

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


def astar(graph: IncidenceGraph, i: int, j: int,
          heuristic: Optional[Callable[[int, int, Any, Any], int | float]]=None,
          ver_heur: bool=False) -> Tuple[Optional[float], int]:
    """
    Performs A* shortest paths search on a graph between two vertices.
    Args:
        graph: An incidence graph.
        i: Index of the source vertex.
        j: Index of the destination vertex.
        heuristic: A heuristic function that takes as input the indices of two vertices and their data values and
            returns an estimate of the distance between them. The heuristic must be symmetric: h_ij = h_ji.

            If None, then the heuristic will always be 0.
        ver_heur: Whether to verify the heuristic function. If True, then an error will be raised whenever the A*
            algorithm encounters discrepancies in its distance estimates (i.e., if the heuristic function is either
            inconsistent or inadmissible).

    Returns:
        A tuple of two values. The first value is the shortest path distance from vertex i to vertex j, or None if no
        path is found. The second value is the number of nodes expanded by the A* algorithm (a.k.a. the number of loop
        iterations), useful for benchmarking/ranking tests.

    Raises:
        ValueError: If the heuristic function is inconsistent or inadmissible (only if ver_heur is True).
    """
    if heuristic is None:
        heuristic = lambda x_i, y_i, x, y: 0.0

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
        for neighbor, _ in node.neighbor_relations():
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


def dijkstra_sp(graph: IncidenceGraph, i: int) -> Tuple[List[float], List[int]]:
    """
    Finds the shortest path from the given vertex to all other vertices in a graph.

    Args:
        graph: An incidence graph.
        i: The index of the vertex to start from.

    Returns:
        A tuple of two lists. The first list contains the shortest path distances from the origin vertex: dist[j] is the
        shortest path distance from vertex i to vertex j, or -1 if no path is found. The second list contains the
        number of edges in the shortest path from the origin: edges[j] is the minimum number of edges in the shortest
        path from vertex i to vertex j (-1 if no path is found).
    """
    pq = []
    heapq.heappush(pq, (0.0, 0, i))
    dists = [-1] * graph.size(0)
    edges = [-1] * graph.size(0)
    while len(pq) > 0:
        dist, e, v = heapq.heappop(pq)
        if dists[v] != -1:
            if dist <= dists[v]:
                edges[v] = min(edges[v], e)
            continue
        dists[v] = dist
        edges[v] = e

        _, node = graph._IncidenceGraph__get((v,))
        for neighbor, _ in node.neighbor_relations():
            if dists[neighbor] != -1:
                continue
            edge = graph.get((v, neighbor))
            if edge < 0:
                raise ValueError('Negative edge weight')
            heapq.heappush(pq, (dist + edge, e + 1, neighbor))

    return dists, edges


def all_pairs_sp(graph: IncidenceGraph) -> Tuple[List[List[float]], List[List[int]]]:
    """
    Performs all-pairs shortest paths on a graph.

    Args:
        graph: An incidence graph.

    Returns:
        A tuple of two lists. The first list contains the shortest path distances between all pairs of vertices:
        dists[i][j] is the shortest path distance from vertex i to vertex j. The second list contains the number of
        edges in the shortest path between all pairs of vertices: edges[i][j] is the minimum number of edges in the
        shortest path from vertex i to vertex j. Both values are -1 if no path is found between i and j.
    """
    dists = []
    edges = []
    for i in tqdm(range(graph.size(0)), desc='All-pairs shortest paths'):
        sp = dijkstra_sp(graph, i)
        dists.append(sp[0])
        edges.append(sp[1])
    return dists, edges


def graph_heuristic(graphrep: IncidenceGraph, dist_metric: Callable[[Any, Any], int | float]) -> Callable[[int, int, Any, Any], int | float]:
    """
    Creates a heuristic function for A* shortest paths search on a graph from a graph representation and a distance
    metric.

    Args:
        graphrep: An incidence graph.
        dist_metric: A distance metric that takes as input two vertex data values and returns the distance between them.

    Returns:
        A heuristic function that takes as input the indices of two vertices and their data values and returns the
        distance between the two vertices in the incidence graph based on the provided distance metric.
    """
    def heuristic(x_i, y_i, x, y):
        return dist_metric(graphrep[x_i], graphrep[y_i])

    return heuristic


def dist_heuristic(dist_metric: Callable[[Any, Any], int | float]) -> Callable[[int, int, Any, Any], int | float]:
    """
    Creates a heuristic function for A* shortest paths search from a distance metric.

    Args:
        dist_metric: A distance metric that takes as input two vertex data values and returns the distance between them.

    Returns:
        A heuristic function that takes as input the indices of two vertices and their data values and returns the
        distance between the two vertices based on the provided distance metric.
    """
    def heuristic(x_i, y_i, x, y):
        return dist_metric(x, y)

    return heuristic
