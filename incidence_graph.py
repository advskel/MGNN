import itertools
from collections.abc import Iterable
import numpy as np


class _IncidenceNode:
    def __init__(self, vertices, data, index, graph):
        self.vertices = vertices
        self.d = len(vertices) - 1
        self.upper = set()  # list of indices of upper dimension connections
        self.lower = set()  # list of indices of lower dimension neighbors
        self.neighbors = set()  # list of indices of neighbors
        self.index = index
        self.graph = graph
        self.data = data

    def adjacency_set(self, rel_dim):
        # TODO rel_dim always positive but connect both ways
        adj = set()
        if rel_dim == 0:
            adj.update(self.neighbors)
        elif rel_dim == 1:
            adj.update(self.upper)
        elif rel_dim == -1:
            adj.update(self.lower)
        elif rel_dim > 1:
            for i in self.upper:
                node = self.graph.dimensions[self.d + 1][i]
                adj.update(node.adjacency_set(rel_dim - 1))
        else:
            for i in self.lower:
                node = self.graph.dimensions[self.d - 1][i]
                adj.update(node.adjacency_set(rel_dim + 1))
        return adj

    # TODO neighbor sets for points d connections away
    # TODO generalize with some probability (so not all connections)

    def non_neighbors(self, rel_dim):
        if self.d + rel_dim < 0 or self.d + rel_dim >= len(self.graph.dimensions):
            return set()

        adj = set(range(len(self.graph.dimensions[self.d + rel_dim])))
        adj.difference_update(self.adjacency_set(rel_dim))
        if rel_dim == 0:
            adj.discard(self.index)
        return adj

    def apply_data_gen(self, data_gen):
        if not callable(data_gen):
            return
        data = []
        for i in self.lower:
            data.append(self.graph.dimensions[self.d - 1][i].data)
        self.data = data_gen(data)

    def generalize(self, data_gen):
        if self.d == 0:
            return

        counts = dict()
        for lower_index in self.lower:
            lower_node = self.graph.dimensions[self.d - 1][lower_index]
            for index in lower_node.neighbors:
                node = self.graph.dimensions[self.d - 1][index]
                for vertex in node.vertices:
                    if vertex in self.vertices:
                        continue
                    if vertex not in counts:
                        counts[vertex] = 1
                    else:
                        counts[vertex] += 1

        for upper_index in self.upper:
            upper_node = self.graph.dimensions[self.d + 1][upper_index]
            for vertex in upper_node.vertices:
                if vertex in counts:
                    counts.pop(vertex)

        for vertex, count in counts.items():
            if count == (self.d + 1) * self.d:
                self.graph.put_simplex(self.vertices + (vertex,), data_gen=data_gen)

    def __repr__(self):
        return f'{self.vertices}: {self.data}'


class IncidenceGraph:
    def __init__(self):
        self.facet_to_index = dict()  # maps facets (tuples of vertices) to indices

        self.dimensions = []  # dimensions[d] stores list of nodes of dimension d
        self.dimensions.append([])

    def put_simplex(self, vertices, data=None, data_gen=None):
        vertices = self.__type_check(vertices)
        d = len(vertices) - 1
        index, _ = self.__get(vertices)
        if index == -1:
            index, _ = self.__make_node(vertices, data)
            self.__make_simplex(index, d, data, data_gen)
        else:
            self.dimensions[d][index].data = data
            self.dimensions[d][index].apply_data_gen(data_gen)

    def get(self, vertices):
        vertices = self.__type_check(vertices)
        index, node = self.__get(vertices)
        if index == -1:
            raise KeyError(f'No node with vertices {vertices}')

        return node.data

    def get_many(self, v_list):
        return [self.get(v) for v in v_list]

    def neighbors(self, vertices, rel_dim=0):
        vertices = self.__type_check(vertices)
        _, node = self.__get(vertices)

        return [self.dimensions[node.d + rel_dim][i].vertices for i in node.adjacency_set(rel_dim)]

    def adjacency_list(self, rel_dim=0, nodes=None):
        start_indices = self.__get_offsets()

        adj = []
        for d in range(len(self.dimensions)):
            for i, node in enumerate(self.dimensions[d]):
                node_adj = node.adjacency_set(rel_dim)
                for j in node_adj:
                    adj.append((start_indices[d] + i, start_indices[d + rel_dim] + j))

        return adj

    def adjacency_matrix(self, rel_dim=0):
        return next(self.partial_matrix(rel_dim))

    def partial_matrix(self, rel_dim, partial_size=None):
        for adj in self.partial_matrices([rel_dim], partial_size):
            yield adj[0]

    def adjacency_matrices(self, rel_dims):
        return next(self.partial_matrices(rel_dims))

    def partial_matrices(self, rel_dims, partial_size=None):
        start_indices = self.__get_offsets()
        flattened = self.__flatten()
        A, N = len(rel_dims), len(self)

        if partial_size is None or partial_size > N:
            partial_size = N
        elif partial_size <= 0:
            raise ValueError('Partial matrix size must be positive')

        adj = np.zeros((A, partial_size, N), dtype=np.float32)
        batch_index = 0
        for node in flattened:
            for a, rel_dim in enumerate(rel_dims):
                node_adj = node.adjacency_set(rel_dim)
                for j in node_adj:
                    adj[a, batch_index, start_indices[node.d + rel_dim] + j] = 1.0
            batch_index += 1
            if batch_index == partial_size:
                yield adj
                adj = np.zeros((A, partial_size, N), dtype=np.float32)
                batch_index = 0

        if batch_index != 0:
            yield adj[:, :batch_index, :]

    def shape(self):
        return [len(d) for d in self.dimensions]

    def generalize(self, dimension=None, data_gen=None):
        # data can be function that generates data based on existing data from lower nodes
        if dimension is None:
            prev_size = len(self.dimensions)
            for d in range(1, len(self.dimensions)):
                for node in self.dimensions[d]:
                    node.generalize(data_gen)
            while len(self.dimensions) != prev_size:
                prev_size = len(self.dimensions)
                for node in self.dimensions[-1]:
                    node.generalize(data_gen)

        elif dimension >= len(self.dimensions) or dimension < 0:
            raise IndexError(f'Invalid dimension {dimension}')
        else:
            for node in self.dimensions[dimension]:
                node.generalize(data_gen)

    def size(self, dim=None):
        if dim is None:
            return len(self)
        if dim < 0:
            raise IndexError(f'Invalid dimension {dim}')
        elif dim >= len(self.dimensions):
            return 0

        return len(self.dimensions[dim])

    def __new_node(self, vertices, data, index):
        # can overload
        return _IncidenceNode(vertices, data, index, self)

    def __type_check(self, vertices):
        if isinstance(vertices, int):
            vs = (vertices,)
        elif isinstance(vertices, Iterable):
            vs = tuple(sorted(vertices))
        else:
            raise TypeError("Vertices must be int or iterable")

        if len(vs) == 0:
            raise ValueError("Vertices must be non-empty")
        if vs[0] < 0:
            raise IndexError("Vertices must be non-negative")

        return vs

    def __get(self, vertices):
        if vertices[-1] >= len(self.dimensions[0]):
            return -1, None

        if len(vertices) == 1:
            index = vertices[0]
            return index, self.dimensions[0][index]

        if vertices in self.facet_to_index:
            index = self.facet_to_index[vertices]
            d = len(vertices) - 1
            return index, self.dimensions[d][index]
        return -1, None

    def __make_node(self, vertices, data):
        d = len(vertices) - 1

        while len(self.dimensions) <= d:
            self.dimensions.append([])

        while len(self.dimensions[0]) < vertices[-1]:
            v_id = len(self.dimensions[0])
            vertex = self.__new_node((len(self.dimensions[0]),), None, v_id)
            self.dimensions[0].append(vertex)

        index = len(self.dimensions[d])

        node = self.__new_node(vertices, data, index)
        self.dimensions[d].append(node)
        if d > 0:
            self.facet_to_index[vertices] = index
        return index, node

    def __make_simplex(self, index, d, data, gen):
        # turn existing node with dimension d and index into simplex
        node = self.dimensions[d][index]
        if d == 0:
            return

        sub_faces = []
        for sub_face in itertools.combinations(node.vertices, d):
            sub_index, sub_node = self.__get(sub_face)
            if sub_index == -1:
                sub_index, sub_node = self.__make_node(sub_face, data)
                self.__make_simplex(sub_index, d - 1, data, gen)

            node.lower.add(sub_index)
            sub_node.upper.add(index)
            sub_faces.append(sub_node)

        node.apply_data_gen(gen)

        for i in range(len(sub_faces)):
            for j in range(i + 1, len(sub_faces)):
                sub_faces[i].neighbors.add(sub_faces[j].index)
                sub_faces[j].neighbors.add(sub_faces[i].index)

    def __get_offsets(self):
        start_indices = [0]
        for d in range(1, len(self.dimensions)):
            start_indices.append(start_indices[d - 1] + len(self.dimensions[d - 1]))

        return start_indices

    def __flatten(self):
        nodes = []
        for d in range(len(self.dimensions)):
            for node in self.dimensions[d]:
                nodes.append(node)
        return nodes

    def __iter__(self):
        return IncidenceGraphIterator(self)

    def __len__(self):
        return len(self.facet_to_index) + len(self.dimensions[0])

    def __getitem__(self, item):
        return self.get(item)

    def __contains__(self, item):
        try:
            vertices = self.__type_check(item)
            index, _ = self.__get(vertices)
            return index != -1
        except:
            return False


class IncidenceGraphIterator:
    def __init__(self, graph, d=0, i=0):
        self.graph = graph
        _d, _i = d, i
        if _d < 0:
            _d += len(graph.dimensions)
        while 0 <= _d < len(graph.dimensions):
            if _i < 0:
                _i += len(graph.dimensions[_d])
            if 0 <= _i < len(graph.dimensions[_d]):
                self.d = _d
                self.i = _i
                return
            if _i > 0:
                _i -= len(graph.dimensions[_d])
                _d += 1
            else:
                _d -= 1

        raise IndexError(f'Invalid starting index {i} from dimension {d}')


    def __iter__(self):
        return self

    def __next__(self):
        if self.d >= len(self.graph.dimensions):
            raise StopIteration

        node = self.graph.dimensions[self.d][self.i]
        self.i += 1
        if self.i >= len(self.graph.dimensions[self.d]):
            self.d += 1
            self.i = 0
        return node.vertices, node.data

