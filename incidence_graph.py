from __future__ import annotations

import itertools
from collections.abc import Iterable, Iterator, Sized, Container
from collections import deque
from typing import List, Tuple, Set, Dict, Optional, Any, Callable  # TODO use built-in list, tuple etc.
import numpy as np
from numpy.typing import NDArray


class IncidenceGraph(Sized, Container, Iterable):
    class _IncidenceNode:
        def __init__(self, vertices: Tuple[int, ...], data: Any, index: int, graph: IncidenceGraph):
            self.vertices: Tuple[int, ...] = vertices
            self.d: int = len(vertices) - 1
            self.upper: Set[int] = set()  # list of indices of upper dimension connections
            self.lower: Set[int] = set()  # list of indices of lower dimension neighbors
            self.index: int = index
            self.graph: IncidenceGraph = graph
            self.data: Any = data
            self.is_simplex: bool = False

        def get_total_index(self) -> int:
            offset = 0
            for i in range(self.d):
                offset += len(self.graph._dimensions[i])
            return offset + self.index

        def neighbor_relations(self, dist: int | Iterable[int] = 1) -> Iterable[Tuple[int, int]]:
            if isinstance(dist, int):
                dist = set().add(dist)
            else:
                dist = set(dist)

            max_dist = -1
            for d in dist:
                if d < 0:
                    raise ValueError('Distances must be non-negative')
                max_dist = max(max_dist, d)

            upper_visited = set()
            visited = set()
            bfs = deque()
            bfs.append((self.index, self.d, 0))
            while bfs:
                index, dim, d = bfs.popleft()
                node = self.graph._dimensions[dim][index]
                if dim == self.d:
                    if index in visited:
                        continue
                    visited.add(index)
                    if d in dist:
                        yield index, d
                    if d < max_dist:
                        for i in node.upper:
                            if i not in upper_visited:
                                bfs.append((i, dim + 1, d + 1))
                elif dim == self.d + 1:
                    if index in upper_visited:
                        continue
                    upper_visited.add(index)

                    for i in node.lower:
                        if i not in visited:
                            bfs.append((i, dim - 1, d))

        def incidence_relations(self, rel_dim: int | Iterable[int]) -> Iterable[Tuple[int, int]]:
            if isinstance(rel_dim, int):
                rel_dim = set().add(rel_dim)
            else:
                rel_dim = set(rel_dim)

            min_dim, max_dim = 0, 0
            for d in rel_dim:
                if d < 0:
                    min_dim = min(min_dim, d)
                else:
                    max_dim = max(max_dim, d)

            offsets = list(itertools.accumulate([0] + self.graph.shape()))
            bfs = deque()
            bfs.append((self, 0))
            visited = set()
            while bfs:
                node, d = bfs.popleft()
                id = node.index + offsets[node.d]
                if id in visited:
                    continue
                if d in rel_dim:
                    yield id, d
                visited.add(id)

                if 0 <= d < max_dim:
                    for j in node.upper:
                        if j + offsets[node.d + 1] not in visited:
                            bfs.append((self.graph._dimensions[node.d + 1][j], d + 1))
                if 0 >= d > min_dim:
                    for j in node.lower:
                        if j + offsets[node.d - 1] not in visited:
                            bfs.append((self.graph._dimensions[node.d - 1][j], d - 1))

        # TODO generalize with some probability (so not all connections)

        def non_adjacency_set(self, rel_dim: int, include_self: bool = False) -> Set[int]:
            if self.d + rel_dim < 0 or self.d + rel_dim >= len(self.graph._dimensions):
                return set()

            adj = set(range(len(self.graph._dimensions[self.d + rel_dim])))
            adj.difference_update(self.incidence_relations(rel_dim))
            if rel_dim == 0 and not include_self:
                adj.discard(self.index)
            return adj

        def apply_data_gen(self, data_gen: Optional[Callable[[List[Any]], Any]]) -> None:
            if data_gen is None:
                return
            data = []
            for i in self.lower:
                data.append(self.graph._dimensions[self.d - 1][i].data)
            self.data = data_gen(data)

        def generalize(self, data_gen: Optional[Callable[[List[Any]], Any]]) -> None:
            if self.d == 0:
                return

            counts = dict()
            for lower_index in self.lower:
                lower_node = self.graph._dimensions[self.d - 1][lower_index]
                for index, _ in lower_node.neighbor_relations():
                    node = self.graph._dimensions[self.d - 1][index]
                    for vertex in node.vertices:
                        if vertex in self.vertices:
                            continue
                        if vertex not in counts:
                            counts[vertex] = 1
                        else:
                            counts[vertex] += 1

            for upper_index in self.upper:
                upper_node = self.graph._dimensions[self.d + 1][upper_index]
                for vertex in upper_node.vertices:
                    if vertex in counts:
                        counts.pop(vertex)

            for vertex, count in counts.items():
                if count == (self.d + 1) * self.d:
                    self.graph.put_simplex(self.vertices + (vertex,), data_gen=data_gen)

        def remove_self(self) -> None:
            self.unset_simplex()
            for i in self.upper:
                self.graph._dimensions[self.d + 1][i].lower.discard(self.index)
            for i in self.lower:
                self.graph._dimensions[self.d - 1][i].upper.discard(self.index)

        def unset_simplex(self) -> None:
            if not self.is_simplex:
                return
            self.is_simplex = False
            for i in self.upper:
                self.graph._dimensions[self.d + 1][i].unset_simplex()

        def __repr__(self) -> str:
            return f'{self.vertices}: {self.data}'

    class IncidenceGraphIterator(Iterator):
        def __init__(self, graph: IncidenceGraph, d: int = 0, i: int = 0, size: Optional[int] = None):
            self.graph: IncidenceGraph = graph
            _d, _i = d, i
            if _d < 0:
                _d += len(graph._dimensions)
            while 0 <= _d < len(graph._dimensions):
                if _i < 0:
                    _i += len(graph._dimensions[_d])
                if 0 <= _i < len(graph._dimensions[_d]):
                    break
                if _i > 0:
                    _i -= len(graph._dimensions[_d])
                    _d += 1
                else:
                    _d -= 1

            self.d: int = _d
            self.i: int = _i

            if size is None:
                self.size: int = len(graph)
            else:
                self.size: int = size

        def __next__(self) -> Tuple[Tuple[int, ...], Any]:
            if self.d >= len(self.graph._dimensions) or self.d < 0 or self.i < 0 or self.size <= 0:
                raise StopIteration

            node = self.graph._dimensions[self.d][self.i]
            self.i += 1
            if self.i >= len(self.graph._dimensions[self.d]):
                self.d += 1
                self.i = 0
            self.size -= 1
            return node.vertices, node.data

    def __init__(self):
        self._facet_to_index: Dict[Tuple[int, ...], int] = dict()  # maps facets (tuples of vertices) to indices

        self._dimensions: List[
            List[IncidenceGraph._IncidenceNode]] = []  # dimensions[d] stores list of nodes of dimension d
        self._dimensions.append([])

    # TODO neighbor sets for points d connections away

    def put_simplex(self, vertices: int | Iterable[int], data: Any = None,
                    data_gen: Optional[Callable[[List[Any]], Any]] = None) -> None:
        vertices = IncidenceGraph.__type_check(vertices)
        d = len(vertices) - 1
        index, node = self.__get(vertices)
        if index == -1:
            index, _ = self.__make_node(vertices, data)
            self.__make_simplex(index, d, data, data_gen)
        else:
            node = self._dimensions[d][index]
            if not node.is_simplex:
                self.__make_simplex(index, d, data, data_gen)
            else:
                node.data = data
                node.apply_data_gen(data_gen)

    def remove_node(self, vertices: int | Iterable[int], as_simplex: bool = False) -> None:
        vertices = IncidenceGraph.__type_check(vertices)
        index, node = self.__get(vertices)
        if index == -1:
            return

        if as_simplex:
            self.__remove_node(index, node.d, True)
        else:
            self.__remove_node(index, node.d, False)
            node.unset_simplex()

    def remove_relation(self, va: int | Iterable[int], vb: int | Iterable[int]) -> None:
        va = IncidenceGraph.__type_check(va)
        vb = IncidenceGraph.__type_check(vb)
        ia, na = self.__get(va)
        ib, nb = self.__get(vb)
        if ia == -1:
            raise ValueError(f'Vertex {va} not in graph')
        if ib == -1:
            raise ValueError(f'Vertex {vb} not in graph')

        if na.d > nb.d:
            if ia in nb.upper and ib in na.lower:
                nb.upper.discard(ia)
                na.lower.discard(ib)
                na.unset_simplex()
            else:
                raise ValueError(f'No relation between {va} and {vb}')
        elif na.d < nb.d:
            self.remove_relation(vb, va)

    def put_incidence_relation(self, vertex_list: Iterable[int | Iterable[int]], data: Any = None,
                               data_gen: Optional[Callable[[List[Any]], Any]] = None):
        vertices = set()
        indices = set()
        for vs in vertex_list:
            vs = IncidenceGraph.__type_check(vs)
            i, _ = self.__get(vs)
            if i == -1:
                raise KeyError(f'No node with vertices {vs}')
            indices.add(i)
            for v in vs:
                vertices.add(v)
        vertices = tuple(sorted(vertices))

        index, node = self.__make_node(vertices, data)
        node.lower.update(indices)
        node.apply_data_gen(data_gen)

    def get(self, vertices: int | Iterable[int]) -> Any:
        vs = IncidenceGraph.__type_check(vertices)
        index, node = self.__get(vs)
        if index == -1:
            raise KeyError(f'No node with vertices {vertices}')

        return node.data

    def get_many(self, v_list: Iterable[int | Iterable[int]]) -> List[Any]:
        return [self.get(v) for v in v_list]

    def neighbors(self, vertices: int | Iterable[int], dist: int = 1) -> Iterable[Tuple[int, ...]]:
        if dist < 0:
            raise ValueError(f'Invalid distance {dist}')

        vs = IncidenceGraph.__type_check(vertices)
        i, node = self.__get(vs)

        if i == -1:
            raise KeyError(f'No node with vertices {vertices}')

        for i, _ in node.neighbor_relations(dist):
            yield self._dimensions[node.d][i].vertices

    def incidence_neighbors(self, vertices: int | Iterable[int], rel_dim: int = 0) -> Iterable[Tuple[int, ...]]:
        vertices = IncidenceGraph.__type_check(vertices)
        i, node = self.__get(vertices)

        if i == -1:
            raise KeyError(f'No node with vertices {vertices}')

        for i, _ in node.incidence_relations(rel_dim):
            yield self._dimensions[node.d + rel_dim][i].vertices

    def adjacency_list(self, rel: int = 0, incidence: bool = True,
                       vertex_list: Optional[Iterable[int | Iterable[int]]] = None) -> Iterable[Tuple[int, int]]:
        offsets = list(itertools.accumulate([0] + self.shape()))

        if vertex_list is None:
            nodes = self.__flatten()
        else:
            nodes = []
            for v in vertex_list:
                i, n = self.__get(IncidenceGraph.__type_check(v))
                if i == -1:
                    raise KeyError(f'No node with vertices {v}')
                nodes.append(n)

        for node in nodes:
            if incidence:
                iter = node.incidence_relations(rel)
            else:
                iter = node.neighbor_relations(rel)
            for j, _ in iter:
                yield offsets[node.d] + node.index, offsets[node.d + rel] + j

    def adjacency_matrix(self, rel: int = 0, incidence: bool = True) -> NDArray[np.float32]:
        return self.partial_matrix(rel, incidence)()

    def partial_matrix(self, rel: int, incidence: bool = True, partial_size: Optional[int] = None) -> NDArray[
        np.float32]:
        if incidence:
            return self.partial_matrices([], [rel], partial_size)
        else:
            return self.partial_matrices([rel], [], partial_size)

    def adjacency_matrices(self, neighbor_dists: Iterable[int], rel_dims: Iterable[int]) -> NDArray[np.float32]:
        return self.partial_matrices(neighbor_dists, rel_dims)()

    def partial_matrices(self, neighbor_dists: Iterable[int], rel_dims: Iterable[int],
                         partial_size: Optional[int] = None) -> Callable[[], NDArray[np.float32]]:
        flattened = self.__flatten()
        A, N = 0, len(self)

        if partial_size is None or partial_size > N:
            partial_size = N
        elif partial_size <= 0:
            raise ValueError('Partial matrix size must be positive')

        starts = iter(range(0, N, partial_size))
        reverse_rel_dims = dict()
        reverse_neighbors = dict()
        max_dim, min_dim = 0, 0
        max_dist = 0
        for i, d in enumerate(neighbor_dists):
            A += 1
            if d in reverse_neighbors:
                reverse_neighbors[d].append(i)
            else:
                reverse_neighbors[d] = [i]
                max_dist = max(max_dist, d)
        for i, d in enumerate(rel_dims):
            A += 1
            if d in reverse_rel_dims:
                reverse_rel_dims[d].append(i)
            else:
                reverse_rel_dims[d] = [i]
                max_dim = max(max_dim, d)
                min_dim = min(min_dim, d)

        def next_partial() -> NDArray[np.float32]:
            i = next(starts)
            P = min(partial_size, N - i)

            adj = np.zeros((A, P, N), dtype=np.float32)
            for n, node in enumerate(flattened[i:i + P]):
                for j, d in node.neighbor_relations(neighbor_dists):
                    for k in reverse_rel_dims[d]:
                        adj[k, n, j] = 1.0
                for j, d in node.incidence_relations(rel_dims):
                    for k in reverse_neighbors[d]:
                        adj[k, n, j] = 1.0

            return adj

        return next_partial

    def shape(self) -> List[int]:
        return [len(d) for d in self._dimensions]

    def generalize(self, dimension: Optional[int] = None,
                   data_gen: Optional[Callable[[List[Any]], Any]] = None) -> None:
        # data can be function that generates data based on existing data from lower nodes
        if dimension is None:
            prev_size = len(self._dimensions)
            for d in range(1, len(self._dimensions)):
                for node in self._dimensions[d]:
                    node.generalize(data_gen)
            while len(self._dimensions) != prev_size:
                prev_size = len(self._dimensions)
                for node in self._dimensions[-1]:
                    node.generalize(data_gen)

        elif dimension >= len(self._dimensions) or dimension < 0:
            raise IndexError(f'Invalid dimension {dimension}')
        else:
            for node in self._dimensions[dimension]:
                node.generalize(data_gen)

    def size(self, dim: Optional[int] = None) -> int:
        """
        Returns the number of nodes in a given dimension in the graph.

        Args: dim (int or None): The dimension from which to get the size. If None, returns the total number of nodes
                                 in the graph (though this is equivalent to using `len` on the graph).

        Returns (int): The number of nodes in the given dimension, or the total number of nodes in the graph if no
                       dimension is given.

        """
        if dim is None:
            return len(self)
        if dim < 0:
            raise IndexError(f'Invalid dimension {dim}')
        elif dim >= len(self._dimensions):
            return 0

        return len(self._dimensions[dim])

    def __new_node(self, vertices: Tuple[int, ...], data: Any, index: int) -> IncidenceGraph._IncidenceNode:
        # can overload
        return IncidenceGraph._IncidenceNode(vertices, data, index, self)

    @staticmethod
    def __type_check(vertices: int | Iterable[int]) -> Tuple[int, ...]:
        # converts an int or iterable of ints to a sorted tuple of ints
        # raises errors if vertices is not an int or iterable of ints

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

    def __get(self, vertices: Tuple[int, ...]) -> Tuple[int, Optional[_IncidenceNode]]:
        # retrieves index and node with given vertices (if existing)
        if vertices[-1] >= len(self._dimensions[0]):
            return -1, None

        if len(vertices) == 1:
            index = vertices[0]
            return index, self._dimensions[0][index]

        if vertices in self._facet_to_index:
            index = self._facet_to_index[vertices]
            d = len(vertices) - 1
            return index, self._dimensions[d][index]
        return -1, None

    def __make_node(self, vertices: Tuple[int, ...], data: Any) -> Tuple[int, _IncidenceNode]:
        # creates a node from a list of vertices and assigns it an index
        d = len(vertices) - 1

        while len(self._dimensions) <= d:
            self._dimensions.append([])

        while len(self._dimensions[0]) < vertices[-1]:
            v_id = len(self._dimensions[0])
            vertex = self.__new_node((len(self._dimensions[0]),), None, v_id)
            self._dimensions[0].append(vertex)

        index = len(self._dimensions[d])

        node = self.__new_node(vertices, data, index)
        self._dimensions[d].append(node)
        if d > 0:
            self._facet_to_index[vertices] = index
        return index, node

    def __swap_indices(self, d: int, i: int, j: int) -> None:
        # swaps positions/indices of two nodes from the same dimension
        node_i = self._dimensions[d][i]
        node_j = self._dimensions[d][j]
        if node_i is not None:
            node_i.index = j
            self._facet_to_index[node_i.vertices] = j
            for k in node_i.lower:
                self._dimensions[d - 1][k].upper.remove(i)
                self._dimensions[d - 1][k].upper.add(j)
            for k in node_i.upper:
                self._dimensions[d + 1][k].lower.remove(i)
                self._dimensions[d + 1][k].lower.add(j)
        if node_j is not None:
            node_j.index = i
            self._facet_to_index[node_j.vertices] = i
            for k in node_j.lower:
                self._dimensions[d - 1][k].upper.remove(j)
                self._dimensions[d - 1][k].upper.add(i)
            for k in node_j.upper:
                self._dimensions[d + 1][k].lower.remove(j)
                self._dimensions[d + 1][k].lower.add(i)

        self._dimensions[d][i] = node_j
        self._dimensions[d][j] = node_i

    def __remove_node(self, d: int, i: int, recursive: bool = False) -> None:
        # removes a node from the graph
        node = self._dimensions[d][i]
        self.__swap_indices(d, i, -1)
        self._dimensions[d].pop()
        node.remove_self()
        if d > 0:
            del self._facet_to_index[node.vertices]
        if recursive:
            for j in node.upper:
                self.__remove_node(d + 1, j, recursive)

    def __make_simplex(self, index: int, d: int, data: Any, gen: Optional[Callable[[List[Any]], Any]]) -> None:
        # turn existing node with dimension d and index into simplex
        node = self._dimensions[d][index]
        if d == 0:
            node.is_simplex = True
            return

        for sub_face in itertools.combinations(node.vertices, d):
            sub_index, sub_node = self.__get(sub_face)
            if sub_index == -1:
                sub_index, sub_node = self.__make_node(sub_face, data)
                self.__make_simplex(sub_index, d - 1, data, gen)

            node.lower.add(sub_index)
            sub_node.upper.add(index)

        node.apply_data_gen(gen)

        node.is_simplex = True

    def __flatten(self) -> List[IncidenceGraph._IncidenceNode]:
        # flattens the graph into a list of nodes
        nodes = []
        for d in range(len(self._dimensions)):
            for node in self._dimensions[d]:
                nodes.append(node)
        return nodes

    def __iter__(self) -> Iterator[IncidenceGraph._IncidenceNode]:
        return IncidenceGraph.IncidenceGraphIterator(self)

    def __len__(self) -> int:
        return len(self._facet_to_index) + len(self._dimensions[0])

    def __getitem__(self, item: Any) -> Any:
        return self.get(item)

    def __contains__(self, item: Any) -> bool:
        try:
            vertices = IncidenceGraph.__type_check(item)
            index, _ = self.__get(vertices)
            return index != -1
        except:
            return False
