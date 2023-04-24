from __future__ import annotations

import itertools
from collections.abc import Iterable, Iterator, Collection
from collections import deque
from typing import List, Tuple, Set, Dict, Optional, Any, Callable, Sequence  # TODO use built-in list, tuple etc.
import numpy as np
from numpy.typing import NDArray
import torch


# TODO: read/write graph files
# TODO: import from networkx
class IncidenceGraph(Collection):
    class _IncidenceNode:
        # Used to store the data for a node in an IncidenceGraph. This class should be hidden from the user.

        def __init__(self, vertices: Tuple[int, ...], data: Any, index: int, graph: IncidenceGraph):
            self.vertices: Tuple[int, ...] = vertices  # the zero-dimensional vertices that make up the node
            self.d: int = len(vertices) - 1
            self.upper: Set[int] = set()  # list of indices of upper dimension connections
            self.lower: Set[int] = set()  # list of indices of lower dimension neighbors
            self.index: int = index  # the index of the node in the IncidenceGraph
            self.graph: IncidenceGraph = graph
            self.data: Any = data
            self.is_simplex: bool = False  # if current node and ALL lower dim nodes are simplexes

            # a binary number representing inbound/outbound connections (for directed node) on lower dim
            # a 1 on ith position means the lower-dimensional node that does NOT include vertex i is connected
            # so if we have face (1, 2, 3) connecting (1, 3) and (2, 3), then the binary number is 110
            self.in_connections = 0
            self.out_connections = 0

        def get_total_index(self) -> int:
            # returns the absolute index of this node in the entire incidence graph

            offset = 0
            for i in range(self.d):
                offset += len(self.graph._dimensions[i])
            return offset + self.index

        def neighbor_relations(self, dist: int | Iterable[int] = 1) -> Iterable[Tuple[int, int]]:
            # returns indices (with distance) of all neighbors with given distance(s)
            # neighbors are nodes in the same dimension as the current one

            if isinstance(dist, int):
                dist = {dist}
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
            # returns all indices (with distance) of nodes that are rel_dim dimensions away from the current one
            # a positive rel_dim means a higher dimension, a negative one means a lower dimension

            if isinstance(rel_dim, int):
                rel_dim = {rel_dim}
            else:
                rel_dim = set(rel_dim)

            min_dim, max_dim = 0, 0
            for d in rel_dim:
                if d < 0:
                    min_dim = min(min_dim, d)
                else:
                    max_dim = max(max_dim, d)

            bfs = deque()
            bfs.append((self.index, 0))
            visited = set()
            while bfs:
                id, d = bfs.popleft()
                node = self.graph._dimensions[self.d + d][id]
                if (id, d) in visited:
                    continue
                if d in rel_dim:
                    yield id, d
                visited.add((id, d))

                if 0 <= d < max_dim:
                    for j in node.upper:
                        if (j, d + 1) not in visited:
                            bfs.append((j, d + 1))
                if 0 >= d > min_dim:
                    for j in node.lower:
                        if (j, d - 1) not in visited:
                            bfs.append((j, d - 1))

        # TODO generalize with some probability (so not all connections)

        def non_adjacency_set(self, rel_dim: int, include_self: bool = False) -> Set[int]:
            # returns all indices of nodes that are NOT connected to the current one in the incidence graph
            # if rel_dim is 0, then it returns all nodes that are not neighbors

            if self.d + rel_dim < 0 or self.d + rel_dim >= len(self.graph._dimensions):
                return set()

            adj = set(range(len(self.graph._dimensions[self.d + rel_dim])))

            if rel_dim == 0:
                adj.difference_update(self.neighbor_relations())
                if not include_self:
                    adj.discard(self.index)
            else:
                adj.difference_update(self.incidence_relations(rel_dim))

            return adj

        def generalize(self) -> None:
            # generalizes this node (see generalize function in IncidenceGraph)

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
                    self.graph.put_simplex(self.vertices + (vertex,))

        def remove_lower(self, index: int, dir_in: Optional[bool] = None) -> None:
            # removes a lower-dimensional node from this node
            # dir_in means remove inbound, False remove out, None remove both
            node = self.graph._dimensions[self.d - 1][index]
            if dir_in is None:
                self.lower.discard(index)

            missing = 0
            while missing < self.d and self.vertices[missing] == node.vertices[missing]:
                missing += 1
            missing = self.d - missing
            mask = (self.__max()) ^ (1 << missing)

            if dir_in is None or dir_in:
                self.in_connections &= mask
            if dir_in is None or not dir_in:
                self.out_connections &= mask

            if self.in_connections != self.__max() or self.out_connections != self.__max():
                self.unset_simplex()
            if self.in_connections == 0 and self.out_connections == 0:
                self.graph.remove_node(self.vertices)

        def add_lower(self, index: int, dir_in: Optional[bool] = None) -> None:
            # adds a lower-dimensional node to this node
            # dir_in means add inbound, False add out, None add both
            node = self.graph._dimensions[self.d - 1][index]

            missing = 0
            while missing < self.d and self.vertices[missing] == node.vertices[missing]:
                missing += 1
            missing = self.d - missing
            mask = 1 << missing

            if dir_in is None or dir_in:
                self.in_connections |= mask
            if dir_in is None or not dir_in:
                self.lower.add(index)
                self.out_connections |= mask

            self.check_simplex()

        def remove_self(self) -> None:
            # removes this node from the parent incidence graph
            self.unset_simplex()
            for i in self.upper:
                self.graph._dimensions[self.d + 1][i].remove_lower(self.index)
            for i in self.lower:
                self.graph._dimensions[self.d - 1][i].upper.discard(self.index)

        def unset_simplex(self) -> None:
            # marks this node and all upper-dimensional connections as not a simplex
            if not self.is_simplex:
                return
            self.is_simplex = False
            for i in self.upper:
                self.graph._dimensions[self.d + 1][i].unset_simplex()

        def check_simplex(self) -> None:
            # marks this node and all upper-dimensional connections as a simplex
            if self.is_simplex:
                return
            if self.in_connections != self.__max() or self.out_connections != self.__max():
                return
            for i in self.lower:
                if not self.graph._dimensions[self.d - 1][i].is_simplex:
                    return

            self.is_simplex = True
            for i in self.upper:
                self.graph._dimensions[self.d + 1][i].check_simplex()

        def __max(self):
            return (1 << (self.d + 1)) - 1

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

    def put_simplex(self, vertices: int | Iterable[int], data: Any = None) -> None:
        """Adds a simplex (node) to the graph with the given vertices and data, or updates an existing one with the
        provided data.

        A k-dimensional simplex, represented by a tuple of `k` 0-dimensional vertices, is a connection between
        all (k-1)-dimensional simplexes that contain the vertices. For example, the 2-simplex (0, 1, 2) is a face
        that connects the 1-simplexes (a.k.a. edges) (0, 1) and (0, 2) and (1, 2).

        Simplex vertices are order-invariant. For example, the face (0, 1, 2) is the same as (2, 1, 0).

        When adding a simplex, if lower-dimensional nodes do not already exist, they are created (recursively, if
        necessary). For example, when adding (0, 1, 2) to an empty graph, the vertices (0), (1), and (2) are added
        to the graph, and then the edges (0, 1) and (0, 2) and (1, 2).

        Args:
            vertices: The node to add as an integer (for 0-dimensional nodes) or an iterable of integers (for higher-
                dimensional nodes).

                For example, 2 or (2,) refers to the vertex (index) 2 from the graph. (2, 3) refers to an edge between
                vertices 2 and 3.
            data: The data to store in the simplex. Note that this data will be overwritten by the data generator if
                provided.
        """
        vertices = IncidenceGraph.__type_check(vertices)
        d = len(vertices) - 1
        index, node = self.__get(vertices)
        if index == -1:
            index, _ = self.__make_node(vertices, data)
            self.__make_simplex(index, d)
        else:
            node = self._dimensions[d][index]
            node.data = data
            if not node.is_simplex:
                self.__make_simplex(index, d)

    def remove_node(self, vertices: int | Iterable[int], as_simplex: bool = False) -> None:
        """Removes a node from the graph.

        Args:
            vertices: The given node as an integer (for 0-dimensional nodes) or an iterable of integers (for higher-
                dimensional nodes).

                For example, 2 or (2,) refers to the vertex (index) 2 from the graph. (2, 3) refers to an edge between
                vertices 2 and 3.
            as_simplex: If True, removes the node and all its upper-dimensional neighbors recursively. If False, removes
                only the provided node; note that any higher-dimensional neighbors that include this node will no longer
                be simplexes.

        Raises:
            ValueError: If the node does not exist.
        """
        vertices = IncidenceGraph.__type_check(vertices)
        index, node = self.__get(vertices)
        if index == -1:
            raise ValueError(f'Node {vertices} not found in graph.')

        if as_simplex:
            self.__remove_node(index, node.d, True)
        else:
            self.__remove_node(index, node.d, False)
            node.unset_simplex()

    # TODO documentation !!!
    def remove_relation(self, va: int | Iterable[int], vb: int | Iterable[int], dir_in: Optional[bool] = None) -> None:
        """Removes an incidence relation between two nodes. (This does not remove the nodes themselves.) Note that the
        node in the upper-dimension will no longer be a simplex.

        Args:
            va: The first node as an integer (for 0-dimensional nodes) or an iterable of integers (for higher-
                dimensional nodes).

                For example, 2 or (2,) refers to the vertex (index) 2 from the graph. (2, 3) refers to an edge between
                vertices 2 and 3.
            vb: The second node as an integer (for 0-dimensional nodes) or an iterable of integers (for higher-
                dimensional nodes).

        Raises:
            ValueError: If either node does not exist or if the nodes are not connected by an incidence relation.
        """
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
                na.remove_lower(ib, dir_in)
            else:
                raise ValueError(f'No relation between {va} and {vb}')
        elif na.d < nb.d:
            self.remove_relation(vb, va)

    def put_incidence_relation(self, src_list: Iterable[int | Iterable[int]], dest_list: Optional[Iterable[int | Iterable[int]]] = None,
                               data: Any = None) -> None:
        """Creates a higher-dimensional node as a connection of a list of (existing) lower-dimensional nodes with the
        same dimension. In other words, creates a non-simplex node.

        For example, the node (0, 1, 2) can be created by connecting the edges (0, 1) and (1, 2). Note that this is not
        a simplex because it is missing the connection to (0, 2).

        The data of the added node will be imputed with the data generator (first priority, if provided) or the given
        data (second priority, if provided). If neither is provided, the data will be None.

        Args:
            vertex_list: A list of nodes, each as an integer (for 0-dimensional nodes) or an iterable of integers (for
                higher-dimensional nodes).

                For example, 2 or (2,) refers to the vertex (index) 2 from the graph. (2, 3) refers to an edge between
                vertices 2 and 3.
            data: The data to store in the simplex. Note that this data will be overwritten by the data generator if
                provided.

        Raises:
            ValueError: If the nodes are not of the same dimension.
            KeyError: If any node does not exist in the graph.
        """
        vertices = set()
        src_indices = set()

        d = -1
        for vs in src_list:
            vs = IncidenceGraph.__type_check(vs)
            if d == -1:
                d = len(vs) - 1
            elif d != len(vs) - 1:
                raise ValueError(f'All nodes must be of the same dimension.')
            i, _ = self.__get(vs)
            if i == -1:
                raise KeyError(f'No node with vertices {vs}')
            src_indices.add(i)
            for v in vs:
                vertices.add(v)

        if dest_list is None:
            dest_indices = src_indices
        else:
            dest_indices = set()
            for vs in dest_list:
                vs = IncidenceGraph.__type_check(vs)
                if d == -1:
                    d = len(vs) - 1
                elif d != len(vs) - 1:
                    raise ValueError(f'All nodes must be of the same dimension.')
                i, _ = self.__get(vs)
                if i == -1:
                    raise KeyError(f'No node with vertices {vs}')
                dest_indices.add(i)
                for v in vs:
                    vertices.add(v)
        vertices = tuple(sorted(vertices))
        if len(vertices) - 1 != d + 1:
            raise ValueError(f'List of nodes of dimension {d} forms a node of dimension {len(vertices) - 1} instead of {d + 1}')
        index, node = self.__get(vertices)
        if index == -1:
            index, node = self.__make_node(vertices, data)
        else:
            node.data = data

        for i in src_indices:
            node.add_lower(i, dir_in=True)
            self._dimensions[d][i].upper.add(index)
        for i in dest_indices:
            node.add_lower(i, dir_in=False)

    def generalize_data(self, vertices: int | Iterable[int], neighbor_dists: Iterable[int], rel_dims: Iterable[int],
                        func: Optional[Callable[[List[Any]], Any]]) -> Any:
        """Generates data for a node based on the data of its neighbors.
        """
        vs = IncidenceGraph.__type_check(vertices)
        index, node = self.__get(vs)
        if index == -1:
            raise KeyError(f'No node with vertices {vertices}')
        data = []
        for id, _ in node.neighbor_relations(neighbor_dists):
            data.append(self._dimensions[node.d][id].data)
        for id, d in node.incidence_relations(rel_dims):
            data.append(self._dimensions[node.d + d][id].data)
        new_data = func(data)
        node.data = new_data
        return new_data

    def propagate_data(self, vertices: int | Iterable[int], data: Any, neighbor_dists: Iterable[int], rel_dims: Iterable[int]
                       , update: Optional[Callable[[Any], Any]]=None) -> None:
        """Propagates data from a node to its neighbors.
        by default replaces data
        """
        vs = IncidenceGraph.__type_check(vertices)
        index, node = self.__get(vs)
        if index == -1:
            raise KeyError(f'No node with vertices {vertices}')
        if update is None:
            update = lambda x, y: y
        for id, _ in node.neighbor_relations(neighbor_dists):
            n = self._dimensions[node.d][id]
            n.data = data if n.data is None else update(n.data, data)
        for id, d in node.incidence_relations(rel_dims):
            n = self._dimensions[node.d + d][id]
            n.data = data if n.data is None else update(n.data, data)

    def node_list(self, dims: Optional[int | Iterable[int]] = None, simplex_only: bool = False) -> List[List[int]]:
        """Returns a list of all nodes in the graph.

        Args:
            dims: The dimensions of the nodes to return. If None, returns all nodes.
            simplex_only: If True, only returns nodes that are simplexes.

        Returns:
            A list of all nodes in the graph.
        """
        if dims is None:
            dims = range(1, len(self._dimensions))
        elif isinstance(dims, int):
            dims = (dims,)
        return [list(node.vertices) for d in dims for node in self._dimensions[d] if not simplex_only or simplex_only and node.is_simplex]

    def get(self, vertices: int | Iterable[int]) -> Any:
        """Returns the data associated with the given node.

        Args:
            vertices: The given node as an integer (for 0-dimensional nodes) or an iterable of integers (for higher-
                dimensional nodes).

                For example, 2 or (2,) refers to the vertex (index) 2 from the graph. (2, 3) refers to an edge between
                vertices 2 and 3.

        Returns:
            The data associated with the given node.

        Raises:
            KeyError: If the given node is not in the graph.
        """
        vs = IncidenceGraph.__type_check(vertices)
        index, node = self.__get(vs)
        if index == -1:
            raise KeyError(f'No node with vertices {vertices}')

        return node.data

    def get_many(self, v_list: Iterable[int | Iterable[int]]) -> Iterable[Any]:
        """Returns an iterable of the data associated with the given nodes.

        Args:
            v_list: An iterable of nodes. Each node is an integer (for 0-dimensional nodes) or an iterable of integers
                (for higher-dimensional nodes).

                For example, 2 or (2,) refers to the vertex (index) 2 from the graph. (2, 3) refers to an edge between
                vertices 2 and 3.

        Returns: An iterable of the data associated with the given nodes, in the order that the nodes are provided.

        Raises:
            KeyError: If any of the given nodes are not in the graph.

        """
        for v in v_list:
            yield self.get(v)

    def neighbors(self, vertices: int | Iterable[int], dist: int = 1) -> Iterable[Tuple[int, ...]]:
        """Returns the neighbors of the given node at the given distance.

        Args:
            vertices: The given node as an integer (for 0-dimensional nodes) or an iterable of integers (for higher
                dimensional nodes).

                For example, 2 or (2,) refers to the vertex (index) 2 from the graph. (2, 3) refers to an edge between
                vertices 2 and 3.
            dist: The distance from the given node to the neighbors.

        Returns:
            A list of the neighbors of the given node at the given distance. Each node is a tuple of integers
            representing the vertices of that node.

        Raises:
            ValueError: If the distance is negative.
            KeyError: If the given node is not in the graph.

        """
        if dist < 0:
            raise ValueError(f'Invalid distance {dist}')

        vs = IncidenceGraph.__type_check(vertices)
        i, node = self.__get(vs)

        if i == -1:
            raise KeyError(f'No node with vertices {vertices}')

        for i, _ in node.neighbor_relations(dist):
            yield self._dimensions[node.d][i].vertices

    def incidence_neighbors(self, vertices: int | Iterable[int], rel_dim: int = 1) -> Iterable[Tuple[int, ...]]:
        """Returns all nodes that are incident to the given node by a relation of the given dimension.

        Args:
            vertices: The given node as an integer (for 0-dimensional nodes) or an iterable of integers (for higher
                dimensional nodes).

                For example, 2 or (2,) refers to the vertex (index) 2 from the graph. (2, 3) refers to an edge between
                vertices 2 and 3.
            rel_dim: The number of dimensions away in the relation, where a positive number indicates a higher
                dimensional relation and a negative number indicates a lower dimensional relation.

        Returns: An iterable of tuples of integers, where each tuple represents a node (as its vertices) in the graph.

        Raises:
            KeyError: If the given node is not in the graph.
        """

        vertices = IncidenceGraph.__type_check(vertices)
        i, node = self.__get(vertices)

        if i == -1:
            raise KeyError(f'No node with vertices {vertices}')

        for i, _ in node.incidence_relations(rel_dim):
            yield self._dimensions[node.d + rel_dim][i].vertices

    def degree_list(self, neighbor_dists: Iterable[int], rel_dims: Iterable[int],
                       node_list: Optional[Iterable[int | Iterable[int]]] = None,
                    pow: float = 1.0, add: float = 0.0) -> List[float]:
        degrees = []
        for node in self.__get_many(node_list):
            degree = add + sum(1 for _ in node.neighbor_relations(neighbor_dists)) + \
                        sum(1 for _ in node.incidence_relations(rel_dims))

            degrees.append(degree ** pow if degree != 0.0 else 0.0)
        return degrees

    def degree_lists(self, neighbor_dists: Iterable[int], rel_dims: Iterable[int],
                       node_list: Optional[Iterable[int | Iterable[int]]] = None,
                    pow: float = 1.0, add: float = 0.0) -> Tuple[Dict[int, List[float]], Dict[int, List[float]]]:
        neighbor_degrees = {}
        incidence_degrees = {}
        for n in neighbor_dists:
            neighbor_degrees[n] = []
        for r in rel_dims:
            incidence_degrees[r] = []
        for node in self.__get_many(node_list):
            for key in neighbor_degrees.keys():
                neighbor_degrees[key].append(add)
            for key in incidence_degrees.keys():
                incidence_degrees[key].append(add)

            for _, d in node.neighbor_relations(neighbor_dists):
                neighbor_degrees[d][-1] += 1
            for _, d in node.incidence_relations(rel_dims):
                incidence_degrees[d][-1] += 1

            for key in neighbor_degrees.keys():
                if neighbor_degrees[key][-1] != 0.0:
                    neighbor_degrees[key][-1] **= pow
            for key in incidence_degrees.keys():
                if incidence_degrees[key][-1] != 0.0:
                    incidence_degrees[key][-1] **= pow

        return neighbor_degrees, incidence_degrees

    def adjacency_list(self, rel: int = 1, incidence: bool = False,
                       node_list: Optional[Iterable[int | Iterable[int]]] = None) -> Iterable[Tuple[int, int]]:
        """Returns an adjacency list from a vertex list or the whole graph for the given relation distance and type.

        Args:
            rel: The distance or number of dimensions away in the relation.
            incidence: Whether to encode incidence (True) or neighbor (False) relations.
            node_list: An optional list of nodes, represented as integers (for 0-dimensional vertices) or iterables
                of integers (for higher dimensional nodes) to search for the adjacency list. If None, all nodes in this
                graph are searched.

                For example, 2 or (2,) refers to the vertex (index) 2 from the graph. (2, 3) refers to an edge between
                vertices 2 and 3.

        Returns: An adjacency list. The adjacency list is a list of tuples (I, J) where I and J each are tuples that
        represent the vertices that make up the node.

            For example, ((1, 2, 3), (2, 4, 5)) represents a relationship between two nodes, one with vertices
            (1, 2, 3) and the other with vertices (2, 4, 5).

        Raises:
            KeyError: If a vertex in `vertex_list` is not in the graph.
        """

        offsets = self.__offsets()

        for node in self.__get_many(node_list):
            if incidence:
                iter = node.incidence_relations(rel)
            else:
                iter = node.neighbor_relations(rel)
            for j, _ in iter:
                yield offsets[node.d] + node.index, offsets[node.d + rel] + j

    def adjacency_matrix(self, neighbor_dists: Iterable[int], rel_dims: Iterable[int], dims: Optional[Iterable[int]] = None,
                         src_weights: Optional[Sequence[Any]] = None,
                         dest_weights: Optional[Sequence[Any]] = None) -> NDArray[np.float32]:
        """
        Returns an adjacency matrix for the given relation distance and type.

        Args:
            src_weights: If nodes i and j are adjacent and src_weights is not none, multiplies src_weights[i] to the
                adjacency matrix at position i, j.
            dest_weights: If nodes i and j are adjacent and dest_weights is not none, multiplies dest_weights[j] to the
                adjacency matrix at position i, j.

        Returns: A (1, n, n) adjacency matrix, where `n` is the number of nodes. To remove the first dimension, use
            numpy's squeeze() function.
        """
        return self.partial_matrix(neighbor_dists, rel_dims, dims=dims, src_weights=src_weights, dest_weights=dest_weights)()

    def partial_matrix(self, neighbor_dists: Iterable[int], rel_dims: Iterable[int],
                       dims: Optional[Iterable[int]] = None,
                       partial_size: Optional[int] = None, src_weights: Optional[Sequence[Any]] = None,
                         dest_weights: Optional[Sequence[Any]] = None) -> NDArray[
        np.float32]:
        """
        Returns a function that generates partial adjacency matrices for the given relation distance and type.

        Args:
            partial_size: The size `p` of the (a, p, n) matrices to return. If `None`, the full matrices are returned.
            src_weights: If nodes i and j are adjacent and src_weights is not none, multiplies src_weights[i] to the
                adjacency matrix at position i, j.
            dest_weights: If nodes i and j are adjacent and dest_weights is not none, multiplies dest_weights[j] to the
                adjacency matrix at position i, j.

        Returns:
            A function that returns a (1, p, n) partial matrix when invoked, where `p` is the partial size and `n` is
            the number of nodes in the graph. To remove the first dimension, use numpy's squeeze() function.

        Raises:
            ValueError: If `partial_size` is not `None` and not a positive integer.
            StopIteration: Once all partial matrices have been returned.
        """
        gen = self.partial_matrices(neighbor_dists, rel_dims, dims=dims, partial_size=partial_size, src_weights=src_weights, dest_weights=dest_weights)
        def partial_matrix():
            adj = gen()
            return np.expand_dims(np.sum(adj, axis=0), axis=0)
        return partial_matrix

    def adjacency_matrices(self, neighbor_dists: Iterable[int], rel_dims: Iterable[int],
                           dims: Optional[Iterable[int]] = None,
                           src_weights: Optional[Sequence[Any]] = None,
                         dest_weights: Optional[Sequence[Any]] = None) -> NDArray[np.float32]:
        """Returns the full adjacency matrices for the given neighbor distances and incidence dimensions.

        Args:
            neighbor_dists: Distances of neighbor relations to encode as adjacency matrices.

                For example, if given [0, 1, 2], the first adjacency matrix will be the identity matrix, the second
                one will have all neighbor relations, and the third one will have all neighbor relations of distance 2.
            rel_dims: Relative dimensions of incidence relations to encode as adjacency matrices.

                For example, if given [-1, 0, 1, 2], the first adjacency matrix will have lower incidence relations,
                the second one will be the identity matrix, the third one will have upper incidence relations, and the
                fourth one will have upper incidence relations that are 2 dimensions away.
            src_weights: If nodes i and j are adjacent and src_weights is not none, multiplies src_weights[i] to the
                adjacency matrix at position i, j.
            dest_weights: If nodes i and j are adjacent and dest_weights is not none, multiplies dest_weights[j] to the
                adjacency matrix at position i, j.

        Returns: An (a, n, n) array, where `a` is the number of adjacency matrices and `n` is the number of nodes in
            the graph.
        """
        return self.partial_matrices(neighbor_dists, rel_dims, dims=dims, src_weights=src_weights, dest_weights=dest_weights)()

    # TODO invalidate generator if graph changes
    def partial_matrices(self, neighbor_dists: Iterable[int], rel_dims: Iterable[int],
                         dims: Optional[Iterable[int]] = None,
                         partial_size: Optional[int] = None,
                         src_weights: Optional[Sequence[Any]] = None,
                         dest_weights: Optional[Sequence[Any]] = None) -> Callable[[], NDArray[np.float32]]:
        """Returns a function that returns partial adjacency matrices of this graph, like an iterator,
        for the given neighbor distances and incidence dimensions.

        Args:
            neighbor_dists: Distances of neighbor relations to encode as adjacency matrices.

                For example, if given [0, 1, 2], the first adjacency matrix will be the identity matrix, the second
                one will have all neighbor relations, and the third one will have all neighbor relations of distance 2.
            rel_dims: Relative dimensions of incidence relations to encode as adjacency matrices.

                For example, if given [-1, 0, 1, 2], the first adjacency matrix will have lower incidence relations,
                the second one will be the identity matrix, the third one will have upper incidence relations, and the
                fourth one will have upper incidence relations that are 2 dimensions away.
            partial_size: The size `p` of the (a, p, n) matrices to return. If `None`, the full matrices are returned.
            src_weights: If nodes i and j are adjacent and src_weights is not none, multiplies src_weights[i] to the
                adjacency matrix at position i, j.
            dest_weights: If nodes i and j are adjacent and dest_weights is not none, multiplies dest_weights[j] to the
                adjacency matrix at position i, j.

        Returns:
            A function that returns the adjacency matrices as an (a, p, n) array, where `a` is the number of adjacency
            matrices, `p` is the partial size, and `n` is the number of nodes in the graph. This function can be used
            like an iterator. When all matrices have been returned, the returned function raises `StopIteration`.

        Raises:
            ValueError: If `partial_size` is not `None` and is not a positive integer.
        """
        if dims is None:
            dims = set(range(len(self._dimensions)))
        else:
            dims = set(dims)

        flattened = self.__flatten(dims)
        A, N = 0, len(flattened)

        if partial_size is None or partial_size > N:
            partial_size = N
        elif partial_size <= 0:
            raise ValueError('Partial matrix size must be positive')

        starts = iter(range(0, N, partial_size))
        offsets = self.__offsets(dims)

        # maps distances to lists of indices of adjacency matrices
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
        num_neighbors = A
        for i, d in enumerate(rel_dims):
            A += 1
            if d in reverse_rel_dims:
                reverse_rel_dims[d].append(i)
            else:
                reverse_rel_dims[d] = [i]
                max_dim = max(max_dim, d)
                min_dim = min(min_dim, d)

        def next_partial() -> NDArray[np.float32]:
            # returns the next partial adjacency matrix when called

            i = next(starts)
            P = min(partial_size, N - i)

            adj = np.zeros((A, P, N), dtype=np.float32)
            for n, node in enumerate(flattened[i:i + P]):
                for j, d in node.neighbor_relations(neighbor_dists):
                    j += offsets[node.d]
                    for k in reverse_neighbors[d]:
                        adj[k, j, n] = 1.0
                        if src_weights is not None:
                            adj[k, j, n] *= src_weights[i + n]
                        if dest_weights is not None:
                            adj[k, j, n] *= dest_weights[j]
                for j, d in node.incidence_relations(filter(lambda x: node.d+x in dims, rel_dims)):
                    j += offsets[node.d + d]
                    for k in reverse_rel_dims[d]:
                        adj[k + num_neighbors, j, n] = 1.0
                        if src_weights is not None:
                            adj[k + num_neighbors, j, n] *= src_weights[i + n]
                        if dest_weights is not None:
                            adj[k + num_neighbors, j, n] *= dest_weights[j]

            return adj

        return next_partial

    def sparse_matrix(self, neighbor_dists: Iterable[int], rel_dims: Iterable[int],
                        dims: Optional[Iterable[int]] = None,
                        src_weights: Optional[Sequence[Any]] = None,
                        dest_weights: Optional[Sequence[Any]] = None) -> torch.Tensor:
        adjs = self.sparse_matrices(neighbor_dists, rel_dims, dims, src_weights, dest_weights)
        sum = adjs[0]
        for i in range(1, len(adjs)):
            sum += adjs[i]
        return sum

    def sparse_matrices(self, neighbor_dists: Iterable[int], rel_dims: Iterable[int],
                         dims: Optional[Iterable[int]] = None,
                         src_weights: Optional[Sequence[Any]] = None,
                         dest_weights: Optional[Sequence[Any]] = None) -> List[torch.Tensor]:
        if dims is None:
            dims = set(range(len(self._dimensions)))
        else:
            dims = set(dims)

        A, N = 0, sum(len(self._dimensions[d]) for d in dims)
        offsets = self.__offsets(dims)

        # maps distances to lists of indices of adjacency matrices
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
        num_neighbors = A
        for i, d in enumerate(rel_dims):
            A += 1
            if d in reverse_rel_dims:
                reverse_rel_dims[d].append(i)
            else:
                reverse_rel_dims[d] = [i]
                max_dim = max(max_dim, d)
                min_dim = min(min_dim, d)

        coords = []
        values = []
        for _ in range(A):
            coords.append([[], []])
            values.append([])
        for n, node in enumerate(self.__iter(dims)):
            for j, d in node.neighbor_relations(neighbor_dists):
                j += offsets[node.d]
                for k in reverse_neighbors[d]:
                    val = 1.0
                    if src_weights is not None:
                        val *= src_weights[n]
                    if dest_weights is not None:
                        val *= dest_weights[j]
                    coords[k][0].append(j)
                    coords[k][1].append(n)
                    values[k].append(val)
            for j, d in node.incidence_relations(filter(lambda x: node.d+x in dims, rel_dims)):
                j += offsets[node.d + d]
                for k in reverse_rel_dims[d]:
                    val = 1.0
                    if src_weights is not None:
                        val *= src_weights[n]
                    if dest_weights is not None:
                        val *= dest_weights[j]
                    coords[k + num_neighbors][0].append(j)
                    coords[k + num_neighbors][1].append(n)
                    values[k + num_neighbors].append(val)
        output = []
        for i in range(A):
            output.append(torch.sparse_coo_tensor(torch.as_tensor(coords[i]), values[i], (N, N)).coalesce())
        return output

    @staticmethod
    def concat_partials(*partial_generators: Callable[[], NDArray[np.float32]])-> Callable[[], NDArray[np.float32]]:
        """Concatenates partial adjacency matrices from multiple generators into a single generator.

        Args:
            partial_generators: The partial adjacency matrix generators to concatenate.

        Returns:
            A generator that returns the adjacency matrices as an (a, p, n) array, where `a` is the number of adjacency
            matrices, `p` is the partial size, and `n` is the number of nodes in the graph. This function can be used
            like an iterator. When all matrices have been returned, the returned function raises `StopIteration`.
        """
        def next_partial():
            adj = []
            for g in partial_generators:
                adj.append(g())
            return np.concatenate(adj, axis=0)
        return next_partial

    def shape(self) -> List[int]:
        """
        Returns the shape of the incidence graph as a list of integers.

        Returns:
            The number of nodes in each dimension

        """
        return [len(d) for d in self._dimensions]

    def generalize(self, dim: Optional[int | Iterable[int]] = None) -> None:
        """"Completes" higher-dimensional simplexes if their lower-dimensional connections already exist.

        For example, in a graph with vertices 0, 1, 2, and 3, if there exists edges (0, 1), (1, 2), (0, 2), and (1, 3),
        then a face (0, 1, 2) will be created. The face (0, 1, 3), for example, will not be created because the edge
        (0, 3) does not exist.

        This method is useful for generating higher-dimensional graphs from lower-dimensional ones.

        Args:
            dim: The dimension or dimensions to generalize. If None, all dimensions will be generalized, including
                newly-created dimensions from this very process.

        Raises:
            IndexError: If `dim` is not `None` and is not a valid dimension.
            TypeError: If `dim` is not `None` and is not an integer or iterable of integers.
        """
        if dim is None:
            d = 1
            while d < len(self._dimensions):
                self.generalize(d)
                d += 1
        elif isinstance(dim, int):
            if dim >= len(self._dimensions) or dim < 0:
                raise IndexError(f'Invalid dimension {dim}')
            else:
                for node in self._dimensions[dim]:
                    node.generalize()
        elif isinstance(dim, Iterable):
            for d in dim:
                self.generalize(d)
        else:
            raise TypeError(f'Invalid type {type(dim)} for `dim`')

    def size(self, dim: Optional[int] = None) -> int:
        """Returns the number of nodes in a given dimension in the graph.

        Args:
            dim: The dimension from which to get the size. If None, returns the total number of nodes in the graph
                (though this is equivalent to using `len` on the graph).

        Returns:
            The number of nodes in the given dimension, or the total number of nodes in the graph if no dimension is
            given.

        Raises:
            IndexError: If the given dimension is invalid.
            TypeError: If the given dimension is not an integer.
        """
        if dim is None:
            return len(self)
        elif not isinstance(dim, int):
            raise TypeError(f'Invalid type {type(dim)} for `dim`')
        if dim < 0:
            raise IndexError(f'Invalid dimension {dim}')
        elif dim >= len(self._dimensions):
            return 0

        return len(self._dimensions[dim])

    def __new_node(self, vertices: Tuple[int, ...], data: Any, index: int) -> IncidenceGraph._IncidenceNode:
        # separate method for overloading
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

    def __get_many(self, node_list: Optional[Iterable[int | Iterable[int]]] = None) -> Iterator[_IncidenceNode]:
        # retrieves nodes with list of vertices, with error checking
        # if no list, returns all nodes

        if node_list is None:
            for d in range(len(self._dimensions)):
                for node in self._dimensions[d]:
                    yield node
        else:
            for v in node_list:
                i, n = self.__get(IncidenceGraph.__type_check(v))
                if i == -1:
                    raise KeyError(f'No node with vertices {v}')
                yield n

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

    def __make_simplex(self, index: int, d: int) -> None:
        # turn existing node with dimension d and index into simplex
        # TODO reimplement bottom-up to avoid recursion
        node = self._dimensions[d][index]
        if node.is_simplex and node.in_connections == node.out_connections == (1 << (d + 1)) - 1:
            return
        if d == 0:
            node.is_simplex = True
            node.in_connections = node.out_connections = 1
            return

        for sub_face in itertools.combinations(node.vertices, d):
            sub_index, sub_node = self.__get(sub_face)
            if sub_index == -1:
                sub_index, sub_node = self.__make_node(sub_face, None)
            self.__make_simplex(sub_index, d - 1)

            node.lower.add(sub_index)
            sub_node.upper.add(index)

        node.is_simplex = True
        node.in_connections = node.out_connections = (1 << (d + 1)) - 1

    def __flatten(self, dims: Optional[Iterable[int]] = None) -> List[_IncidenceNode]:
        # flattens the graph into a list of nodes
        nodes = []
        if dims is None:
            dims = range(len(self._dimensions))
        for d in dims:
            if d >= len(self._dimensions) or d < 0:
                continue
            for node in self._dimensions[d]:
                nodes.append(node)
        return nodes

    def __iter(self, dims: Optional[Iterable[int]] = None) -> Iterable[_IncidenceNode]:
        # iterates over nodes in the graph
        if dims is None:
            dims = range(len(self._dimensions))
        for d in dims:
            if d >= len(self._dimensions) or d < 0:
                continue
            for node in self._dimensions[d]:
                yield node

    def __offsets(self, dims: Optional[Iterable[int]] = None) -> List[int]:
        # returns a list of offsets for each dimension
        offsets = [0]
        for d in range(len(self._dimensions)):
            if dims is None or d in dims:
                offsets.append(offsets[-1] + len(self._dimensions[d]))
            else:
                offsets.append(offsets[-1])
        return offsets

    def __iter__(self) -> Iterator[Tuple[Tuple[int, ...], Any]]:
        return IncidenceGraph.IncidenceGraphIterator(self)

    def __len__(self) -> int:
        return len(self._facet_to_index) + len(self._dimensions[0])

    def __getitem__(self, item: Any) -> Any:
        return self.get(item)

    def __setitem__(self, key: Any, value: Any) -> None:
        vertices = IncidenceGraph.__type_check(key)
        index, node = self.__get(vertices)
        if index == -1:
            self.put_simplex(vertices, data=value)
        else:
            node.data = value

    def __contains__(self, item: Any) -> bool:
        try:
            vertices = IncidenceGraph.__type_check(item)
            index, _ = self.__get(vertices)
            return index != -1
        except:
            return False
