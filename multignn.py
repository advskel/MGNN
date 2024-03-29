import math
from typing import Optional, Callable, Any, Tuple, List

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

_tensor_func = Optional[Callable[[Tensor | List[Tensor]], Tensor | List[Tensor]]]
_two_tensor_func = Optional[Callable[[Tensor, Tensor], Tensor]]

class PartialForwardNN(nn.Module):
    """
    A neural network container that supports partial forwarding and hasty partial forwarding.
    """
    def __init__(self, *args):
        super().__init__()
        self.layers = nn.ModuleList()
        for l in args:
            self.layers.append(l)
        self._curr_layer = 0

    def forward(self, x: Tensor, *args: Any) -> Optional[Tensor]:
        done = False
        curr_layer = self._curr_layer
        while self._curr_layer < len(self.layers) and curr_layer < len(self.layers):
            x, done = self.layers[self._curr_layer](x, *args)
            if x is None:
                return None
            curr_layer += 1
            if done:
                self._curr_layer += 1

        if done:
            self._curr_layer = 0
        return x


class MultiGraphLayer(nn.Module):
    """
    A Multi-graph layer is a message-passing layer for higher-dimensional graphs that consists of four steps:
    1. (Optional) Transform some initial input into a graph embedding
    2. Generate messages from each adjacency matrix (which supports partial adjacency matrices)
    3. Combine the messages into a single message
    4. Update the original graph embedding with the message to create a new graph embedding

    The user is expected to provide all functions listed above, though this module provides defaults (that do not
    involve learning).
    """
    # A: (a, n, n) tensor
    # a: number of adjacency matrices
    # n x n: adjacency matrix
    # H: (n, d)
    # W: (a, d, d) tensor
    def __init__(self, transform_func: _tensor_func = None,
                 adj_transform_func: _tensor_func = None,
                 vertex_agg_func: Optional[Callable[[Tensor, Tensor | List[Tensor]], Tensor]] = None,
                 graph_agg_func: _tensor_func = None, update_func: _two_tensor_func = None,
                 output_func: _tensor_func = None,
                 num_vertices: Optional[int] = None):
        """
        Args:
            transform_func: Inputs an (n, *) original graph and outputs an (n, d) transformed graph. If None, then this
                layer is skipped.
            vertex_agg_func: Inputs an (n, d) graph and an (a, n, n) adjacency tensor and outputs a (a, n, d) tensor of
                messages, one message per adjacency matrix. This function should also support partial updates with
                 the shapes (n, d) (a, p, n) -> (a, p, d). If None, then the adjacency tensor is multiplied (matrix
                 multiplication) to the graph tensor.
            graph_agg_func: Inputs an (a, n, d) tensor and outputs an (n, d) tensor, in which the goal is to aggregate
                all `a` messages then activate. If None, then the `a` messages are summed and activated with ReLU.
            update_func: Inputs two (n, d) tensors and outputs an (n, d) tensor, in which the goal is to aggregate both
                tensors then activate. Note that the updates are always performed with full inputs, not partial ones.
                If None, then the two tensors are averaged and activated with ReLU.
            num_vertices: The size `n` to expect from all inputs. If None, then a standard input dimension will not be
                enforced.
        """
        super().__init__()
        if transform_func is None:
            transform_func = lambda x: x
        self.transform = transform_func

        if adj_transform_func is None:
            adj_transform_func = lambda x: x
        self.adj_transform = adj_transform_func

        if vertex_agg_func is None:
            vertex_agg_func = lambda g, a: a @ g
        self.v_agg = vertex_agg_func  # aggregate messages per vertex

        if graph_agg_func is None:
            graph_agg_func = lambda a: F.relu(torch.sum(a, dim=0))
        self.g_agg = graph_agg_func  # aggregate messages per graph/adj matrix

        if update_func is None:
            update_func = lambda g, m: F.relu((g + m) / 2)
        self.update = update_func

        if output_func is None:
            output_func = lambda x: x
        self.output = output_func

        if num_vertices is not None and (num_vertices <= 0 or not isinstance(num_vertices, int)):
            raise ValueError(f'Number of vertices must be a positive integer or None, but got {num_vertices}')

        self.n = num_vertices
        self._partial = None  # (a, p, d) tensor
        self._hasty = 0

    def reset_forward(self) -> None:
        self._partial = None
        self._hasty = 0

    def forward(self, x: Tensor, adjs: Tensor | List[Tensor]) -> Tuple[Optional[Tensor], bool]:
        """
        Args:
            x: an (n, *) tensor representing the vertex embeddings for a 2D graph
                      with `n` vertices
            adjs: an (a, n, n) or partial (a, p, n) tensor representing `a` (n, n) adjacency lists:
                     a[i, j] = 1 if there is an edge from i to j (specifically, if j contributes a message to i),
                     a[i, j] = 0 otherwise

        Returns:
            a new (n, d) tensor representing new vertex embeddings for input graph
        """
        graph = self.transform(x)
        if graph.shape[0] != x.shape[0]:
            raise ValueError(f'Graph and transformed graph must have same number of vertices, but got'
                             f'{x.shape[0]} (input) and {graph.shape[0]} (transformed)')

        adjs = self.adj_transform(adjs)

        if self.n is None:
            return self.output(self.update(graph, self.g_agg(self.v_agg(graph, adjs)))), True

        if len(graph.shape) != 2:
            raise ValueError(f'Graph must be a 2D tensor, but got {graph.shape}')
        v, d = graph.shape

        if len(adjs.shape) != 3:
            raise ValueError(f'Adjacency matrices must be a 3D tensor, but got {adjs.shape}')
        a, p, n = adjs.shape
        if n != self.n:
            raise ValueError(f'Adjacency matrix has {n} vertices, but this model expects {self.n}')
        if p > self.n:
            raise PartialForwardError(f'Adjacency matrix has more "partial" vertices {p} than expected vertices {self.n}')

        # if p == v < n then 'hasty partial', forward part of graph completely
        if self._partial is None:
            if p == self.n:
                # no partial updates
                return self.output(self.update(graph, self.g_agg(self.v_agg(graph, adjs)))), True
            elif p == v:
                end = self._hasty + p
                self._hasty = end % self.n
                if end > self.n:
                    raise PartialForwardError(
                        f'Total number of partial vertices {self._hasty} exceeds number of total vertices {self.n}')
                return self.output(self.update(graph, self.g_agg(self.v_agg(graph, adjs)))), self._hasty == 0

        if v != self.n:
            raise ValueError(f'Graph has {v} vertices, but this model expects {self.n}')

        if self._partial is not None:
            a2, p2, d2 = self._partial.shape
            if a2 != a:
                raise PartialForwardError(f'Adjacency matrix has {a} adjacency matrices, but past input(s) had {a2}')
            if d2 != d:
                raise PartialForwardError(f'Graph has {d} embedding dimensions, but past input(s) had {d2}')
            if p2 + p > n:
                raise PartialForwardError(
                    f'The graph should have {n} total vertices, but after this partial update, it will have {p2 + p}')

        partial_vertex_agg = self.v_agg(graph, adjs)
        if len(partial_vertex_agg.shape) != 3:
            raise ValueError(f'Result of vertex aggregation must be a 3D tensor, but got {partial_vertex_agg.shape}')
        a2, p2, d2 = partial_vertex_agg.shape
        if a2 != a:
            raise ValueError(
                f'Vertex aggregation function outputted {a2} adjacency matrices, but this model expects {a} based on the input')
        if p2 != p:
            raise PartialForwardError(
                f'Vertex aggregation function outputted {p2} partial vertices, but this model expects {p} based on the input')
        if d2 != d:
            raise ValueError(
                f'Vertex aggregation function outputted {d2} embedding dimensions, but this model expects {d} based on the input')

        if self._partial is None:
            self._partial = partial_vertex_agg
            return None, False

        self._partial = torch.cat((self._partial, partial_vertex_agg), dim=1)
        if self._partial.shape[1] < n:
            return None, False

        messages = self.g_agg(self._partial)
        self._partial = None

        if len(messages.shape) != 2:
            raise ValueError(f'Result of multi-graph aggregation must be a 2D tensor, but got {messages.shape}')
        n2, d2 = messages.shape
        if n2 != self.n:
            raise ValueError(f'Graph aggregation function outputted {n2} vertices, but this model expects {self.n}')
        if d2 != d:
            raise ValueError(
                f'Graph aggregation function outputted {d2} embedding dimensions, but this model expects {d} based on the input')

        new_graph = self.update(graph, messages)
        if len(new_graph.shape) != 2:
            raise ValueError(f'Result of message update function must be a 2D tensor, but got {new_graph.shape}')
        v2, d2 = new_graph.shape
        if v2 != self.n:
            raise ValueError(f'Update function outputted {v2} vertices, but this model expects {self.n}')
        if d2 != d:
            raise ValueError(
                f'Update function outputted {d2} embedding dimensions, but this model expects {d} based on the input')

        output = self.output(new_graph)
        if output.shape[0] != self.n:
            raise ValueError(f'Output function outputted {output.shape[0]} vertices, but this model expects {self.n}')
        return output, True


class WeightedGraphAggregate(nn.Module):
    def __init__(self, num_graphs: int, num_nodes: int, num_features: int, agg_func: _tensor_func = None,
                 activation_func: _tensor_func = None):
        super().__init__()

        if agg_func is None:
            agg_func = lambda x: torch.sum(x, dim=0)
        self.aggregate = agg_func

        if activation_func is None:
            activation_func = lambda x: x
        self.activate = activation_func

        r = math.sqrt(1.0 / (num_graphs * num_nodes * num_features))
        self.weights = nn.Parameter(torch.rand((num_graphs, num_nodes, num_features)) * 2.0 * r - r)

    def forward(self, graphs: Tensor) -> Tensor:
        """
        Args:
            graphs: (a, n, d) tensor representing `num_layers` graphs/messages with `n` vertices
                       represented by embeddings of dimension `num_features`

        Returns:
            (n, d) tensor representing combined message from all layers

        """
        return self.activate(self.aggregate(self.weights * graphs))

class LinearGraphAggregate(nn.Module):
    """
    Layer that applies a linear transformation to each of `a` matrices in an (a, n, d) tensor and aggregates them.
    """
    def __init__(self, num_graphs: int, in_features: int, out_features: Optional[int] = None, agg_func: _tensor_func = None,
                 activation_func: _tensor_func = None, use_bias: bool = False):
        super().__init__()
        self.a = num_graphs
        if out_features is None:
            out_features = in_features

        if agg_func is None:
            agg_func = lambda x: torch.sum(x, dim=0)
        self.aggregate = agg_func

        if activation_func is None:
            activation_func = lambda x: x
        self.activate = activation_func

        r = math.sqrt(1.0 / (in_features * out_features))
        self.W = nn.Parameter(torch.rand((self.a, in_features, out_features)) * -2.0 * r + r)

        self.B = None
        if use_bias:
            r = math.sqrt(1.0 / out_features)
            self.B = nn.Parameter(torch.rand((self.a, 1, out_features)) * -2.0 * r + r)

    def forward(self, graphs: Tensor) -> Tensor:
        """
        Args:
            graphs: (a, n, d) tensor representing `num_layers` graphs/messages with `n` vertices
                       represented by embeddings of dimension `num_features`

        Returns:
            (n, d) tensor representing combined message from all layers

        """
        if self.B is not None:
            return self.activate(self.aggregate(graphs @ self.W + self.B))
        else:
            return self.activate(self.aggregate(graphs @ self.W))


class LinearMessageUpdate(nn.Module):
    """
    A layer that combines a graph embedding with a message by applying a linear transformation and summing them.
    """
    def __init__(self, in_graph_features: int, in_msg_features: int, out_features: int,
                 activation_func: _tensor_func = None, use_bias: bool = False):
        super().__init__()
        if activation_func is None:
            activation_func = lambda x: x
        self.activate = activation_func

        self.g_layer = lambda g: 0
        if in_graph_features != 0:
            self.g_layer = nn.Linear(in_graph_features, out_features, bias=use_bias)

        self.m_layer = nn.Linear(in_msg_features, out_features, bias=use_bias)

    def forward(self, graph: Tensor, message: Tensor) -> Tensor:
        """
        Applies a linear transformation to the graph and to the message and sums them.

        Args:
            graph: (n, d) tensor representing a graph with *n* vertices,
                   each vertex with embedding dimension *num_features*
            message: (n, d) tensor representing messages for all *n* vertices,
                        each message with embedding dimension *num_features*

        Returns:
            A new (n, d) tensor representing final graph embeddings

        """
        return self.activate(self.g_layer(graph) + self.m_layer(message))


class AdjacencyScale(nn.Module):
    def __init__(self, num_matrices: int, num_nodes: int, pre_weight: bool = True, post_weight: bool = True):
        super().__init__()
        self.A = num_matrices

        r = math.sqrt(1.0 / num_nodes)
        self.pre = None
        if pre_weight:
            self.pre = nn.Parameter(torch.stack([torch.eye(num_nodes) * r for _ in range(self.A)]))
        self.post = None
        if post_weight:
            self.post = nn.Parameter(torch.stack([torch.eye(num_nodes) * r for _ in range(self.A)]))

    def forward(self, adj: Tensor) -> Tensor:
        if self.pre is None and self.post is None:
            return adj
        outputs = []
        for i in range(self.A):
            if self.pre is not None and self.post is not None:
                outputs.append(self.pre[i] @ adj[i] @ self.post[i])
            elif self.pre is not None:
                outputs.append(self.pre[i] @ adj[i])
            elif self.post is not None:
                outputs.append(adj[i] @ self.post[i])

        return torch.stack(outputs)

class EmbeddingGenerator(nn.Module):
    """
    A layer that outputs a learnable (n, d) embedding matrix.
    """
    def __init__(self, size: int, out_features: int):
        super().__init__()
        self.N = size
        self.d = out_features

        r = math.sqrt(1.0 / out_features)
        self.W = nn.Parameter(torch.rand((size, out_features)) * -2.0 * r + r)

    def forward(self, *args: Any) -> Tensor:
        """
        Outputs an (n, d) learnable embedding matrix.

        Args:
            *args: Dummy/unused parameters that do not affect the output.

        Returns:
            an (n, d) embedding matrix.

        """
        return self.W


class PartialForwardError(Exception):
    """Raised when a partial forward input is invalid."""

def sparse_vertex_agg(graph: Tensor, adj: Tensor | List[Tensor]) -> Tensor:
    if not isinstance(adj, list):
        adj = [adj]
    if len(adj) == 0:
        raise ValueError('Must provide at least one adjacency matrix')
    outputs = []
    for a in adj:
        outputs.append(torch.sparse.mm(a, graph))
    return torch.stack(outputs)