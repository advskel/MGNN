import math
from typing import Optional, Callable, Any, Tuple

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

_tensor_func = Optional[Callable[[Tensor], Tensor]]
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
    def __init__(self, transform_func: _tensor_func = None, vertex_agg_func: _two_tensor_func = None, graph_agg_func: _tensor_func = None, update_func: _two_tensor_func = None,
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

        if vertex_agg_func is None:
            vertex_agg_func = lambda g, a: a @ g
        self.v_agg = vertex_agg_func  # aggregate messages per vertex

        if graph_agg_func is None:
            graph_agg_func = lambda a: F.relu(torch.sum(a, dim=0))
        self.g_agg = graph_agg_func  # aggregate messages per graph/adj matrix

        if update_func is None:
            update_func = lambda g, m: F.relu((g + m) / 2)
        self.update = update_func

        if num_vertices is not None and (num_vertices <= 0 or not isinstance(num_vertices, int)):
            raise ValueError(f'Number of vertices must be a positive integer or None, but got {num_vertices}')

        self.n = num_vertices
        self._partial = None  # (a, p, d) tensor
        self._hasty = 0

    def reset_forward(self) -> None:
        self._partial = None
        self._hasty = 0

    def forward(self, x: Tensor, adjs: Tensor) -> Tuple[Optional[Tensor], bool]:
        # TODO batch forward? where input is (b, n, *) and adjs is (b, a, n, n)
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

        if self.n is None:
            return self.update(graph, self.g_agg(self.v_agg(graph, adjs))), True

        if len(graph.shape) != 2:
            raise ValueError(f'Graph must be a 2D tensor, but got {graph.shape}')
        v, d = graph.shape

        if len(adjs.shape) != 3:
            raise ValueError(f'Adjacency matrices must be a 3D tensor, but got {adjs.shape}')
        a, p, n = adjs.shape
        if n != self.n:
            raise ValueError(f'Adjacency matrix has {n} vertices, but this model expects {self.n}')
        if p > self.n:
            raise ValueError(f'Adjacency matrix has more "partial" vertices {p} than expected vertices {self.n}')

        # if p == v < n then 'hasty partial', forward part of graph completely
        if self._partial is None:
            if p == self.n:
                # no partial updates
                return self.update(graph, self.g_agg(self.v_agg(graph, adjs))), True
            elif p == v:
                self._hasty += p
                if self._hasty > self.n:
                    raise ValueError(
                        f'Total number of partial vertices {self._hasty} exceeds number of total vertices {self.n}')
                elif self._hasty == self.n:
                    self._hasty = 0
                    return self.update(graph, self.g_agg(self.v_agg(graph, adjs))), True
                else:
                    return self.update(graph, self.g_agg(self.v_agg(graph, adjs))), False

        if v != self.n:
            raise ValueError(f'Graph has {v} vertices, but this model expects {self.n}')

        if self._partial is not None:
            a2, p2, d2 = self._partial.shape
            if a2 != a:
                raise ValueError(f'Adjacency matrix has {a} adjacency matrices, but past input(s) had {a2}')
            if d2 != d:
                raise ValueError(f'Graph has {d} embedding dimensions, but past input(s) had {d2}')
            if p2 + p > n:
                raise ValueError(
                    f'The graph should have {n} vertices, but after this partial update, it will have {p2 + p}')

        partial_vertex_agg = self.v_agg(graph, adjs)
        if len(partial_vertex_agg.shape) != 3:
            raise ValueError(f'Result of vertex aggregation must be a 3D tensor, but got {partial_vertex_agg.shape}')
        a2, p2, d2 = partial_vertex_agg.shape
        if a2 != a:
            raise ValueError(
                f'Vertex aggregation function outputted {a2} adjacency matrices, but this model expects {a} based on the input')
        if p2 != p:
            raise ValueError(
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
        return new_graph, True


class LinearAggregate(nn.Module):
    """
    Layer that applies a linear transformation to each of `a` matrices in an (a, n, d) tensor and sums them.
    """
    def __init__(self, num_layers: int, num_features: int, activation_func: _tensor_func = None, use_weights: bool = True, use_bias: bool = True):
        super().__init__()
        self.a = num_layers
        self.d = num_features

        if activation_func is None:
            activation_func = lambda x: x
        self.activate = activation_func

        r = math.sqrt(1.0 / self.d)

        self.use_weights = use_weights
        if use_weights:
            self.W = nn.Parameter(torch.rand((self.a, self.d, self.d)) * -2.0 * r + r)

        self.use_bias = use_bias
        if use_bias:
            self.B = nn.Parameter(torch.rand((self.a, 1, self.d)) * -2.0 * r + r)

    def forward(self, graphs: Tensor) -> Tensor:
        """
        Args:
            graphs: (a, n, d) tensor representing `num_layers` graphs/messages with `n` vertices
                       represented by embeddings of dimension `num_features`

        Returns:
            (n, d) tensor representing combined message from all layers

        """
        if self.use_weights and self.use_bias:
            return self.activate(torch.sum(graphs @ self.W + self.B, dim=0))
        elif self.use_weights:
            return self.activate(torch.sum(graphs @ self.W, dim=0))
        elif self.use_bias:
            return self.activate(torch.sum(graphs + self.B, dim=0))
        else:
            return self.activate(torch.sum(graphs, dim=0))


class LinearMessageUpdate(nn.Module):
    """
    A layer that combines a graph embedding with a message by applying a linear transformation and summing them.
    """
    def __init__(self, num_features: int, activation_func: _tensor_func = None, use_weights: bool = True, use_bias: bool = True):
        super().__init__()
        if activation_func is None:
            activation_func = lambda x: x
        self.activate = activation_func
        self.d = num_features

        r = math.sqrt(1.0 / self.d)

        self.use_weights = use_weights
        if use_weights:
            self.W1 = nn.Parameter(torch.rand((self.d, self.d)) * -2.0 * r + r)
            self.W2 = nn.Parameter(torch.rand((self.d, self.d)) * -2.0 * r + r)

        self.use_bias = use_bias
        if use_bias:
            self.B = nn.Parameter(torch.rand(self.d) * -2.0 * r + r)

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
        if self.use_weights and self.use_bias:
            return self.activate(graph @ self.W1 + message @ self.W2 + self.B)
        elif self.use_weights:
            return self.activate(graph @ self.W1 + message @ self.W2)
        elif self.use_bias:
            return self.activate(graph + message + self.B)
        else:
            return self.activate(graph + message)

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
