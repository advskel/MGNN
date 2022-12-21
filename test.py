import gc
import logging
import random
import time
import multiprocessing

import incidence_graph
import incidence_mp
import multignn
import torch
import coloredlogs
from tqdm import tqdm
import torch.nn as nn
import graphutils
import numpy as np

def main():
    g = incidence_graph.IncidenceGraph()
    g.put_simplex((0, 1))
    g.put_simplex((1, 2, 3))
    print(g.adjacency_matrix(1))

def gridtest(width, height, rel_dims, partial_size):
    A, D = 1, 16

    log.debug('Generating grid...')
    test, index, reverse_index = incidence_graph.grid(width, height)
    N, V = len(test), test.num_vertices()
    P = partial_size

    a_gen = test.partial_matrices(rel_dims=rel_dims, partial_size=P)
    graph = torch.rand((N, D), device=device)
    model = multignn.MultiGraphLayer(lambda g, a: a @ g,
                                     multignn.LinearAggregate(A, D, nn.ReLU(), use_bias=False),
                                     multignn.LinearMessageUpdate(D, nn.ReLU(), use_bias=False),
                                     num_vertices=N).to(device)

    post = {'gen time': 0.0, 'nn time': 0.0}
    g_count, f_count = 0, 0
    iterator = tqdm(a_gen, total=N // P, desc='Forward')
    end = time.time()
    for adj in iterator:
        adj = torch.as_tensor(adj, device=device)
        start = time.time()
        g_count += 1
        post['gen time'] += (start - end - post['gen time']) / g_count

        _ = model.forward(graph, adj)
        end = time.time()
        f_count += 1
        post['nn time'] += (end - start - post['nn time']) / f_count
        iterator.set_postfix(ordered_dict=post)

    log.info('done')


def gridtest_mp(width, height, rel_dims, partial_size):
    A, D = 1, 16

    log.debug('Generating grid...')
    test, index, reverse_index = incidence_graph.grid(width, height)
    test_mp = incidence_mp.IGraphMP(test, 8, tasks_per_process=500)
    N, V = len(test), test.num_vertices()
    P = partial_size

    a_gen = test_mp.partial_matrices(rel_dims=rel_dims, partial_size=P)
    graph = torch.rand((N, D), device=device)
    model = multignn.MultiGraphLayer(lambda g, a: a @ g,
                                     multignn.LinearAggregate(A, D, nn.ReLU(), use_bias=False),
                                     multignn.LinearMessageUpdate(D, nn.ReLU(), use_bias=False),
                                     num_vertices=N).to(device)

    post = {'gen time': 0.0, 'nn time': 0.0}
    g_count, f_count = 0, 0
    iterator = tqdm(a_gen, total=N // P, desc='Forward MP')
    end = time.time()
    for shared in iterator:
        adj = torch.as_tensor(shared.read(), device=device)
        start = time.time()
        g_count += 1
        post['gen time'] += (start - end - post['gen time']) / g_count

        _ = model.forward(graph, adj)
        end = time.time()
        f_count += 1
        post['nn time'] += (end - start - post['nn time']) / f_count
        iterator.set_postfix(ordered_dict=post)
        shared.unlink()

    test_mp.close()
    log.info('done')


if __name__ == "__main__":
    log = logging.getLogger(__name__)
    coloredlogs.install(level='DEBUG')  # Change this to DEBUG to see more info.

    multiprocessing.freeze_support()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.debug(f'Using device: {device}')
    main()

