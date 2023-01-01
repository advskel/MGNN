import gc
import logging
import random
import time
import multiprocessing

from incidence_graph import IncidenceGraph
import incidence_mp
import multignn
import torch
import coloredlogs
from tqdm import tqdm
import torch.nn as nn
import graphutils
import numpy as np


def main():
    g = IncidenceGraph()
    g.put_simplex((0, 1))
    g.put_simplex((1, 2, 3))
    print(g.neighbors(0, -1))


def gridtest(width, height, rel_dims, partial_size, iters):
    A, D = 1, 16

    log.debug('Generating grid...')
    test, index, reverse_index = graphutils.grid(width, height)
    N, V = len(test), test.size(0)
    P = partial_size

    a_gen = test.partial_matrices(rel_dims=rel_dims, partial_size=P)
    graph = torch.rand((N, D), device=device)
    model = multignn.MultiGraphLayer(graph_agg_func=multignn.LinearAggregate(A, D, nn.ReLU(), use_bias=False),
                                     update_func=multignn.LinearMessageUpdate(D, nn.ReLU(), use_bias=False),
                                     num_vertices=N).to(device)

    post = {'gen time': 0.0, 'nn time': 0.0}
    g_count, f_count = 0, 0
    iterator = tqdm(range(iters), desc='Forward')
    for _ in iterator:
        start = time.time()
        adj = torch.as_tensor(a_gen(), device=device)
        end = time.time()
        g_count += 1
        post['gen time'] += (end - start - post['gen time']) / g_count

        start = time.time()
        _ = model.forward(graph, adj)
        end = time.time()
        f_count += 1
        post['nn time'] += (end - start - post['nn time']) / f_count
        iterator.set_postfix(ordered_dict=post)

    log.info('done')


def gridtest_mp(width, height, rel_dims, partial_size, iters):
    A, D = 1, 16

    log.debug('Generating grid...')
    test, index, reverse_index = graphutils.grid(width, height)
    test_mp = incidence_mp.IGraphMP(test, tasks_per_process=1500)
    N, V = len(test), test.size(0)
    P = partial_size

    a_gen = test_mp.partial_matrices(rel_dims=rel_dims, partial_size=P)
    graph = torch.rand((N, D), device=device)
    model = multignn.MultiGraphLayer(graph_agg_func=multignn.LinearAggregate(A, D, nn.ReLU(), use_bias=False),
                                     update_func=multignn.LinearMessageUpdate(D, nn.ReLU(), use_bias=False),
                                     num_vertices=N).to(device)

    post = {'gen time': 0.0, 'nn time': 0.0}
    g_count, f_count = 0, 0
    iterator = tqdm(range(iters), desc='Forward MP')
    for _ in iterator:
        start = time.time()
        with a_gen() as shared:
            adj = torch.as_tensor(shared, device=device)
            end = time.time()
            g_count += 1
            post['gen time'] += (end - start - post['gen time']) / g_count

            start = time.time()
            _ = model.forward(graph, adj)
            end = time.time()
            f_count += 1
            post['nn time'] += (end - start - post['nn time']) / f_count
            iterator.set_postfix(ordered_dict=post)

    test_mp.close()
    log.info('done')


if __name__ == "__main__":
    log = logging.getLogger(__name__)
    coloredlogs.install(level='DEBUG')  # Change this to DEBUG to see more info.

    multiprocessing.freeze_support()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.debug(f'Using device: {device}')
    main()
