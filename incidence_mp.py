from queue import Empty
import numpy as np
import multiprocessing
import mp
from tqdm import tqdm


class _INodeMP:
    def __init__(self, node, graph, offsets):
        self.graph = graph

        self.upper = None
        if len(node.upper) > 0:
            self.upper = mp.SharedNumpyArray(len(node.upper), np.int32)
            for i, u in enumerate(node.upper):
                self.upper[i] = u + offsets[node.d + 1]

        self.neighbors = None
        if len(node.neighbors) > 0:
            self.neighbors = mp.SharedNumpyArray(len(node.neighbors), np.int32)
            for i, n in enumerate(node.neighbors):
                self.neighbors[i] = n + offsets[node.d]

        self.lower = None
        if len(node.lower) > 0:
            self.lower = mp.SharedNumpyArray(len(node.lower), np.int32)
            for i, l in enumerate(node.lower):
                self.lower[i] = l + offsets[node.d - 1]

    def adjacency_set(self, rel_dim):
        if rel_dim == 0:
            if self.neighbors is None:
                return set()
            return set(self.neighbors.read())
        elif rel_dim == 1:
            if self.upper is None:
                return set()
            return set(self.upper.read())
        elif rel_dim == -1:
            if self.lower is None:
                return set()
            return set(self.lower.read())

        adj = set()
        if rel_dim > 1 and self.upper is not None:
            for i in self.upper:
                adj.update(self.graph[i].adjacency_set(rel_dim - 1))
        elif rel_dim < -1 and self.lower is not None:
            for i in self.lower:
                adj.update(self.graph[i].adjacency_set(rel_dim + 1))
        return adj

    def close(self):
        if self.upper is not None:
            self.upper.unlink()
        if self.neighbors is not None:
            self.neighbors.unlink()
        if self.lower is not None:
            self.lower.unlink()

    def load_graph(self, graph):
        self.graph = graph

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['graph']
        return state


class IGraphMP:
    def __init__(self, graph, num_processes=None, work_per_process=20000000, tasks_per_process=None):
        # work per process should be adjusted to core speed
        # task per process should be adjusted to memory & pagefile size

        self.nodes = []
        offsets = graph._IncidenceGraph__get_offsets()
        flattened = graph._IncidenceGraph__flatten()
        for node in tqdm(flattened, desc='Importing nodes to MP unit'):
            self.nodes.append(_INodeMP(node, self.nodes, offsets))

        if work_per_process is not None and (work_per_process <= 0 or not isinstance(work_per_process, int)):
            raise ValueError('Work per process must be `None` or a positive integer')
        self._work_per_process = work_per_process

        if tasks_per_process is not None and (tasks_per_process <= 0 or not isinstance(tasks_per_process, int)):
            raise ValueError('Tasks per process must be `None` or a positive integer')
        self._tasks_per_process = tasks_per_process

        self._processes = []
        self._work = multiprocessing.JoinableQueue()
        if num_processes is None:
            num_processes = max(multiprocessing.cpu_count() - 1, 1)
        elif num_processes <= 0 or not isinstance(num_processes, int):
            raise ValueError('Number of processes must be `None` or a positive integer')

        for i in tqdm(range(num_processes), desc='Starting processes'):
            self._processes.append(mp.Process(target=IGraphMP._worker, args=(i, self.nodes, self._work, self._tasks_per_process)))
            self._processes[-1].start()

    def adjacency_matrix(self, rel_dim=0):
        return next(self.partial_matrix(rel_dim))

    def partial_matrix(self, rel_dim, partial_size=None):
        for adj in self.partial_matrices([rel_dim], partial_size):
            yield adj[0]

    def adjacency_matrices(self, rel_dims):
        return next(self.partial_matrices(rel_dims))

    def partial_matrices(self, rel_dims, partial_size=None):
        shared_dims = mp.SharedNumpyArray(len(rel_dims), np.int32, arr=np.array(rel_dims, dtype=np.int32))
        A, N = len(rel_dims), len(self)

        if partial_size is None or partial_size > N:
            partial_size = N
        elif partial_size <= 0:
            raise ValueError('Partial matrix size must be positive')

        if self._work_per_process is None:
            a_step, p_step = A, partial_size
        elif self._work_per_process // N < A:
            a_step, p_step = max(1, self._work_per_process // N), 1
        else:
            a_step, p_step = A, max(1, self._work_per_process // (A * N))
        for i in range(0, N, partial_size):
            P = min(partial_size, N - i)
            shared = mp.SharedNumpyArray((A, P, N), np.float32)
            for a in range(0, A, a_step):
                for partial_index, index in enumerate(range(i, i+P, p_step)):
                    self._work.put((shared, a, min(a_step, A - a), partial_index, index, min(p_step, i + P - index), shared_dims))
            self.__check_processes()
            while True:
                # main process will do work in case other processes quit
                try:
                    shared, a_start, a_count, p_start, n_start, p_count, rel_dims = self._work.get(False)
                    for c in range(p_count):
                        for a in range(a_count):
                            adj = self.nodes[n_start + c].adjacency_set(rel_dims[a + a_start])
                            for j in adj:
                                shared[a + a_start, p_start + c, j] = 1.0
                    self._work.task_done()
                except Empty:
                    break

            self._work.join()
            yield shared

        shared_dims.unlink()

    @staticmethod
    def _worker(worker_id, graph, queue, max_tasks):
        for node in graph:
            node.load_graph(graph)
        tasks_done = 0
        while True:
            if max_tasks is not None and tasks_done >= max_tasks:
                break

            shared, a_start, a_count, p_start, n_start, p_count, rel_dims = queue.get()
            if shared is None:
                queue.task_done()
                break
            for c in range(p_count):
                for a in range(a_count):
                    adj = graph[n_start + c].adjacency_set(rel_dims[a + a_start])
                    for j in adj:
                        shared[a + a_start, p_start + c, j] = 1.0
            queue.task_done()
            tasks_done += 1

    def __check_processes(self):
        for i in range(len(self._processes)):
            if not self._processes[i].is_alive():
                self._processes[i].close()
                self._processes[i] = mp.Process(target=IGraphMP._worker, args=(i, self.nodes, self._work, self._tasks_per_process))
                self._processes[i].start()

    def close(self):
        for _ in range(len(self._processes)):
            self._work.put((None, None, None, None, None, None, None))
        for process in self._processes:
            process.join()
            process.close()
        for node in self.nodes:
            node.close()

    def __len__(self):
        return len(self.nodes)
