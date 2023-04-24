from collections import deque
import numpy as np
from . import utils as mp
from tqdm import tqdm
import multiprocessing


class IGraphMP:
    def __init__(self, graph, num_processes=None, tasks_per_process=None):
        # work per process should be adjusted to core speed
        # task per process should be adjusted to memory & pagefile size
        neighbors = []
        lookup = []
        offsets = graph._IncidenceGraph__get_offsets()
        flattened = graph._IncidenceGraph__flatten()
        for node in tqdm(flattened, desc='Importing nodes to MP unit'):
            lookup.append(len(neighbors))
            for n in node.neighbors:
                neighbors.append(n + offsets[node.d])
            neighbors.append(-1)

            lookup.append(len(neighbors))
            for u in node.upper:
                neighbors.append(u + offsets[node.d + 1])
            neighbors.append(-1)

            lookup.append(len(neighbors))
            for l in node.lower:
                neighbors.append(l + offsets[node.d - 1])
            neighbors.append(-1)

        self._neighbors = mp.SharedNumpyArray(len(neighbors), np.int32, arr=neighbors)
        self._lookup = mp.SharedNumpyArray(len(lookup), np.int32, arr=lookup)

        if tasks_per_process is not None and (tasks_per_process <= 0 or not isinstance(tasks_per_process, int)):
            raise ValueError('Tasks per process must be `None` or a positive integer')
        self._tasks_per_process = tasks_per_process

        if num_processes is None:
            num_processes = max(multiprocessing.cpu_count() - 1, 1)
        elif num_processes <= 0 or not isinstance(num_processes, int):
            raise ValueError('Number of processes must be `None` or a positive integer')

        self._processes = []
        self._pipes = []
        self._main_queue = multiprocessing.Queue()
        for i in tqdm(range(num_processes), desc='Starting processes'):
            rec, src = multiprocessing.Pipe()
            self._pipes.append((rec, src))
            self._processes.append(mp.Process(target=IGraphMP._worker, args=(
            i, num_processes, self._neighbors, self._lookup, self._main_queue, rec, self._tasks_per_process)))

            self._processes[-1].start()

        self._sleep_time = 0.01

    def adjacency_matrix(self, rel_dim=0):
        return self.partial_matrix(rel_dim)()

    def partial_matrix(self, rel_dim, partial_size=None):
        return self.partial_matrices([rel_dim], partial_size)

    def adjacency_matrices(self, rel_dims):
        return self.partial_matrices(rel_dims)()

    def partial_matrices(self, rel_dims, partial_size=None):
        A, N = len(rel_dims), len(self)

        if partial_size is None or partial_size > N:
            partial_size = N
        elif partial_size <= 0:
            raise ValueError('Partial matrix size must be positive')

        starts = iter(range(0, N, partial_size))

        def next_partial():
            start = next(starts)
            P = min(partial_size, N - start)
            shared = mp.SharedNumpyArray((A, P, N), np.float32)

            for p, i in enumerate(range(start, start + P)):
                for a, rel_dim in enumerate(rel_dims):
                    id = self._main_queue.get()
                    while id < 0:
                        id = -id - 1
                        self._processes[id].close()
                        self._processes[id] = mp.Process(target=IGraphMP._worker, args=(
                            id, len(self._processes), self._neighbors, self._lookup, self._main_queue, self._pipes[0],
                            self._tasks_per_process))
                        self._processes[id].start()

                        id = self._main_queue.get()

                    self._pipes[id][1].send((shared, a, rel_dim, p, i))

            for _, src in self._pipes:
                src.send(True)
                src.recv()

            return shared

        return next_partial

    @staticmethod
    def _worker(worker_id, total_workers, neighbors, lookup, queue, comm, max_tasks):
        tasks_done = 0
        while True:
            if max_tasks is not None and tasks_done >= max_tasks:
                queue.put(-worker_id - 1)
                break

            queue.put(worker_id)
            work = comm.recv()
            if work is None:
                comm.close()
                break
            elif work:
                comm.send(True)
                continue

            shared, a, rel_dim, p, n = work

            if rel_dim == 0:
                i = lookup[n * 3]
                while neighbors[i] != -1:
                    shared[a, p, neighbors[i]] = 1.0
                    i += 1
                continue

            dfs = deque()
            dfs.append((n, 0))
            visited = set()
            while len(dfs) > 0:
                node, depth = dfs.pop()
                if node in visited:
                    continue
                visited.add(node)
                if depth == rel_dim:
                    shared[a, p, node] = 1.0
                    continue

                if 0 <= depth < rel_dim:
                    i = lookup[node * 3 + 1]
                else:
                    i = lookup[node * 3 + 2]

                while neighbors[i] != -1:
                    if neighbors[i] not in visited:
                        dfs.append((neighbors[i], depth + 1))
                    i += 1

            tasks_done += 1

    def close(self):
        for _, src in self._pipes:
            src.send(None)
        for process in self._processes:
            process.join()
            process.close()
        self._neighbors.unlink()
        self._lookup.unlink()
        self._main_queue.close()

    def __len__(self):
        return len(self._lookup) // 3
