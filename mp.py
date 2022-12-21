import multiprocessing
from multiprocessing.process import AuthenticationString
from multiprocessing.shared_memory import SharedMemory
import numpy as np


class SharedNumpyArray:
    """
    Wraps a numpy array so that it can be shared quickly among processes,
    avoiding unnecessary copying and (de)serializing.

    https://e-dorigatti.github.io/python/2020/06/19/multiprocessing-large-objects.html
    """

    def __init__(self, shape, dtype, arr=None):
        """
        Creates the shared memory and copies the array therein
        """
        if isinstance(shape, int):
            shape = (shape,)
        size = 1
        for s in shape:
            size *= s

        # create the shared memory location of the same size of the array
        self._shared = SharedMemory(create=True, size=int(np.dtype(dtype).itemsize) * size)
        self.name = self._shared.name

        # save data type and shape, necessary to read the data correctly
        self._dtype, self._shape = dtype, shape

        # create a new numpy array that uses the shared memory we created.
        # at first, it is filled with zeros
        self._res = np.ndarray(self._shape, dtype=self._dtype, buffer=self._shared.buf)

        if arr is not None:
            self._res[:] = arr[:]

    def read(self):
        """
        Reads the array from the shared memory without unnecessary copying.
        """
        # simply create an array of the correct shape and type,
        # using the shared memory location we created earlier
        return np.ndarray(self._shape, self._dtype, buffer=self._shared.buf)

    def copy(self):
        """
        Returns a new copy of the array stored in shared memory.
        """
        return np.copy(self.read())

    def unlink(self):
        """
        Releases the allocated memory. Call when finished using the data,
        or when the data was copied somewhere else.
        """
        self._shared.close()
        self._shared.unlink()

    def __getitem__(self, item):
        """
        Allows to access the array using the [] operator
        """
        return self._res[item]

    def __setitem__(self, key, value):
        self._res[key] = value

    def __len__(self):
        return len(self._res)

    def __iter__(self):
        return iter(self._res)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['_shared']
        del state['_res']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._shared = SharedMemory(name=self.name)
        self._res = np.ndarray(self._shape, self._dtype, buffer=self._shared.buf)


class Process(multiprocessing.Process):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, *, daemon=None):
        if kwargs is None:
            kwargs = {}
        super(Process, self).__init__(group=group, target=target, name=name, args=args, kwargs=kwargs, daemon=daemon)

    def __getstate__(self):
        """called when pickling - this hack allows subprocesses to
           be spawned without the AuthenticationString raising an error"""
        state = self.__dict__.copy()
        conf = state['_config']
        if 'authkey' in conf:
            conf['authkey'] = bytes(conf['authkey'])
        return state

    def __setstate__(self, state):
        """for unpickling"""
        state['_config']['authkey'] = AuthenticationString(state['_config']['authkey'])
        self.__dict__.update(state)
