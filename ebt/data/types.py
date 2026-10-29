import os
import pickle
import shutil
import tempfile
import uuid
from collections.abc import MutableMapping
from urllib.parse import quote
from ebt.simulate.tricks import is_dask_process


class LazyDiskDict(MutableMapping):
    """A dict whose values live on disk (one pickle per key) and are loaded on access.

    Each instance owns a private directory ``<folder>/lazy_disk_dict_<uuid>/`` that is
    removed when the instance is garbage collected (unless ``monitor_dask`` is set and the
    instance lives inside a dask worker, where the files may be shared). Keys are
    ``str()``-ed and percent-encoded for the file name, so any key is safe ('a/b' does not
    become a sub-directory) and distinct keys never collide. The directory name is a uuid,
    not ``id(self)``: ``id`` is recycled by CPython, so a later instance could otherwise
    read or overwrite the files an earlier (undeleted) one left behind.
    """

    def __init__(self, d=None, folder=None, monitor_dask=False):
        self._folder = folder or tempfile.gettempdir()
        self._dir = os.path.join(self._folder, f'lazy_disk_dict_{uuid.uuid4().hex}')
        os.makedirs(self._dir, exist_ok=True)
        self._keys = set()  # Keep track of keys explicitly
        self.is_dask = is_dask_process() if monitor_dask else False
        if d is not None:
            for k, v in d.items():
                self[k] = v

    def _filepath(self, key):
        """A safe file name for a given key (percent-encoded, no path separators)."""
        return os.path.join(self._dir, f'{quote(str(key), safe="")}.pkl')

    def __getitem__(self, key):
        if key not in self._keys:
            raise KeyError(key)
        with open(self._filepath(key), "rb") as f:
            return pickle.load(f)

    def __setitem__(self, key, value):
        os.makedirs(self._dir, exist_ok=True)
        with open(self._filepath(key), "wb") as f:
            pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
        self._keys.add(key)

    def __delitem__(self, key):
        if key not in self._keys:
            raise KeyError(key)
        os.remove(self._filepath(key))
        self._keys.remove(key)

    def __del__(self):
        if getattr(self, 'is_dask', False):
            return  # files may be shared with dask workers; leave them
        d = getattr(self, '_dir', None)
        if d is not None:
            shutil.rmtree(d, ignore_errors=True)

    def __iter__(self):
        return iter(self._keys)

    def __len__(self):
        return len(self._keys)

    def __repr__(self):
        return f"{self.__class__.__name__}({list(self._keys)})"
