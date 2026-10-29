"""Parallel map utilities for single-machine workloads.

Two execution modes:

**pmap (fork-based)** — workers inherit parent memory via copy-on-write.
Good when the parent process is small. Becomes slow when the parent is large
(>10GB RSS) because fork() must copy the entire page table per worker.

**pmap with shared={...}** — launches a lean subprocess that loads data from
Arrow IPC files, then forks workers from there. Fork is fast because the
subprocess has a small page table. Data sharing is zero-copy via mmap.
Use this when the parent process is large. The worker function is unchanged —
shared dict keys are set as module-level globals on the worker's module:

    _DATA = load_big_data()

    def worker(config):
        return process(_DATA, config)

    # Without shared (fork from fat parent — slow):
    _DATA = big_df
    pmap(worker, configs)

    # With shared (lean subprocess — fast, worker unchanged):
    pmap(worker, configs, shared={'_DATA': big_df})

Use `use_threads=True` when worker functions spend most of their time in code
that releases the GIL (Rust extensions, numpy BLAS, I/O).
"""

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import logging
import os
import pickle
import subprocess
import sys
import tempfile
import time

log = logging.getLogger(__name__)


def _noop(_):
    return True


# ---------------------------------------------------------------------------
# Arrow IPC shared data helpers
# ---------------------------------------------------------------------------

def _serialize_shared_data(shared, shm_dir):
    """Write shared DataFrames/Series/arrays to Arrow IPC files.

    Returns a metadata dict mapping names to file paths + type info.
    """
    import pyarrow as pa
    import pyarrow.ipc as ipc
    import numpy as np
    import pandas as pd

    meta = {}
    for name, obj in shared.items():
        safe_name = str(name)
        path = os.path.join(shm_dir, f'{safe_name}.arrow')
        if isinstance(obj, pd.DataFrame):
            table = pa.Table.from_pandas(obj, preserve_index=True)
            with pa.OSFile(path, 'wb') as sink:
                with ipc.new_file(sink, table.schema) as writer:
                    writer.write_table(table)
            meta[name] = {'path': path, 'type': 'dataframe'}
        elif isinstance(obj, pd.Series):
            df = obj.to_frame('__series__')
            table = pa.Table.from_pandas(df, preserve_index=True)
            with pa.OSFile(path, 'wb') as sink:
                with ipc.new_file(sink, table.schema) as writer:
                    writer.write_table(table)
            meta[name] = {'path': path, 'type': 'series', 'name': obj.name}
        elif isinstance(obj, np.ndarray):
            mm = np.memmap(path, dtype=obj.dtype, mode='w+', shape=obj.shape)
            mm[:] = obj
            mm.flush()
            del mm
            meta[name] = {'path': path, 'type': 'ndarray',
                          'dtype': str(obj.dtype), 'shape': list(obj.shape)}
        elif isinstance(obj, dict):
            sub_dir = os.path.join(shm_dir, safe_name)
            os.makedirs(sub_dir, exist_ok=True)
            meta[name] = {'type': 'dict',
                          'children': _serialize_shared_data(obj, sub_dir)}
        else:
            pkl_path = os.path.join(shm_dir, f'{safe_name}.pkl')
            with open(pkl_path, 'wb') as f:
                pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
            meta[name] = {'path': pkl_path, 'type': 'pickle'}

    return meta


def _deserialize_shared_data(meta):
    """Reconstruct shared data from Arrow IPC files (zero-copy via mmap)."""
    import pyarrow as pa
    import pyarrow.ipc as ipc
    import numpy as np

    data = {}
    for name, info in meta.items():
        t = info['type']
        if t == 'dataframe':
            source = pa.memory_map(info['path'], 'r')
            table = ipc.open_file(source).read_all()
            # split_blocks + self_destruct: the mmap+read_all is lazy, but a
            # plain to_pandas() materializes a single consolidated block.
            # Converting column by column and releasing each Arrow buffer as it
            # is consumed keeps the peak near 1x. Same values; the frame is not
            # one block, which only matters if a caller mutates it in place
            # (shared data is read-only by contract).
            data[name] = table.to_pandas(split_blocks=True, self_destruct=True)
            del table
        elif t == 'series':
            source = pa.memory_map(info['path'], 'r')
            table = ipc.open_file(source).read_all()
            df = table.to_pandas(split_blocks=True, self_destruct=True)
            del table
            data[name] = df['__series__'].rename(info['name'])
        elif t == 'ndarray':
            data[name] = np.memmap(info['path'], dtype=info['dtype'],
                                   mode='r', shape=tuple(info['shape']))
        elif t == 'dict':
            data[name] = _deserialize_shared_data(info['children'])
        elif t == 'pickle':
            with open(info['path'], 'rb') as f:
                data[name] = pickle.load(f)

    return data


# ---------------------------------------------------------------------------
# Result externalization — write large numpy arrays (memmap) and pandas
# DataFrames/Series (Arrow IPC) to files so they don't go through pickle
# (neither the executor pipe nor the results file). Without this, a worker
# returning a position frame is pickled TWICE (worker→pool pipe, then the
# aggregate _results.pkl) and the pool process holds every result frame in
# RAM simultaneously while it accumulates them.
# ---------------------------------------------------------------------------

_MEMMAP_TAG = '__pmap_mm__'
_ARROW_TAG = '__pmap_arrow__'
_RESULTS_DIR = None       # Set by _subprocess_main before forking
_EXTERN_COUNTER = 0       # Per-process counter (each fork gets its own copy)

# PMAP_NO_FRAME_EXTERN=1 uses the pickle path for
# DataFrame/Series results (ndarray memmap externalization is unaffected).
# Read at import time in each process — the env propagates to the pool
# subprocess and its forked workers.
_NO_FRAME_EXTERN = os.environ.get('PMAP_NO_FRAME_EXTERN', '0') == '1'


def _frame_arrow_safe(obj):
    """True if a DataFrame/Series provably round-trips through Arrow IPC.

    Deliberately strict: plain numpy numeric/bool/datetime/timedelta dtypes
    plus tz-aware datetimes, for both values and index levels. Everything
    else (object, categorical, nullable extension dtypes, MultiIndex
    columns, ...) keeps the existing pickle path — identical behavior, just
    without the shortcut. The hot case (float position/prediction frames on
    a DatetimeIndex) is fully covered.
    """
    import numpy as np
    import pandas as pd

    def _ok(d):
        if isinstance(d, pd.DatetimeTZDtype):
            return True
        return isinstance(d, np.dtype) and d.kind in 'biufMm'

    if isinstance(obj, pd.DataFrame):
        if isinstance(obj.columns, pd.MultiIndex):
            return False
        if obj.columns.has_duplicates:
            return False
        dtypes = list(obj.dtypes)
    else:
        dtypes = [obj.dtype]
    index = obj.index
    idx_dtypes = list(index.dtypes) if isinstance(index, pd.MultiIndex) \
        else [index.dtype]
    return all(_ok(d) for d in dtypes + idx_dtypes)


def _externalize(obj, prefix):
    """Recursively replace large numpy arrays (memmap) and DataFrames/Series
    (Arrow IPC) with file references. Runs in the WORKER process."""
    import numpy as np
    import pandas as pd
    global _EXTERN_COUNTER
    if isinstance(obj, np.ndarray) and obj.nbytes > 4096:
        _EXTERN_COUNTER += 1
        path = os.path.join(_RESULTS_DIR,
                            f'{prefix}_{os.getpid()}_{_EXTERN_COUNTER}.dat')
        mm = np.memmap(path, dtype=obj.dtype, mode='w+', shape=obj.shape)
        mm[:] = obj
        mm.flush()
        del mm
        return {_MEMMAP_TAG: path, 'd': str(obj.dtype), 's': list(obj.shape)}
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        is_series = isinstance(obj, pd.Series)
        nbytes = int(obj.memory_usage(index=True)) if is_series \
            else int(obj.memory_usage(index=True).sum())
        if nbytes > 4096 and not _NO_FRAME_EXTERN and _frame_arrow_safe(obj):
            path = os.path.join(_RESULTS_DIR,
                                f'{prefix}_{os.getpid()}_{_EXTERN_COUNTER + 1}.arrow')
            try:
                import pyarrow as pa
                import pyarrow.ipc as ipc
                df = obj.to_frame('__series__') if is_series else obj
                table = pa.Table.from_pandas(df, preserve_index=True)
                with pa.OSFile(path, 'wb') as sink:
                    with ipc.new_file(sink, table.schema) as writer:
                        writer.write_table(table)
            except Exception:
                # Not Arrow-representable after all — fall back to pickle,
                # never fail the task over a shortcut.
                if os.path.exists(path):
                    os.unlink(path)
                return obj
            _EXTERN_COUNTER += 1
            # Arrow drops DatetimeIndex.freq; pickle preserves it. Carry it in
            # the metadata so round-trip stays pickle-equivalent.
            freq = getattr(obj.index, 'freqstr', None)
            return {_ARROW_TAG: path, 'series': is_series,
                    'name': obj.name if is_series else None, 'freq': freq}
        return obj
    if isinstance(obj, tuple):
        return tuple(_externalize(x, prefix) for x in obj)
    if isinstance(obj, list):
        return [_externalize(x, prefix) for x in obj]
    if isinstance(obj, dict) and _MEMMAP_TAG not in obj and _ARROW_TAG not in obj:
        return {k: _externalize(v, prefix) for k, v in obj.items()}
    return obj


def _internalize(obj):
    """Reconstruct numpy arrays / DataFrames / Series from file references.
    Runs in the CALLING process (before shm_dir cleanup)."""
    import numpy as np
    if isinstance(obj, dict) and _MEMMAP_TAG in obj:
        return np.array(np.memmap(obj[_MEMMAP_TAG], dtype=obj['d'],
                                  mode='r', shape=tuple(obj['s'])))
    if isinstance(obj, dict) and _ARROW_TAG in obj:
        import pyarrow as pa
        import pyarrow.ipc as ipc
        source = pa.memory_map(obj[_ARROW_TAG], 'r')
        table = ipc.open_file(source).read_all()
        df = table.to_pandas(split_blocks=True, self_destruct=True)
        del table
        out = df['__series__'].rename(obj['name']) if obj['series'] else df
        # copy(): pickle-parity — the caller gets a writable frame that owns
        # its memory, independent of the backing file (deleted right after).
        out = out.copy()
        if obj.get('freq') is not None:
            try:
                out.index.freq = obj['freq']
            except (AttributeError, ValueError):
                pass
        return out
    if isinstance(obj, tuple):
        return tuple(_internalize(x) for x in obj)
    if isinstance(obj, list):
        return [_internalize(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _internalize(v) for k, v in obj.items()}
    return obj


# ---------------------------------------------------------------------------
# Subprocess worker dispatch (for kwargs injection mode)
# ---------------------------------------------------------------------------

# Set by _subprocess_main before forking. Workers inherit via fork COW.
_DISPATCH_FN = None
_DISPATCH_SHARED = None
_DISPATCH_UNPACK = False


def _dispatch(item):
    """Calls fn(item, **shared) or fn(*item, **shared).
    Externalizes numpy arrays (memmap) and DataFrames/Series (Arrow IPC)
    in results to files instead of returning them through pickle.

    Module-level and picklable. fn and shared are inherited via fork COW.
    """
    if _DISPATCH_UNPACK:
        result = _DISPATCH_FN(*item, **_DISPATCH_SHARED)
    else:
        result = _DISPATCH_FN(item, **_DISPATCH_SHARED)
    if _RESULTS_DIR is not None:
        return _externalize(result, 'r')
    return result


# ---------------------------------------------------------------------------
# Subprocess worker pool
# ---------------------------------------------------------------------------

def _subprocess_main():
    """Entry point for the lean subprocess. Runs the process pool."""
    import logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [pool] %(message)s')
    slog = logging.getLogger('pmap_subprocess')

    args_path = sys.argv[2]
    with open(args_path, 'rb') as f:
        args = pickle.load(f)

    shm_meta = args['shm_meta']
    items = args['items']
    max_workers = args['max_workers']
    fn_module = args['fn_module']
    fn_name = args['fn_name']
    filter_none = args['filter_none']
    unpack = args['unpack']
    log_every = args['log_every']
    desc = args['desc']
    results_path = args['results_path']

    # Load shared data from Arrow IPC (zero-copy mmap)
    t0 = time.time()
    shared_data = _deserialize_shared_data(shm_meta)
    t_load = time.time()
    slog.info("shared data loaded: %.2fs", t_load - t0)

    # Restore parent's sys.argv before importing the worker module.
    # Module-level code (e.g. CONFIG = sys.argv[1]) must see the parent's
    # args, not the subprocess's ['...', '--subprocess', ...].
    parent_argv = args.get('parent_argv')
    if parent_argv is not None:
        sys.argv = parent_argv

    # Import the worker function's module
    import importlib
    fn_file = args.get('fn_file')
    if fn_module == '__main__' and fn_file:
        import importlib.util
        spec = importlib.util.spec_from_file_location('_worker_main', fn_file)
        mod = importlib.util.module_from_spec(spec)
        sys.modules['_worker_main'] = mod
        spec.loader.exec_module(mod)
    else:
        mod = importlib.import_module(fn_module)

    fn = getattr(mod, fn_name)
    if fn_module == '__main__' and fn_file:
        fn.__module__ = '_worker_main'

    # Decide how to pass shared data to the worker function:
    # If shared keys match function parameter names → inject as kwargs
    # Otherwise → set as module-level globals (globals pattern)
    import inspect
    try:
        params = set(inspect.signature(fn).parameters.keys())
    except (ValueError, TypeError):
        params = set()

    shared_keys = set(shared_data.keys())
    use_kwargs = bool(shared_keys & params)

    # Set up result externalization directory — workers write large numpy
    # arrays (memmap) and DataFrames/Series (Arrow IPC) here instead of
    # returning them through pickle, so this process never holds every
    # result frame in RAM at once.
    import ebt.parallel as _ep
    results_data_dir = os.path.join(os.path.dirname(results_path), '_results_data')
    os.makedirs(results_data_dir, exist_ok=True)
    _ep._RESULTS_DIR = results_data_dir

    if use_kwargs:
        # kwargs pattern: fn(item, key1=val1, key2=val2, ...)
        _ep._DISPATCH_FN = fn
        _ep._DISPATCH_SHARED = shared_data
        _ep._DISPATCH_UNPACK = unpack
        submit_fn = _ep._dispatch
        slog.info("shared mode: kwargs injection (%s)",
                  ', '.join(str(k) for k in shared_keys & params))
    else:
        # globals pattern: set on worker's module, fn reads them directly
        for name, value in shared_data.items():
            setattr(mod, name, value)
        _ep._DISPATCH_FN = fn
        _ep._DISPATCH_SHARED = {}
        _ep._DISPATCH_UNPACK = unpack
        submit_fn = _ep._dispatch
        slog.info("shared mode: module globals (%s)",
                  ', '.join(str(k) for k in shared_keys))

    t_import = time.time()
    slog.info("function imported: %.2fs", t_import - t_load)

    # Fork workers from this lean process
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futs = [executor.submit(_noop, i) for i in range(max_workers)]
        [f.result() for f in futs]
        t_fork = time.time()
        slog.info("forked %d workers: %.2fs", max_workers, t_fork - t_import)

        # All paths go through _dispatch which handles unpack + externalization
        fut_to_item = {executor.submit(submit_fn, item): item
                       for item in items}

        results = []
        errors = 0
        done = 0
        n = len(items)
        for fut in as_completed(fut_to_item):
            done += 1
            item = fut_to_item[fut]
            try:
                result = fut.result()
                if filter_none:
                    if result is None:
                        continue
                    if hasattr(result, '__len__') and len(result) == 0:
                        continue
                results.append((item, result))
            except Exception as e:
                errors += 1
                slog.warning("task error: %s", e, exc_info=True)

            if done == 1:
                slog.info("first result: %.1fs", time.time() - t0)
            if done % log_every == 0 or done == n:
                elapsed = time.time() - t_fork
                rate = done / elapsed * 60 if elapsed > 0 else 0
                slog.info("%d/%d done, %d results, %d errors | %.1f/min",
                          done, n, len(results), errors, rate)

    t_compute = time.time()
    slog.info("compute done: %.1fs", t_compute - t_fork)

    # Results already have numpy arrays externalized to memmap by _dispatch.
    # Pickle now only contains lightweight metadata — fast.
    with open(results_path, 'wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    t_write = time.time()

    total = time.time() - t0
    slog.info("total: %.1fs (load=%.1f, fork=%.1f, compute=%.1f, results=%.1f)",
              total, t_load - t0, t_fork - t_import,
              t_compute - t_fork, t_write - t_compute)

    # 100% failure must not look like success: parent raises on nonzero exit
    if n > 0 and errors == n:
        slog.error("all %d items failed — exiting nonzero", n)
        sys.exit(3)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def pmap(fn, items, *, max_workers=50, use_threads=False, desc=None,
         log_every=20, filter_none=True, unpack=False, shared=None):
    """Parallel map with progress tracking.

    Args:
        fn: Function to apply to each item. Called as fn(item), or fn(*item)
            if unpack=True. Must be a module-level function (not a lambda
            or closure) when shared={...} is used.
        items: Iterable of items to process.
        max_workers: Number of parallel workers.
        use_threads: Use ThreadPoolExecutor instead of ProcessPoolExecutor.
        desc: Description for progress logging.
        log_every: Log progress every N completions.
        filter_none: If True, skip None results.
        unpack: If True, call fn(*item) instead of fn(item).
        shared: Dict of {global_name: value} to share with workers via
            Arrow IPC. Keys are set as module-level globals on the worker
            function's module before forking, so the worker reads them
            exactly as with fork-based COW. The worker function signature
            is unchanged.

            Launches a lean subprocess to avoid slow fork() from a large
            parent process. Recommended when parent has >10GB RSS.

    Returns:
        List of (item, result) tuples for successful, non-filtered calls.

    Example:
        # kwargs pattern — shared keys match function parameters:
        def worker(config, prices, features):
            return predict(features, prices, config)

        pmap(worker, configs, max_workers=100,
             shared={'prices': big_prices_df, 'features': big_feat_df})

        # globals pattern — shared keys are module-level variable names:
        def worker(config):
            return process(_DATA, config)

        pmap(worker, configs, max_workers=100,
             shared={'_DATA': big_dataframe})
    """
    if shared is not None and not use_threads:
        return _pmap_lean(fn, items, shared=shared, max_workers=max_workers,
                          desc=desc, log_every=log_every,
                          filter_none=filter_none, unpack=unpack)

    return _pmap_fork(fn, items, max_workers=max_workers,
                      use_threads=use_threads, desc=desc,
                      log_every=log_every, filter_none=filter_none,
                      unpack=unpack)


def _pmap_fork(fn, items, *, max_workers=50, use_threads=False, desc=None,
               log_every=20, filter_none=True, unpack=False):
    """Fork-based pmap."""
    items = list(items)
    n = len(items)
    if n == 0:
        return []

    desc = desc or getattr(fn, '__name__', 'pmap')
    Executor = ThreadPoolExecutor if use_threads else ProcessPoolExecutor

    results = []
    errors = 0
    t0 = time.time()

    with Executor(max_workers=max_workers) as executor:
        if unpack:
            fut_to_item = {executor.submit(fn, *item): item for item in items}
        else:
            fut_to_item = {executor.submit(fn, item): item for item in items}

        done = 0
        for fut in as_completed(fut_to_item):
            done += 1
            item = fut_to_item[fut]
            try:
                result = fut.result()
                if filter_none:
                    if result is None:
                        continue
                    if hasattr(result, '__len__') and len(result) == 0:
                        continue
                results.append((item, result))
            except Exception as e:
                errors += 1
                # exc_info: a swallowed worker error with no traceback silently
                # drops models/configs from results
                log.warning("%s failed: %s", desc, e, exc_info=True)

            if done % log_every == 0 or done == n:
                elapsed = time.time() - t0
                rate = done / elapsed * 60 if elapsed > 0 else 0
                log.info("  %s: %d/%d done, %d results, %d errors | %.1f/min",
                         desc, done, n, len(results), errors, rate)

    if n > 0 and errors == n:
        raise RuntimeError(
            f"{desc}: all {n} pmap items failed — see logged tracebacks")

    elapsed = time.time() - t0
    log.info("%s: %d results from %d items in %.1fs", desc, len(results), n, elapsed)
    return results


def _pmap_lean(fn, items, *, shared, max_workers=50, desc=None,
               log_every=20, filter_none=True, unpack=False):
    """Lean subprocess pmap: Arrow IPC shared data + fork from clean process."""
    import shutil

    items = list(items)
    n = len(items)
    if n == 0:
        return []

    desc = desc or getattr(fn, '__name__', 'pmap')
    t0 = time.time()

    shm_dir = tempfile.mkdtemp(prefix='pmap_shm_')

    try:
        # Serialize shared data to Arrow IPC
        shm_meta = _serialize_shared_data(shared, shm_dir)
        t_ser = time.time()
        log.info("  %s: arrow serialization: %.1fs", desc, t_ser - t0)

        # Resolve function location
        fn_module = fn.__module__
        fn_file = None
        if fn_module == '__main__':
            import inspect
            try:
                fn_file = inspect.getfile(fn)
            except TypeError:
                pass

        # Write subprocess arguments
        args_path = os.path.join(shm_dir, '_args.pkl')
        results_path = os.path.join(shm_dir, '_results.pkl')
        with open(args_path, 'wb') as f:
            pickle.dump({
                'shm_meta': shm_meta,
                'items': items,
                'max_workers': max_workers,
                'fn_module': fn_module,
                'fn_name': fn.__qualname__,
                'fn_file': fn_file,
                'filter_none': filter_none,
                'unpack': unpack,
                'log_every': log_every,
                'desc': desc,
                'results_path': results_path,
                'parent_argv': sys.argv,
            }, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Launch lean subprocess
        proc = subprocess.Popen(
            [sys.executable, '-m', 'ebt.parallel', '--subprocess', args_path],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        for line in proc.stdout:
            log.info("  %s: %s", desc, line.rstrip())
        proc.wait()

        if proc.returncode != 0:
            raise RuntimeError(
                f"pmap subprocess failed with code {proc.returncode}")

        # Load results — lightweight pickle + memmap'd numpy arrays
        with open(results_path, 'rb') as f:
            results_raw = pickle.load(f)
        # Reconstruct numpy arrays from memmap files (reads into memory
        # before shm_dir cleanup in finally block)
        results = [(item, _internalize(result)) for item, result in results_raw]

        elapsed = time.time() - t0
        log.info("%s: %d results from %d items in %.1fs (lean subprocess)",
                 desc, len(results), n, elapsed)
        return results

    finally:
        shutil.rmtree(shm_dir, ignore_errors=True)


def pmap_values(fn, items, **kwargs):
    """Like pmap but returns only the results (not the input items).

    Example:
        signals = pmap_values(run_config, configs, max_workers=50)
    """
    return [result for _, result in pmap(fn, items, **kwargs)]


def pmap_unzip(fn, items, **kwargs):
    """Like pmap but returns (items_list, results_list) separately.

    Example:
        valid_configs, signals = pmap_unzip(run_config, configs, max_workers=50)
        configs_df = pd.DataFrame(valid_configs)
    """
    pairs = pmap(fn, items, **kwargs)
    if not pairs:
        return [], []
    items_out, results_out = zip(*pairs)
    return list(items_out), list(results_out)


# ---------------------------------------------------------------------------
# Module entry point for subprocess
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--subprocess':
        _subprocess_main()
