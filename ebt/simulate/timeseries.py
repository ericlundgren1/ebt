from numpy.lib.stride_tricks import as_strided as stride
from functools import reduce
import pandas as pd
import numpy as np
import tqdm


def drop_duplicates_index(df):
    return df[~df.index.duplicated()]


def undict(df, columns=None, sep='.'):
    '''
    Turns 
            a       b
        0   foo     {'a': {'d': 'e'}}
        1   baz    {'foo': 'bar'}
        
    into 
    
            a     b.a.d  b.foo
        0   foo   e      NaN
        1   baz  NaN    bar
        
    '''
    nm = df.columns.name
    columns = columns or df.apply(lambda s: s.apply(type) == dict).any().pipe(lambda x: x.index[x])
    for c in columns:
        df = pd.concat([df.drop(c, axis=1), pd.json_normalize(df[c]).set_axis(df.index).add_prefix(f'{c}{sep}')], axis=1)
    return df.rename_axis(nm, axis=1)


def rolling(df, max_lb=1280, min_lb=None, step=1):
    min_lb = min_lb or max_lb
    it = iter(df.expanding())
    first_valid_i = None
    for i, _df in enumerate(it):
        if i > min_lb - 2:
            first_valid_i = first_valid_i or i
            if (i - first_valid_i) % step == 0:
                yield _df.index[-1], _df.iloc[-max_lb:]


def add_index(df, k, axis=0, **kwargs):
    return pd.concat([df], keys=[k], axis=axis, **kwargs)


def apply_to_secondary_index(df, f, progress_bar=True):
    progress = tqdm.tqdm if progress_bar else lambda x: x
    return pd.concat([f(v.loc[k]).pipe(add_index, k, df.index.names[0]) for k, v in progress(df.groupby(level=0))])


def merge_asof(s_or_df, ix, **merge_asof_kwargs):
    '''
    Create a series with index ix with the data that was available at the time from s_or_df
    '''
    if isinstance(s_or_df, (pd.Series, pd.DataFrame)):
        assert s_or_df.index.is_monotonic_increasing, "s_or_df index must be sorted in increasing order for merge_asof"
    if isinstance(ix, (pd.Series, pd.DataFrame)):
        assert ix.index.is_monotonic_increasing, "ix index must be sorted in increasing order for merge_asof"
    elif isinstance(ix, pd.Index):
        assert ix.is_monotonic_increasing, "ix must be sorted in increasing order for merge_asof"
    ix = ix.copy()
    s_or_df = s_or_df.copy()
    drop_index_placeholder = False
    
    if type(s_or_df) == pd.Series:
        if s_or_df.name is None:
            s_or_df.name = 'placeholder-s'

    if isinstance(ix, pd.Index):
        ix = pd.Series(index=ix, name=s_or_df.name)
        drop_index_placeholder = True
        
    merge_asof_default_kwargs = {
        'suffixes': ('_ix', ''),
        'left_index': True, 
        'right_index': True
    }
    merge_asof_default_kwargs.update(**merge_asof_kwargs)

    merged = pd.merge_asof(ix, s_or_df, **merge_asof_default_kwargs)

    if type(s_or_df) == pd.Series:
        return merged.iloc[:, 1]

    if drop_index_placeholder:
        return merged.drop('placeholder-ix', axis=1)
        
    return merged
