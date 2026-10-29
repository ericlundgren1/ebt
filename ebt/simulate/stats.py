try:
    from numba import jit
except ModuleNotFoundError:
    print('Optional dependency numba, install to use ebt.simulate.stats')

import pandas as pd
import numpy as np
import logging

_LOGGER = logging.getLogger(__name__)


def calculate_z_score(df, mean_f=None, std_f=None, **ewm_kwargs):
    mean_f = mean_f or (lambda x: x.ewm(**ewm_kwargs).mean())
    std_f = std_f or (lambda x: x.ewm(**ewm_kwargs).std())
    return (df - df.pipe(mean_f)) / df.pipe(std_f)


def logistic(x, l=1, k=1, x0=0):
    z = np.clip(-k * (x - x0), -500, 500)  # prevent overflow in exp
    ne = np.exp(z)
    return l / (1 + ne)


def standard_centered_logistic(x):
    return logistic(x, l=2) - 1


@jit(nopython=True)
def _stop_break(target_positions, threshold=0.01):
    last_real_value = target_positions[0]
    results = [last_real_value]
    for target_v in target_positions[1:]:
        delta = abs(target_v - last_real_value)
        if delta > threshold:
            last_real_value = target_v
            results.append(target_v)
        else:
            results.append(np.nan)
    return results


def stop_break(target_positions, threshold=0.01):
    if type(target_positions) == pd.Series:
        target_positions_no_na = target_positions.dropna()
        return pd.Series(_stop_break(target_positions_no_na.values, threshold=threshold), index=target_positions_no_na.index)
    elif type(target_positions) == pd.DataFrame:
        return target_positions.apply(stop_break, threshold=threshold)


def shrink_cov(cov, shrinkage=0.5):
    '''
    Function for shrinking a covariance matrix. If shrinkage=0 that means no shrinkage is applied.
    '''
    return (1 - shrinkage) * cov + shrinkage * np.eye(cov.shape[1]) * cov


def nearestPD(A):
    def isPD(B):
        """Returns true when input is positive-definite, via Cholesky"""
        try:
            _ = la.cholesky(B)
            return True
        except la.LinAlgError:
            return False

    import numpy as np
    from numpy import linalg as la
    """Find the nearest positive-definite matrix to input
    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].
    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = la.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(la.norm(A))
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(la.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3


def yang_zhang(ohlc, smooth=None):
    '''
    Expects the columns to be a multiindex where the top level columns are called open, high, low and close.
    '''
    ohlc = ohlc.copy()

    assert 'close' in ohlc.columns, 'Top level column should be contain columns open high low close'

    
    smooth = (lambda x: x.ewm(com=60, adjust=False, min_periods=30).mean()) if smooth is None else smooth
    if ohlc.loc[:, ['open', 'high', 'low']].isnull().any().any():
        _LOGGER.warning(f'Found missing values in open, high or low prices, replacing missing values with close prices')

    ohlc = ohlc.apply(lambda s: s.ffill() if s.name[0] == 'close' else s) \
        .pipe(lambda x: x.apply(lambda s: s.fillna(x['close'][s.name[1]]) if s.name[0] != 'close' else s))
    
    op = ohlc.loc[:, 'open']
    hi = ohlc.loc[:, 'high']
    lo = ohlc.loc[:, 'low']
    cl = ohlc.loc[:, 'close']

    day = (0.5 * (hi - lo) ** 2 - (2 * np.log(2) - 1) * (cl - op) ** 2)
    night = ((op - cl.shift()) ** 2).iloc[1:]
    night.loc[day.index[0]] = 0
    var = (day + night).ffill()
    z_var = smooth(var).replace(0, np.nan).ffill()
    return z_var ** 0.5
    