from sklearn.model_selection import train_test_split
from ebt.simulate.timeseries import rolling
from ebt.simulate.model import all_models
from functools import partial
from tqdm.auto import tqdm

import pandas as pd
import numpy as np
import logging
import math

_LOGGER = logging.getLogger(__name__)

# Row chunk for the preprocessor leak check — bounds the comparison temporary
# instead of materializing the whole overlap twice (see predict()).
_CHECK_CHUNK_ROWS = 4096


def generator(df, max_lb='100D', min_lb='10D', embargo='1D', retrain_interval='5D', use_progress=False):
    '''
    Generates slices out of df.

    Each slice has a training window and a prediction window as well as an optional embargo window.

    The training window has a minimum duration of min_lb and maximum duration of max_lb. The embargo
    period (if any) sits in between the training window and the subsequent prediction window. The
    prediction window stretches to the start of the prediction window of the next slice.

    The slice is made up of a tuple consisting of
    - a dataframe with the whole training, embargo and prediction window in it
    - a timestamp called retrain_date which specifies where the data for the training window stops and the embargo begins.
        The timestamp can be viewed as the embargo start time (inclusive). Meaning that it's not part of the training data.
    - a timestamp caled prediction_start date which denotes where the embargo period stops and the prediction period starts
    '''
    progress = tqdm if use_progress else lambda x: x
    max_lb = pd.Timedelta(max_lb) if type(max_lb) == str else max_lb
    min_lb = pd.Timedelta(min_lb) if type(min_lb) == str else min_lb
    embargo = pd.Timedelta(embargo) if type(embargo) == str else embargo
    retrain_interval = pd.Timedelta(retrain_interval) if type(retrain_interval) == str else retrain_interval

    index = df.index
    n = len(index)
    retrain_dates = pd.date_range(start=index[0] + min_lb, end=index[-1], freq=retrain_interval)

    # Pre-compute all boundary positions via vectorized searchsorted
    embargo_end_ts = retrain_dates + embargo
    window_end_ts = retrain_dates + embargo + retrain_interval
    embargo_end_pos = index.searchsorted(embargo_end_ts, side='right')  # first index > embargo_end
    window_end_pos = index.searchsorted(window_end_ts, side='right')   # exclusive end for iloc

    start_date = index[0]
    for i, retrain_date in enumerate(progress(retrain_dates)):
        if retrain_date - start_date > max_lb:
            start_date = retrain_date - max_lb
        if retrain_date - start_date >= min_lb:
            if embargo_end_pos[i] < n:
                start_pos = index.searchsorted(start_date, side='left')
                end_pos = min(window_end_pos[i], n)
                train_embargo_predict = df.iloc[start_pos:end_pos]
                prediction_start_date = index[embargo_end_pos[i]]
                yield train_embargo_predict, retrain_date, prediction_start_date



def enough_data(x_in=None, 
                y_in=None, 
                x_fraction_present_in_column_full=None, # Fraction of values needed to be present for a column to be considered (0, 1)
                x_number_present_per_row=None, # Number of values needed to be present for a row to be considered (0, x.shape[1])
                x_fraction_present_per_row=None, # Fraction of values needed to be present for a row to be considered (0, 1)
                x_window=None, # Rolling window size (2, x.shape[1])
                x_fraction_present_per_row_inwindow=None, # Fraction of values needed to be present within window_x (0, 1)

                y_fraction_present_in_column_full=None, 
                y_number_present_per_row=None,
                y_fraction_present_per_row=None,
                y_window=None,
                y_fraction_present_per_row_inwindow=None,

                fill=True):
    '''
    """
    Filters input feature and label DataFrames based on data completeness criteria.

    This function is used to ensure that enough data is present in the input features (x) and labels (y)
    before proceeding with model training or prediction. It applies a series of filters to remove columns
    and rows that do not meet specified thresholds for data presence.

    Parameters
    ----------
    x : pd.DataFrame, optional
        Feature matrix to filter. If None, returns a partial function with the provided arguments.
    y : pd.DataFrame, optional
        Label matrix to filter.
    x_fraction_present_in_column_full : float, optional
        Minimum fraction of non-NaN values required for a column in x to be kept.
    x_number_present_per_row : int, optional
        Minimum number of non-NaN values required for a row in x to be kept.
    x_fraction_present_per_row : float, optional
        Minimum fraction of non-NaN values required for a row in x to be kept.
    x_window : int, optional
        Rolling window size for evaluating x_fraction_present_per_row_inwindow.
    x_fraction_present_per_row_inwindow : float, optional
        Minimum fraction of non-NaN values required within the rolling window for a row in x to be kept.
    y_fraction_present_in_column_full : float, optional
        Minimum fraction of non-NaN values required for a column in y to be kept.
    y_number_present_per_row : int, optional
        Minimum number of non-NaN values required for a row in y to be kept.
    y_fraction_present_per_row : float, optional
        Minimum fraction of non-NaN values required for a row in y to be kept.
    y_window : int, optional
        Rolling window size for evaluating y_fraction_present_per_row_inwindow.
    y_fraction_present_per_row_inwindow : float, optional
        Minimum fraction of non-NaN values required within the rolling window for a row in y to be kept.
    fill : bool or int, default True
        Whether to forward-fill and fill remaining NaNs with 0 after filtering. If int, it is used as a limit.

    Returns
    -------
    tuple
        (x_filtered, y_filtered) where x_filtered and y_filtered are the filtered DataFrames.
        If y is None, returns (x_filtered, None).
        If x is None, returns a partial function with the provided arguments.

    Notes
    -----
    - This function is typically used as a preprocessing step before model training or prediction.
    - If called with x=None, returns a partial function for later use.
    '''
    
    if x_in is None:
        return partial(enough_data, **{k: v for k, v in locals().items() if v is not None})
    
    _LOGGER.debug(f'x_fraction_present_in_column_full: {x_fraction_present_in_column_full}, y_fraction_present_in_column_full: {y_fraction_present_in_column_full}')

    x = x_in.copy()
    y = y_in

    if x_number_present_per_row is not None:
        x = x.loc[np.isfinite(x).sum(axis=1) >= x_number_present_per_row, :]
    if x_fraction_present_in_column_full is not None:
        x = x.loc[:, np.isfinite(x).mean() > x_fraction_present_in_column_full]
    if x_fraction_present_per_row is not None:
        x = x.loc[np.isfinite(x).mean(axis=1) > x_fraction_present_per_row, :]
    if x_window is not None:
        if x_fraction_present_per_row_inwindow is None:
            x_fraction_present_per_row_inwindow = 0
        x = x.loc[np.isfinite(x).sum(axis=1) > np.isfinite(x).sum(axis=1).rolling(x_window).mean() * x_fraction_present_per_row_inwindow, :]

    if y_in is not None:
        y = y_in.copy()
        if y_fraction_present_per_row is not None:
            y = y.loc[np.isfinite(y).mean(axis=1) > y_fraction_present_per_row, :]
        if y_fraction_present_in_column_full is not None:
            y = y.loc[:, np.isfinite(y).mean() > y_fraction_present_in_column_full]
        if y_number_present_per_row is not None:
            y = y.loc[np.isfinite(y).sum(axis=1) > y_number_present_per_row, :]
        if y_window is not None:
            if y_fraction_present_per_row_inwindow is None:
                y_fraction_present_per_row_inwindow = 0
            y = y.loc[np.isfinite(y).sum(axis=1) > np.isfinite(y).sum(axis=1).rolling(y_window).mean() * y_fraction_present_per_row_inwindow, :]

    if fill:
        limit = None if fill == True else fill
        x = x.ffill(limit=limit).fillna(0)
        y = y.fillna(0) if y is not None else y

    return x, y


def _fit_predict_one(args):
    """Fit one model and return predictions. Top-level for picklability."""
    model, x_train, y_train, x_predict, prediction_start_dt = args
    return model.fit(x_train, y_train).predict(x_predict, y_train)[prediction_start_dt:]


def flatten_predictions(predictions_dict):
    '''
    Concatenates the nested {train_end: {pred_end: df}} dict from predict() into a single DataFrame.
    Predictions are already sliced to [prediction_start_dt:] at creation, so we just concat all frames.
    '''
    frames = []
    for train_end_dt, predict_dict in predictions_dict.items():
        for predict_end_dt, df in predict_dict.items():
            frames.append(df)
    if not frames:
        return pd.DataFrame(index=pd.DatetimeIndex([]))
    # Drop `frames` before the sort, sort in place, and select with a
    # positional take to bound peak memory. Duplicate timestamps: last wins.
    out = pd.concat(frames)
    del frames
    # NOTE: no `kind=` — with duplicate timestamps the sort's stability
    # decides which row keep='last' selects.
    out.sort_index(inplace=True)
    dup = out.index.duplicated(keep='last')
    if dup.any():
        out = out.iloc[np.flatnonzero(~dup)]
    return out


def predict(features: pd.DataFrame,
            labels: pd.DataFrame,
            models=None,
            prediction_horizon=2,
            max_lookback=None,
            min_lookback='10D',
            retrain_interval='30D',
            embargo='10D',
            preprocess_training_data=None,
            minimum_required_points=64,
            turbo=False,
            use_string_keys=False,
            update_model_state=False,
            predictions_dict=None,
            parallel=False,
            verbose=False,
            progressbar=True,
            log=True,
            **kwargs):
    '''Walk-forward prediction: every ``retrain_interval`` each model is refit on the
    trailing ``min_lookback``..``max_lookback`` window and predicts forward, with an
    ``embargo`` gap between the training window and the first predicted bar.'''

    assert features.index.is_monotonic_increasing, 'features index must be sorted'
    assert labels.index.is_monotonic_increasing, 'labels index must be sorted'

    min_lookback = pd.Timedelta(f'{min_lookback}D' if type(min_lookback) == int else min_lookback)
    retrain_interval = pd.Timedelta(f'{retrain_interval}D' if type(retrain_interval) == int else retrain_interval)
    embargo = pd.Timedelta(f'{embargo}D' if type(embargo) == int else embargo)
    max_lookback = pd.Timedelta(f'{max_lookback}D' if type(max_lookback) == int else max_lookback)
    max_lookback = pd.Timedelta((features.index[-1] - features.index[0]) if max_lookback is None else max_lookback)

    models = models or all_models
    predictions_dict = {} if predictions_dict is None else predictions_dict

    training_sets = generator(features, max_lb=max_lookback, min_lb=min_lookback, retrain_interval=retrain_interval, embargo=embargo)
    if turbo:
        n_turbo = 2 if type(turbo) is bool else turbo
        training_sets = list(training_sets)[-n_turbo:]

    progress = (lambda x: tqdm(x, total=math.ceil((features.index[-1] - features.index[0] - min_lookback - embargo) / retrain_interval))) if progressbar else lambda x: x

    # How many windows the preprocessor leak check inspects (see the check
    # itself). A leaky preprocessor leaks on the first window, so checking
    # every window is unnecessary. 0 disables, as does
    # SUPPRESS_PREPROCESSOR_CHECK=True.
    _check_budget = int(kwargs.get('PREPROCESSOR_CHECK_WINDOWS', 2))
    _checks_done = 0

    for i, (x_window, embargo_start_dt, prediction_start_dt) in enumerate(progress(training_sets)):
        x_train = x_window[:embargo_start_dt].iloc[:-1]
        y_train = labels.loc[x_train.index[0]:x_train.index[-1]].shift(-prediction_horizon)

        # Drop rows where y_train is all-NaN (caused by shift) to avoid training on fabricated zeros
        x_train, y_train = x_train.align(y_train, join='inner', axis=0)
        valid_mask = y_train.notna().any(axis=1)
        x_train = x_train.loc[valid_mask]
        y_train = y_train.loc[valid_mask]

        x_train, y_train = preprocess_training_data(x_train, y_train) if preprocess_training_data is not None else (x_train, y_train)
        x_predict, _ = preprocess_training_data(x_window) if preprocess_training_data is not None else (x_window, None)

        if (not kwargs.get('SUPPRESS_PREPROCESSOR_CHECK', False)
                and _checks_done < _check_budget):
            # To bound cost: (1) run only on the first
            # PREPROCESSOR_CHECK_WINDOWS windows — a preprocessor that uses
            # future info does so on the first window; (2) compare in row
            # chunks via positional take, so the temporary is bounded by the
            # chunk and a mismatch short-circuits.
            _checks_done += 1
            x_pred_overlap = x_predict[:embargo_start_dt].iloc[:-1]
            common_cols = x_pred_overlap.columns.intersection(x_train.columns)
            common_idx = x_pred_overlap.index.intersection(x_train.index)
            if not common_cols.equals(x_train.columns) or not common_cols.equals(x_pred_overlap.columns):
                diff_cols = x_pred_overlap.columns.symmetric_difference(x_train.columns)
                _LOGGER.warning(f'Preprocessor produced different columns for train vs predict: {list(diff_cols)}')
            if len(common_cols) > 0 and len(common_idx) > 0:
                _a_rows = x_pred_overlap.index.get_indexer(common_idx)
                _b_rows = x_train.index.get_indexer(common_idx)
                _a_cols = x_pred_overlap.columns.get_indexer(common_cols)
                _b_cols = x_train.columns.get_indexer(common_cols)
                _av, _bv = x_pred_overlap.values, x_train.values
                _leaky = False
                for _s in range(0, len(common_idx), _CHECK_CHUNK_ROWS):
                    _sl = slice(_s, _s + _CHECK_CHUNK_ROWS)
                    if not np.allclose(_av[np.ix_(_a_rows[_sl], _a_cols)],
                                       _bv[np.ix_(_b_rows[_sl], _b_cols)],
                                       atol=1e-10, equal_nan=True):
                        _leaky = True
                        break
                if _leaky:
                    _LOGGER.warning('Prediction preprocessor uses future info, use the kwarg SUPPRESS_PREPROCESSOR_CHECK=True to silence this warning.')

        # .copy() is deliberate: the predict-side preprocessor runs over the
        # WHOLE train+embargo+predict window, of which only the tail is used
        # here. Under pandas Copy-on-Write this slice is a VIEW that would pin
        # the entire preprocessed window alive for the whole fit. Copying the
        # tail releases the rest. Values are unchanged.
        #
        # NOT done here: preprocessing the
        # window once and slicing BOTH train and predict out of it. The
        # preprocessor is range-dependent — enough_data's column filter keeps a
        # different column set for a train slice than for the full window — so
        # that would feed window-level statistics into training, i.e. exactly
        # the lookahead the check above exists to detect.
        x_predict = x_predict[embargo_start_dt:].copy()

        if x_predict.shape[0] == 0:
            _LOGGER.info(f'Not training on window: predict window empty after embargo + preprocessing') if log else None
            continue

        if x_train.shape[0] < minimum_required_points:
            _LOGGER.info(f'Not training on window due to low amount of observations on features: {x_train.shape[0]}') if log else None
            continue

        if y_train.shape[1] < 1:
            _LOGGER.info(f'Not training on window due to low amount of columns on labels: {y_train.shape[1]}') if log else None
            continue
        
        if y_train.shape[0] < minimum_required_points:
            _LOGGER.info(f'Not training on window due to low amount of observations on labels: {y_train.shape[0]}') if log else None
            continue

        _LOGGER.info(f'Training on {x_train.shape} datapoints from {x_train.index[0]} to {x_train.index[-1]}, predicting {y_train.shape[1]} labels on {x_predict.shape} datapoints between {x_predict.index[0]} ({prediction_start_dt} excluding embargo) and {x_predict.index[-1]}') if log else None

        if parallel:
            from ebt.parallel import pmap

            items = [(m, x_train, y_train, x_predict, prediction_start_dt) for m in models]
            pairs = pmap(_fit_predict_one, items, max_workers=len(models),
                         filter_none=False, desc="models")

            for item, pred in pairs:
                model = item[0]
                key = str(model) if use_string_keys else model
                if key not in predictions_dict:
                    predictions_dict[key] = {}
                train_end = x_train.index[-1]
                if train_end not in predictions_dict[key]:
                    predictions_dict[key][train_end] = {}
                predictions_dict[key][train_end][x_predict.index[-1]] = pred
        else:
            for model in models:
                key = str(model) if use_string_keys else model
                if key not in predictions_dict:
                    predictions_dict[key] = {}
                train_end = x_train.index[-1]
                if train_end not in predictions_dict[key]:
                    predictions_dict[key][train_end] = {}
                    model = model.fit(x_train, y_train)
                elif x_predict.index[-1] not in predictions_dict[key][train_end]:
                    # Boundary window: train_end cached but pred_end is new
                    # (data was extended). Refit to get a usable model.
                    model = model.fit(x_train, y_train)
                if x_predict.index[-1] not in predictions_dict[key][train_end]:
                    predictions_dict[key][train_end][x_predict.index[-1]] = model.predict(x_predict, y_train)[prediction_start_dt:]

    if update_model_state:
        for model in models:
            key = str(model) if use_string_keys else model
            model.state = predictions_dict[key]


    predictions_flattened = {
        m: flatten_predictions(p) for m, p in predictions_dict.items()
    }

    return predictions_flattened if not verbose else {'predictions_dict': predictions_dict, 'predictions_flattened': predictions_flattened}

