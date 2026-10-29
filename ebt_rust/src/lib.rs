use pyo3::prelude::*;

mod backtest;
mod combine_positions;
mod execution;
pub mod outlier;
pub mod rows;


use crate::backtest::{ColumnInput, backtest_columns, backtest_multi_shift, backtest_matrix, backtest_matrix_multi_shift, backtest_stacked_summed, backtest_stacked, backtest_stacked_multi_shift_summed, holding_curve, correlation_to_trend};
use numpy::IntoPyArray;

fn verr(msg: String) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(msg)
}

fn check_len(name: &str, actual: usize, expected: usize) -> PyResult<()> {
    if actual != expected {
        return Err(verr(format!("{}: expected length {}, got {}", name, expected, actual)));
    }
    Ok(())
}

fn check_sorted(name: &str, ts: &[i64]) -> PyResult<()> {
    if !backtest::is_sorted_ts(ts) {
        return Err(verr(format!("{} must be sorted (non-decreasing) by time", name)));
    }
    Ok(())
}

/// Like `check_sorted`, over the rows the detector will actually see: invalid rows
/// (e.g. NaT, which arrives as i64::MIN) are skipped rather than failing the check.
fn check_sorted_valid(name: &str, ts: &[i64], valid: &[bool]) -> PyResult<()> {
    let mut prev: Option<i64> = None;
    for (i, &t) in ts.iter().enumerate() {
        if !valid[i] {
            continue;
        }
        if let Some(p) = prev {
            if t < p {
                return Err(verr(format!(
                    "{} must be sorted (non-decreasing) by time; row {} goes backwards", name, i)));
            }
        }
        prev = Some(t);
    }
    Ok(())
}

/// Shape + ordering checks shared by the column-major matrix kernels. Without them a
/// mismatch panics inside rayon (a PanicException, which `except Exception` does not
/// catch) and an unsorted index silently returns wrong numbers from the as-of merge.
fn check_matrix(
    pos_ts: &[i64], pos_flat: &[f64], pos_n_rows: usize,
    price_ts: &[i64], price_flat: &[f64], price_n_rows: usize, n_cols: usize,
) -> PyResult<()> {
    check_len("pos_timestamps", pos_ts.len(), pos_n_rows)?;
    check_len("price_timestamps", price_ts.len(), price_n_rows)?;
    check_len("pos_data", pos_flat.len(), pos_n_rows * n_cols)?;
    check_len("price_data", price_flat.len(), price_n_rows * n_cols)?;
    check_sorted("pos_timestamps", pos_ts)?;
    check_sorted("price_timestamps", price_ts)?;
    Ok(())
}

/// Per-column (names, timestamps, values) lists -> ColumnInput, with length and ordering checks.
fn build_columns(
    names: Vec<String>,
    pos_timestamps_list: Vec<numpy::PyReadonlyArray1<i64>>,
    pos_values_list: Vec<numpy::PyReadonlyArray1<f64>>,
    price_timestamps_list: Vec<numpy::PyReadonlyArray1<i64>>,
    price_values_list: Vec<numpy::PyReadonlyArray1<f64>>,
) -> PyResult<Vec<ColumnInput>> {
    let n = names.len();
    check_len("pos_timestamps_list", pos_timestamps_list.len(), n)?;
    check_len("pos_values_list", pos_values_list.len(), n)?;
    check_len("price_timestamps_list", price_timestamps_list.len(), n)?;
    check_len("price_values_list", price_values_list.len(), n)?;
    let mut out = Vec::with_capacity(n);
    for (i, name) in names.into_iter().enumerate() {
        let pt = pos_timestamps_list[i].as_slice().map_err(|e| verr(format!("{}", e)))?;
        let pv = pos_values_list[i].as_slice().map_err(|e| verr(format!("{}", e)))?;
        let prt = price_timestamps_list[i].as_slice().map_err(|e| verr(format!("{}", e)))?;
        let prv = price_values_list[i].as_slice().map_err(|e| verr(format!("{}", e)))?;
        check_len(&format!("pos_values[{}]", name), pv.len(), pt.len())?;
        check_len(&format!("price_values[{}]", name), prv.len(), prt.len())?;
        check_sorted(&format!("pos_timestamps[{}]", name), pt)?;
        check_sorted(&format!("price_timestamps[{}]", name), prt)?;
        out.push(ColumnInput {
            name,
            pos_timestamps: pt.to_vec(),
            pos_values: pv.to_vec(),
            price_timestamps: prt.to_vec(),
            price_values: prv.to_vec(),
        });
    }
    Ok(out)
}

/// Compute backtest for multiple columns in parallel (Rust implementation).
///
/// Args:
///     names: list of column name strings
///     pos_timestamps_list: list of 1-d int64 numpy arrays (nanosecond timestamps, non-NaN only)
///     pos_values_list: list of 1-d float64 numpy arrays (non-NaN position values)
///     price_timestamps_list: list of 1-d int64 numpy arrays (nanosecond timestamps, non-NaN only)
///     price_values_list: list of 1-d float64 numpy arrays (non-NaN price values)
///     shift: execution lag in bars
///     return_variable: 0=pnl, 1=pos, 2=market_return
///
/// Returns:
///     dict mapping column name -> (timestamps_array, values_array)
#[pyfunction]
#[pyo3(signature = (names, pos_timestamps_list, pos_values_list, price_timestamps_list, price_values_list, shift, return_variable))]
fn backtest_core(
    py: Python<'_>,
    names: Vec<String>,
    pos_timestamps_list: Vec<numpy::PyReadonlyArray1<i64>>,
    pos_values_list: Vec<numpy::PyReadonlyArray1<f64>>,
    price_timestamps_list: Vec<numpy::PyReadonlyArray1<i64>>,
    price_values_list: Vec<numpy::PyReadonlyArray1<f64>>,
    shift: i64,
    return_variable: u8,
) -> PyResult<PyObject> {
    let columns = build_columns(names, pos_timestamps_list, pos_values_list, price_timestamps_list, price_values_list)?;

    let results = py.allow_threads(|| backtest_columns(columns, shift, return_variable));

    let dict = pyo3::types::PyDict::new_bound(py);
    for r in results {
        let ts_arr = r.timestamps.into_pyarray_bound(py);
        let vals_arr = r.values.into_pyarray_bound(py);
        let tuple = pyo3::types::PyTuple::new_bound(py, &[ts_arr.as_any(), vals_arr.as_any()]);
        dict.set_item(r.name, tuple)?;
    }
    Ok(dict.into_py(py))
}

/// Batch backtest across multiple shift values (merge_asof done once).
///
/// Returns:
///     dict mapping shift (int) -> dict mapping column name -> (timestamps_array, values_array)
#[pyfunction]
#[pyo3(signature = (names, pos_timestamps_list, pos_values_list, price_timestamps_list, price_values_list, shifts))]
fn backtest_multi_shift_core(
    py: Python<'_>,
    names: Vec<String>,
    pos_timestamps_list: Vec<numpy::PyReadonlyArray1<i64>>,
    pos_values_list: Vec<numpy::PyReadonlyArray1<f64>>,
    price_timestamps_list: Vec<numpy::PyReadonlyArray1<i64>>,
    price_values_list: Vec<numpy::PyReadonlyArray1<f64>>,
    shifts: Vec<i64>,
) -> PyResult<PyObject> {
    let columns = build_columns(names, pos_timestamps_list, pos_values_list, price_timestamps_list, price_values_list)?;

    let results = py.allow_threads(|| backtest_multi_shift(columns, &shifts));

    let outer_dict = pyo3::types::PyDict::new_bound(py);
    for (shift, outputs) in results {
        let inner_dict = pyo3::types::PyDict::new_bound(py);
        for r in outputs {
            let ts_arr = r.timestamps.into_pyarray_bound(py);
            let vals_arr = r.values.into_pyarray_bound(py);
            let tuple = pyo3::types::PyTuple::new_bound(py, &[ts_arr.as_any(), vals_arr.as_any()]);
            inner_dict.set_item(r.name, tuple)?;
        }
        outer_dict.set_item(shift, inner_dict)?;
    }
    Ok(outer_dict.into_py(py))
}

/// Fast backtest on 2D numpy arrays (avoids per-column Python overhead).
///
/// Takes full position/price matrices with NaNs; Rust handles NaN filtering internally.
/// Returns a 2D numpy array [price_n_rows x n_cols].
#[pyfunction]
#[pyo3(signature = (pos_timestamps, pos_data, price_timestamps, price_data, shift, return_variable, pos_n_rows, price_n_rows, n_cols))]
fn backtest_matrix_core(
    py: Python<'_>,
    pos_timestamps: numpy::PyReadonlyArray1<i64>,
    pos_data: numpy::PyReadonlyArray1<f64>,    // flattened column-major
    price_timestamps: numpy::PyReadonlyArray1<i64>,
    price_data: numpy::PyReadonlyArray1<f64>,  // flattened column-major
    shift: i64,
    return_variable: u8,
    pos_n_rows: usize,
    price_n_rows: usize,
    n_cols: usize,
) -> PyResult<PyObject> {
    let pos_ts = pos_timestamps.as_slice().map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
    let price_ts = price_timestamps.as_slice().map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
    let pos_flat = pos_data.as_slice().map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
    let price_flat = price_data.as_slice().map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
    check_matrix(pos_ts, pos_flat, pos_n_rows, price_ts, price_flat, price_n_rows, n_cols)?;

    let result = py.allow_threads(|| {
        backtest_matrix(
            pos_ts, pos_flat, pos_n_rows,
            price_ts, price_flat, price_n_rows,
            n_cols, shift, return_variable,
        )
    });

    let py_arr = numpy::PyArray1::from_vec_bound(py, result);
    Ok(py_arr.into_py(py))
}

/// Fast multi-shift backtest on 2D numpy arrays.
/// Returns a 3D numpy array [n_shifts x price_n_rows x n_cols].
#[pyfunction]
#[pyo3(signature = (pos_timestamps, pos_data, price_timestamps, price_data, shifts, pos_n_rows, price_n_rows, n_cols))]
fn backtest_matrix_multi_shift_core(
    py: Python<'_>,
    pos_timestamps: numpy::PyReadonlyArray1<i64>,
    pos_data: numpy::PyReadonlyArray1<f64>,    // flattened column-major
    price_timestamps: numpy::PyReadonlyArray1<i64>,
    price_data: numpy::PyReadonlyArray1<f64>,  // flattened column-major
    shifts: Vec<i64>,
    pos_n_rows: usize,
    price_n_rows: usize,
    n_cols: usize,
) -> PyResult<PyObject> {
    let pos_ts = pos_timestamps.as_slice().map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
    let price_ts = price_timestamps.as_slice().map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
    let pos_flat = pos_data.as_slice().map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
    let price_flat = price_data.as_slice().map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
    check_matrix(pos_ts, pos_flat, pos_n_rows, price_ts, price_flat, price_n_rows, n_cols)?;

    let result = py.allow_threads(|| {
        backtest_matrix_multi_shift(
            pos_ts, pos_flat, pos_n_rows,
            price_ts, price_flat, price_n_rows,
            n_cols, &shifts,
        )
    });

    let py_arr = numpy::PyArray1::from_vec_bound(py, result);
    Ok(py_arr.into_py(py))
}

/// Compute Sharpe ratios for 4 position variants (exbias, bias, long, short) in one Rust call.
/// Returns [exbias, bias, long, short] as a numpy array.
#[pyfunction]
#[pyo3(signature = (pos_timestamps, pos_data, price_timestamps, price_data, shift, resample_daily, pos_n_rows, price_n_rows, n_cols, flat_rtol = 0.0))]
fn variant_sharpes_core(
    py: Python<'_>,
    pos_timestamps: numpy::PyReadonlyArray1<i64>,
    pos_data: numpy::PyReadonlyArray1<f64>,
    price_timestamps: numpy::PyReadonlyArray1<i64>,
    price_data: numpy::PyReadonlyArray1<f64>,
    shift: i64,
    resample_daily: bool,
    pos_n_rows: usize,
    price_n_rows: usize,
    n_cols: usize,
    flat_rtol: f64,
) -> PyResult<PyObject> {
    let pos_ts = pos_timestamps.as_slice().map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
    let price_ts = price_timestamps.as_slice().map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
    let pos_flat = pos_data.as_slice().map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
    let price_flat = price_data.as_slice().map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
    check_matrix(pos_ts, pos_flat, pos_n_rows, price_ts, price_flat, price_n_rows, n_cols)?;

    let result = py.allow_threads(|| {
        backtest::variant_sharpes_matrix(
            pos_ts, pos_flat, pos_n_rows,
            price_ts, price_flat, price_n_rows,
            n_cols, shift, resample_daily, flat_rtol,
        )
    });

    let py_arr = numpy::PyArray1::from_vec_bound(py, result);
    Ok(py_arr.into_py(py))
}

/// Compute Sharpe ratio per execution lag directly in Rust (no intermediate PnL matrix).
/// Returns a 1D numpy array of Sharpe ratios (one per shift).
#[pyfunction]
#[pyo3(signature = (pos_timestamps, pos_data, price_timestamps, price_data, shifts, resample_daily, pos_n_rows, price_n_rows, n_cols, flat_rtol = 0.0))]
fn sharpe_per_lag_core(
    py: Python<'_>,
    pos_timestamps: numpy::PyReadonlyArray1<i64>,
    pos_data: numpy::PyReadonlyArray1<f64>,
    price_timestamps: numpy::PyReadonlyArray1<i64>,
    price_data: numpy::PyReadonlyArray1<f64>,
    shifts: Vec<i64>,
    resample_daily: bool,
    pos_n_rows: usize,
    price_n_rows: usize,
    n_cols: usize,
    flat_rtol: f64,
) -> PyResult<PyObject> {
    let pos_ts = pos_timestamps.as_slice().map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
    let price_ts = price_timestamps.as_slice().map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
    let pos_flat = pos_data.as_slice().map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
    let price_flat = price_data.as_slice().map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
    check_matrix(pos_ts, pos_flat, pos_n_rows, price_ts, price_flat, price_n_rows, n_cols)?;

    let result = py.allow_threads(|| {
        backtest::sharpe_per_lag_matrix(
            pos_ts, pos_flat, pos_n_rows,
            price_ts, price_flat, price_n_rows,
            n_cols, &shifts, resample_daily, flat_rtol,
        )
    });

    let py_arr = numpy::PyArray1::from_vec_bound(py, result);
    Ok(py_arr.into_py(py))
}

/// Backtest from stacked/long-format arrays — everything done in Rust.
/// Contract IDs are pre-encoded integers; names maps id -> display name.
/// Returns dict mapping contract_name -> (timestamps_array, values_array).
#[pyfunction]
#[pyo3(signature = (names, pos_ids, pos_timestamps, pos_values, price_ids, price_timestamps, price_values, shift, return_variable))]
fn backtest_stacked_core(
    py: Python<'_>,
    names: Vec<String>,
    pos_ids: numpy::PyReadonlyArray1<i32>,
    pos_timestamps: numpy::PyReadonlyArray1<i64>,
    pos_values: numpy::PyReadonlyArray1<f64>,
    price_ids: numpy::PyReadonlyArray1<i32>,
    price_timestamps: numpy::PyReadonlyArray1<i64>,
    price_values: numpy::PyReadonlyArray1<f64>,
    shift: i64,
    return_variable: u8,
) -> PyResult<PyObject> {
    let pos_id_s = pos_ids.as_slice().map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
    let pos_ts = pos_timestamps.as_slice().map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
    let pos_vals = pos_values.as_slice().map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
    let price_id_s = price_ids.as_slice().map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
    let price_ts = price_timestamps.as_slice().map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
    let price_vals = price_values.as_slice().map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
    check_len("pos_timestamps", pos_ts.len(), pos_id_s.len())?;
    check_len("pos_values", pos_vals.len(), pos_id_s.len())?;
    check_len("price_timestamps", price_ts.len(), price_id_s.len())?;
    check_len("price_values", price_vals.len(), price_id_s.len())?;

    let results = py.allow_threads(|| {
        backtest_stacked(&names, pos_id_s, pos_ts, pos_vals, price_id_s, price_ts, price_vals, shift, return_variable)
    }).map_err(verr)?;

    let dict = pyo3::types::PyDict::new_bound(py);
    for r in results {
        let ts_arr = r.timestamps.into_pyarray_bound(py);
        let vals_arr = r.values.into_pyarray_bound(py);
        let tuple = pyo3::types::PyTuple::new_bound(py, &[ts_arr.as_any(), vals_arr.as_any()]);
        dict.set_item(r.name, tuple)?;
    }
    Ok(dict.into_py(py))
}

/// Backtest from stacked arrays, summed across contracts.
/// Returns (timestamps_array, values_array) with per-timestamp sums.
#[pyfunction]
#[pyo3(signature = (names, pos_ids, pos_timestamps, pos_values, price_ids, price_timestamps, price_values, shift, return_variable))]
fn backtest_stacked_summed_core(
    py: Python<'_>,
    names: Vec<String>,
    pos_ids: numpy::PyReadonlyArray1<i32>,
    pos_timestamps: numpy::PyReadonlyArray1<i64>,
    pos_values: numpy::PyReadonlyArray1<f64>,
    price_ids: numpy::PyReadonlyArray1<i32>,
    price_timestamps: numpy::PyReadonlyArray1<i64>,
    price_values: numpy::PyReadonlyArray1<f64>,
    shift: i64,
    return_variable: u8,
) -> PyResult<PyObject> {
    let pos_id_s = pos_ids.as_slice().map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
    let pos_ts = pos_timestamps.as_slice().map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
    let pos_vals = pos_values.as_slice().map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
    let price_id_s = price_ids.as_slice().map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
    let price_ts = price_timestamps.as_slice().map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
    let price_vals = price_values.as_slice().map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
    check_len("pos_timestamps", pos_ts.len(), pos_id_s.len())?;
    check_len("pos_values", pos_vals.len(), pos_id_s.len())?;
    check_len("price_timestamps", price_ts.len(), price_id_s.len())?;
    check_len("price_values", price_vals.len(), price_id_s.len())?;

    let (out_ts, out_vals) = py.allow_threads(|| {
        backtest_stacked_summed(&names, pos_id_s, pos_ts, pos_vals, price_id_s, price_ts, price_vals, shift, return_variable)
    }).map_err(verr)?;

    let ts_arr = out_ts.into_pyarray_bound(py);
    let vals_arr = out_vals.into_pyarray_bound(py);
    let tuple = pyo3::types::PyTuple::new_bound(py, &[ts_arr.as_any(), vals_arr.as_any()]);
    Ok(tuple.into_py(py))
}

/// Multi-shift backtest from stacked arrays, summed across contracts.
/// Returns dict mapping shift -> (timestamps_array, values_array).
#[pyfunction]
#[pyo3(signature = (names, pos_ids, pos_timestamps, pos_values, price_ids, price_timestamps, price_values, shifts))]
fn backtest_stacked_multi_shift_summed_core(
    py: Python<'_>,
    names: Vec<String>,
    pos_ids: numpy::PyReadonlyArray1<i32>,
    pos_timestamps: numpy::PyReadonlyArray1<i64>,
    pos_values: numpy::PyReadonlyArray1<f64>,
    price_ids: numpy::PyReadonlyArray1<i32>,
    price_timestamps: numpy::PyReadonlyArray1<i64>,
    price_values: numpy::PyReadonlyArray1<f64>,
    shifts: Vec<i64>,
) -> PyResult<PyObject> {
    let pos_id_s = pos_ids.as_slice().map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
    let pos_ts = pos_timestamps.as_slice().map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
    let pos_vals = pos_values.as_slice().map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
    let price_id_s = price_ids.as_slice().map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
    let price_ts = price_timestamps.as_slice().map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
    let price_vals = price_values.as_slice().map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
    check_len("pos_timestamps", pos_ts.len(), pos_id_s.len())?;
    check_len("pos_values", pos_vals.len(), pos_id_s.len())?;
    check_len("price_timestamps", price_ts.len(), price_id_s.len())?;
    check_len("price_values", price_vals.len(), price_id_s.len())?;

    let results = py.allow_threads(|| {
        backtest_stacked_multi_shift_summed(&names, pos_id_s, pos_ts, pos_vals, price_id_s, price_ts, price_vals, &shifts)
    }).map_err(verr)?;

    let dict = pyo3::types::PyDict::new_bound(py);
    for (shift, out_ts, out_vals) in results {
        let ts_arr = out_ts.into_pyarray_bound(py);
        let vals_arr = out_vals.into_pyarray_bound(py);
        let tuple = pyo3::types::PyTuple::new_bound(py, &[ts_arr.as_any(), vals_arr.as_any()]);
        dict.set_item(shift, tuple)?;
    }
    Ok(dict.into_py(py))
}

/// Compute holding curve: sharpe and holding_days for each EWM smoothing span.
/// Returns list of (sharpe, holding_days) tuples.
#[pyfunction]
#[pyo3(signature = (pos_timestamps, pos_data, price_timestamps, price_data, shift, ewm_spans, pos_n_rows, price_n_rows, n_cols, flat_rtol = 0.0))]
fn holding_curve_core(
    py: Python<'_>,
    pos_timestamps: numpy::PyReadonlyArray1<i64>,
    pos_data: numpy::PyReadonlyArray1<f64>,
    price_timestamps: numpy::PyReadonlyArray1<i64>,
    price_data: numpy::PyReadonlyArray1<f64>,
    shift: i64,
    ewm_spans: Vec<f64>,
    pos_n_rows: usize,
    price_n_rows: usize,
    n_cols: usize,
    flat_rtol: f64,
) -> PyResult<PyObject> {
    let pos_ts = pos_timestamps.as_slice().map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
    let price_ts = price_timestamps.as_slice().map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
    let pos_flat = pos_data.as_slice().map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
    let price_flat = price_data.as_slice().map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
    check_matrix(pos_ts, pos_flat, pos_n_rows, price_ts, price_flat, price_n_rows, n_cols)?;

    let results = py.allow_threads(|| {
        holding_curve(
            pos_ts, pos_flat, pos_n_rows,
            price_ts, price_flat, price_n_rows,
            n_cols, shift, &ewm_spans, flat_rtol,
        )
    });

    let list = pyo3::types::PyList::empty_bound(py);
    for r in results {
        let tuple = pyo3::types::PyTuple::new_bound(py, &[r.sharpe.into_py(py), r.holding_days.into_py(py)]);
        list.append(tuple)?;
    }
    Ok(list.into_py(py))
}

/// Compute correlation to trend: correlation of strategy PnL to trend-following at various filter lengths.
/// Returns list of correlation values.
#[pyfunction]
#[pyo3(signature = (pos_timestamps, pos_data, price_timestamps, price_data, shift, filter_lengths, pos_n_rows, price_n_rows, n_cols))]
fn correlation_to_trend_core(
    py: Python<'_>,
    pos_timestamps: numpy::PyReadonlyArray1<i64>,
    pos_data: numpy::PyReadonlyArray1<f64>,
    price_timestamps: numpy::PyReadonlyArray1<i64>,
    price_data: numpy::PyReadonlyArray1<f64>,
    shift: i64,
    filter_lengths: Vec<f64>,
    pos_n_rows: usize,
    price_n_rows: usize,
    n_cols: usize,
) -> PyResult<PyObject> {
    let pos_ts = pos_timestamps.as_slice().map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
    let price_ts = price_timestamps.as_slice().map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
    let pos_flat = pos_data.as_slice().map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
    let price_flat = price_data.as_slice().map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
    check_matrix(pos_ts, pos_flat, pos_n_rows, price_ts, price_flat, price_n_rows, n_cols)?;

    let results = py.allow_threads(|| {
        correlation_to_trend(
            pos_ts, pos_flat, pos_n_rows,
            price_ts, price_flat, price_n_rows,
            n_cols, shift, &filter_lengths,
        )
    });

    let py_arr = numpy::PyArray1::from_vec_bound(py, results);
    Ok(py_arr.into_py(py))
}

/// Fused combine_positions: ffill → diff → scatter → cumsum in a single pass per df.
///
/// Args:
///   target_timestamps: 1D i64 array of target timestamps (sorted, nanoseconds)
///   df_timestamps_list: list of 1D i64 arrays, one per input df
///   df_values_list: list of 1D f64 arrays (row-major flattened), one per input df
///   df_n_rows_list: list of int, number of rows per df
///   df_n_cols_list: list of int, number of columns per df
///   df_col_indices_list: list of 1D u32 arrays mapping source columns → target column indices
///   n_target_cols: total number of columns in result
///   is_mean: whether to compute mean instead of sum
///
/// Returns: 1D f64 array (row-major flattened result, n_target * n_target_cols)
#[pyfunction]
#[pyo3(signature = (target_timestamps, df_timestamps_list, df_values_list, df_n_rows_list, df_n_cols_list, df_col_indices_list, n_target_cols, is_mean))]
fn combine_positions_core(
    py: Python<'_>,
    target_timestamps: numpy::PyReadonlyArray1<i64>,
    df_timestamps_list: Vec<numpy::PyReadonlyArray1<i64>>,
    df_values_list: Vec<numpy::PyReadonlyArray1<f64>>,
    df_n_rows_list: Vec<usize>,
    df_n_cols_list: Vec<usize>,
    df_col_indices_list: Vec<numpy::PyReadonlyArray1<u32>>,
    n_target_cols: usize,
    is_mean: bool,
) -> PyResult<PyObject> {
    let target_ts = target_timestamps.as_slice()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
    check_sorted("target_timestamps", target_ts)?;
    let n_dfs = df_timestamps_list.len();
    check_len("df_values_list", df_values_list.len(), n_dfs)?;
    check_len("df_n_rows_list", df_n_rows_list.len(), n_dfs)?;
    check_len("df_n_cols_list", df_n_cols_list.len(), n_dfs)?;
    check_len("df_col_indices_list", df_col_indices_list.len(), n_dfs)?;

    // Build DfInput list from Python arrays
    let mut dfs = Vec::with_capacity(n_dfs);
    for i in 0..n_dfs {
        let ts = df_timestamps_list[i].as_slice()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?
            .to_vec();
        let vals = df_values_list[i].as_slice()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?
            .to_vec();
        let col_indices = df_col_indices_list[i].as_slice()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?
            .to_vec();
        check_len(&format!("df_timestamps_list[{}]", i), ts.len(), df_n_rows_list[i])?;
        check_len(&format!("df_values_list[{}]", i), vals.len(), df_n_rows_list[i] * df_n_cols_list[i])?;
        check_len(&format!("df_col_indices_list[{}]", i), col_indices.len(), df_n_cols_list[i])?;
        check_sorted(&format!("df_timestamps_list[{}]", i), &ts)?;
        if let Some(&c) = col_indices.iter().find(|&&c| c as usize >= n_target_cols) {
            return Err(verr(format!("df_col_indices_list[{}]: column index {} >= n_target_cols {}", i, c, n_target_cols)));
        }
        // Every df stamp must be a target stamp (the kernel scatters trades onto the target
        // row with the same stamp; the equal-length fast path assumes identical stamps).
        let mut ti = 0usize;
        for &t in &ts {
            while ti < target_ts.len() && target_ts[ti] < t { ti += 1; }
            if ti == target_ts.len() || target_ts[ti] != t {
                return Err(verr(format!("df_timestamps_list[{}]: timestamp {} is not in target_timestamps", i, t)));
            }
        }
        dfs.push(combine_positions::DfInput {
            timestamps: ts,
            values: vals,
            n_rows: df_n_rows_list[i],
            n_cols: df_n_cols_list[i],
            col_indices,
        });
    }

    let (result, first_valid, count_starts) = py.allow_threads(|| {
        combine_positions::combine_positions_core(
            target_ts, &dfs, n_target_cols, is_mean,
        )
    });

    let py_result = numpy::PyArray1::from_vec_bound(py, result);
    let py_first_valid = numpy::PyArray1::from_vec_bound(
        py, first_valid.into_iter().map(|x| x as u64).collect::<Vec<_>>(),
    );
    let py_count_starts = numpy::PyArray1::from_vec_bound(py, count_starts);
    Ok((py_result, py_first_valid, py_count_starts).into_py(py))
}


/// Simulate VWAP execution on column-major 2D numpy arrays.
///
/// Input: timestamps (i64 ns), target positions (f64 col-major), volume (f64 col-major),
///        n_rows, n_cols, window_ns (i64 nanoseconds).
/// Returns: 1D f64 array (col-major flattened result, n_rows * n_cols).
#[pyfunction]
#[pyo3(signature = (timestamps, target_data, volume_data, n_rows, n_cols, window_ns))]
fn simulate_vwap_execution_core(
    py: Python<'_>,
    timestamps: numpy::PyReadonlyArray1<i64>,
    target_data: numpy::PyReadonlyArray1<f64>,
    volume_data: numpy::PyReadonlyArray1<f64>,
    n_rows: usize,
    n_cols: usize,
    window_ns: i64,
) -> PyResult<PyObject> {
    let ts = timestamps.as_slice()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
    let target = target_data.as_slice()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
    let volume = volume_data.as_slice()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
    check_len("timestamps", ts.len(), n_rows)?;
    check_len("target_data", target.len(), n_rows * n_cols)?;
    check_len("volume_data", volume.len(), n_rows * n_cols)?;
    check_sorted("timestamps", ts)?;

    let result = py.allow_threads(|| {
        execution::simulate_vwap_matrix(ts, target, volume, n_rows, n_cols, window_ns)
    });

    let py_arr = numpy::PyArray1::from_vec_bound(py, result);
    Ok(py_arr.into_py(py))
}

// ---------------------------------------------------------------------------
// Outlier detection
// ---------------------------------------------------------------------------

fn valid_mask(valid: Option<numpy::PyReadonlyArray1<bool>>, n: usize) -> PyResult<Vec<bool>> {
    match valid {
        Some(v) => {
            let v = v.as_slice()?.to_vec();
            check_len("valid", v.len(), n)?;
            Ok(v)
        }
        None => Ok(vec![true; n]),
    }
}

fn debug_dict<'py>(py: Python<'py>, cols: Vec<(&str, Vec<f64>)>, accepted: Vec<bool>) -> PyResult<PyObject> {
    let dict = pyo3::types::PyDict::new_bound(py);
    dict.set_item("accepted", accepted.into_pyarray_bound(py))?;
    for (name, v) in cols {
        dict.set_item(name, v.into_pyarray_bound(py))?;
    }
    Ok(dict.into_py(py))
}

fn ensemble_debug_cols(d: crate::outlier::DebugBuffers) -> Vec<(&'static str, Vec<f64>)> {
    vec![
        ("level", d.level), ("scale", d.scale), ("u", d.u), ("tau", d.tau),
        ("variance_rate", d.variance_rate), ("zscore", d.zscore),
        ("iqr_lower", d.iqr_lower), ("iqr_upper", d.iqr_upper),
        ("price_change_pct", d.price_change_pct), ("time_gap_seconds", d.time_gap_seconds),
        ("volume_change", d.volume_change), ("votes_against", d.votes_against),
        ("cand_level", d.cand_level), ("cand_n_confirmed", d.cand_n_confirmed),
        ("regime_switch", d.regime_switch),
    ]
}

#[allow(clippy::too_many_arguments)]
fn ensemble_config(
    zscore_threshold: Option<f64>, iqr_multiplier: Option<f64>, min_votes_to_reject: Option<usize>,
    window_size: Option<usize>, merge_stat_votes: Option<bool>, gap_widen_threshold_sec: Option<f64>,
    warmup_threshold: Option<usize>, min_scale: Option<f64>, min_scale_ticks: Option<f64>,
    n_switch: Option<usize>, min_switch_seconds: Option<f64>, warmup_hard_limit: Option<f64>,
) -> crate::outlier::OutlierConfig {
    let mut c = crate::outlier::OutlierConfig::default();
    if let Some(v) = zscore_threshold { c.zscore_threshold = v; }
    if let Some(v) = iqr_multiplier { c.iqr_multiplier = v; }
    if let Some(v) = min_votes_to_reject { c.min_votes_to_reject = v; }
    if let Some(v) = window_size { c.window_size = v; }
    if let Some(v) = merge_stat_votes { c.merge_stat_votes = v; }
    if let Some(v) = gap_widen_threshold_sec { c.gap_widen_threshold_sec = v; }
    if let Some(v) = warmup_threshold { c.warmup_threshold = v; }
    if let Some(v) = min_scale { c.min_scale = v; }
    if let Some(v) = min_scale_ticks { c.min_scale_ticks = v; }
    if let Some(v) = n_switch { c.n_switch = v; }
    if let Some(v) = min_switch_seconds { c.min_switch_seconds = v; }
    if let Some(v) = warmup_hard_limit { c.warmup_hard_limit = v; }
    c
}

/// Ensemble outlier detection on trades (adaptive level filter, rolling z-score,
/// IQR, price jump, time gap, volume consistency; a row is rejected when at
/// least `min_votes_to_reject` methods vote against it).
///
/// Args:
///     time_ns: int64 nanosecond timestamps, sorted
///     price, volume: float64 arrays
///     valid: optional bool mask; False rows are treated as missing (not accepted,
///            detector state unchanged)
///     return_debug: return a dict of numpy arrays (accepted + 15 debug columns)
///
/// Returns: numpy bool array (True = accepted), or the debug dict.
#[pyfunction]
#[pyo3(signature = (time_ns, price, volume, valid = None, zscore_threshold = None, iqr_multiplier = None,
                    min_votes_to_reject = None, window_size = None, merge_stat_votes = None,
                    gap_widen_threshold_sec = None, warmup_threshold = None, min_scale = None,
                    min_scale_ticks = None, n_switch = None, min_switch_seconds = None,
                    warmup_hard_limit = None, return_debug = false))]
#[allow(clippy::too_many_arguments)]
fn detect_outliers(
    py: Python<'_>,
    time_ns: numpy::PyReadonlyArray1<i64>,
    price: numpy::PyReadonlyArray1<f64>,
    volume: numpy::PyReadonlyArray1<f64>,
    valid: Option<numpy::PyReadonlyArray1<bool>>,
    zscore_threshold: Option<f64>,
    iqr_multiplier: Option<f64>,
    min_votes_to_reject: Option<usize>,
    window_size: Option<usize>,
    merge_stat_votes: Option<bool>,
    gap_widen_threshold_sec: Option<f64>,
    warmup_threshold: Option<usize>,
    min_scale: Option<f64>,
    min_scale_ticks: Option<f64>,
    n_switch: Option<usize>,
    min_switch_seconds: Option<f64>,
    warmup_hard_limit: Option<f64>,
    return_debug: bool,
) -> PyResult<PyObject> {
    let t = time_ns.as_slice()?;
    let p = price.as_slice()?;
    let v = volume.as_slice()?;
    check_len("price", p.len(), t.len())?;
    check_len("volume", v.len(), t.len())?;
    let valid = valid_mask(valid, t.len())?;
    check_sorted_valid("time_ns", t, &valid)?;
    let config = ensemble_config(zscore_threshold, iqr_multiplier, min_votes_to_reject, window_size,
                                 merge_stat_votes, gap_widen_threshold_sec, warmup_threshold,
                                 min_scale, min_scale_ticks, n_switch, min_switch_seconds, warmup_hard_limit);
    if return_debug {
        let r = py.allow_threads(|| crate::outlier::detect_outliers_arrays_debug(t, p, v, &valid, config));
        debug_dict(py, ensemble_debug_cols(r.debug), r.accepted)
    } else {
        let r = py.allow_threads(|| crate::outlier::detect_outliers_arrays(t, p, v, &valid, config));
        Ok(r.into_pyarray_bound(py).into_py(py))
    }
}

/// Quote outlier detection: mid-price level filter cascaded with a spread
/// upper-bound filter.
///
/// Args:
///     time_ns: int64 nanosecond timestamps, sorted
///     bid, ask: float64 arrays
///     valid: optional bool mask of usable rows
///     return_debug: return a dict of numpy arrays (accepted + 15 debug columns)
///
/// Returns: numpy bool array (True = accepted), or the debug dict.
#[pyfunction]
#[pyo3(signature = (time_ns, bid, ask, valid = None, gap_widen_threshold_sec = None, mid_tau_min = None,
                    mid_tau_max = None, mid_n_switch = None, spread_max_consecutive_reject = None,
                    max_spread = None, warmup_count = None, allow_crossed = None, mid_min_scale = None,
                    mid_min_scale_ticks = None, mid_min_switch_seconds = None, spread_min_scale = None,
                    spread_min_scale_ticks = None, warmup_hard_limit = None, return_debug = false))]
#[allow(clippy::too_many_arguments)]
fn detect_outliers_quotes(
    py: Python<'_>,
    time_ns: numpy::PyReadonlyArray1<i64>,
    bid: numpy::PyReadonlyArray1<f64>,
    ask: numpy::PyReadonlyArray1<f64>,
    valid: Option<numpy::PyReadonlyArray1<bool>>,
    gap_widen_threshold_sec: Option<f64>,
    mid_tau_min: Option<f64>,
    mid_tau_max: Option<f64>,
    mid_n_switch: Option<usize>,
    spread_max_consecutive_reject: Option<usize>,
    max_spread: Option<f64>,
    warmup_count: Option<usize>,
    allow_crossed: Option<bool>,
    mid_min_scale: Option<f64>,
    mid_min_scale_ticks: Option<f64>,
    mid_min_switch_seconds: Option<f64>,
    spread_min_scale: Option<f64>,
    spread_min_scale_ticks: Option<f64>,
    warmup_hard_limit: Option<f64>,
    return_debug: bool,
) -> PyResult<PyObject> {
    let t = time_ns.as_slice()?;
    let b = bid.as_slice()?;
    let a = ask.as_slice()?;
    check_len("bid", b.len(), t.len())?;
    check_len("ask", a.len(), t.len())?;
    let valid = valid_mask(valid, t.len())?;
    check_sorted_valid("time_ns", t, &valid)?;
    let mut c = crate::outlier::QuoteOutlierConfig::default();
    if let Some(v) = gap_widen_threshold_sec { c.gap_widen_threshold_sec = v; }
    if let Some(v) = mid_tau_min { c.mid_tau_min = v; }
    if let Some(v) = mid_tau_max { c.mid_tau_max = v; }
    if let Some(v) = mid_n_switch { c.mid_n_switch = v; }
    if let Some(v) = spread_max_consecutive_reject { c.spread_max_consecutive_reject = v; }
    if let Some(v) = max_spread { c.max_spread = v; }
    if let Some(v) = warmup_count { c.warmup_count = v; }
    if let Some(v) = allow_crossed { c.allow_crossed = v; }
    if let Some(v) = mid_min_scale { c.mid_min_scale = v; }
    if let Some(v) = mid_min_scale_ticks { c.mid_min_scale_ticks = v; }
    if let Some(v) = mid_min_switch_seconds { c.mid_min_switch_seconds = v; }
    if let Some(v) = spread_min_scale { c.spread_min_scale = v; }
    if let Some(v) = spread_min_scale_ticks { c.spread_min_scale_ticks = v; }
    if let Some(v) = warmup_hard_limit { c.warmup_hard_limit = v; }
    if return_debug {
        let r = py.allow_threads(|| crate::outlier::detect_outliers_quotes_arrays_debug(t, b, a, &valid, c));
        let d = r.debug;
        debug_dict(py, vec![
            ("mid_level", d.mid_level), ("mid_scale", d.mid_scale), ("mid_u", d.mid_u),
            ("mid_tau", d.mid_tau), ("mid_variance_rate", d.mid_variance_rate),
            ("mid_cand_level", d.mid_cand_level), ("mid_cand_n_confirmed", d.mid_cand_n_confirmed),
            ("mid_regime_switch", d.mid_regime_switch), ("spread_level", d.spread_level),
            ("spread_scale", d.spread_scale), ("spread_u", d.spread_u), ("spread_tau", d.spread_tau),
            ("spread_variance_rate", d.spread_variance_rate), ("spread_upper_bound", d.spread_upper_bound),
            ("gate", d.gate),
        ], r.accepted)
    } else {
        let r = py.allow_threads(|| crate::outlier::detect_outliers_quotes_arrays(t, b, a, &valid, c));
        Ok(r.into_pyarray_bound(py).into_py(py))
    }
}

/// OHLCV bar outlier detection: bar-consistency checks (non-finite prices,
/// high < low) followed by the trade ensemble on the close price.
///
/// Args:
///     time_ns: int64 nanosecond timestamps, sorted
///     open, high, low, close, volume: float64 arrays
///     valid: optional bool mask of usable rows
///     return_debug: return a dict of numpy arrays (accepted + 15 ensemble + 6 bar columns)
///
/// Returns: numpy bool array (True = accepted), or the debug dict.
#[pyfunction]
#[pyo3(signature = (time_ns, open, high, low, close, volume, valid = None, zscore_threshold = None,
                    iqr_multiplier = None, min_votes_to_reject = None, window_size = None,
                    merge_stat_votes = None, gap_widen_threshold_sec = None, warmup_threshold = None,
                    min_scale = None, min_scale_ticks = None, n_switch = None, min_switch_seconds = None,
                    warmup_hard_limit = None, return_debug = false))]
#[allow(clippy::too_many_arguments)]
fn detect_outliers_ohlcv(
    py: Python<'_>,
    time_ns: numpy::PyReadonlyArray1<i64>,
    open: numpy::PyReadonlyArray1<f64>,
    high: numpy::PyReadonlyArray1<f64>,
    low: numpy::PyReadonlyArray1<f64>,
    close: numpy::PyReadonlyArray1<f64>,
    volume: numpy::PyReadonlyArray1<f64>,
    valid: Option<numpy::PyReadonlyArray1<bool>>,
    zscore_threshold: Option<f64>,
    iqr_multiplier: Option<f64>,
    min_votes_to_reject: Option<usize>,
    window_size: Option<usize>,
    merge_stat_votes: Option<bool>,
    gap_widen_threshold_sec: Option<f64>,
    warmup_threshold: Option<usize>,
    min_scale: Option<f64>,
    min_scale_ticks: Option<f64>,
    n_switch: Option<usize>,
    min_switch_seconds: Option<f64>,
    warmup_hard_limit: Option<f64>,
    return_debug: bool,
) -> PyResult<PyObject> {
    let t = time_ns.as_slice()?;
    let (o, h, l, c, v) = (open.as_slice()?, high.as_slice()?, low.as_slice()?, close.as_slice()?, volume.as_slice()?);
    for (name, x) in [("open", o), ("high", h), ("low", l), ("close", c), ("volume", v)] {
        check_len(name, x.len(), t.len())?;
    }
    let valid = valid_mask(valid, t.len())?;
    check_sorted_valid("time_ns", t, &valid)?;
    let config = ensemble_config(zscore_threshold, iqr_multiplier, min_votes_to_reject, window_size,
                                 merge_stat_votes, gap_widen_threshold_sec, warmup_threshold,
                                 min_scale, min_scale_ticks, n_switch, min_switch_seconds, warmup_hard_limit);
    if return_debug {
        let r = py.allow_threads(|| crate::outlier::detect_outliers_ohlcv_arrays_debug(t, o, h, l, c, v, &valid, config));
        let mut cols = ensemble_debug_cols(r.inner_debug);
        let b = r.ohlcv_debug;
        cols.extend([
            ("bar_open", b.bar_open), ("bar_high", b.bar_high), ("bar_low", b.bar_low),
            ("bar_close", b.bar_close), ("bar_volume", b.bar_volume), ("bar_invalid_reason", b.bar_invalid_reason),
        ]);
        debug_dict(py, cols, r.accepted)
    } else {
        let r = py.allow_threads(|| crate::outlier::detect_outliers_ohlcv_arrays(t, o, h, l, c, v, &valid, config));
        Ok(r.into_pyarray_bound(py).into_py(py))
    }
}

/// Python module definition
#[pymodule]
fn ebt_rust(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(backtest_core, m)?)?;
    m.add_function(wrap_pyfunction!(backtest_multi_shift_core, m)?)?;
    m.add_function(wrap_pyfunction!(backtest_matrix_core, m)?)?;
    m.add_function(wrap_pyfunction!(backtest_matrix_multi_shift_core, m)?)?;
    m.add_function(wrap_pyfunction!(sharpe_per_lag_core, m)?)?;
    m.add_function(wrap_pyfunction!(variant_sharpes_core, m)?)?;
    m.add_function(wrap_pyfunction!(backtest_stacked_core, m)?)?;
    m.add_function(wrap_pyfunction!(backtest_stacked_summed_core, m)?)?;
    m.add_function(wrap_pyfunction!(backtest_stacked_multi_shift_summed_core, m)?)?;
    m.add_function(wrap_pyfunction!(holding_curve_core, m)?)?;
    m.add_function(wrap_pyfunction!(correlation_to_trend_core, m)?)?;
    m.add_function(wrap_pyfunction!(combine_positions_core, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_vwap_execution_core, m)?)?;
    m.add_function(wrap_pyfunction!(detect_outliers, m)?)?;
    m.add_function(wrap_pyfunction!(detect_outliers_quotes, m)?)?;
    m.add_function(wrap_pyfunction!(detect_outliers_ohlcv, m)?)?;
    Ok(())
}
