use rayon::prelude::*;

/// Input for a single column's backtest computation.
pub struct ColumnInput {
    pub name: String,
    pub pos_timestamps: Vec<i64>,  // nanosecond timestamps (non-NaN positions only)
    pub pos_values: Vec<f64>,
    pub price_timestamps: Vec<i64>,  // nanosecond timestamps (non-NaN prices only)
    pub price_values: Vec<f64>,
}

/// Output from a single column's backtest computation.
pub struct ColumnOutput {
    pub name: String,
    pub timestamps: Vec<i64>,
    pub values: Vec<f64>,
}

/// As-of merge: for each price timestamp, find the latest position timestamp <= it.
/// Returns aligned (timestamps, values) on the price timeline, starting from the first
/// price timestamp that has a valid position.
fn merge_asof_1d(
    pos_ts: &[i64],
    pos_vals: &[f64],
    price_ts: &[i64],
) -> (Vec<i64>, Vec<f64>) {
    if pos_ts.is_empty() || price_ts.is_empty() {
        return (Vec::new(), Vec::new());
    }

    let n = price_ts.len();
    let mut out_ts = Vec::with_capacity(n);
    let mut out_vals = Vec::with_capacity(n);

    let mut pos_idx: usize = 0;
    let mut started = false;

    for i in 0..n {
        let pt = price_ts[i];

        // Advance pos_idx to the last position timestamp <= pt
        while pos_idx + 1 < pos_ts.len() && pos_ts[pos_idx + 1] <= pt {
            pos_idx += 1;
        }

        if pos_ts[pos_idx] <= pt {
            started = true;
            out_ts.push(pt);
            out_vals.push(pos_vals[pos_idx]);
        } else if started {
            // Price timestamp is before current pos but we already started — shouldn't happen
            // with sorted data, but handle gracefully
            out_ts.push(pt);
            out_vals.push(f64::NAN);
        }
        // else: price timestamp is before first position, skip
    }

    (out_ts, out_vals)
}

/// Compute backtest for a single column.
/// return_variable: 0 = pnl, 1 = pos, 2 = market_return
fn backtest_column(
    pos_ts: &[i64],
    pos_vals: &[f64],
    price_ts: &[i64],
    price_vals: &[f64],
    shift: i64,
    return_variable: u8,
) -> (Vec<i64>, Vec<f64>) {
    // Step 1: merge_asof positions onto price timeline
    let (merged_ts, merged_pos) = merge_asof_1d(pos_ts, pos_vals, price_ts);

    if merged_ts.is_empty() {
        return (Vec::new(), Vec::new());
    }

    // We need to find where merged_ts starts in price_ts to compute returns correctly
    // merged_ts is a subset of price_ts starting from some offset
    let n = merged_ts.len();

    // Step 2: compute returns (price.diff()) aligned to merged_ts
    // Find the price values corresponding to merged_ts timestamps
    // Since merged_ts is a contiguous suffix of price_ts, find the offset
    let price_offset = price_ts.partition_point(|&t| t < merged_ts[0]);

    let mut returns = vec![f64::NAN; n];
    for i in 1..n {
        let price_cur = price_vals[price_offset + i];
        let price_prev = price_vals[price_offset + i - 1];
        returns[i] = price_cur - price_prev;
    }
    // First return needs the price before the merge window (if available)
    if price_offset > 0 {
        returns[0] = price_vals[price_offset] - price_vals[price_offset - 1];
    }

    // Step 3: shift positions
    // Step 4: multiply or return requested variable
    match return_variable {
        2 => {
            // market_return
            (merged_ts, returns)
        }
        1 => {
            // pos (shifted)
            let mut shifted_pos = vec![f64::NAN; n];
            if shift >= 0 {
                let s = shift as usize;
                for i in s..n {
                    shifted_pos[i] = merged_pos[i - s];
                }
            } else {
                let s = (-shift) as usize;
                if n > s {
                    for i in 0..n - s {
                        shifted_pos[i] = merged_pos[i + s];
                    }
                }
            }
            (merged_ts, shifted_pos)
        }
        _ => {
            // pnl = shifted_pos * returns
            let mut pnl = vec![f64::NAN; n];
            if shift >= 0 {
                let s = shift as usize;
                for i in s..n {
                    pnl[i] = merged_pos[i - s] * returns[i];
                }
            } else {
                let s = (-shift) as usize;
                if n > s {
                    for i in 0..n - s {
                        pnl[i] = merged_pos[i + s] * returns[i];
                    }
                }
            }
            (merged_ts, pnl)
        }
    }
}

/// Compute backtest for multiple columns in parallel.
pub fn backtest_columns(
    columns: Vec<ColumnInput>,
    shift: i64,
    return_variable: u8,
) -> Vec<ColumnOutput> {
    columns
        .into_par_iter()
        .map(|col| {
            let (ts, vals) = backtest_column(
                &col.pos_timestamps,
                &col.pos_values,
                &col.price_timestamps,
                &col.price_values,
                shift,
                return_variable,
            );
            ColumnOutput {
                name: col.name,
                timestamps: ts,
                values: vals,
            }
        })
        .collect()
}

/// Batch multiple shift values: merge_asof + diff done once, then shift+multiply per shift.
pub fn backtest_multi_shift(
    columns: Vec<ColumnInput>,
    shifts: &[i64],
) -> Vec<(i64, Vec<ColumnOutput>)> {
    // Pre-compute merged positions and returns for each column
    struct PreComputed {
        name: String,
        timestamps: Vec<i64>,
        merged_pos: Vec<f64>,
        returns: Vec<f64>,
    }

    let pre: Vec<PreComputed> = columns
        .into_par_iter()
        .map(|col| {
            let (merged_ts, merged_pos) =
                merge_asof_1d(&col.pos_timestamps, &col.pos_values, &col.price_timestamps);

            if merged_ts.is_empty() {
                return PreComputed {
                    name: col.name,
                    timestamps: Vec::new(),
                    merged_pos: Vec::new(),
                    returns: Vec::new(),
                };
            }

            let n = merged_ts.len();
            let price_offset = col.price_timestamps.partition_point(|&t| t < merged_ts[0]);

            let mut returns = vec![f64::NAN; n];
            for i in 1..n {
                returns[i] = col.price_values[price_offset + i] - col.price_values[price_offset + i - 1];
            }
            if price_offset > 0 {
                returns[0] = col.price_values[price_offset] - col.price_values[price_offset - 1];
            }

            PreComputed {
                name: col.name,
                timestamps: merged_ts,
                merged_pos,
                returns,
            }
        })
        .collect();

    // For each shift, compute pnl
    shifts
        .par_iter()
        .map(|&shift| {
            let outputs: Vec<ColumnOutput> = pre
                .iter()
                .map(|pc| {
                    let n = pc.timestamps.len();
                    if n == 0 {
                        return ColumnOutput {
                            name: pc.name.clone(),
                            timestamps: Vec::new(),
                            values: Vec::new(),
                        };
                    }

                    let mut pnl = vec![f64::NAN; n];
                    if shift >= 0 {
                        let s = shift as usize;
                        for i in s..n {
                            pnl[i] = pc.merged_pos[i - s] * pc.returns[i];
                        }
                    } else {
                        let s = (-shift) as usize;
                        if n > s {
                            for i in 0..n - s {
                                pnl[i] = pc.merged_pos[i + s] * pc.returns[i];
                            }
                        }
                    }

                    ColumnOutput {
                        name: pc.name.clone(),
                        timestamps: pc.timestamps.clone(),
                        values: pnl,
                    }
                })
                .collect();

            (shift, outputs)
        })
        .collect()
}

/// Check if a column has any NaN values.
#[inline]
fn has_nan(data: &[f64]) -> bool {
    data.iter().any(|v| v.is_nan())
}

/// Extract non-NaN timestamps and values from a column.
#[inline]
fn extract_non_nan(ts: &[i64], vals: &[f64]) -> (Vec<i64>, Vec<f64>) {
    let mut out_ts = Vec::with_capacity(vals.len());
    let mut out_vals = Vec::with_capacity(vals.len());
    for (i, &v) in vals.iter().enumerate() {
        if !v.is_nan() {
            out_ts.push(ts[i]);
            out_vals.push(v);
        }
    }
    (out_ts, out_vals)
}

/// Compute backtest result for a single column given non-NaN positions and prices.
/// Writes results directly into col_out (column-major slice of result).
fn backtest_column_into(
    col_pos_ts: &[i64],
    col_pos_vals: &[f64],
    col_price_ts: &[i64],
    col_price_vals: &[f64],
    price_ts: &[i64],
    price_n_rows: usize,
    shift: i64,
    return_variable: u8,
    col_out: &mut [f64],
) {
    if col_pos_ts.is_empty() || col_price_ts.is_empty() {
        return;
    }

    if col_pos_ts[0] > *col_price_ts.last().unwrap()
        || *col_pos_ts.last().unwrap() < col_price_ts[0]
    {
        return;
    }

    let (merged_ts, merged_pos) = merge_asof_1d(col_pos_ts, col_pos_vals, col_price_ts);
    if merged_ts.is_empty() {
        return;
    }

    let n = merged_ts.len();
    let price_offset = col_price_ts.partition_point(|&t| t < merged_ts[0]);

    // Compute returns
    let mut returns = vec![f64::NAN; n];
    for i in 1..n {
        returns[i] = col_price_vals[price_offset + i] - col_price_vals[price_offset + i - 1];
    }
    if price_offset > 0 {
        returns[0] = col_price_vals[price_offset] - col_price_vals[price_offset - 1];
    }

    // Map results back to price_ts grid while computing
    let mut price_row = 0;
    match return_variable {
        2 => {
            for (mi, &mt) in merged_ts.iter().enumerate() {
                while price_row < price_n_rows && price_ts[price_row] < mt { price_row += 1; }
                if price_row < price_n_rows && price_ts[price_row] == mt {
                    col_out[price_row] = returns[mi];
                }
            }
        }
        1 => {
            for (mi, &mt) in merged_ts.iter().enumerate() {
                while price_row < price_n_rows && price_ts[price_row] < mt { price_row += 1; }
                if price_row < price_n_rows && price_ts[price_row] == mt {
                    let src = if shift >= 0 {
                        let s = shift as usize;
                        if mi >= s { merged_pos[mi - s] } else { f64::NAN }
                    } else {
                        let s = (-shift) as usize;
                        if mi + s < n { merged_pos[mi + s] } else { f64::NAN }
                    };
                    col_out[price_row] = src;
                }
            }
        }
        _ => {
            for (mi, &mt) in merged_ts.iter().enumerate() {
                while price_row < price_n_rows && price_ts[price_row] < mt { price_row += 1; }
                if price_row < price_n_rows && price_ts[price_row] == mt {
                    let pos_val = if shift >= 0 {
                        let s = shift as usize;
                        if mi >= s { merged_pos[mi - s] } else { f64::NAN }
                    } else {
                        let s = (-shift) as usize;
                        if mi + s < n { merged_pos[mi + s] } else { f64::NAN }
                    };
                    col_out[price_row] = pos_val * returns[mi];
                }
            }
        }
    }
}

/// Fast backtest operating on column-major (Fortran-order) 2D matrices with NaNs.
/// pos_data and price_data are column-major: [col][row] = data[col * n_rows + row].
/// Returns a column-major result matrix [price_n_rows x n_cols].
pub fn backtest_matrix(
    pos_ts: &[i64],
    pos_data: &[f64],
    pos_n_rows: usize,
    price_ts: &[i64],
    price_data: &[f64],
    price_n_rows: usize,
    n_cols: usize,
    shift: i64,
    return_variable: u8,
) -> Vec<f64> {
    let mut result = vec![f64::NAN; price_n_rows * n_cols];

    // Split result into column slices for parallel write
    let col_slices: Vec<&mut [f64]> = result.chunks_mut(price_n_rows).collect();

    // Use scoped threads via rayon for mutable slices
    col_slices.into_par_iter().enumerate().for_each(|(col, col_out)| {
        let pos_col = &pos_data[col * pos_n_rows..(col + 1) * pos_n_rows];
        let price_col = &price_data[col * price_n_rows..(col + 1) * price_n_rows];

        // Avoid allocations when no NaNs
        let pos_has_nan = has_nan(pos_col);
        let price_has_nan = has_nan(price_col);

        if !pos_has_nan && !price_has_nan {
            // Fast path: no NaN filtering needed
            backtest_column_into(
                pos_ts, pos_col, price_ts, price_col,
                price_ts, price_n_rows, shift, return_variable, col_out,
            );
        } else {
            let (filt_pos_ts, filt_pos_vals) = if pos_has_nan {
                extract_non_nan(pos_ts, pos_col)
            } else {
                (pos_ts.to_vec(), pos_col.to_vec())
            };
            let (filt_price_ts, filt_price_vals) = if price_has_nan {
                extract_non_nan(price_ts, price_col)
            } else {
                (price_ts.to_vec(), price_col.to_vec())
            };
            backtest_column_into(
                &filt_pos_ts, &filt_pos_vals, &filt_price_ts, &filt_price_vals,
                price_ts, price_n_rows, shift, return_variable, col_out,
            );
        }
    });

    result
}

/// Batch multi-shift on column-major 2D matrices.
/// merge_asof + diff done once, shift+multiply per shift.
/// Returns flat column-major vec: [n_shifts][n_cols][price_n_rows].
pub fn backtest_matrix_multi_shift(
    pos_ts: &[i64],
    pos_data: &[f64],
    pos_n_rows: usize,
    price_ts: &[i64],
    price_data: &[f64],
    price_n_rows: usize,
    n_cols: usize,
    shifts: &[i64],
) -> Vec<f64> {
    let n_shifts = shifts.len();
    let sheet_size = price_n_rows * n_cols;

    struct ColPrecomputed {
        merged_pos: Vec<f64>,
        returns: Vec<f64>,
        row_indices: Vec<usize>,
    }

    let pre: Vec<ColPrecomputed> = (0..n_cols)
        .into_par_iter()
        .map(|col| {
            let pos_col = &pos_data[col * pos_n_rows..(col + 1) * pos_n_rows];
            let price_col = &price_data[col * price_n_rows..(col + 1) * price_n_rows];

            let pos_has_nan = has_nan(pos_col);
            let price_has_nan = has_nan(price_col);

            let (col_pos_ts, col_pos_vals) = if pos_has_nan {
                extract_non_nan(pos_ts, pos_col)
            } else {
                (pos_ts.to_vec(), pos_col.to_vec())
            };
            let (col_price_ts, col_price_vals) = if price_has_nan {
                extract_non_nan(price_ts, price_col)
            } else {
                (price_ts.to_vec(), price_col.to_vec())
            };

            if col_pos_ts.is_empty() || col_price_ts.is_empty()
                || col_pos_ts[0] > *col_price_ts.last().unwrap()
                || *col_pos_ts.last().unwrap() < col_price_ts[0]
            {
                return ColPrecomputed { merged_pos: Vec::new(), returns: Vec::new(), row_indices: Vec::new() };
            }

            let (merged_ts, merged_pos) = merge_asof_1d(&col_pos_ts, &col_pos_vals, &col_price_ts);
            if merged_ts.is_empty() {
                return ColPrecomputed { merged_pos: Vec::new(), returns: Vec::new(), row_indices: Vec::new() };
            }

            let n = merged_ts.len();
            let price_offset = col_price_ts.partition_point(|&t| t < merged_ts[0]);

            let mut returns = vec![f64::NAN; n];
            for i in 1..n { returns[i] = col_price_vals[price_offset + i] - col_price_vals[price_offset + i - 1]; }
            if price_offset > 0 { returns[0] = col_price_vals[price_offset] - col_price_vals[price_offset - 1]; }

            let mut row_indices = Vec::with_capacity(n);
            let mut price_row = 0;
            for &mt in &merged_ts {
                while price_row < price_n_rows && price_ts[price_row] < mt { price_row += 1; }
                row_indices.push(price_row);
            }

            ColPrecomputed { merged_pos, returns, row_indices }
        })
        .collect();

    // Parallelize over (shift, col) pairs for maximum parallelism on many-core machines
    let mut result = vec![f64::NAN; n_shifts * sheet_size];

    let tasks: Vec<(usize, usize)> = (0..n_shifts)
        .flat_map(|si| (0..n_cols).map(move |ci| (si, ci)))
        .collect();

    let partial_results: Vec<(usize, usize, Vec<(usize, f64)>)> = tasks
        .into_par_iter()
        .map(|(si, ci)| {
            let pc = &pre[ci];
            let shift = shifts[si];
            let n = pc.merged_pos.len();
            if n == 0 { return (si, ci, Vec::new()); }

            let mut entries = Vec::with_capacity(n);
            if shift >= 0 {
                let s = shift as usize;
                for i in s..n { entries.push((pc.row_indices[i], pc.merged_pos[i - s] * pc.returns[i])); }
            } else {
                let s = (-shift) as usize;
                if n > s { for i in 0..n - s { entries.push((pc.row_indices[i], pc.merged_pos[i + s] * pc.returns[i])); } }
            }
            (si, ci, entries)
        })
        .collect();

    for (si, ci, entries) in partial_results {
        let col_start = si * sheet_size + ci * price_n_rows;
        for (row, val) in entries {
            result[col_start + row] = val;
        }
    }

    result
}

/// Flat-day threshold: `rtol * median(|v|)` over the non-NaN entries of `vals`.
/// The median matches numpy's `np.median` (sort; middle element for odd n, mean of the
/// two middle elements for even n). Empty (or all-NaN) input, or `rtol == 0.0`, gives 0.0.
pub fn flat_threshold(vals: &[f64], rtol: f64) -> f64 {
    if rtol == 0.0 {
        return 0.0;
    }
    // median over the NONZERO days (a book flat by design most days must still measure its
    // rounding-residue days against its real days); mirrors kf._flat_mask
    let mut abs: Vec<f64> = vals.iter().filter(|v| !v.is_nan() && **v != 0.0).map(|v| v.abs()).collect();
    if abs.is_empty() {
        return 0.0;
    }
    abs.sort_unstable_by(f64::total_cmp);
    let n = abs.len();
    let median = if n % 2 == 1 {
        abs[n / 2]
    } else {
        (abs[n / 2 - 1] + abs[n / 2]) / 2.0
    };
    rtol * median
}

/// Is a day with summed PnL `v` a trading day under threshold `thr` (from
/// `flat_threshold(.., rtol)`)? With `rtol == 0.0` this is exactly `v != 0.0` — the
/// exact-zero rule, where only a day with no PnL at all is a non-trading day.
#[inline]
fn is_active_day(v: f64, thr: f64, rtol: f64) -> bool {
    if rtol == 0.0 { v != 0.0 } else { v.abs() > thr }
}

/// Drop flat days from a daily PnL sample: keeps the values with
/// `|v| > flat_threshold(days, rtol)`. With `rtol == 0.0` keeps exactly the values
/// `!= 0.0` (identical to the exact-zero rule).
pub fn drop_flat(days: Vec<f64>, rtol: f64) -> Vec<f64> {
    let thr = flat_threshold(&days, rtol);
    days.into_iter().filter(|&v| is_active_day(v, thr, rtol)).collect()
}

/// Compute Sharpe ratio per shift directly, without materializing the full 3D PnL array.
/// Sums PnL across columns per row, optionally resamples to daily, then computes Sharpe.
/// A day is dropped from the Sharpe sample when |daily PnL| <= flat_rtol * median(|daily PnL|)
/// (flat_rtol = 0.0: exact zeros only, i.e. non-trading days).
/// Returns Vec<f64> of length n_shifts (one Sharpe per shift).
pub fn sharpe_per_lag_matrix(
    pos_ts: &[i64],
    pos_data: &[f64],
    pos_n_rows: usize,
    price_ts: &[i64],
    price_data: &[f64],
    price_n_rows: usize,
    n_cols: usize,
    shifts: &[i64],
    resample_daily: bool,
    flat_rtol: f64,
) -> Vec<f64> {
    let n_shifts = shifts.len();

    struct ColPrecomputed {
        merged_pos: Vec<f64>,
        returns: Vec<f64>,
        row_indices: Vec<usize>,
    }

    // Step 1: Precompute per-column merge_asof and returns (same as backtest_matrix_multi_shift)
    let pre: Vec<ColPrecomputed> = (0..n_cols)
        .into_par_iter()
        .map(|col| {
            let pos_col = &pos_data[col * pos_n_rows..(col + 1) * pos_n_rows];
            let price_col = &price_data[col * price_n_rows..(col + 1) * price_n_rows];

            let pos_has_nan = has_nan(pos_col);
            let price_has_nan = has_nan(price_col);

            let (col_pos_ts, col_pos_vals) = if pos_has_nan {
                extract_non_nan(pos_ts, pos_col)
            } else {
                (pos_ts.to_vec(), pos_col.to_vec())
            };
            let (col_price_ts, col_price_vals) = if price_has_nan {
                extract_non_nan(price_ts, price_col)
            } else {
                (price_ts.to_vec(), price_col.to_vec())
            };

            if col_pos_ts.is_empty() || col_price_ts.is_empty()
                || col_pos_ts[0] > *col_price_ts.last().unwrap()
                || *col_pos_ts.last().unwrap() < col_price_ts[0]
            {
                return ColPrecomputed { merged_pos: Vec::new(), returns: Vec::new(), row_indices: Vec::new() };
            }

            let (merged_ts, merged_pos) = merge_asof_1d(&col_pos_ts, &col_pos_vals, &col_price_ts);
            if merged_ts.is_empty() {
                return ColPrecomputed { merged_pos: Vec::new(), returns: Vec::new(), row_indices: Vec::new() };
            }

            let n = merged_ts.len();
            let price_offset = col_price_ts.partition_point(|&t| t < merged_ts[0]);

            let mut returns = vec![f64::NAN; n];
            for i in 1..n { returns[i] = col_price_vals[price_offset + i] - col_price_vals[price_offset + i - 1]; }
            if price_offset > 0 { returns[0] = col_price_vals[price_offset] - col_price_vals[price_offset - 1]; }

            let mut row_indices = Vec::with_capacity(n);
            let mut price_row = 0;
            for &mt in &merged_ts {
                while price_row < price_n_rows && price_ts[price_row] < mt { price_row += 1; }
                row_indices.push(price_row);
            }

            ColPrecomputed { merged_pos, returns, row_indices }
        })
        .collect();

    // Step 2: For each shift, accumulate PnL per row across all columns, then compute Sharpe
    // Precompute day keys if resampling
    let day_keys: Vec<i64> = if resample_daily {
        price_ts.iter().map(|&t| t / 86_400_000_000_000).collect()
    } else {
        Vec::new()
    };

    // Parallelize over shifts
    let sharpes: Vec<f64> = (0..n_shifts)
        .into_par_iter()
        .map(|si| {
            let shift = shifts[si];

            // Accumulate PnL per row across all columns
            let mut row_pnl = vec![0.0f64; price_n_rows];
            let mut row_has_data = vec![false; price_n_rows];

            for ci in 0..n_cols {
                let pc = &pre[ci];
                let n = pc.merged_pos.len();
                if n == 0 { continue; }

                if shift >= 0 {
                    let s = shift as usize;
                    for i in s..n {
                        let ret = pc.returns[i];
                        if ret.is_nan() { continue; }
                        let pnl = pc.merged_pos[i - s] * ret;
                        if pnl.is_nan() { continue; }
                        let row = pc.row_indices[i];
                        row_pnl[row] += pnl;
                        row_has_data[row] = true;
                    }
                } else {
                    let s = (-shift) as usize;
                    if n > s {
                        for i in 0..n - s {
                            let ret = pc.returns[i];
                            if ret.is_nan() { continue; }
                            let pnl = pc.merged_pos[i + s] * ret;
                            if pnl.is_nan() { continue; }
                            let row = pc.row_indices[i];
                            row_pnl[row] += pnl;
                            row_has_data[row] = true;
                        }
                    }
                }
            }

            if resample_daily {
                // Bucket by day, sum PnL per day
                let mut daily_pnl: Vec<f64> = Vec::new();
                let mut current_day = i64::MIN;
                let mut current_sum = 0.0f64;
                let mut current_has_data = false;

                for row in 0..price_n_rows {
                    let day = day_keys[row];
                    if day != current_day {
                        if current_has_data {
                            daily_pnl.push(current_sum);
                        }
                        current_day = day;
                        current_sum = 0.0;
                        current_has_data = false;
                    }
                    if row_has_data[row] {
                        current_sum += row_pnl[row];
                        current_has_data = true;
                    }
                }
                // Flush last day
                if current_has_data {
                    daily_pnl.push(current_sum);
                }
                // Flat-day tolerance (flat_rtol = 0.0 drops exact zeros only)
                let daily_pnl = drop_flat(daily_pnl, flat_rtol);

                if daily_pnl.len() < 2 {
                    return f64::NAN;
                }

                let n = daily_pnl.len() as f64;
                let mean = daily_pnl.iter().sum::<f64>() / n;
                let var = daily_pnl.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / (n - 1.0);
                let std = var.sqrt();
                if std == 0.0 { return f64::NAN; }
                mean / std * (252.0_f64).sqrt()
            } else {
                // Sharpe on row-level PnL. resample=False is for already-daily input, so a
                // row is a day and the same flat-day rule applies (variant_sharpes_matrix
                // does the same; without it the two kernels disagree on the same input).
                let active: Vec<f64> = (0..price_n_rows)
                    .filter(|&r| row_has_data[r])
                    .map(|r| row_pnl[r])
                    .collect();
                let active = drop_flat(active, flat_rtol);

                if active.len() < 2 {
                    return f64::NAN;
                }

                let n = active.len() as f64;
                let mean = active.iter().sum::<f64>() / n;
                let var = active.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / (n - 1.0);
                let std = var.sqrt();
                if std == 0.0 { return f64::NAN; }
                mean / std * (252.0_f64).sqrt()
            }
        })
        .collect();

    sharpes
}

/// Compute Sharpe ratios for 4 position variants (exbias, bias, long, short) in one pass.
/// Reuses price precomputation across variants.
/// For long/short: the book's two sides, flat when the book is not on that side (clip at 0).
/// Returns [exbias_sharpe, bias_sharpe, long_sharpe, short_sharpe].
pub fn variant_sharpes_matrix(
    pos_ts: &[i64],
    pos_data: &[f64],
    pos_n_rows: usize,
    price_ts: &[i64],
    price_data: &[f64],
    price_n_rows: usize,
    n_cols: usize,
    shift: i64,
    resample_daily: bool,
    flat_rtol: f64,
) -> Vec<f64> {
    struct ColPrecomputed {
        merged_pos: Vec<f64>,       // original positions on price timeline
        merged_pos_long: Vec<f64>,  // long leg: max(pos, 0), NaN where pos is NaN
        merged_pos_short: Vec<f64>, // short leg: min(pos, 0), NaN where pos is NaN
        returns: Vec<f64>,
        row_indices: Vec<usize>,
        col_mean: f64,
    }

    // Step 1: Precompute merge_asof + returns + variants per column
    let pre: Vec<ColPrecomputed> = (0..n_cols)
        .into_par_iter()
        .map(|col| {
            let pos_col = &pos_data[col * pos_n_rows..(col + 1) * pos_n_rows];
            let price_col = &price_data[col * price_n_rows..(col + 1) * price_n_rows];

            let pos_has_nan = has_nan(pos_col);
            let price_has_nan = has_nan(price_col);

            let (col_pos_ts, col_pos_vals) = if pos_has_nan {
                extract_non_nan(pos_ts, pos_col)
            } else {
                (pos_ts.to_vec(), pos_col.to_vec())
            };
            let (col_price_ts, col_price_vals) = if price_has_nan {
                extract_non_nan(price_ts, price_col)
            } else {
                (price_ts.to_vec(), price_col.to_vec())
            };

            if col_pos_ts.is_empty() || col_price_ts.is_empty()
                || col_pos_ts[0] > *col_price_ts.last().unwrap()
                || *col_pos_ts.last().unwrap() < col_price_ts[0]
            {
                return ColPrecomputed {
                    merged_pos: Vec::new(), merged_pos_long: Vec::new(),
                    merged_pos_short: Vec::new(), returns: Vec::new(),
                    row_indices: Vec::new(), col_mean: 0.0,
                };
            }

            let (merged_ts, merged_pos) = merge_asof_1d(&col_pos_ts, &col_pos_vals, &col_price_ts);
            if merged_ts.is_empty() {
                return ColPrecomputed {
                    merged_pos: Vec::new(), merged_pos_long: Vec::new(),
                    merged_pos_short: Vec::new(), returns: Vec::new(),
                    row_indices: Vec::new(), col_mean: 0.0,
                };
            }

            let n = merged_ts.len();
            let price_offset = col_price_ts.partition_point(|&t| t < merged_ts[0]);

            let mut returns = vec![f64::NAN; n];
            for i in 1..n { returns[i] = col_price_vals[price_offset + i] - col_price_vals[price_offset + i - 1]; }
            if price_offset > 0 { returns[0] = col_price_vals[price_offset] - col_price_vals[price_offset - 1]; }

            let mut row_indices = Vec::with_capacity(n);
            let mut price_row = 0;
            for &mt in &merged_ts {
                while price_row < price_n_rows && price_ts[price_row] < mt { price_row += 1; }
                row_indices.push(price_row);
            }

            // Column mean (for exbias/bias)
            let mut sum = 0.0;
            let mut count = 0usize;
            for &v in &merged_pos {
                if !v.is_nan() { sum += v; count += 1; }
            }
            let col_mean = if count > 0 { sum / count as f64 } else { 0.0 };

            // Long leg: the book's long side only, FLAT when the book is not long.
            // Matches Python: pos.clip(lower=0).
            //
            // Clipping (rather than masking negatives and forward-filling the last long
            // size) keeps a genuine flat, so the legs are a decomposition of the book:
            // PnL(long) + PnL(short) == PnL(total) exactly. NaN must survive as NaN
            // (f64::max would fold it to 0.0 and add phantom flat rows to the Sharpe sample
            // before the book starts).
            let mut merged_pos_long = vec![f64::NAN; n];
            let mut merged_pos_short = vec![f64::NAN; n];
            for i in 0..n {
                let v = merged_pos[i];
                if v.is_nan() {
                    continue;                       // leave both legs NaN
                }
                merged_pos_long[i] = if v > 0.0 { v } else { 0.0 };
                merged_pos_short[i] = if v < 0.0 { v } else { 0.0 };
            }

            ColPrecomputed { merged_pos, merged_pos_long, merged_pos_short, returns, row_indices, col_mean }
        })
        .collect();

    // Day keys for resampling
    let day_keys: Vec<i64> = if resample_daily {
        price_ts.iter().map(|&t| t / 86_400_000_000_000).collect()
    } else {
        Vec::new()
    };

    // Step 2: For each variant, accumulate PnL per row and compute Sharpe
    // Variants: 0=exbias, 1=bias, 2=long, 3=short
    let sharpes: Vec<f64> = (0..4usize)
        .into_par_iter()
        .map(|variant| {
            let mut row_pnl = vec![0.0f64; price_n_rows];
            let mut row_has_data = vec![false; price_n_rows];

            for ci in 0..n_cols {
                let pc = &pre[ci];
                let n = pc.merged_pos.len();
                if n == 0 { continue; }

                // Select the right position array for this variant
                let pos_arr: &[f64] = match variant {
                    2 => &pc.merged_pos_long,
                    3 => &pc.merged_pos_short,
                    _ => &pc.merged_pos,
                };

                if shift >= 0 {
                    let s = shift as usize;
                    for i in s..n {
                        let ret = pc.returns[i];
                        if ret.is_nan() { continue; }
                        let raw_pos = pos_arr[i - s];
                        if raw_pos.is_nan() { continue; }
                        let variant_pos = match variant {
                            0 => raw_pos - pc.col_mean,
                            1 => pc.col_mean,
                            _ => raw_pos,  // long/short already pre-computed
                        };
                        let pnl = variant_pos * ret;
                        if pnl.is_nan() { continue; }
                        let row = pc.row_indices[i];
                        row_pnl[row] += pnl;
                        row_has_data[row] = true;
                    }
                } else {
                    let s = (-shift) as usize;
                    if n > s {
                        for i in 0..n - s {
                            let ret = pc.returns[i];
                            if ret.is_nan() { continue; }
                            let raw_pos = pos_arr[i + s];
                            if raw_pos.is_nan() { continue; }
                            let variant_pos = match variant {
                                0 => raw_pos - pc.col_mean,
                                1 => pc.col_mean,
                                _ => raw_pos,
                            };
                            let pnl = variant_pos * ret;
                            if pnl.is_nan() { continue; }
                            let row = pc.row_indices[i];
                            row_pnl[row] += pnl;
                            row_has_data[row] = true;
                        }
                    }
                }
            }

            // Compute Sharpe
            if resample_daily {
                let mut daily_pnl: Vec<f64> = Vec::new();
                let mut current_day = i64::MIN;
                let mut current_sum = 0.0f64;
                let mut current_has_data = false;

                for row in 0..price_n_rows {
                    let day = day_keys[row];
                    if day != current_day {
                        if current_has_data {
                            daily_pnl.push(current_sum);
                        }
                        current_day = day;
                        current_sum = 0.0;
                        current_has_data = false;
                    }
                    if row_has_data[row] {
                        current_sum += row_pnl[row];
                        current_has_data = true;
                    }
                }
                if current_has_data {
                    daily_pnl.push(current_sum);
                }
                // Flat-day tolerance (flat_rtol = 0.0 drops exact zeros only)
                let daily_pnl = drop_flat(daily_pnl, flat_rtol);

                if daily_pnl.len() < 2 { return f64::NAN; }
                let nn = daily_pnl.len() as f64;
                let mean = daily_pnl.iter().sum::<f64>() / nn;
                let var = daily_pnl.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / (nn - 1.0);
                let std = var.sqrt();
                if std == 0.0 { return f64::NAN; }
                mean / std * (252.0_f64).sqrt()
            } else {
                // Same flat-day rule as the daily branch above: a row whose PnL is flat
                // (exactly 0.0 at flat_rtol = 0.0, or within flat_rtol of the median |PnL|)
                // is a period the variant held nothing, and counting it would drag both
                // the mean and the std toward zero. resample=False is for already-daily
                // input, so a row here is a day.
                let active: Vec<f64> = (0..price_n_rows)
                    .filter(|&r| row_has_data[r])
                    .map(|r| row_pnl[r])
                    .collect();
                let active = drop_flat(active, flat_rtol);
                if active.len() < 2 { return f64::NAN; }
                let nn = active.len() as f64;
                let mean = active.iter().sum::<f64>() / nn;
                let var = active.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / (nn - 1.0);
                let std = var.sqrt();
                if std == 0.0 { return f64::NAN; }
                mean / std * (252.0_f64).sqrt()
            }
        })
        .collect();

    sharpes
}

/// Is `ts` non-decreasing? (merge_asof_1d silently returns garbage on unsorted input.)
pub fn is_sorted_ts(ts: &[i64]) -> bool {
    ts.windows(2).all(|w| w[0] <= w[1])
}

/// Partition stacked arrays by integer contract ID into per-contract groups.
/// Contract IDs are pre-encoded integers (0..n_contracts-1).
/// Names maps contract_id -> display name.
/// Errors when a contract's rows are not in time order (either a globally time-sorted
/// frame or one grouped by contract is fine; anything else cannot be as-of joined).
fn partition_stacked_int(
    names: &[String],
    pos_ids: &[i32],
    pos_ts: &[i64],
    pos_vals: &[f64],
    price_ids: &[i32],
    price_ts: &[i64],
    price_vals: &[f64],
) -> Result<Vec<ColumnInput>, String> {
    let n_contracts = names.len();

    let mut pos_data: Vec<(Vec<i64>, Vec<f64>)> = (0..n_contracts).map(|_| (Vec::new(), Vec::new())).collect();
    for i in 0..pos_ids.len() {
        let v = pos_vals[i];
        if v.is_nan() { continue; }
        let id = pos_ids[i] as usize;
        if id < n_contracts {
            pos_data[id].0.push(pos_ts[i]);
            pos_data[id].1.push(v);
        }
    }

    let mut price_data: Vec<(Vec<i64>, Vec<f64>)> = (0..n_contracts).map(|_| (Vec::new(), Vec::new())).collect();
    for i in 0..price_ids.len() {
        let v = price_vals[i];
        if v.is_nan() { continue; }
        let id = price_ids[i] as usize;
        if id < n_contracts {
            price_data[id].0.push(price_ts[i]);
            price_data[id].1.push(v);
        }
    }

    let mut columns = Vec::new();
    for id in 0..n_contracts {
        if !pos_data[id].0.is_empty() && !price_data[id].0.is_empty() {
            let (pt, pv) = std::mem::take(&mut pos_data[id]);
            let (prt, prv) = std::mem::take(&mut price_data[id]);
            if !is_sorted_ts(&pt) {
                return Err(format!("positions of contract {:?} are not sorted by time", names[id]));
            }
            if !is_sorted_ts(&prt) {
                return Err(format!("prices of contract {:?} are not sorted by time", names[id]));
            }
            columns.push(ColumnInput {
                name: names[id].clone(),
                pos_timestamps: pt,
                pos_values: pv,
                price_timestamps: prt,
                price_values: prv,
            });
        }
    }
    Ok(columns)
}

/// Backtest from stacked/long-format arrays with integer contract IDs.
pub fn backtest_stacked(
    names: &[String],
    pos_ids: &[i32],
    pos_ts: &[i64],
    pos_vals: &[f64],
    price_ids: &[i32],
    price_ts: &[i64],
    price_vals: &[f64],
    shift: i64,
    return_variable: u8,
) -> Result<Vec<ColumnOutput>, String> {
    let columns = partition_stacked_int(names, pos_ids, pos_ts, pos_vals, price_ids, price_ts, price_vals)?;
    Ok(backtest_columns(columns, shift, return_variable))
}

/// Backtest from stacked arrays with integer contract IDs, summed across contracts.
pub fn backtest_stacked_summed(
    names: &[String],
    pos_ids: &[i32],
    pos_ts: &[i64],
    pos_vals: &[f64],
    price_ids: &[i32],
    price_ts: &[i64],
    price_vals: &[f64],
    shift: i64,
    return_variable: u8,
) -> Result<(Vec<i64>, Vec<f64>), String> {
    let results = backtest_stacked(
        names, pos_ids, pos_ts, pos_vals,
        price_ids, price_ts, price_vals,
        shift, return_variable,
    )?;

    if results.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }

    let total_len: usize = results.iter().map(|r| r.timestamps.len()).sum();
    let mut pairs: Vec<(i64, f64)> = Vec::with_capacity(total_len);
    for r in &results {
        for i in 0..r.timestamps.len() {
            let v = r.values[i];
            if !v.is_nan() {
                pairs.push((r.timestamps[i], v));
            }
        }
    }
    pairs.sort_unstable_by_key(|&(ts, _)| ts);

    if pairs.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }

    let mut out_ts = Vec::with_capacity(pairs.len());
    let mut out_vals = Vec::with_capacity(pairs.len());
    let mut cur_ts = pairs[0].0;
    let mut cur_sum = pairs[0].1;
    for &(ts, val) in &pairs[1..] {
        if ts == cur_ts {
            cur_sum += val;
        } else {
            out_ts.push(cur_ts);
            out_vals.push(cur_sum);
            cur_ts = ts;
            cur_sum = val;
        }
    }
    out_ts.push(cur_ts);
    out_vals.push(cur_sum);
    Ok((out_ts, out_vals))
}

/// Multi-shift backtest from stacked arrays with integer contract IDs, summed across contracts.
pub fn backtest_stacked_multi_shift_summed(
    names: &[String],
    pos_ids: &[i32],
    pos_ts: &[i64],
    pos_vals: &[f64],
    price_ids: &[i32],
    price_ts: &[i64],
    price_vals: &[f64],
    shifts: &[i64],
) -> Result<Vec<(i64, Vec<i64>, Vec<f64>)>, String> {
    let columns = partition_stacked_int(names, pos_ids, pos_ts, pos_vals, price_ids, price_ts, price_vals)?;
    let results = backtest_multi_shift(columns, shifts);

    Ok(results.into_iter().map(|(shift, outputs)| {
        let total_len: usize = outputs.iter().map(|r| r.timestamps.len()).sum();
        let mut pairs: Vec<(i64, f64)> = Vec::with_capacity(total_len);
        for r in &outputs {
            for i in 0..r.timestamps.len() {
                let v = r.values[i];
                if !v.is_nan() {
                    pairs.push((r.timestamps[i], v));
                }
            }
        }
        pairs.sort_unstable_by_key(|&(ts, _)| ts);

        let mut out_ts = Vec::with_capacity(pairs.len());
        let mut out_vals = Vec::with_capacity(pairs.len());
        if !pairs.is_empty() {
            let mut cur_ts = pairs[0].0;
            let mut cur_sum = pairs[0].1;
            for &(ts, val) in &pairs[1..] {
                if ts == cur_ts {
                    cur_sum += val;
                } else {
                    out_ts.push(cur_ts);
                    out_vals.push(cur_sum);
                    cur_ts = ts;
                    cur_sum = val;
                }
            }
            out_ts.push(cur_ts);
            out_vals.push(cur_sum);
        }
        (shift, out_ts, out_vals)
    }).collect())
}

/// Apply EWM (exponential weighted moving average) smoothing in-place, pandas semantics
/// (`ewm(span=span, adjust=True).mean()`): y[k] = sum_j (1-a)^(k-j) x_j / sum_j (1-a)^(k-j),
/// a = 2/(span+1). The same EWM the pandas/numba holding axis (kf._ewm_pandas_col) is built
/// with, so the Sharpe of a smoothed book and its holding time describe the same positions.
/// NaN entries are skipped (left as NaN, no weight decay across them; the merged position
/// series this is applied to has none). span <= 0 means no smoothing.
fn ewm_smooth(data: &mut [f64], span: f64) {
    if span <= 0.0 || data.is_empty() {
        return;
    }
    let decay = 1.0 - 2.0 / (span + 1.0);
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for v in data.iter_mut() {
        if v.is_nan() {
            continue;
        }
        num = decay * num + *v;
        den = decay * den + 1.0;
        *v = num / den;
    }
}

/// Result from holding_curve: (sharpe, turnover_mean, holding_days) per EWM span
pub struct HoldingCurveResult {
    pub sharpe: f64,
    pub holding_days: f64,
}

/// Compute holding curve: for each EWM span, smooth positions, backtest, compute sharpe + holding.
/// pos_data and price_data are column-major: [col][row] = data[col * n_rows + row].
/// ewm_spans: list of spans (0.0 = no smoothing).
/// flat_rtol: a day is flat (dropped from the Sharpe sample and the turnover mean) when
/// |daily PnL| <= flat_rtol * median(|daily PnL|); 0.0 drops exact zeros only.
/// Returns Vec of (sharpe, holding_days) per span.
pub fn holding_curve(
    pos_ts: &[i64],
    pos_data: &[f64],
    pos_n_rows: usize,
    price_ts: &[i64],
    price_data: &[f64],
    price_n_rows: usize,
    n_cols: usize,
    shift: i64,
    ewm_spans: &[f64],
    flat_rtol: f64,
) -> Vec<HoldingCurveResult> {
    // Pre-compute per-column: merge_asof positions onto price grid, compute returns (done once)
    struct ColPre {
        merged_pos: Vec<f64>,  // position values on price grid
        returns: Vec<f64>,     // price.diff() on price grid
        row_indices: Vec<usize>, // mapping from merged index to price_ts index
        n: usize,
    }

    let pre: Vec<ColPre> = (0..n_cols)
        .into_par_iter()
        .map(|col| {
            let pos_col = &pos_data[col * pos_n_rows..(col + 1) * pos_n_rows];
            let price_col = &price_data[col * price_n_rows..(col + 1) * price_n_rows];

            let pos_has_nan = has_nan(pos_col);

            let (col_pos_ts, col_pos_vals) = if pos_has_nan {
                extract_non_nan(pos_ts, pos_col)
            } else {
                (pos_ts.to_vec(), pos_col.to_vec())
            };

            if col_pos_ts.is_empty() || price_n_rows == 0
                || col_pos_ts[0] > price_ts[price_n_rows - 1]
                || *col_pos_ts.last().unwrap() < price_ts[0]
            {
                return ColPre { merged_pos: Vec::new(), returns: Vec::new(), row_indices: Vec::new(), n: 0 };
            }

            // Merge positions onto the FULL price grid (not NaN-stripped)
            // to match Python's pos.ffill() behavior
            let (merged_ts, merged_pos) = merge_asof_1d(&col_pos_ts, &col_pos_vals, price_ts);
            if merged_ts.is_empty() {
                return ColPre { merged_pos: Vec::new(), returns: Vec::new(), row_indices: Vec::new(), n: 0 };
            }

            let n = merged_ts.len();
            let price_offset = price_ts.partition_point(|&t| t < merged_ts[0]);

            // Compute returns from the RAW price column (preserving NaN for gaps)
            // This matches Python's prices.diff() which produces NaN at gaps
            let mut returns = vec![f64::NAN; n];
            for i in 1..n {
                let pi = price_offset + i;
                let p_cur = price_col[pi];
                let p_prev = price_col[pi - 1];
                if !p_cur.is_nan() && !p_prev.is_nan() {
                    returns[i] = p_cur - p_prev;
                }
            }
            if price_offset > 0 {
                let p_cur = price_col[price_offset];
                let p_prev = price_col[price_offset - 1];
                if !p_cur.is_nan() && !p_prev.is_nan() {
                    returns[0] = p_cur - p_prev;
                }
            }

            // Map merged indices to price_ts global indices (1:1 since we use full grid)
            let mut row_indices = Vec::with_capacity(n);
            for i in 0..n {
                row_indices.push(price_offset + i);
            }

            ColPre { merged_pos, returns, row_indices, n }
        })
        .collect();

    // Precompute bars per calendar day from price_ts (for correct mean(|pos|) denominator)
    // This counts unique TIME steps per day, NOT column*bar evaluations
    let mut bars_per_day: Vec<(i64, u32)> = Vec::new();
    for &ts in price_ts {
        let day_ns = ts / 86_400_000_000_000 * 86_400_000_000_000;
        match bars_per_day.binary_search_by_key(&day_ns, |&(d, _)| d) {
            Ok(idx) => bars_per_day[idx].1 += 1,
            Err(idx) => bars_per_day.insert(idx, (day_ns, 1)),
        }
    }

    // For each EWM span, compute sharpe and holding
    ewm_spans
        .par_iter()
        .map(|&span| {
            // For each column: apply EWM to merged positions, then shift, compute pnl
            // Accumulate daily PnL across all columns
            let mut daily_pnl: Vec<(i64, f64)> = Vec::new(); // (day_ns, pnl_sum)
            // Daily turnover: track TV=sum(|diff|) and sum(|pos|) per day
            // turnover = TV / mean(|pos|), where mean uses bars_per_day (time steps, not column*bars)
            // Using mean(|pos|) (not per-bar |pos|) avoids blow-up near zero crossings
            let mut daily_tv: Vec<(i64, f64, f64)> = Vec::new(); // (day_ns, sum_abs_diff, sum_abs_pos)

            for col_pre in &pre {
                if col_pre.n == 0 {
                    continue;
                }

                // Clone and smooth positions
                let mut smoothed = col_pre.merged_pos.clone();
                if span > 0.0 {
                    ewm_smooth(&mut smoothed, span);
                }

                let n = col_pre.n;

                // Accumulate daily TV and sum(|pos|) across all columns
                for i in 1..n {
                    if smoothed[i].is_nan() || smoothed[i - 1].is_nan() {
                        continue;
                    }
                    let day_ns = price_ts[col_pre.row_indices[i]] / 86_400_000_000_000 * 86_400_000_000_000;
                    let abs_diff = (smoothed[i] - smoothed[i - 1]).abs();
                    let abs_pos = smoothed[i].abs();
                    match daily_tv.binary_search_by_key(&day_ns, |&(d, _, _)| d) {
                        Ok(idx) => {
                            daily_tv[idx].1 += abs_diff;
                            daily_tv[idx].2 += abs_pos;
                        }
                        Err(idx) => daily_tv.insert(idx, (day_ns, abs_diff, abs_pos)),
                    }
                }

                // Shift and compute PnL
                for i in 0..n {
                    let pos_val = if shift >= 0 {
                        let s = shift as usize;
                        if i >= s { smoothed[i - s] } else { continue; }
                    } else {
                        let s = (-shift) as usize;
                        if i + s < n { smoothed[i + s] } else { continue; }
                    };

                    let ret = col_pre.returns[i];
                    if pos_val.is_nan() || ret.is_nan() {
                        continue;
                    }

                    let pnl_val = pos_val * ret;
                    let day_ns = price_ts[col_pre.row_indices[i]] / 86_400_000_000_000 * 86_400_000_000_000;

                    // Accumulate into daily_pnl (sorted insert or scan)
                    match daily_pnl.binary_search_by_key(&day_ns, |&(d, _)| d) {
                        Ok(idx) => daily_pnl[idx].1 += pnl_val,
                        Err(idx) => daily_pnl.insert(idx, (day_ns, pnl_val)),
                    }
                }
            }

            if daily_pnl.is_empty() {
                return HoldingCurveResult { sharpe: f64::NAN, holding_days: f64::NAN };
            }

            // Compute sharpe = mean/std * sqrt(252) with ddof=1
            // Filter out flat-PnL days (weekends/holidays) — at flat_rtol = 0.0 this is exact
            // zeros only, matching Python's replace(0, NaN).dropna()
            let all_daily: Vec<f64> = daily_pnl.iter().map(|(_, v)| *v).collect();
            let flat_thr = flat_threshold(&all_daily, flat_rtol);
            let nonzero_pnl: Vec<f64> = drop_flat(all_daily, flat_rtol);
            let n_days = nonzero_pnl.len() as f64;
            let sum: f64 = nonzero_pnl.iter().sum();
            let mean = sum / n_days;
            let sharpe = if n_days < 2.0 {
                f64::NAN
            } else {
                let var: f64 = nonzero_pnl.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n_days - 1.0);
                let std = var.sqrt();
                if std > 0.0 { mean / std * 252.0_f64.sqrt() } else { 0.0 }
            };

            // Daily turnover = TV / mean(|pos|) where mean uses time-step count (not column*bar count)
            // bars_per_day is precomputed from price_ts to count unique time steps per day
            let mean_daily_turnover = if daily_tv.is_empty() {
                0.0
            } else {
                let mut turnover_sum = 0.0;
                let mut n_turnover_days = 0usize;
                for &(day_ns, tv, pos_sum) in &daily_tv {
                    // Skip non-trading days (flat PnL, same rule as the Sharpe sample above)
                    // — matches Python's business-day filter
                    let has_pnl = match daily_pnl.binary_search_by_key(&day_ns, |&(d, _)| d) {
                        Ok(idx) => is_active_day(daily_pnl[idx].1, flat_thr, flat_rtol),
                        Err(_) => false,
                    };
                    if !has_pnl {
                        continue;
                    }
                    // Look up bars-per-day for this calendar day
                    let n_time_bars = match bars_per_day.binary_search_by_key(&day_ns, |&(d, _)| d) {
                        Ok(idx) => bars_per_day[idx].1 as f64,
                        Err(_) => 1.0,  // shouldn't happen
                    };
                    let mean_pos = pos_sum / n_time_bars;
                    if mean_pos > 0.0 {
                        turnover_sum += tv / mean_pos;
                        n_turnover_days += 1;
                    }
                }
                if n_turnover_days > 0 { turnover_sum / n_turnover_days as f64 } else { 0.0 }
            };
            let holding_days = if mean_daily_turnover > 0.0 { 1.0 / mean_daily_turnover } else { f64::INFINITY };

            HoldingCurveResult { sharpe, holding_days }
        })
        .collect()
}

/// Compute correlation_to_trend: correlation of strategy returns to trend-following at various filter lengths.
/// Returns Vec of correlation values, one per filter_length.
pub fn correlation_to_trend(
    pos_ts: &[i64],
    pos_data: &[f64],
    pos_n_rows: usize,
    price_ts: &[i64],
    price_data: &[f64],
    price_n_rows: usize,
    n_cols: usize,
    shift: i64,
    filter_lengths: &[f64],
) -> Vec<f64> {
    // Step 1: Backtest pos vs prices -> base daily PnL (summed across columns)
    let base_result = backtest_matrix(
        pos_ts, pos_data, pos_n_rows,
        price_ts, price_data, price_n_rows,
        n_cols, shift, 0, // return_variable=0 (pnl)
    );

    // Sum across columns per row, then aggregate to daily
    let mut base_daily: Vec<(i64, f64)> = Vec::new();
    for row in 0..price_n_rows {
        let mut row_sum = 0.0f64;
        let mut has_value = false;
        for col in 0..n_cols {
            let v = base_result[col * price_n_rows + row];
            if !v.is_nan() {
                row_sum += v;
                has_value = true;
            }
        }
        if has_value {
            let day_ns = price_ts[row] / 86_400_000_000_000 * 86_400_000_000_000;
            match base_daily.binary_search_by_key(&day_ns, |&(d, _)| d) {
                Ok(idx) => base_daily[idx].1 += row_sum,
                Err(idx) => base_daily.insert(idx, (day_ns, row_sum)),
            }
        }
    }

    if base_daily.is_empty() {
        return vec![f64::NAN; filter_lengths.len()];
    }

    // Step 2: For each filter length, compute EWM trend signal from prices, backtest, correlate
    filter_lengths
        .par_iter()
        .map(|&fl| {
            // Trend signal per column: EWM of the price diffs on the column's own non-NaN
            // prints, pandas semantics (`s.dropna().diff().ewm(span=fl, adjust=True,
            // min_periods=min(fl // 2, 32)).mean()`, the Python fallback in kf.py).
            let min_periods = ((fl / 2.0).min(32.0) as usize).max(1);

            // Create smoothed position data (column-major)
            let mut trend_pos = vec![f64::NAN; price_n_rows * n_cols];

            for col in 0..n_cols {
                let price_col = &price_data[col * price_n_rows..(col + 1) * price_n_rows];
                let trend_col = &mut trend_pos[col * price_n_rows..(col + 1) * price_n_rows];

                // adjust=True: ewm = sum_j (1-a)^(k-j) r_j / sum_j (1-a)^(k-j)
                let decay = 1.0 - 2.0 / (fl + 1.0);
                let mut num = 0.0f64;
                let mut den = 0.0f64;
                let mut count = 0usize;
                let mut prev_price = f64::NAN;

                for i in 0..price_n_rows {
                    let p = price_col[i];
                    if p.is_nan() {
                        continue;
                    }
                    if !prev_price.is_nan() {
                        let ret = p - prev_price;
                        num = decay * num + ret;
                        den = decay * den + 1.0;
                        count += 1;
                        if count >= min_periods {
                            trend_col[i] = num / den;
                        }
                    }
                    prev_price = p;
                }
            }

            // Backtest the trend positions vs prices with shift=1: trend_pos[i] already
            // contains the diff INTO print i, so it is only tradable from the next print.
            // With shift=0 the trend leg would earn alpha * ret^2 on every bar (a look-ahead
            // "strategy" with Sharpe > 10 on a random walk) and the correlation would be
            // measured against that artefact rather than against trend-following PnL.
            let trend_result = backtest_matrix(
                price_ts, &trend_pos, price_n_rows,
                price_ts, price_data, price_n_rows,
                n_cols, 1, 0, // shift=1, pnl
            );

            // Sum across columns per row, aggregate to daily
            let mut trend_daily: Vec<(i64, f64)> = Vec::new();
            for row in 0..price_n_rows {
                let mut row_sum = 0.0f64;
                let mut has_value = false;
                for col in 0..n_cols {
                    let v = trend_result[col * price_n_rows + row];
                    if !v.is_nan() {
                        row_sum += v;
                        has_value = true;
                    }
                }
                if has_value {
                    let day_ns = price_ts[row] / 86_400_000_000_000 * 86_400_000_000_000;
                    match trend_daily.binary_search_by_key(&day_ns, |&(d, _)| d) {
                        Ok(idx) => trend_daily[idx].1 += row_sum,
                        Err(idx) => trend_daily.insert(idx, (day_ns, row_sum)),
                    }
                }
            }

            // Pearson correlation between base_daily and trend_daily (aligned by day)
            let mut pairs: Vec<(f64, f64)> = Vec::new();
            let mut ti = 0usize;
            for &(day, base_val) in &base_daily {
                while ti < trend_daily.len() && trend_daily[ti].0 < day {
                    ti += 1;
                }
                if ti < trend_daily.len() && trend_daily[ti].0 == day {
                    pairs.push((base_val, trend_daily[ti].1));
                }
            }

            if pairs.len() < 3 {
                return f64::NAN;
            }

            let n = pairs.len() as f64;
            let sum_x: f64 = pairs.iter().map(|(x, _)| x).sum();
            let sum_y: f64 = pairs.iter().map(|(_, y)| y).sum();
            let mean_x = sum_x / n;
            let mean_y = sum_y / n;
            let mut cov = 0.0f64;
            let mut var_x = 0.0f64;
            let mut var_y = 0.0f64;
            for &(x, y) in &pairs {
                let dx = x - mean_x;
                let dy = y - mean_y;
                cov += dx * dy;
                var_x += dx * dx;
                var_y += dy * dy;
            }
            let denom = (var_x * var_y).sqrt();
            if denom > 0.0 { cov / denom } else { 0.0 }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merge_asof_basic() {
        // pos: [10, 30, 50] with values [1.0, 2.0, 3.0]
        // prices: [5, 10, 20, 30, 40, 50, 60]
        // Expected: starts at 10 (first price >= first pos)
        //   price=10 -> pos_ts=10 -> 1.0
        //   price=20 -> pos_ts=10 -> 1.0
        //   price=30 -> pos_ts=30 -> 2.0
        //   price=40 -> pos_ts=30 -> 2.0
        //   price=50 -> pos_ts=50 -> 3.0
        //   price=60 -> pos_ts=50 -> 3.0
        let pos_ts = vec![10, 30, 50];
        let pos_vals = vec![1.0, 2.0, 3.0];
        let price_ts = vec![5, 10, 20, 30, 40, 50, 60];

        let (ts, vals) = merge_asof_1d(&pos_ts, &pos_vals, &price_ts);
        assert_eq!(ts, vec![10, 20, 30, 40, 50, 60]);
        assert_eq!(vals, vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
    }

    #[test]
    fn test_merge_asof_empty() {
        let (ts, vals) = merge_asof_1d(&[], &[], &[1, 2, 3]);
        assert!(ts.is_empty());
        assert!(vals.is_empty());

        let (ts, vals) = merge_asof_1d(&[1], &[1.0], &[]);
        assert!(ts.is_empty());
        assert!(vals.is_empty());
    }

    #[test]
    fn test_merge_asof_no_overlap() {
        // All prices before any position
        let pos_ts = vec![100, 200];
        let pos_vals = vec![1.0, 2.0];
        let price_ts = vec![10, 20, 30];

        let (ts, vals) = merge_asof_1d(&pos_ts, &pos_vals, &price_ts);
        assert!(ts.is_empty());
        assert!(vals.is_empty());
    }

    #[test]
    fn test_merge_asof_single_element() {
        let pos_ts = vec![10];
        let pos_vals = vec![5.0];
        let price_ts = vec![10, 20, 30];

        let (ts, vals) = merge_asof_1d(&pos_ts, &pos_vals, &price_ts);
        assert_eq!(ts, vec![10, 20, 30]);
        assert_eq!(vals, vec![5.0, 5.0, 5.0]);
    }

    #[test]
    fn test_backtest_column_pnl() {
        // Simple: pos=[1.0] at ts=0, prices=[100, 101, 103] at ts=[0,1,2]
        // merged_pos = [1.0, 1.0, 1.0]
        // returns = [NaN, 1.0, 2.0]
        // pnl (shift=0) = [NaN, 1.0, 2.0]
        let pos_ts = vec![0];
        let pos_vals = vec![1.0];
        let price_ts = vec![0, 1, 2];
        let price_vals = vec![100.0, 101.0, 103.0];

        let (ts, vals) = backtest_column(&pos_ts, &pos_vals, &price_ts, &price_vals, 0, 0);
        assert_eq!(ts, vec![0, 1, 2]);
        assert!(vals[0].is_nan());
        assert_eq!(vals[1], 1.0);
        assert_eq!(vals[2], 2.0);
    }

    #[test]
    fn test_backtest_column_with_shift() {
        let pos_ts = vec![0, 1, 2];
        let pos_vals = vec![1.0, 2.0, 3.0];
        let price_ts = vec![0, 1, 2];
        let price_vals = vec![100.0, 101.0, 103.0];

        // shift=1: pos shifted right by 1
        // merged_pos = [1.0, 2.0, 3.0]
        // shifted_pos = [NaN, 1.0, 2.0]
        // returns = [NaN, 1.0, 2.0]
        // pnl = [NaN, 1.0, 4.0]
        let (_, vals) = backtest_column(&pos_ts, &pos_vals, &price_ts, &price_vals, 1, 0);
        assert!(vals[0].is_nan());
        assert_eq!(vals[1], 1.0);
        assert_eq!(vals[2], 4.0);
    }

    #[test]
    fn test_backtest_column_market_return() {
        let pos_ts = vec![0];
        let pos_vals = vec![1.0];
        let price_ts = vec![0, 1, 2];
        let price_vals = vec![100.0, 102.0, 105.0];

        let (_, vals) = backtest_column(&pos_ts, &pos_vals, &price_ts, &price_vals, 0, 2);
        assert!(vals[0].is_nan());
        assert_eq!(vals[1], 2.0);
        assert_eq!(vals[2], 3.0);
    }

    #[test]
    fn test_backtest_column_pos() {
        let pos_ts = vec![0, 2];
        let pos_vals = vec![1.0, 3.0];
        let price_ts = vec![0, 1, 2, 3];
        let price_vals = vec![100.0, 101.0, 102.0, 103.0];

        // merged_pos = [1.0, 1.0, 3.0, 3.0], shift=1
        // shifted = [NaN, 1.0, 1.0, 3.0]
        let (_, vals) = backtest_column(&pos_ts, &pos_vals, &price_ts, &price_vals, 1, 1);
        assert!(vals[0].is_nan());
        assert_eq!(vals[1], 1.0);
        assert_eq!(vals[2], 1.0);
        assert_eq!(vals[3], 3.0);
    }

    #[test]
    fn test_backtest_columns_parallel() {
        let cols = vec![
            ColumnInput {
                name: "A".to_string(),
                pos_timestamps: vec![0, 1],
                pos_values: vec![1.0, 2.0],
                price_timestamps: vec![0, 1, 2],
                price_values: vec![100.0, 101.0, 103.0],
            },
            ColumnInput {
                name: "B".to_string(),
                pos_timestamps: vec![0],
                pos_values: vec![0.5],
                price_timestamps: vec![0, 1, 2],
                price_values: vec![200.0, 202.0, 201.0],
            },
        ];

        let results = backtest_columns(cols, 0, 0);
        assert_eq!(results.len(), 2);

        // Find results by name
        let a = results.iter().find(|r| r.name == "A").unwrap();
        let b = results.iter().find(|r| r.name == "B").unwrap();

        // A: merged_pos=[1,2,2], returns=[NaN,1,2], pnl=[NaN,2*1,2*2]=[NaN,2,4]
        assert_eq!(a.values[1], 2.0);
        assert_eq!(a.values[2], 4.0);

        // B: merged_pos=[0.5,0.5,0.5], returns=[NaN,2,-1], pnl=[NaN,1,-0.5]
        assert_eq!(b.values[1], 1.0);
        assert_eq!(b.values[2], -0.5);
    }

    #[test]
    fn test_backtest_multi_shift() {
        let cols = vec![ColumnInput {
            name: "X".to_string(),
            pos_timestamps: vec![0, 1, 2, 3],
            pos_values: vec![1.0, 2.0, 3.0, 4.0],
            price_timestamps: vec![0, 1, 2, 3],
            price_values: vec![100.0, 101.0, 103.0, 106.0],
        }];

        let shifts = vec![0, 1, 2];
        let results = backtest_multi_shift(cols, &shifts);
        assert_eq!(results.len(), 3);

        // Verify we get results for each shift
        for (shift, outputs) in &results {
            assert_eq!(outputs.len(), 1);
            assert_eq!(outputs[0].name, "X");
            assert!(*shift >= 0 && *shift <= 2);
        }
    }

    #[test]
    fn test_ewm_smooth() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        ewm_smooth(&mut data, 3.0);
        // pandas: pd.Series([1,2,3,4,5]).ewm(span=3).mean()  (adjust=True)
        let expected = [1.0, 1.6666666666666667, 2.4285714285714284, 3.2666666666666666, 4.161290322580645];
        for (got, want) in data.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1e-12, "{} vs {}", got, want);
        }
    }

    #[test]
    fn test_ewm_smooth_no_op() {
        let mut data = vec![1.0, 2.0, 3.0];
        let original = data.clone();
        ewm_smooth(&mut data, 0.0);
        assert_eq!(data, original);
    }

    #[test]
    fn test_holding_curve_basic() {
        // 1 column, 10 DAILY price points (the Sharpe is a daily statistic and needs
        // at least two days with non-flat PnL), constant position, varying increments
        let n = 10;
        let day = 86_400_000_000_000i64;
        let pos_ts: Vec<i64> = (0..n).map(|i| i as i64 * day).collect();
        let price_ts = pos_ts.clone();
        let pos_data: Vec<f64> = vec![1.0; n];
        let price_data: Vec<f64> = (0..n).map(|i| 100.0 + i as f64 + (i % 3) as f64).collect();

        let spans = vec![0.0, 4.0];
        let results = holding_curve(
            &pos_ts, &pos_data, n,
            &price_ts, &price_data, n,
            1, 0, &spans, 0.0,
        );

        assert_eq!(results.len(), 2);
        // With constant position of 1.0, EWM smoothing doesn't change anything
        // so both spans should give similar results
        assert!(!results[0].sharpe.is_nan());
        assert!(!results[1].sharpe.is_nan());
        assert!(results[0].holding_days > 0.0);
    }

    /// pandas `ewm(span, adjust=True).mean()` of `x`, for the tests below.
    fn ewm_adjust(x: &[f64], span: f64) -> Vec<f64> {
        let decay = 1.0 - 2.0 / (span + 1.0);
        let (mut num, mut den) = (0.0, 0.0);
        x.iter().map(|&v| { num = decay * num + v; den = decay * den + 1.0; num / den }).collect()
    }

    #[test]
    fn test_correlation_to_trend_trend_leg_is_lagged() {
        // Flat price with a single jump. The trend signal is exactly 0 before the jump, so
        // a LAGGED trend leg earns nothing on the jump bar (and nothing afterwards, the
        // price is flat again): zero trend PnL everywhere -> correlation 0.0 by the
        // zero-variance rule. A shift=0 trend leg would earn alpha * jump^2 on the jump
        // day, the same day the long book earns its only PnL, and report corr = 1.0.
        let n = 60;
        let day = 86_400_000_000_000i64;
        let ts: Vec<i64> = (0..n).map(|i| i as i64 * day).collect();
        let price: Vec<f64> = (0..n).map(|i| if i < 30 { 100.0 } else { 101.0 }).collect();
        let pos: Vec<f64> = vec![1.0; n];
        let r = correlation_to_trend(&ts, &pos, n, &ts, &price, n, 1, 0, &[4.0, 16.0]);
        assert_eq!(r, vec![0.0, 0.0]);
    }

    #[test]
    fn test_correlation_to_trend_matches_lagged_ewm_position() {
        // A book that IS the lagged trend signal of filter length fl correlates 1.0 with
        // the fl trend leg: pins the EWM (adjust=True, min_periods = fl/2) and the lag.
        let n = 400;
        let day = 86_400_000_000_000i64;
        let ts: Vec<i64> = (0..n).map(|i| i as i64 * day).collect();
        let mut state: u64 = 0x9E3779B97F4A7C15;
        let mut price = vec![100.0f64; n];
        for i in 1..n {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = ((state >> 11) as f64) / ((1u64 << 53) as f64);
            price[i] = price[i - 1] + (u - 0.5);
        }
        let fl = 16.0;
        let rets: Vec<f64> = (1..n).map(|i| price[i] - price[i - 1]).collect();
        let ewm = ewm_adjust(&rets, fl);
        let min_periods = (fl as usize) / 2;
        // realised position at print i = EWM through print i-1 (i.e. rets[..i-1], ewm[i-2])
        let mut pos = vec![f64::NAN; n];
        for i in 2..n {
            if i - 1 >= min_periods { pos[i] = ewm[i - 2]; }
        }
        let r = correlation_to_trend(&ts, &pos, n, &ts, &price, n, 1, 0, &[fl, 4.0]);
        assert!((r[0] - 1.0).abs() < 1e-9, "corr with own filter length = {}", r[0]);
        assert!(r[1] < 1.0 - 1e-6, "a different filter length is not a perfect match: {}", r[1]);
    }

    #[test]
    fn test_stacked_rejects_unsorted_contract_rows() {
        let names = vec!["A".to_string()];
        let ids = vec![0, 0, 0];
        let ts_bad = vec![2i64, 1, 3];
        let ts_ok = vec![1i64, 2, 3];
        let vals = vec![1.0, 1.0, 1.0];
        let price = vec![100.0, 101.0, 103.0];
        assert!(backtest_stacked(&names, &ids, &ts_bad, &vals, &ids, &ts_ok, &price, 0, 0).is_err());
        assert!(backtest_stacked(&names, &ids, &ts_ok, &vals, &ids, &ts_bad, &price, 0, 0).is_err());
        assert_eq!(backtest_stacked(&names, &ids, &ts_ok, &vals, &ids, &ts_ok, &price, 0, 0).unwrap().len(), 1);
    }

    #[test]
    fn test_correlation_to_trend_basic() {
        // Simple test: trend-following position should correlate with trend
        let n = 100;
        let price_ts: Vec<i64> = (0..n).map(|i| i as i64 * 86_400_000_000_000).collect(); // daily
        let price_data: Vec<f64> = (0..n).map(|i| 100.0 + i as f64 * 0.5).collect(); // uptrend

        // Position = 1.0 (long), should correlate positively with trend
        let pos_data: Vec<f64> = vec![1.0; n];

        let filter_lengths = vec![4.0, 16.0, 64.0];
        let results = correlation_to_trend(
            &price_ts, &pos_data, n,
            &price_ts, &price_data, n,
            1, 0, &filter_lengths,
        );

        assert_eq!(results.len(), 3);
        // All should be defined
        for r in &results {
            assert!(!r.is_nan(), "Expected non-NaN correlation");
        }
    }
    #[test]
    fn test_flat_threshold_odd_even() {
        // odd n: median of |v| = middle element
        assert_eq!(flat_threshold(&[3.0, -1.0, 2.0], 1.0), 2.0);
        assert_eq!(flat_threshold(&[3.0, -1.0, 2.0], 0.5), 1.0);
        // even n: mean of the two middle elements (np.median)
        assert_eq!(flat_threshold(&[4.0, 1.0, -3.0, 2.0], 1.0), 2.5);
        assert_eq!(flat_threshold(&[4.0, 1.0, -3.0, 2.0], 0.1), 0.25);
        // empty input -> 0.0
        assert_eq!(flat_threshold(&[], 1.0), 0.0);
    }

    #[test]
    fn test_flat_threshold_all_zeros() {
        assert_eq!(flat_threshold(&[0.0, 0.0, 0.0], 1e-3), 0.0);
        assert_eq!(flat_threshold(&[0.0, 0.0, 0.0, 0.0], 1e-3), 0.0);
        // and rtol 0 is always 0
        assert_eq!(flat_threshold(&[1.0, 2.0, 3.0], 0.0), 0.0);
    }

    #[test]
    fn test_flat_threshold_skips_nan() {
        // NaNs are not part of the median sample
        assert_eq!(flat_threshold(&[f64::NAN, 1.0, f64::NAN, 3.0, 2.0], 1.0), 2.0);
        assert_eq!(flat_threshold(&[f64::NAN, f64::NAN], 1.0), 0.0);
    }

    #[test]
    fn test_drop_flat_rtol_zero_keeps_nonzeros_only() {
        let days = vec![1.0, 0.0, -2.0, 0.0, 1e-300, -0.0];
        let kept = drop_flat(days, 0.0);
        assert_eq!(kept, vec![1.0, -2.0, 1e-300]);
    }

    #[test]
    fn test_drop_flat_tolerance() {
        // |v| sorted: [0, 1e-16, 1e-16, 1, 2, 3] -> np.median = (1e-16 + 1) / 2 ~ 0.5,
        // threshold 5e-13: drops 1e-16, -1e-16 and 0.0
        let days = vec![1.0, 2.0, 3.0, 1e-16, -1e-16, 0.0];
        let kept = drop_flat(days, 1e-12);
        assert_eq!(kept, vec![1.0, 2.0, 3.0]);
        // The median is taken over the NONZERO days, so exact zeros (a book flat by design)
        // never dilute it: nonzero |v| = [1e-16, 1e-16, 1, 2] -> median ~0.5, threshold
        // 5e-13, the two residue days and the zero go.
        let days = vec![1.0, 2.0, 1e-16, -1e-16, 0.0];
        let kept = drop_flat(days, 1e-12);
        assert_eq!(kept, vec![1.0, 2.0]);
        // ... but residue days that are the MAJORITY of the nonzero days survive: median 1e-16
        let days = vec![1.0, 1e-16, -1e-16, 1e-16, 0.0];
        let kept = drop_flat(days, 1e-12);
        assert_eq!(kept, vec![1.0, 1e-16, -1e-16, 1e-16]);
        // odd/even paths through drop_flat
        assert_eq!(drop_flat(vec![5.0, 1.0, 3.0], 0.5), vec![5.0, 3.0]); // thr 1.5
        assert_eq!(drop_flat(vec![4.0, 1.0, 3.0, 2.0], 1.0), vec![4.0, 3.0]); // thr 2.5
        // all zeros: threshold 0, nothing survives
        assert!(drop_flat(vec![0.0, 0.0, 0.0], 1e-3).is_empty());
    }

    #[test]
    fn test_sharpe_kernels_rtol_zero_match_and_tolerance_drops() {
        // 1 column, daily grid: PnL per day = pos * diff(price). Days 2 and 4 have
        // tiny PnL (1e-10 price moves), day 3 has none (flat price), the rest are O(1).
        let n = 7;
        let day = 86_400_000_000_000i64;
        let ts: Vec<i64> = (0..n).map(|i| i as i64 * day).collect();
        let pos: Vec<f64> = vec![1.0; n];
        let price: Vec<f64> = vec![100.0, 101.0, 101.0 + 1e-10, 101.0 + 1e-10, 101.0 + 2e-10, 103.0, 102.0];

        let s0 = sharpe_per_lag_matrix(&ts, &pos, n, &ts, &price, n, 1, &[0], true, 0.0)[0];
        let s1 = sharpe_per_lag_matrix(&ts, &pos, n, &ts, &price, n, 1, &[0], true, 1e-9)[0];
        // rtol 0 keeps the tiny days; the tolerance drops them (only the 1.0/2.0/-1.0 days survive)
        assert!(!s0.is_nan() && !s1.is_nan());
        assert!(s0 != s1);
        let v0 = variant_sharpes_matrix(&ts, &pos, n, &ts, &price, n, 1, 0, true, 0.0);
        let v1 = variant_sharpes_matrix(&ts, &pos, n, &ts, &price, n, 1, 0, true, 1e-9);
        // bias variant (pos == col_mean) equals the plain Sharpe in each regime
        assert!((v0[1] - s0).abs() < 1e-12);
        assert!((v1[1] - s1).abs() < 1e-12);
        // resample=False on daily input agrees with the daily branch
        let r0 = variant_sharpes_matrix(&ts, &pos, n, &ts, &price, n, 1, 0, false, 0.0);
        let r1 = variant_sharpes_matrix(&ts, &pos, n, &ts, &price, n, 1, 0, false, 1e-9);
        assert!((r0[1] - v0[1]).abs() < 1e-12);
        assert!((r1[1] - v1[1]).abs() < 1e-12);
        // ... and so does sharpe_per_lag with resample=False, in both regimes
        let l0 = sharpe_per_lag_matrix(&ts, &pos, n, &ts, &price, n, 1, &[0], false, 0.0)[0];
        let l1 = sharpe_per_lag_matrix(&ts, &pos, n, &ts, &price, n, 1, &[0], false, 1e-9)[0];
        assert!((l0 - r0[1]).abs() < 1e-12);
        assert!((l1 - r1[1]).abs() < 1e-12);
        let h0 = holding_curve(&ts, &pos, n, &ts, &price, n, 1, 0, &[0.0], 0.0);
        let h1 = holding_curve(&ts, &pos, n, &ts, &price, n, 1, 0, &[0.0], 1e-9);
        assert!((h0[0].sharpe - s0).abs() < 1e-12);
        assert!((h1[0].sharpe - s1).abs() < 1e-12);
    }
}
